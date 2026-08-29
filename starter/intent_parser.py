"""LLM-first natural-language understanding layer for the shopping copilot.

Provides three IntentParser implementations:
- DeterministicIntentParser: minimal fallback, zero network calls
- OpenAIIntentParser: calls OpenAI gpt-4.1-mini for structured intent extraction
- HybridIntentParser: LLM-first -> deterministic fallback

The LLM extracts structured intent; retrieval and ranking remain deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

VALID_MODES = frozenset({"browsing", "buying", "override"})
VALID_OPERATIONS = frozenset({"add", "replace", "remove", "none"})

LLM_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class IntentResult:
    """Structured output from any intent parser."""

    mode: str = "browsing"
    operation: str = "add"
    category: str | None = None
    add_constraints: dict[str, str] = field(default_factory=dict)
    remove_constraints: list[str] = field(default_factory=list)
    negative_constraints: list[str] = field(default_factory=list)
    no_preference: list[str] = field(default_factory=list)
    referenced_previous_item: bool = False
    reference_description: str | None = None
    confidence: float = 0.0
    source: str = "deterministic"
    prompt_tokens: int = 0
    completion_tokens: int = 0


def validate_intent_result(data: object) -> IntentResult | None:
    """Validate a raw dict into an IntentResult. Returns None on invalid input."""
    if not isinstance(data, dict):
        return None

    mode = data.get("mode", "browsing")
    if mode not in VALID_MODES:
        return None

    operation = data.get("operation", "add")
    if operation not in VALID_OPERATIONS:
        return None

    category = data.get("category")
    if category is not None:
        category = str(category).strip() if category else None

    add_constraints = data.get("add_constraints") or {}
    if not isinstance(add_constraints, dict):
        add_constraints = {}
    add_constraints = {
        str(k): str(v)
        for k, v in add_constraints.items()
        if k is not None and v is not None
    }

    def _str_list(raw: object) -> list[str]:
        if not isinstance(raw, list):
            return []
        return [str(x).strip() for x in raw if x is not None and str(x).strip()]

    remove_constraints = _str_list(data.get("remove_constraints"))
    negative_constraints = _str_list(data.get("negative_constraints"))
    no_preference = _str_list(data.get("no_preference"))

    referenced = bool(data.get("referenced_previous_item", False))

    ref_desc = data.get("reference_description")
    if ref_desc is not None:
        ref_desc = str(ref_desc).strip() if ref_desc else None

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    return IntentResult(
        mode=mode,
        operation=operation,
        category=category,
        add_constraints=add_constraints,
        remove_constraints=remove_constraints,
        negative_constraints=negative_constraints,
        no_preference=no_preference,
        referenced_previous_item=referenced,
        reference_description=ref_desc,
        confidence=confidence,
        source="llm",
    )



def load_api_key() -> str | None:
    """Load OPENAI_API_KEY from environment or ``.env`` file. Never logs the key.

    Checks ``os.environ`` first, then falls back to reading from a ``.env``
    file next to the project root.  When found in ``.env`` the value is
    injected into ``os.environ`` so downstream code sees it too.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.is_file():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("OPENAI_API_KEY=") and not stripped.startswith("#"):
                    value = stripped.split("=", 1)[1].strip()
                    if value:
                        os.environ["OPENAI_API_KEY"] = value
                        return value
        except OSError:
            pass
    return None


class IntentParser(ABC):
    """Abstract interface for intent extraction."""

    @abstractmethod
    def parse(self, message: str, conversation_state: dict) -> IntentResult: ...


class DeterministicIntentParser(IntentParser):
    """Regex-based intent parser using the existing IntentClassifier."""

    def __init__(self, classifier: object) -> None:
        self._classifier = classifier

    def parse(self, message: str, conversation_state: dict) -> IntentResult:
        active_query = conversation_state.get("active_query", "")
        mode = self._classifier.classify(message, active_query)
        return IntentResult(mode=mode, confidence=1.0, source="deterministic")


_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"

_SYSTEM_PROMPT = """\
You are a shopping intent parser. Parse the customer message into structured JSON.

Shopping context:
- Current preferences: {active_query}
- Turn: {turn}
- Session mode: {session_mode}
- Last question asked: {last_question}

Return this exact JSON structure:
{{
  "mode": "browsing" or "buying" or "override",
  "operation": "add" or "replace" or "remove" or "none",
  "category": "product category" or null,
  "add_constraints": {{"attribute": "value"}},
  "remove_constraints": ["removed preferences"],
  "negative_constraints": ["things explicitly excluded like materials or colors"],
  "no_preference": ["attributes the customer has no preference for"],
  "referenced_previous_item": true or false,
  "reference_description": "what they reference" or null,
  "confidence": 0.0 to 1.0
}}

Intent rules:
- override = contradicting or replacing earlier stated preferences (e.g. "on second thought", "actually", "instead", "changed my mind")
- buying = stating specific requirements (materials, colors, features, budget)
- browsing = exploring with no firm requirements
- negative_constraints = explicit exclusions ("anything except X", "not Y", "no Z")
- no_preference = explicit indifference ("don't care about color", "not fussed", "doesn't matter")
- confidence = how clearly the intent is expressed (0.0 = very unclear, 1.0 = crystal clear)"""


class OpenAIIntentParser(IntentParser):
    """Calls OpenAI API for structured intent extraction with caching."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        timeout: float = 5.0,
        cache_size: int = 512,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._cache: dict[str, IntentResult] = {}
        self._cache_size = cache_size

    def parse(self, message: str, conversation_state: dict) -> IntentResult:
        system_prompt = _SYSTEM_PROMPT.format(
            active_query=conversation_state.get("active_query") or "none",
            turn=conversation_state.get("turn", 1),
            session_mode=conversation_state.get("session_mode", "browsing"),
            last_question=conversation_state.get("last_question") or "none",
        )

        cache_key = hashlib.sha256(
            (system_prompt + message).encode("utf-8")
        ).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        parsed, prompt_tokens, completion_tokens = self._call_api(
            system_prompt, message
        )
        if parsed is None:
            raise RuntimeError("OpenAI API returned no usable response")

        result = validate_intent_result(parsed)
        if result is None:
            raise ValueError("OpenAI response failed schema validation")

        result.prompt_tokens = prompt_tokens
        result.completion_tokens = completion_tokens

        if len(self._cache) >= self._cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[cache_key] = result
        return result

    def _call_api(
        self, system_prompt: str, user_message: str
    ) -> tuple[dict | None, int, int]:
        body = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")

        request = urllib.request.Request(
            _OPENAI_ENDPOINT,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(
                request, timeout=self._timeout, context=ctx
            ) as resp:
                raw = json.loads(resp.read())
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            TimeoutError,
            OSError,
        ):
            return None, 0, 0

        usage = raw.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))

        choices = raw.get("choices") or []
        if not choices:
            return None, prompt_tokens, completion_tokens

        text = (choices[0].get("message") or {}).get("content", "")
        if not text:
            return None, prompt_tokens, completion_tokens

        try:
            return json.loads(text), prompt_tokens, completion_tokens
        except json.JSONDecodeError:
            return None, prompt_tokens, completion_tokens


class HybridIntentParser(IntentParser):
    """LLM-first intent parser with deterministic fallback.

    All messages are sent to the LLM. On any failure (no API key, timeout,
    malformed response, low confidence) the deterministic parser is used.
    """

    def __init__(
        self,
        deterministic: DeterministicIntentParser,
        llm: OpenAIIntentParser | None = None,
    ) -> None:
        self.deterministic = deterministic
        self.llm = llm

    def parse(self, message: str, conversation_state: dict) -> IntentResult:
        if self.llm is not None:
            try:
                result = self.llm.parse(message, conversation_state)
                if result.confidence >= LLM_CONFIDENCE_THRESHOLD:
                    return result
            except Exception:
                pass

        return self.deterministic.parse(message, conversation_state)
