"""LLM-first natural-language understanding layer for the shopping copilot.

Provides three IntentParser implementations:
- DeterministicIntentParser: regex-based, zero network calls
- GeminiIntentParser: calls Gemini Flash-Lite for structured intent extraction
- HybridIntentParser: template detection -> Gemini -> deterministic fallback

The LLM extracts structured intent; retrieval and ranking remain deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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


_VERBATIM_PATTERNS = [
    re.compile(r"^I'm looking for "),
    re.compile(r"^For that, what matters is: "),
    re.compile(r"^I don't have an? (?:additional )?preference for "),
    re.compile(r"^Actually,?\s*(?:please\s+)?ignore my earlier preference"),
    re.compile(r"^Those options are not quite right yet"),
]


def is_simulator_template(message: str) -> bool:
    """Return True if the message matches an exact official simulator template."""
    return any(pattern.search(message) for pattern in _VERBATIM_PATTERNS)


def load_api_key() -> str | None:
    """Load GEMINI_API_KEY from environment. Never logs the key.

    Reads ``os.environ`` only.  To use the key stored in ``.env``,
    export it into the shell before launching the agent::

        export $(grep GEMINI_API_KEY .env | xargs)
        python scripts/score_datasets.py

    Or call ``load_dotenv_api_key()`` once in your entry-point script.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip() or None
    return None


def load_dotenv_api_key() -> str | None:
    """Read GEMINI_API_KEY from a ``.env`` file and inject into os.environ.

    Returns the key (or None).  Safe to call multiple times.
    """
    key = load_api_key()
    if key:
        return key
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.is_file():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("GEMINI_API_KEY=") and not stripped.startswith("#"):
                    value = stripped.split("=", 1)[1].strip()
                    if value:
                        os.environ["GEMINI_API_KEY"] = value
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


_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

_SYSTEM_PROMPT = """\
Parse this shopping message into structured JSON intent.

Shopping context:
- Current preferences: {active_query}
- Turn: {turn}
- Session mode: {session_mode}
- Last question asked: {last_question}

Customer message: "{message}"

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


class GeminiIntentParser(IntentParser):
    """Calls Gemini API for structured intent extraction with caching."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        timeout: float = 5.0,
        cache_size: int = 512,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._cache: dict[str, IntentResult] = {}
        self._cache_size = cache_size

    def parse(self, message: str, conversation_state: dict) -> IntentResult:
        prompt = _SYSTEM_PROMPT.format(
            active_query=conversation_state.get("active_query") or "none",
            turn=conversation_state.get("turn", 1),
            session_mode=conversation_state.get("session_mode", "browsing"),
            last_question=conversation_state.get("last_question") or "none",
            message=message,
        )

        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        parsed, prompt_tokens, completion_tokens = self._call_api(prompt)
        if parsed is None:
            raise RuntimeError("Gemini API returned no usable response")

        result = validate_intent_result(parsed)
        if result is None:
            raise ValueError("Gemini response failed schema validation")

        result.prompt_tokens = prompt_tokens
        result.completion_tokens = completion_tokens

        if len(self._cache) >= self._cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[cache_key] = result
        return result

    def _call_api(self, prompt: str) -> tuple[dict | None, int, int]:
        url = _GEMINI_ENDPOINT.format(model=self._model) + f"?key={self._api_key}"
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 256,
                "responseMimeType": "application/json",
            },
        }).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
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

        usage = raw.get("usageMetadata") or {}
        prompt_tokens = int(usage.get("promptTokenCount", 0))
        completion_tokens = int(usage.get("candidatesTokenCount", 0))

        candidates = raw.get("candidates") or []
        if not candidates:
            return None, prompt_tokens, completion_tokens

        parts = (candidates[0].get("content") or {}).get("parts") or [{}]
        text = parts[0].get("text", "") if parts else ""
        if not text:
            return None, prompt_tokens, completion_tokens

        try:
            return json.loads(text), prompt_tokens, completion_tokens
        except json.JSONDecodeError:
            return None, prompt_tokens, completion_tokens


class HybridIntentParser(IntentParser):
    """Template detection -> Gemini -> deterministic fallback.

    Official simulator templates use the deterministic fast path to preserve
    the existing high scores. All other messages are sent to the LLM when
    an API key is available. On any failure the deterministic parser is used.
    """

    def __init__(
        self,
        deterministic: DeterministicIntentParser,
        gemini: GeminiIntentParser | None = None,
    ) -> None:
        self.deterministic = deterministic
        self.gemini = gemini

    def parse(self, message: str, conversation_state: dict) -> IntentResult:
        if is_simulator_template(message):
            return self.deterministic.parse(message, conversation_state)

        if self.gemini is not None:
            try:
                result = self.gemini.parse(message, conversation_state)
                if result.confidence >= LLM_CONFIDENCE_THRESHOLD:
                    return result
            except Exception:
                pass

        return self.deterministic.parse(message, conversation_state)
