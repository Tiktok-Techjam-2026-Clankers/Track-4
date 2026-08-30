"""Fusion, dynamic RRF weights, LLM reranking, and response assembly.

Holds ``LLMReranker`` (semantic reorder via OpenAI, with deterministic
fallback), ``HybridRanker`` (RRF fusion + dynamic weight computation), and
``ResponseBuilder`` (final response contract). ``call_openai`` is imported
here so tests patch ``starter.ranking.call_openai``.
"""

from __future__ import annotations

import hashlib
import threading

from starter.intent_parser import call_openai
from starter.text_utils import *  # noqa: F401,F403 — shared helpers/constants


_RERANK_PROMPT = """\
You are a product reranker. Given the shopper's requirements and a numbered list \
of products, return a JSON object with a single key "order" containing the product \
numbers reordered by relevance (most relevant first).

Shopper requirements: {query}
User profile: {user_summary}
Preference tags: {preference_tags}

Return: {{"order": [3, 1, 5, ...]}}"""


class LLMReranker:
    """Reranks top candidates via an LLM call. Falls back to original order on failure."""

    def __init__(
        self,
        api_key: str,
        product_titles: dict[str, str],
        model: str = "gpt-4.1-mini",
        timeout: float = 3.0,
        cache_size: int = 256,
        on_failure: "callable | None" = None,
    ) -> None:
        self._api_key = api_key
        self._product_titles = product_titles
        self._model = model
        self._timeout = timeout
        self._cache: dict[str, list[int]] = {}
        self._cache_size = cache_size
        self._cache_lock = threading.Lock()
        self._on_failure = on_failure

    def rerank(
        self,
        ranked: list[tuple[str, float]],
        query: str,
        user_profile: dict,
        limit: int = 20,
    ) -> tuple[list[tuple[str, float]], int, int]:
        """Rerank top candidates. Returns (reranked_list, prompt_tokens, completion_tokens)."""
        if len(ranked) <= 1:
            return ranked, 0, 0

        head = ranked[:limit]
        items = []
        for i, (identifier, _) in enumerate(head, 1):
            title = self._product_titles.get(identifier, "unknown product")[:120]
            items.append(f"{i}. {title}")
        product_list = "\n".join(items)

        tags = user_profile.get("preference_tags") or []
        system_prompt = _RERANK_PROMPT.format(
            query=query[:300],
            user_summary=str(user_profile.get("summary", "none"))[:200],
            preference_tags=", ".join(tags) if tags else "none",
        )

        cache_key = hashlib.sha256(
            (system_prompt + product_list).encode("utf-8")
        ).hexdigest()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            reordered = self._apply_order(cached, head, ranked[limit:])
            return reordered, 0, 0

        parsed, pt, ct = call_openai(
            self._api_key, system_prompt, product_list,
            model=self._model, timeout=self._timeout, max_tokens=128,
        )
        if parsed is None:
            # Hard failure (network/transport/empty response) — signal the
            # latch so the whole run drops to deterministic mode.
            if self._on_failure is not None:
                self._on_failure()
            return ranked, pt, ct

        order = parsed.get("order")
        if not isinstance(order, list):
            return ranked, pt, ct

        try:
            indices = [int(x) for x in order]
        except (TypeError, ValueError):
            return ranked, pt, ct

        with self._cache_lock:
            if len(self._cache) >= self._cache_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[cache_key] = indices

        reordered = self._apply_order(indices, head, ranked[limit:])
        return reordered, pt, ct

    @staticmethod
    def _apply_order(
        indices: list[int],
        head: list[tuple[str, float]],
        tail: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        seen: set[int] = set()
        reordered: list[tuple[str, float]] = []
        for idx in indices:
            pos = idx - 1
            if 0 <= pos < len(head) and pos not in seen:
                seen.add(pos)
                reordered.append(head[pos])
        for pos in range(len(head)):
            if pos not in seen:
                reordered.append(head[pos])
        return reordered + list(tail)


class HybridRanker:
    """Intent-aware fusion of retrieval, semantic, phrase, and prior ranks."""

    WEIGHTS = {
        "buying": (0.25, 0.10, 0.50, 0.40),
        "browsing": (0.25, 0.10, 0.50, 0.50),
        "override": (0.25, 0.10, 0.50, 0.50),
        "boundary": (0.25, 0.10, 0.50, 0.30),
    }
    LEXICAL_QUOTA = 1
    FUSION_K = {
        "buying": 20,
        "browsing": 10,
        "override": 20,
        "boundary": 10,
    }

    @classmethod
    def compute_weights(
        cls,
        intent: str,
        turn: int = 1,
        constraint_count: int = 0,
        is_post_override: bool = False,
        phase_turn: int = 1,
        has_preference_tags: bool = False,
    ) -> tuple[tuple[float, float, float, float], int]:
        """Compute dynamic RRF weights and fusion_k based on session state."""
        base = cls.WEIGHTS.get(intent, cls.WEIGHTS["browsing"])
        bm25_w, sem_w, ev_w, pop_w = base
        base_k = cls.FUSION_K.get(intent, RRF_K)

        if turn >= 5:
            ev_w += 0.15
            bm25_w -= 0.10
        if constraint_count >= 3:
            ev_w += 0.10
            sem_w -= 0.05
        if is_post_override and phase_turn <= 2:
            sem_w += 0.15
            ev_w -= 0.10
        if has_preference_tags:
            sem_w += 0.05

        clamp = lambda v: max(0.05, min(1.0, v))
        weights = (clamp(bm25_w), clamp(sem_w), clamp(ev_w), clamp(pop_w))
        fusion_k = max(5, base_k - 2 * constraint_count)
        return weights, fusion_k

    @classmethod
    def fuse(
        cls,
        bm25: list[str],
        semantic: list[str],
        phrase: list[str],
        popularity: list[str],
        intent: str,
        limit: int,
        weights: tuple[float, float, float, float] | None = None,
        fusion_k: int | None = None,
    ) -> list[tuple[str, float]]:
        rankings = (bm25, semantic, phrase, popularity)
        if weights is None:
            weights = cls.WEIGHTS.get(intent, cls.WEIGHTS["browsing"])
        if fusion_k is None:
            fusion_k = cls.FUSION_K.get(intent, RRF_K)
        points: dict[str, float] = {}
        first_seen: dict[str, int] = {}
        sequence = 0
        for weight, ranking in zip(weights, rankings):
            for rank, identifier in enumerate(ranking, start=1):
                first_seen.setdefault(identifier, sequence)
                sequence += 1
                points[identifier] = points.get(identifier, 0.0) + weight / (fusion_k + rank)
        ordered = sorted(points, key=lambda item: (-points[item], first_seen[item]))
        if cls.LEXICAL_QUOTA > 0 and limit > 1:
            protected = [
                identifier for identifier in bm25[: cls.LEXICAL_QUOTA]
                if identifier not in ordered[:limit]
            ]
            protected = protected[: min(cls.LEXICAL_QUOTA, limit - 1)]
            head = ordered[: limit - len(protected)]
            selected = set([*head, *protected])
            ordered = [
                *head,
                *protected,
                *(identifier for identifier in ordered if identifier not in selected),
            ]
        return [(identifier, points[identifier]) for identifier in ordered[:limit]]


class ResponseBuilder:
    QUESTIONS = {
        "other": "What else matters most to you about the product?",
        "material": "Do you have a preferred material?",
        "color": "Which colour would you prefer?",
        "style": "What style are you looking for?",
        "use_case": "What will you mainly use it for?",
        "feature": "Is there a must-have feature?",
        "budget": "What is your budget?",
        "brand": "Do you have a preferred brand?",
        "size": "What sizing requirement should I consider?",
        "category": "Which product category is closest to what you need?",
    }

    @classmethod
    def build(
        cls,
        ranked: list[tuple[str, float]],
        ask_attribute: str,
        intent: str,
    ) -> dict:
        prefix = (
            "I refined the shortlist around your requirements. "
            if intent in {"buying", "override"}
            else "Here is a diverse shortlist to explore. "
        )
        return {
            "message": prefix + cls.QUESTIONS[ask_attribute],
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": identifier, "score": round(float(score), 8)}
                for identifier, score in ranked
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
