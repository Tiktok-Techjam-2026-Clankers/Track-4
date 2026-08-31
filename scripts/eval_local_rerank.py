"""Measure a LOCAL pretrained cross-encoder reranker vs the OpenAI reranker.

The shipped LLM reranker (`starter/ranking.LLMReranker`) reorders the plain-path
fusion window via an OpenAI call. This harness swaps in an offline HuggingFace
cross-encoder (fastembed `TextCrossEncoder`, ms-marco-MiniLM) at the *same* seam
and scores it against the official metric, so we can answer: is a local model
better than the OpenAI call for reranking? `starter/` is not modified — the
reranker is plugged into a constructed agent for measurement only.

Configs (all share the deterministic retrieval + intent path, so the ONLY
variable is the reranker; the OpenAI-intent path is disabled even in --mode
openai, isolating the reranker's contribution):

    none    reranker disabled                      (deterministic baseline)
    local   local cross-encoder reorders the window (offline, 0 tokens)
    openai  OpenAI LLMReranker reorders the window  (needs key; intent parser
                                                     forced deterministic)

    .venv/bin/python scripts/eval_local_rerank.py --mode none local
    .venv/bin/python scripts/eval_local_rerank.py --mode local --blend 0.5 --window 30
    .venv/bin/python scripts/eval_local_rerank.py --mode openai      # head-to-head

Scoring reuses scripts/fast_eval.py (official evaluator), so it cannot drift.
Titles only ever enter the cross-encoder — never parent_asin (leakage rule #3).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from scripts.fast_eval import (  # noqa: E402
    Agent,
    CATALOG_PATH,
    HEADERS,
    WIDTHS,
    format_row,
    run_session,
    summarize,
)
from starter.text_utils import RRF_K  # noqa: E402

DATASETS = (
    ("Default", REPO_ROOT / "data" / "public_set.jsonl"),
)


class LocalReranker:
    """Offline cross-encoder reranker; duck-types LLMReranker.rerank().

    Reorders the incoming (deterministic) fusion window by cross-encoder
    relevance of (query, title). `blend` fuses the cross-encoder order with the
    incoming fusion rank via weighted RRF (1.0 = pure cross-encoder, 0.0 = no-op
    = identity). Falls back to identity order if the model can't load, so it can
    never regress the deterministic baseline.
    """

    def __init__(self, product_titles, model="Xenova/ms-marco-MiniLM-L-6-v2",
                 cache_dir="models", blend=1.0, include_tags=False):
        self._titles = product_titles
        self._blend = max(0.0, min(1.0, float(blend)))
        self._include_tags = include_tags
        self._lock = threading.Lock()
        self._cache: dict[tuple, list[int]] = {}
        self.available = False
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._ce = TextCrossEncoder(model_name=model, cache_dir=cache_dir)
            self.available = True
        except Exception as exc:  # pragma: no cover - measurement harness
            print(f"[LocalReranker] unavailable ({exc}); identity fallback")
            self._ce = None

    def rerank(self, ranked, query, user_profile, limit=30):
        if not self.available or len(ranked) <= 1:
            return ranked, 0, 0
        head = ranked[:limit]
        tail = ranked[limit:]
        titles = [self._titles.get(idf, "unknown product")[:160] for idf, _ in head]
        q = query
        if self._include_tags:
            tags = user_profile.get("preference_tags") or []
            if tags:
                q = f"{query} ({', '.join(tags)})"
        key = (q, tuple(titles))
        with self._lock:
            order = self._cache.get(key)
            if order is None:
                scores = list(self._ce.rerank(q, titles))
                # cross-encoder rank: positions sorted by score desc
                order = sorted(range(len(head)), key=lambda i: -float(scores[i]))
                if len(self._cache) > 4096:
                    self._cache.clear()
                self._cache[key] = order
        reordered = self._fuse(order, head)
        return reordered + list(tail), 0, 0

    def _fuse(self, ce_order, head):
        n = len(head)
        if self._blend >= 1.0:
            return [head[i] for i in ce_order]
        ce_rank = {pos: r for r, pos in enumerate(ce_order)}
        k = RRF_K
        scored = sorted(
            range(n),
            key=lambda det: -(
                self._blend / (k + ce_rank.get(det, n) + 1)
                + (1.0 - self._blend) / (k + det + 1)
            ),
        )
        # stable tie-break on det position via secondary sort key already implicit
        return [head[i] for i in scored]


def build_agent(mode, blend, window, include_tags):
    use_llm = mode == "openai"
    agent = Agent(CATALOG_PATH, use_llm=use_llm)
    if mode == "none":
        agent.reranker = None
    elif mode == "local":
        agent.reranker = LocalReranker(
            agent.product_titles, blend=blend, include_tags=include_tags
        )
        if not agent.reranker.available:
            raise SystemExit("cross-encoder unavailable")
    elif mode == "openai":
        if agent.reranker is None:
            raise SystemExit("no OpenAI key — cannot run --mode openai")
        # Isolate the reranker: force deterministic intent parsing and disable
        # the whole-run latch so a transient timeout doesn't null the reranker.
        agent.intent_parser.llm = None
        agent._latch_llm_off = lambda: None
        agent.reranker._timeout = 20.0
    return agent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", nargs="+", default=["none", "local"],
                    choices=["none", "local", "openai"])
    ap.add_argument("--blend", type=float, default=1.0, help="local: cross-encoder vs fusion prior")
    ap.add_argument("--window", type=int, default=30, help="rerank window (patches RERANK_WINDOW)")
    ap.add_argument("--include-tags", action="store_true", help="local: append preference tags to query")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # Widen/narrow the rerank window by patching the constant the agent reads.
    import starter.agent as agent_mod
    agent_mod.RERANK_WINDOW = args.window

    catalog_ids, categories, products = catalog_index(CATALOG_PATH)
    loaded = []
    for label, path in DATASETS:
        s = load_jsonl(path)
        if args.limit:
            s = s[: args.limit]
        loaded.append((label, s))

    print(f"Evaluator: eval-local-rerank  (window={args.window}, blend={args.blend}, tags={args.include_tags})")
    print("=" * 74)
    for mode in args.mode:
        t0 = time.perf_counter()
        agent = build_agent(mode, args.blend, args.window, args.include_tags)
        tag = mode
        if mode == "local":
            tag = f"local(blend={args.blend})"
        print(f"\n--- reranker = {tag} ---")
        print("  ".join(h.rjust(w) for h, w in zip(HEADERS, WIDTHS)))
        print("  ".join("-" * w for w in WIDTHS))
        pt = ct = 0
        for label, samples in loaded:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                sessions = list(pool.map(
                    lambda s: run_session(agent, s, catalog_ids, categories, products),
                    samples,
                ))
            result = summarize(sessions)
            pt += sum(s["prompt_tokens"] for s in sessions)
            ct += sum(s["completion_tokens"] for s in sessions)
            print(format_row(label, len(samples), result))
        print(f"tokens: {pt:,} prompt · {ct:,} completion   ({time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
