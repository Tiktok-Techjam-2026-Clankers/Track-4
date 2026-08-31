"""Measure a PRETRAINED sentence-embedding semantic route vs the lexical hash.

The deterministic `semantic` route is `starter.retrieval.InMemoryVectorIndex`, a
feature-hashing bag-of-words encoder — it can only match shared tokens, so it
fails under paraphrase drift. This harness swaps in a pretrained sentence
embedding (fastembed bge-small, precomputed by scripts/precompute_embeddings.py)
at the *same* seam and scores it on the official + robust sets. `starter/` is
not modified; the index is plugged into a constructed agent for measurement.

    lexical     baseline InMemoryVectorIndex (feature-hashing)
    pretrained  cosine over precomputed bge-small catalog vectors
    blend       RRF fusion of the pretrained order with the lexical order

    .venv/bin/python scripts/eval_semantic_embed.py --mode lexical pretrained
    .venv/bin/python scripts/eval_semantic_embed.py --mode blend --alpha 0.5
    .venv/bin/python scripts/eval_semantic_embed.py --mode pretrained --robust

Titles/attributes only enter the model — never parent_asin (leakage rule #3).
Scoring reuses fast_eval (official) and evaluate_robust, so it cannot drift.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from scripts.fast_eval import (  # noqa: E402
    Agent, CATALOG_PATH, HEADERS, WIDTHS, format_row, run_session, summarize,
)
from starter.text_utils import RRF_K, SEMANTIC_POOL  # noqa: E402

CACHE_DIR = REPO_ROOT / "models"


class _Embedder:
    """Process-wide singleton around a fastembed query embedder (thread-safe)."""

    _lock = threading.Lock()
    _model = None
    _name = None

    @classmethod
    def get(cls, model_name):
        with cls._lock:
            if cls._model is None or cls._name != model_name:
                from fastembed import TextEmbedding
                cls._model = TextEmbedding(model_name=model_name, cache_dir=str(CACHE_DIR))
                cls._name = model_name
            return cls._model

    @classmethod
    def query_vec(cls, model_name, query):
        model = cls.get(model_name)
        with cls._lock:
            vec = np.asarray(list(model.query_embed([query]))[0], dtype=np.float32)
        n = float(np.linalg.norm(vec))
        return vec / n if n else vec


class PretrainedVectorIndex:
    """Drop-in for InMemoryVectorIndex backed by precomputed bge-small vectors.

    ``alpha`` blends the pretrained order with the lexical index's order via
    weighted RRF (1.0 = pure pretrained, 0.0 = pure lexical). ``lexical`` is the
    original index, used only when alpha < 1.0.
    """

    def __init__(self, model_name, lexical=None, alpha=1.0):
        slug = model_name.split("/")[-1].replace(".", "_")
        data = np.load(CACHE_DIR / f"catalog_{slug}.npz", allow_pickle=True)
        self.identifiers = data["identifiers"]
        self.vectors = data["vectors"].astype(np.float32)
        self._model_name = model_name
        self._lexical = lexical
        self._alpha = max(0.0, min(1.0, float(alpha)))
        # Pre-embed the query for the singleton warm-up.
        _Embedder.get(model_name)

    def _pretrained_order(self, query, limit):
        q = _Embedder.query_vec(self._model_name, query)
        if not np.any(q):
            return []
        scores = self.vectors @ q
        count = min(limit, len(scores))
        if count <= 0:
            return []
        cand = np.argpartition(scores, -count)[-count:]
        ordered = cand[np.argsort(scores[cand])[::-1]]
        return [str(self.identifiers[i]) for i in ordered]

    def search(self, query, limit=SEMANTIC_POOL):
        if not query.strip() or len(self.identifiers) == 0:
            return []
        pre = self._pretrained_order(query, limit)
        if self._alpha >= 1.0 or self._lexical is None:
            return pre
        lex = self._lexical.search(query, limit)
        points = {}
        for rank, idf in enumerate(pre, 1):
            points[idf] = points.get(idf, 0.0) + self._alpha / (RRF_K + rank)
        for rank, idf in enumerate(lex, 1):
            points[idf] = points.get(idf, 0.0) + (1.0 - self._alpha) / (RRF_K + rank)
        ordered = sorted(points, key=lambda i: -points[i])
        return ordered[:limit]


def build_agent(mode, model_name, alpha):
    agent = Agent(CATALOG_PATH, use_llm=False)
    if mode == "lexical":
        return agent
    a = 1.0 if mode == "pretrained" else alpha
    agent.semantic = PretrainedVectorIndex(model_name, lexical=agent.semantic, alpha=a)
    return agent


def _run_official(agent, loaded, catalog_ids, categories, products, workers):
    print("  ".join(h.rjust(w) for h, w in zip(HEADERS, WIDTHS)))
    print("  ".join("-" * w for w in WIDTHS))
    for label, samples in loaded:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            sessions = list(pool.map(
                lambda s: run_session(agent, s, catalog_ids, categories, products),
                samples,
            ))
        print(format_row(label, len(samples), summarize(sessions)))


def _run_robust(agent, splits):
    from scripts.evaluate_robust import evaluate, format_row as rformat
    cat_ids = {str(json.loads(l)["parent_asin"])
               for l in Path(CATALOG_PATH).open() if l.strip()}
    for split in splits:
        cases = [json.loads(l) for l in (REPO_ROOT / "data" / "robust" / f"{split}.jsonl").open() if l.strip()]
        result = evaluate(agent, cases, cat_ids)
        print(rformat(split, result))
        for name, m in result["scenario_metrics"].items():
            print(rformat(f"  {name}", m))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", nargs="+", default=["lexical", "pretrained"],
                    choices=["lexical", "pretrained", "blend"])
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--alpha", type=float, default=0.5, help="blend: pretrained vs lexical")
    ap.add_argument("--robust", action="store_true", help="also score robust splits")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    catalog_ids, categories, products = catalog_index(CATALOG_PATH)
    loaded = []
    for label, path in (("Default", REPO_ROOT / "data" / "public_set.jsonl"),
                        ("Extended", REPO_ROOT / "data" / "private_set.jsonl")):
        s = load_jsonl(path)
        loaded.append((label, s[: args.limit] if args.limit else s))

    print(f"Evaluator: eval-semantic-embed  (model={args.model}, alpha={args.alpha})")
    print("=" * 74)
    for mode in args.mode:
        t0 = time.perf_counter()
        agent = build_agent(mode, args.model, args.alpha)
        tag = mode if mode != "blend" else f"blend(alpha={args.alpha})"
        print(f"\n--- semantic = {tag} ---")
        _run_official(agent, loaded, catalog_ids, categories, products, args.workers)
        if args.robust:
            _run_robust(agent, ("stress", "validation"))
        print(f"  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
