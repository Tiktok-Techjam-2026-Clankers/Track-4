"""Prefetch the offline reranker's ONNX weights into the repo-local ./models cache.

The scored run may execute with the network cut. `starter.ranking.LocalReranker`
loads its cross-encoder from `./models` in offline mode, so the weights must be
present *before* scoring. Run this once, with the network up, from the repo root:

    python scripts/prefetch_models.py

It downloads `Xenova/ms-marco-MiniLM-L-6-v2` (~103 MB) into ./models. The cache
is git-ignored. If fastembed is not installed, the reranker degrades to identity
and the deterministic scores are unchanged — this prefetch is only needed to
actually run the reranking stage offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
CACHE_DIR = REPO_ROOT / "models"


def main() -> int:
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except Exception as exc:  # pragma: no cover - setup helper
        print(f"fastembed not installed ({exc}). `pip install -r requirements.txt` first.")
        return 1

    CACHE_DIR.mkdir(exist_ok=True)
    print(f"Fetching {MODEL} into {CACHE_DIR} ...")
    ce = TextCrossEncoder(model_name=MODEL, cache_dir=str(CACHE_DIR))
    # Force a tiny scoring pass so the weights are materialised, not just listed.
    list(ce.rerank("test query", ["a sample product title"]))
    print("Done. Offline reranker weights are cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
