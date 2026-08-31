"""Precompute pretrained sentence-embeddings for the catalog (offline cache).

Measurement support for the "pretrained semantic embeddings vs the lexical
feature-hashing SemanticEncoder" experiment. Embeds each product's text with a
fastembed sentence-embedding model (bge-small, ONNX/CPU) once and caches the
L2-normalised matrix + identifier order to ./models, so the eval harness and a
would-be scored path load vectors instantly instead of re-embedding 50k docs.

    python scripts/precompute_embeddings.py            # bge-small-en-v1.5
    python scripts/precompute_embeddings.py --model BAAI/bge-small-en-v1.5

Titles/attributes only enter the model — never parent_asin (leakage rule #3);
the identifier order is stored separately as an index, not fed to the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CATALOG = REPO_ROOT / "data" / "catalog.jsonl"
CACHE_DIR = REPO_ROOT / "models"
DOC_CHARS = 256  # embed a truncated doc; title/attrs dominate retrieval signal


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_text(v) for v in value.values())
    return str(value)


def _document(product: dict) -> str:
    title = _text(product.get("title"))
    categories = _text(product.get("categories"))
    features = _text(product.get("features"))
    description = _text(product.get("description"))
    # Mirror the lexical index's field emphasis (title + categories weighted),
    # truncated so we embed the discriminative head, not boilerplate tails.
    return " ".join((title, title, categories, features, description))[:DOC_CHARS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    try:
        from fastembed import TextEmbedding
    except Exception as exc:
        print(f"fastembed not installed ({exc}). pip install -r requirements.txt")
        return 1

    ids: list[str] = []
    docs: list[str] = []
    for line in CATALOG.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        product = json.loads(line)
        ids.append(str(product["parent_asin"]))
        docs.append(_document(product))
    print(f"catalog: {len(ids)} products")

    CACHE_DIR.mkdir(exist_ok=True)
    model = TextEmbedding(model_name=args.model, cache_dir=str(CACHE_DIR))
    print(f"embedding with {args.model} ...")
    t0 = time.perf_counter()
    vectors = np.zeros((len(docs), 0), dtype=np.float32)
    rows = []
    for i, emb in enumerate(model.embed(docs, batch_size=args.batch)):
        rows.append(np.asarray(emb, dtype=np.float32))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(docs)}  ({time.perf_counter() - t0:.0f}s)")
    vectors = np.vstack(rows)
    # L2-normalise so cosine == dot product.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    slug = args.model.split("/")[-1].replace(".", "_")
    out = CACHE_DIR / f"catalog_{slug}.npz"
    np.savez(out, identifiers=np.asarray(ids, dtype=object), vectors=vectors)
    print(f"saved {out}  shape={vectors.shape}  ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
