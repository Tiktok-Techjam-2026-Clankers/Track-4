"""Score the agent against the default and extended local datasets."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    Agent,
    catalog_index,
    evaluate,
    load_jsonl,
)


CATALOG_PATH = REPO_ROOT / "data" / "catalog.jsonl"
DATASETS = (
    ("Default", REPO_ROOT / "data" / "public_set.jsonl"),
    ("Extended", REPO_ROOT / "data" / "private_set.jsonl"),
)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Required file not found: {path.relative_to(REPO_ROOT)}")


def main() -> None:
    require_file(CATALOG_PATH)
    for _, dataset_path in DATASETS:
        require_file(dataset_path)

    catalog_ids, categories, products = catalog_index(CATALOG_PATH)
    agent = Agent(CATALOG_PATH)

    for label, dataset_path in DATASETS:
        samples = load_jsonl(dataset_path)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        print(f"{label} dataset ({result['sample_count']} sessions)")
        print(f"  Technical score: {result['recommended_technical_score']:.6f}")
        print(f"  Hit Rate@10:    {result['hit_rate_at_10']:.6f}")
        print(f"  MRR:            {result['mrr']:.6f}")
        print(f"  MTTC:           {result['mttc']:.6f}")


if __name__ == "__main__":
    main()
