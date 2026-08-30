"""Score the agent against the default and extended local datasets."""

from __future__ import annotations

import argparse
import sys
import time
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

HEADERS = ("dataset", "n", "Hit@10", "MRR", "MTTC", "Eff", "Score")
WIDTHS = (16, 5, 8, 8, 8, 8, 8)
KEYS = ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Required file not found: {path.relative_to(REPO_ROOT)}")


def format_row(label: str, result: dict) -> str:
    cells = [
        label,
        str(result["sample_count"]),
        *(f"{float(result[k]):.4f}" for k in KEYS),
    ]
    return "  ".join(c.rjust(w) for c, w in zip(cells, WIDTHS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable all LLM calls; run the deterministic pipeline only.",
    )
    args = parser.parse_args()

    require_file(CATALOG_PATH)
    for _, dataset_path in DATASETS:
        require_file(dataset_path)

    catalog_ids, categories, products = catalog_index(CATALOG_PATH)

    t0 = time.perf_counter()
    agent = Agent(CATALOG_PATH, use_llm=not args.no_llm)
    startup = time.perf_counter() - t0

    prompt_tokens = 0
    completion_tokens = 0
    rows: list[str] = []

    for label, dataset_path in DATASETS:
        samples = load_jsonl(dataset_path)
        result = evaluate(agent, samples, catalog_ids, categories, products)
        rows.append(format_row(label, result))
        usage = result.get("reported_token_usage", {})
        prompt_tokens += usage.get("prompt_tokens", 0)
        completion_tokens += usage.get("completion_tokens", 0)

    elapsed = time.perf_counter() - t0

    print("Evaluator: evaluate-datasets")
    print("=" * 65)
    print()
    print("  ".join(h.rjust(w) for h, w in zip(HEADERS, WIDTHS)))
    print("  ".join("-" * w for w in WIDTHS))
    for row in rows:
        print(row)
    total_tokens = prompt_tokens + completion_tokens
    print()
    print(f"Tokens: {prompt_tokens:,} prompt · {completion_tokens:,} completion · {total_tokens:,} total")
    print(f"Time:   {elapsed:.1f}s (startup {startup:.1f}s)")


if __name__ == "__main__":
    main()
