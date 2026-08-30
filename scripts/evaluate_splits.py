"""Evaluate one Agent instance across named datasets with locked-test safety."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluator.local_evaluator import Agent, catalog_index, evaluate, load_jsonl


DEFAULT_SPLITS = {
    "public": Path("data/public_set.jsonl"),
    "train": Path("data/synthetic/train.jsonl"),
    "validation": Path("data/synthetic/validation.jsonl"),
    "test": Path("data/synthetic/test.jsonl"),
    "stress": Path("data/synthetic/stress.jsonl"),
}

HEADERS = ("split", "n", "Hit@10", "MRR", "MTTC", "Eff", "Score")
WIDTHS = (16, 5, 8, 8, 8, 8, 8)
KEYS = ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")


def format_row(label: str, result: dict) -> str:
    cells = [
        label,
        str(result["sample_count"]),
        *(f"{float(result[k]):.4f}" for k in KEYS),
    ]
    return "  ".join(c.rjust(w) for c, w in zip(cells, WIDTHS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("splits", nargs="+", choices=sorted(DEFAULT_SPLITS))
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output-dir", default="benchmark-results")
    parser.add_argument(
        "--allow-locked-test",
        action="store_true",
        help="required to evaluate the synthetic test split",
    )
    args = parser.parse_args()
    if "test" in args.splits and not args.allow_locked_test:
        raise SystemExit("test is locked; freeze configuration and pass --allow-locked-test")

    catalog_path = Path(args.catalog)
    catalog_ids, categories, products = catalog_index(catalog_path)
    t0 = time.perf_counter()
    agent = Agent(catalog_path)
    startup = time.perf_counter() - t0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_tokens = 0
    completion_tokens = 0
    rows: list[str] = []

    for name in args.splits:
        result = evaluate(
            agent,
            load_jsonl(DEFAULT_SPLITS[name]),
            catalog_ids,
            categories,
            products,
        )
        (output_dir / f"{name}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        rows.append(format_row(name, result))
        usage = result.get("reported_token_usage", {})
        prompt_tokens += usage.get("prompt_tokens", 0)
        completion_tokens += usage.get("completion_tokens", 0)

    elapsed = time.perf_counter() - t0

    print("Evaluator: evaluate-splits")
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
