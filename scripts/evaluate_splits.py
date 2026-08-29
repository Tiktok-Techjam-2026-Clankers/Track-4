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
    started = time.perf_counter()
    agent = Agent(catalog_path)
    startup_seconds = time.perf_counter() - started
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"startup_seconds": round(startup_seconds, 6), "splits": {}}
    for name in args.splits:
        split_started = time.perf_counter()
        result = evaluate(
            agent,
            load_jsonl(DEFAULT_SPLITS[name]),
            catalog_ids,
            categories,
            products,
        )
        elapsed = time.perf_counter() - split_started
        (output_dir / f"{name}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        summary["splits"][name] = {
            key: result[key]
            for key in (
                "sample_count",
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "efficiency",
                "recommended_technical_score",
                "scenario_metrics",
            )
        }
        summary["splits"][name]["elapsed_seconds"] = round(elapsed, 6)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
