"""Run custom evaluator 1: one agent, several customer personas."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluator.local_evaluator import Agent, catalog_index, load_jsonl  # noqa: E402
from evaluator.custom_evaluator_1_personas import CONTROL_PERSONA, PERSONAS  # noqa: E402
from evaluator.custom_evaluator_1_runner import delta_against, run_persona  # noqa: E402


COLUMNS = ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")
HEADERS = ("persona", "n", "Hit@10", "MRR", "MTTC", "Eff", "Score", "d(Score)")


def relative_label(path: Path) -> str:
    """Repo-relative name when the path sits inside the repo, else the path as given."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_agent(catalog_path: Path) -> Agent:
    try:
        return Agent(catalog_path)
    except TypeError:
        return Agent()


def stratified_sample(samples: list[dict], limit: int | None) -> list[dict]:
    """Keep the scenario mix when trimming, and keep the file's ordering."""
    if limit is None or limit >= len(samples):
        return samples
    buckets: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        buckets.setdefault(str(sample.get("scenario_type")), []).append(index)
    chosen: set[int] = set()
    queues = [buckets[name] for name in sorted(buckets)]
    position = 0
    while len(chosen) < limit and any(position < len(queue) for queue in queues):
        for queue in queues:
            if position < len(queue) and len(chosen) < limit:
                chosen.add(queue[position])
        position += 1
    return [sample for index, sample in enumerate(samples) if index in chosen]


def format_table(results: dict[str, dict]) -> str:
    control = results.get(CONTROL_PERSONA)
    widths = [max(len(HEADERS[0]), *(len(name) for name in results)), 5, 8, 8, 8, 8, 8, 9]
    lines = ["  ".join(header.rjust(width) for header, width in zip(HEADERS, widths))]
    lines.append("  ".join("-" * width for width in widths))
    for name, result in results.items():
        delta = "-" if control is None or name == CONTROL_PERSONA else (
            f"{delta_against(control, result)['recommended_technical_score']:+.4f}"
        )
        cells = [
            name,
            str(result["sample_count"]),
            *(f"{float(result[column]):.4f}" for column in COLUMNS),
            delta,
        ]
        lines.append("  ".join(cell.rjust(width) for cell, width in zip(cells, widths)))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom customer-phrasing evaluator 1")
    parser.add_argument("--catalog", default=REPO_ROOT / "data" / "catalog.jsonl", type=Path)
    parser.add_argument("--dataset", default=REPO_ROOT / "data" / "public_set.jsonl", type=Path)
    parser.add_argument(
        "--output", default=REPO_ROOT / "custom_evaluator_1_results.json", type=Path
    )
    parser.add_argument("--personas", nargs="+", default=list(PERSONAS), choices=list(PERSONAS))
    parser.add_argument("--limit", type=int, default=None, help="stratified subsample size")
    parser.add_argument("--transcripts", action="store_true", help="store per-turn dialogue in the report")
    parser.add_argument("--timings", action="store_true", help="record per-turn latency (non-deterministic)")
    parser.add_argument(
        "--shared-agent", action="store_true",
        help="reuse one Agent across personas instead of rebuilding per persona",
    )
    args = parser.parse_args()

    # Resolve before the run so a bad path fails now rather than after the last session.
    args.catalog = args.catalog.expanduser().resolve()
    args.dataset = args.dataset.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    for path in (args.catalog, args.dataset):
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")
    if not args.output.parent.is_dir():
        raise SystemExit(f"Output directory does not exist: {args.output.parent}")

    samples = stratified_sample(load_jsonl(args.dataset), args.limit)
    catalog_ids, categories, products = catalog_index(args.catalog)
    t0 = time.perf_counter()
    shared = build_agent(args.catalog) if args.shared_agent else None
    startup = time.perf_counter() - t0

    results: dict[str, dict] = {}
    for name in args.personas:
        agent = shared if shared is not None else build_agent(args.catalog)
        results[name] = run_persona(
            agent, PERSONAS[name], samples, catalog_ids, categories, products,
            transcripts=args.transcripts, collect_timings=args.timings,
        )

    control = results.get(CONTROL_PERSONA)
    report = {
        "harness": "custom_evaluator_1",
        "dataset": relative_label(args.dataset),
        "sample_count": len(samples),
        "control_persona": CONTROL_PERSONA if control else None,
        "personas": results,
        "deltas_vs_control": {
            name: delta_against(control, result)
            for name, result in results.items()
            if control and name != CONTROL_PERSONA
        },
    }
    elapsed = time.perf_counter() - t0
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    prompt_tokens = sum(r.get("reported_token_usage", {}).get("prompt_tokens", 0) for r in results.values())
    completion_tokens = sum(r.get("reported_token_usage", {}).get("completion_tokens", 0) for r in results.values())

    print("Evaluator: evaluate-personas")
    print("=" * 65)
    print(f"\nDataset: {report['dataset']}  ({len(samples)} sessions per persona)")
    print()
    print(format_table(results))
    for name, result in results.items():
        if result["contract_violations"]:
            print(f"\ncontract violations [{name}]: {json.dumps(result['contract_violations'])}")
    if args.timings:
        for name, result in results.items():
            if "timings_ms" in result:
                print(f"latency ms [{name}]: {json.dumps(result['timings_ms'])}")
    total_tokens = prompt_tokens + completion_tokens
    print()
    print(f"Tokens: {prompt_tokens:,} prompt · {completion_tokens:,} completion · {total_tokens:,} total")
    print(f"Time:   {elapsed:.1f}s (startup {startup:.1f}s)")
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
