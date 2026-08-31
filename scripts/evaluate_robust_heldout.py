"""Generalization guard: re-run the robust eval with a DISJOINT synonym set.

`scripts/evaluate_robust.py` drifts phrasing with one fixed `SYNONYMS` map. Any
improvement that helps only *that* map is overfitting to our own probe, not a
real generalization gain. This wrapper swaps in a held-out synonym map (natural
English synonyms with no overlap in surface form) and reuses the exact same
evaluate() machinery and generated cases. A change is trustworthy only if it
improves (or holds) on BOTH the primary and this held-out eval.

Usage: .venv/bin/python -m scripts.evaluate_robust_heldout stress
"""

from __future__ import annotations

import argparse
from pathlib import Path

import scripts.evaluate_robust as R

# Disjoint from R.SYNONYMS (no shared source or target surface forms). These are
# ordinary synonyms a different paraphraser might reasonably choose.
HELDOUT_SYNONYMS = {
    "shoes": "kicks", "shoe": "kick", "sneakers": "runners",
    "boots": "booties", "comfortable": "cushioned", "comfort": "plushness",
    "waterproof": "water-repellent", "water resistant": "splash-proof",
    "lightweight": "featherweight", "warm": "thermal", "formal": "elegant",
    "casual": "everyday", "running": "road-running", "hiking": "trail",
    "gray": "charcoal", "navy": "midnight blue", "polyester": "poly-blend",
    "breathable": "ventilated", "durable": "long-lasting",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("train", "validation", "test", "stress"))
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--cases-dir", default="data/robust")
    args = ap.parse_args()

    R.SYNONYMS = HELDOUT_SYNONYMS  # _surface reads the module global at call time
    catalog_ids = {str(r["parent_asin"]) for r in R._load(Path(args.catalog))}
    agent = R.Agent(Path(args.catalog), use_llm=False)
    cases = R._load(Path(args.cases_dir) / f"{args.split}.jsonl")
    result = R.evaluate(agent, cases, catalog_ids)

    print("Evaluator: evaluate-robust-HELDOUT-synonyms")
    print("=" * 65)
    print("  ".join(h.rjust(w) for h, w in zip(R.HEADERS, R.WIDTHS)))
    print("  ".join("-" * w for w in R.WIDTHS))
    print(R.format_row(args.split, result))
    for scenario, metrics in sorted(result.get("scenario_metrics", {}).items()):
        print(R.format_row(f"  {scenario}", metrics))


if __name__ == "__main__":
    main()
