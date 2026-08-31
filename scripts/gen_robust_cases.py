"""Materialize robust-evaluator cases from the thin public/private sets.

The public/private sets store only sample_id / scenario_type / ground_truth /
user_profile; the official evaluator derives the intent constraints at runtime
from the target's catalog entry (``intent_card`` + ``behavior_for``). The
paraphrase evaluator (``scripts/evaluate_robust.py``) instead expects those
fields pre-baked in an ``intent.attributes`` / ``decoy`` / ``language_style``
schema.

This bridge reuses the *official* derivation functions so the generated cases
carry exactly the constraints the official simulator would disclose — only the
surface phrasing differs (which is the whole point of the robustness probe). No
hidden labels or IDs beyond the target already present in ground_truth; every
constraint value is catalog-derived.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

from evaluator.local_evaluator import (
    behavior_for,
    catalog_index,
    classify_constraint,
    coarse_category,
    intent_card,
)


def _attributes(card: dict) -> dict[str, str]:
    """Turn hard+soft constraint clauses into an attribute->value dict.

    Colliding attribute classes are kept (suffixed) so no constraint value is
    silently dropped — the paraphrase evaluator surfaces them via its
    open-ended 'other' replies regardless of key.
    """
    values = [
        *[str(v) for v in card.get("hard_constraints", [])],
        *[str(v) for v in card.get("soft_preferences", [])],
    ]
    attributes: dict[str, str] = {}
    for index, value in enumerate(values):
        key = classify_constraint(value)
        if key in attributes:
            key = f"{key}_{index}"
        attributes[key] = value
    return attributes


def _case(sample: dict, categories: dict, products: dict) -> dict:
    target = str(sample["ground_truth"]["parent_asin"])
    product = products[target]
    card = intent_card(product)
    # Reuse the official seeded behavior so override turn/old-value match what
    # the official simulator would pick for this sample.
    import random

    seed = f"{sample.get('sample_id', '')}\0{sample.get('scenario_type', '')}"
    behavior = behavior_for(str(sample["scenario_type"]), card, random.Random(seed))
    override = behavior.get("override") or {}

    soft = card.get("soft_preferences") or card.get("hard_constraints") or ["a different style"]
    decoy_value = str(override.get("old_value") or soft[-1])

    # Spread language styles across all 6 template variants (and thus all three
    # surface modes: verbatim / synonym / synonym+marker-stripped) so the
    # aggregate score reflects a realistic mix of phrasing drift, not one style.
    style = zlib.crc32(str(sample["sample_id"]).encode("utf-8")) % 6

    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "user_profile": sample["user_profile"],
        "ground_truth": {"parent_asin": target},
        "language_style": style,
        "override_turn": int(override.get("turn", 99)),
        "decoy": {"value": decoy_value},
        "intent": {
            "category": coarse_category(categories.get(target, [])),
            "attributes": _attributes(card),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--out-dir", default="data/robust")
    parser.add_argument(
        "--map",
        nargs="+",
        default=["data/public_set.jsonl:stress", "data/private_set.jsonl:validation"],
        help="src.jsonl:split pairs",
    )
    args = parser.parse_args()

    _, categories, products = catalog_index(args.catalog)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pair in args.map:
        src, split = pair.rsplit(":", 1)
        rows = [json.loads(line) for line in Path(src).open(encoding="utf-8") if line.strip()]
        cases = [_case(row, categories, products) for row in rows]
        dest = out_dir / f"{split}.jsonl"
        with dest.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(case) + "\n")
        print(f"{src} -> {dest} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
