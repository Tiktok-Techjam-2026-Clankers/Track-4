"""Measure post-override surfacing latency for robust intent_override cases.

For each intent_override case, records the override turn, the turn the target
first appears in top-10, and the gap (turns after override to hit). High MTTC on
this scenario could be structural (override happens late) or a real lag
(target known post-override but surfaced slowly). This tells which.

Usage: .venv/bin/python -m scripts.trace_override_latency stress
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from starter.agent import Agent
from scripts.evaluate_robust import _initial, _override, _reply, _load, MAX_TURNS, TOP_K


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("train", "validation", "test", "stress"))
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--cases-dir", default="data/robust")
    args = ap.parse_args()

    catalog_ids = {str(r["parent_asin"]) for r in _load(Path(args.catalog))}
    agent = Agent(Path(args.catalog), use_llm=False)
    cases = _load(Path(args.cases_dir) / f"{args.split}.jsonl")

    gaps: Counter[int] = Counter()
    misses = 0
    n = 0
    ov_turns: Counter[int] = Counter()
    for idx, case in enumerate(cases):
        if case["scenario_type"] != "intent_override":
            continue
        n += 1
        sid = f"ovl_{idx:05d}"
        agent.reset(sid, case["user_profile"])
        message, revealed = _initial(case)
        rng = random.Random(str(case["sample_id"]))
        override_applied = False
        override_turn_actual = None
        target = str(case["ground_truth"]["parent_asin"])
        hit_turn = None
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(sid, message, turn, TOP_K)
            ranked = []
            for item in response.get("recommendations", []):
                asin = str(item.get("parent_asin", ""))
                if asin in catalog_ids and asin not in ranked:
                    ranked.append(asin)
            if override_applied and target in ranked[:TOP_K]:
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            if not override_applied and turn + 1 == int(case["override_turn"]):
                message = _override(case, revealed)
                override_applied = True
                override_turn_actual = turn + 1
            else:
                message = _reply(case, response.get("ask_attribute"), revealed, rng)
        if hit_turn is None:
            misses += 1
        else:
            ov_turns[override_turn_actual or 0] += 1
            gaps[hit_turn - (override_turn_actual or hit_turn)] += 1

    print(f"=== {args.split} intent_override :: n={n} misses={misses} ===")
    print("gap = turns from override turn to first hit (0 = hit on override turn)")
    for g in sorted(gaps):
        print(f"  gap {g:+d}: {gaps[g]}")
    print("override turn distribution (hits):")
    for t in sorted(ov_turns):
        print(f"  override@t{t}: {ov_turns[t]}")


if __name__ == "__main__":
    main()
