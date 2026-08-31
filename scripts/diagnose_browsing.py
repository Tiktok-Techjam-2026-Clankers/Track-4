"""Diagnose robust browsing/buying misses: reachability vs ranking.

Read-only. For each browsing/buying case in a robust split, replays the same
conversation the robust evaluator drives, but wraps ``HybridRanker.fuse`` to
capture the *full* fusion ordering (not just top_k). For every session it
records whether the ground-truth target was hit in top-10, and — crucially for
deciding whether a reranker can help — whether the target was *reachable* in the
fused candidate pool at all, and at what full-fusion rank on the last turn and
at its best turn.

Verdict per miss:
  - "unreachable": target never entered the fused pool  -> retrieval problem,
    a reranker cannot help.
  - "rank>10":     target in pool but fusion ranked it below 10 -> a reranker
    (which reorders the pool) *could* rescue it.

Usage: .venv/bin/python -m scripts.diagnose_browsing stress
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import starter.ranking as ranking
from starter.agent import Agent
from scripts.evaluate_robust import (
    _initial, _override, _reply, _load, MAX_TURNS, TOP_K,
)

_FULL_ORDER: list[list[str]] = []
_orig_fuse = ranking.HybridRanker.fuse.__func__


def _wrapped_fuse(cls, bm25, semantic, phrase, popularity, intent, limit,
                  weights=None, fusion_k=None):
    # Full ordering: call the real fuse with a limit large enough to keep all.
    pool = len(set([*bm25, *semantic, *phrase, *popularity]))
    full = _orig_fuse(cls, bm25, semantic, phrase, popularity, intent,
                      max(pool, limit), weights=weights, fusion_k=fusion_k)
    _FULL_ORDER.append([identifier for identifier, _ in full])
    return _orig_fuse(cls, bm25, semantic, phrase, popularity, intent, limit,
                      weights=weights, fusion_k=fusion_k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("train", "validation", "test", "stress"))
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--cases-dir", default="data/robust")
    ap.add_argument("--scenarios", default="browsing,buying")
    args = ap.parse_args()

    ranking.HybridRanker.fuse = classmethod(_wrapped_fuse)
    scenarios = set(args.scenarios.split(","))
    catalog_ids = {str(r["parent_asin"]) for r in _load(Path(args.catalog))}
    agent = Agent(Path(args.catalog), use_llm=False)
    cases = _load(Path(args.cases_dir) / f"{args.split}.jsonl")

    verdicts: Counter[str] = Counter()
    by_scenario: dict[str, Counter] = {}
    miss_details: list[dict] = []

    for idx, case in enumerate(cases):
        if case["scenario_type"] not in scenarios:
            continue
        by_scenario.setdefault(case["scenario_type"], Counter())
        sid = f"diag_{idx:05d}"
        agent.reset(sid, case["user_profile"])
        message, revealed = _initial(case)
        rng = random.Random(str(case["sample_id"]))
        boundary_used = False
        override_applied = case["scenario_type"] != "intent_override"
        target = str(case["ground_truth"]["parent_asin"])
        hit = False
        best_full_rank = None   # best (lowest) rank of target in full fusion, any turn
        last_full_rank = None
        for turn in range(1, MAX_TURNS + 1):
            _FULL_ORDER.clear()
            response = agent.respond(sid, message, turn, TOP_K)
            # full-fusion rank of target this turn (min across fuse calls this turn)
            turn_rank = None
            for order in _FULL_ORDER:
                if target in order:
                    r = order.index(target) + 1
                    turn_rank = r if turn_rank is None else min(turn_rank, r)
            last_full_rank = turn_rank
            if turn_rank is not None:
                best_full_rank = turn_rank if best_full_rank is None else min(best_full_rank, turn_rank)
            ranked = []
            for item in response.get("recommendations", []):
                asin = str(item.get("parent_asin", ""))
                if asin in catalog_ids and asin not in ranked:
                    ranked.append(asin)
                if len(ranked) == TOP_K:
                    break
            if override_applied and target in ranked:
                hit = True
                break
            if turn == MAX_TURNS:
                break
            if not override_applied and turn + 1 == int(case["override_turn"]):
                message = _override(case, revealed); override_applied = True
            elif case["scenario_type"] == "boundary" and not boundary_used:
                attribute = response.get("ask_attribute") or "that"
                from scripts.evaluate_robust import NO_PREFERENCE
                message = rng.choice(NO_PREFERENCE).format(attribute=str(attribute).replace("_", " "))
                boundary_used = True
            else:
                message = _reply(case, response.get("ask_attribute"), revealed, rng)

        if hit:
            verdicts["hit"] += 1
            by_scenario[case["scenario_type"]]["hit"] += 1
        else:
            if best_full_rank is None:
                v = "miss:unreachable"
            elif best_full_rank <= 10:
                v = "miss:reachable_top10_but_lost"  # in top10 of fusion but cascade dropped it
            else:
                v = "miss:reachable_rank>10"
            verdicts[v] += 1
            by_scenario[case["scenario_type"]][v] += 1
            miss_details.append({
                "sample_id": case["sample_id"], "scenario": case["scenario_type"],
                "best_full_rank": best_full_rank, "last_full_rank": last_full_rank,
            })

    total = sum(verdicts.values())
    print(f"=== {args.split} :: scenarios={sorted(scenarios)} :: n={total} ===")
    for k in sorted(verdicts):
        print(f"  {k:35s} {verdicts[k]:4d}  ({100*verdicts[k]/total:.1f}%)")
    print("\n-- by scenario --")
    for sc, c in sorted(by_scenario.items()):
        n = sum(c.values())
        print(f"  {sc}: n={n}  " + "  ".join(f"{k.replace('miss:','')}={v}" for k, v in sorted(c.items())))
    print("\n-- miss rank distribution (best_full_rank across turns) --")
    ranks = [d["best_full_rank"] for d in miss_details if d["best_full_rank"] is not None]
    unreach = sum(1 for d in miss_details if d["best_full_rank"] is None)
    print(f"  unreachable: {unreach}")
    if ranks:
        ranks.sort()
        buckets = Counter()
        for r in ranks:
            if r <= 10: buckets["01-10"] += 1
            elif r <= 20: buckets["11-20"] += 1
            elif r <= 30: buckets["21-30"] += 1
            elif r <= 50: buckets["31-50"] += 1
            else: buckets["51+"] += 1
        for b in ("01-10", "11-20", "21-30", "31-50", "51+"):
            if buckets[b]:
                print(f"  rank {b}: {buckets[b]}")
        print(f"  reachable-miss rank: min={ranks[0]} median={ranks[len(ranks)//2]} max={ranks[-1]}")


if __name__ == "__main__":
    main()
