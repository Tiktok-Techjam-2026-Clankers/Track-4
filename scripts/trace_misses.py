"""Per-turn, path-aware tracer for robust browsing/buying misses.

Read-only. Wraps the retrieval routes and HybridRanker.fuse to record, for each
MISSED session, a per-turn row:

  turn | path | fusion_rank(target) | in_response | resp_len

path is inferred exactly as agent.respond does:
  prefix  -> intent_cards.prefix_search returned non-empty
  fuzzy   -> fuzzy_single_mode (fuzzy_card_results non-empty, no prefix/override)
  plain   -> neither (ladder / fusion path)

Usage: .venv/bin/python -m scripts.trace_misses stress --scenarios browsing,buying
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import starter.ranking as ranking
from starter.agent import Agent
from scripts.evaluate_robust import (
    _initial, _override, _reply, _load, MAX_TURNS, TOP_K, NO_PREFERENCE,
)

_FULL_ORDER: list[list[str]] = []
_ROUTES: dict[str, int] = {}
_orig_fuse = ranking.HybridRanker.fuse.__func__


def _wrapped_fuse(cls, bm25, semantic, phrase, popularity, intent, limit,
                  weights=None, fusion_k=None):
    pool = len(set([*bm25, *semantic, *phrase, *popularity]))
    full = _orig_fuse(cls, bm25, semantic, phrase, popularity, intent,
                      max(pool, limit), weights=weights, fusion_k=fusion_k)
    _FULL_ORDER.append([i for i, _ in full])
    return _orig_fuse(cls, bm25, semantic, phrase, popularity, intent, limit,
                      weights=weights, fusion_k=fusion_k)


def _wrap_route(agent, name):
    fn = getattr(agent.intent_cards, name)

    def wrapper(*a, **k):
        out = fn(*a, **k)
        _ROUTES[name] = len(out)
        return out
    setattr(agent.intent_cards, name, wrapper)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("train", "validation", "test", "stress"))
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--cases-dir", default="data/robust")
    ap.add_argument("--scenarios", default="browsing,buying")
    ap.add_argument("--limit", type=int, default=40, help="max miss sessions to print")
    args = ap.parse_args()

    ranking.HybridRanker.fuse = classmethod(_wrapped_fuse)
    scenarios = set(args.scenarios.split(","))
    catalog_ids = {str(r["parent_asin"]) for r in _load(Path(args.catalog))}
    agent = Agent(Path(args.catalog), use_llm=False)
    for r in ("prefix_search", "fuzzy_search", "override_search"):
        _wrap_route(agent, r)
    cases = _load(Path(args.cases_dir) / f"{args.split}.jsonl")

    printed = 0
    path_at_decisive: dict[str, int] = {}
    for idx, case in enumerate(cases):
        if case["scenario_type"] not in scenarios:
            continue
        sid = f"trace_{idx:05d}"
        agent.reset(sid, case["user_profile"])
        message, revealed = _initial(case)
        rng = random.Random(str(case["sample_id"]))
        boundary_used = False
        override_applied = case["scenario_type"] != "intent_override"
        target = str(case["ground_truth"]["parent_asin"])
        rows = []
        hit = False
        for turn in range(1, MAX_TURNS + 1):
            _FULL_ORDER.clear()
            _ROUTES.clear()
            response = agent.respond(sid, message, turn, TOP_K)
            frank = None
            for order in _FULL_ORDER:
                if target in order:
                    r = order.index(target) + 1
                    frank = r if frank is None else min(frank, r)
            if _ROUTES.get("prefix_search", 0) > 0:
                path = "prefix"
            elif _ROUTES.get("fuzzy_search", 0) > 0 and _ROUTES.get("override_search", 0) == 0:
                path = "fuzzy"
            else:
                path = "plain"
            ranked = []
            for item in response.get("recommendations", []):
                asin = str(item.get("parent_asin", ""))
                if asin in catalog_ids and asin not in ranked:
                    ranked.append(asin)
            in_resp = target in ranked[:TOP_K]
            rows.append((turn, path, frank, in_resp, len(ranked)))
            if override_applied and in_resp:
                hit = True
                break
            if turn == MAX_TURNS:
                break
            if not override_applied and turn + 1 == int(case["override_turn"]):
                message = _override(case, revealed); override_applied = True
            elif case["scenario_type"] == "boundary" and not boundary_used:
                attribute = response.get("ask_attribute") or "that"
                message = rng.choice(NO_PREFERENCE).format(attribute=str(attribute).replace("_", " "))
                boundary_used = True
            else:
                message = _reply(case, response.get("ask_attribute"), revealed, rng)

        if hit:
            continue
        # decisive turn = last turn's path (where it ultimately failed)
        path_at_decisive[rows[-1][1]] = path_at_decisive.get(rows[-1][1], 0) + 1
        if printed < args.limit:
            printed += 1
            print(f"\nMISS {case['scenario_type']} sample={case['sample_id']} "
                  f"style={case['language_style']} target_ranks="
                  f"{sorted({r[2] for r in rows if r[2] is not None})}")
            for (t, p, fr, ir, n) in rows:
                print(f"  t{t:<2} {p:<7} fusion_rank={str(fr):>5} in_resp={str(ir):<5} resp_len={n}")

    print("\n=== decisive-turn path counts (misses) ===")
    for p, c in sorted(path_at_decisive.items()):
        print(f"  {p}: {c}")


if __name__ == "__main__":
    main()
