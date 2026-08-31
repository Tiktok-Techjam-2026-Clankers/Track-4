"""Diagnose the intent_override collapse: recall vs ordering.

Replays the robust evaluator's intent_override conversations against the
deterministic agent, wrapping the override-relevant retrieval routes to record,
each turn, whether the ground-truth target is present in the candidate pool and
at what position. Read-only: patches instance methods at runtime, touches no
files under starter/ or evaluator/.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from starter.agent import Agent
from scripts.evaluate_robust import _initial, _override, _reply, MAX_TURNS, TOP_K


def _pos(lst, target):
    try:
        return lst.index(target) + 1
    except ValueError:
        return None


def main() -> None:
    catalog = Path("data/catalog.jsonl")
    catalog_ids = {str(r["parent_asin"]) for r in
                   (json.loads(l) for l in catalog.open() if l.strip())}
    agent = Agent(catalog, use_llm=False)
    cases = [json.loads(l) for l in Path("data/robust/stress.jsonl").open() if l.strip()]
    cases = [c for c in cases if c["scenario_type"] == "intent_override"]

    # Wrap the override-relevant routes to capture their outputs per call.
    captured: dict[str, list[str]] = {}
    ic = agent.intent_cards

    def wrap(name, fn):
        def inner(*a, **k):
            out = fn(*a, **k)
            captured[name] = list(out)
            return out
        return inner

    ic.fuzzy_search = wrap("fuzzy", ic.fuzzy_search)          # type: ignore
    ic.override_search = wrap("override", ic.override_search)  # type: ignore
    ic.search = wrap("card", ic.search)                        # type: ignore
    agent.constraints.search = wrap("constraint", agent.constraints.search)  # type: ignore
    agent.semantic.search = wrap("semantic", agent.semantic.search)          # type: ignore
    agent.categories.search = wrap("category", agent.categories.search)      # type: ignore

    # Aggregate stats
    hit_sessions = 0
    target_in_any_pool = 0          # target reachable somewhere post-override
    target_in_fuzzy = Counter()     # bucketed fuzzy rank when reachable
    never_reachable = 0
    reach_examples = []
    miss_examples = []
    intents_seen = Counter()        # what intent did the agent classify?
    seen_before_override = 0        # target already in fuzzy_recommended pre-override
    trace_lines = []

    for ci, case in enumerate(cases):
        sid = f"diag_{ci:04d}"
        agent.reset(sid, case["user_profile"])
        message, revealed = _initial(case)
        rng = random.Random(str(case["sample_id"]))
        override_applied = False
        target = str(case["ground_truth"]["parent_asin"])
        session_hit = False
        best_fuzzy = None
        reachable = False
        override_turn = int(case["override_turn"])
        mem = agent.sessions[sid]
        for turn in range(1, MAX_TURNS + 1):
            captured.clear()
            resp = agent.respond(sid, message, turn, TOP_K)
            intents_seen[mem.intent] += 1
            ranked = [str(it.get("parent_asin", "")) for it in resp.get("recommendations", [])]
            if override_applied:
                fpos = _pos(captured.get("fuzzy", []), target)
                fin = _pos(ranked, target)
                seen = target in mem.fuzzy_recommended
                if ci < 4:
                    trace_lines.append(
                        f"    {case['sample_id']} t{turn} pt{turn - (override_turn-1)} "
                        f"intent={mem.intent} fuzzy_rank={fpos} in_ranked={fin} "
                        f"seen={seen} n_rec={len(ranked)}")
                for route in ("fuzzy", "override", "card", "constraint", "category"):
                    p = _pos(captured.get(route, []), target)
                    if p is not None:
                        reachable = True
                        if route == "fuzzy" and (best_fuzzy is None or p < best_fuzzy):
                            best_fuzzy = p
                if target in ranked:
                    session_hit = True
                    break
            if turn == MAX_TURNS:
                break
            if not override_applied and turn + 1 == override_turn:
                if target in mem.fuzzy_recommended:
                    seen_before_override += 1
                message = _override(case, revealed)
                override_applied = True
            else:
                message = _reply(case, resp.get("ask_attribute"), revealed, rng)

        if session_hit:
            hit_sessions += 1
        if reachable:
            target_in_any_pool += 1
            if best_fuzzy is None:
                target_in_fuzzy["not_in_fuzzy"] += 1
            elif best_fuzzy <= 10:
                target_in_fuzzy["fuzzy_1_10"] += 1
            elif best_fuzzy <= 30:
                target_in_fuzzy["fuzzy_11_30"] += 1
            else:
                target_in_fuzzy["fuzzy_31plus"] += 1
            if not session_hit and len(reach_examples) < 6:
                reach_examples.append((case["sample_id"], best_fuzzy, override_turn))
        else:
            never_reachable += 1
            if len(miss_examples) < 6:
                miss_examples.append((case["sample_id"], override_turn))

    n = len(cases)
    print(f"intent_override cases: {n}")
    print(f"  session hit@10           : {hit_sessions}/{n}")
    print(f"  target reachable in pool : {target_in_any_pool}/{n}  "
          f"(=> ordering-recoverable)")
    print(f"  target NEVER reachable   : {never_reachable}/{n}  "
          f"(=> recall failure, rerank cannot help)")
    print(f"  fuzzy-rank buckets (reachable cases): {dict(target_in_fuzzy)}")
    print(f"  reachable-but-missed examples (sample_id, best_fuzzy_rank, override_turn):")
    for ex in reach_examples:
        print("     ", ex)
    print(f"  never-reachable examples (sample_id, override_turn):")
    for ex in miss_examples:
        print("     ", ex)
    print(f"  intents classified across all turns: {dict(intents_seen)}")
    print(f"  target already 'seen' (fuzzy_recommended) before override: {seen_before_override}/{n}")
    print("  per-turn trace (first 4 cases, post-override turns):")
    for line in trace_lines:
        print(line)


if __name__ == "__main__":
    main()
