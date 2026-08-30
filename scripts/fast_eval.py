"""Parallel evaluation harness — same scoring as the official evaluator, faster.

The official loop in ``evaluator.local_evaluator`` runs sessions one at a time.
In LLM mode that loop is dominated by network round-trips, so this runs sessions
concurrently instead. Sessions are fully independent — only the turns *within* a
session are ordered — so one shared ``Agent`` is safe: the SQLite connection and
the LLM caches are lock-guarded (see ``starter/retrieval.py`` and the reranker /
intent parser). Scoring is delegated to the official module, so this cannot drift
from it, and ``evaluator/`` is never modified.

    python scripts/fast_eval.py --workers 8                 # both sets, LLM if key
    python scripts/fast_eval.py --no-llm                    # deterministic (should
                                                            # match the sequential
                                                            # 0.9664 / 0.9599)
    python scripts/fast_eval.py --model gpt-4.1-nano --limit 40   # A/B a model

The single-threaded official path stays authoritative:

    python scripts/evaluate_datasets.py [--no-llm]
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)
from starter.agent import Agent  # noqa: E402

CATALOG_PATH = REPO_ROOT / "data" / "catalog.jsonl"
DATASETS = (
    ("Default", REPO_ROOT / "data" / "public_set.jsonl"),
    ("Extended", REPO_ROOT / "data" / "private_set.jsonl"),
)
EMPTY = {"message": "", "ask_attribute": None, "recommendations": []}

HEADERS = ("dataset", "n", "Hit@10", "MRR", "MTTC", "Eff", "Score")
WIDTHS = (16, 5, 8, 8, 8, 8, 8)
KEYS = ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")


def run_session(agent, sample, catalog_ids, categories, products) -> dict:
    """One session, turn for turn — identical to ``local_evaluator.evaluate``."""
    session_id = f"fast_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(
        effective, coarse_category(categories.get(target, [])), disclosed
    )
    hit_turn = best_rank = None
    prompt_tokens = completion_tokens = 0

    for turn in range(1, MAX_TURNS + 1):
        try:
            response = agent.respond(session_id, user_message, turn, TOP_K)
        except Exception:
            response = dict(EMPTY)
        if not isinstance(response, dict) or not isinstance(response.get("message"), str):
            response = dict(EMPTY)
        usage = response.get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] >= 0:
                prompt_tokens += usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), int) and usage["completion_tokens"] >= 0:
                completion_tokens += usage["completion_tokens"]
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        if override_applied and target in ranked:
            best_rank, hit_turn = ranked.index(target) + 1, turn
            break
        if turn == MAX_TURNS:
            break
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(
                override.get("message", "Actually, please ignore my earlier preference.")
            )
        else:
            user_message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )

    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def summarize(sessions: list[dict]) -> dict:
    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(
            0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency, 6
        ),
    }


def format_row(label: str, n: int, result: dict) -> str:
    cells = [label, str(n), *(f"{float(result[k]):.4f}" for k in KEYS)]
    return "  ".join(c.rjust(w) for c, w in zip(cells, WIDTHS))


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel evaluation (official scoring)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-llm", action="store_true", help="Deterministic only.")
    parser.add_argument("--model", default=None, help="Override the LLM model (e.g. gpt-4.1-nano).")
    parser.add_argument("--limit", type=int, default=None, help="Cap sessions per dataset.")
    args = parser.parse_args()

    catalog_ids, categories, products = catalog_index(CATALOG_PATH)
    t0 = time.perf_counter()
    agent = Agent(CATALOG_PATH, use_llm=not args.no_llm, model=args.model)
    startup = time.perf_counter() - t0
    mode = "deterministic" if not agent._llm_active else f"LLM ({agent.model})"

    prompt_tokens = completion_tokens = 0
    rows: list[str] = []
    for label, dataset_path in DATASETS:
        samples = load_jsonl(dataset_path)
        if args.limit:
            samples = samples[: args.limit]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            sessions = list(pool.map(
                lambda s: run_session(agent, s, catalog_ids, categories, products),
                samples,
            ))
        result = summarize(sessions)
        rows.append(format_row(label, len(sessions), result))
        prompt_tokens += sum(s["prompt_tokens"] for s in sessions)
        completion_tokens += sum(s["completion_tokens"] for s in sessions)

    elapsed = time.perf_counter() - t0

    print(f"Evaluator: fast-eval ({mode}, {args.workers} workers)")
    print("=" * 65)
    print()
    print("  ".join(h.rjust(w) for h, w in zip(HEADERS, WIDTHS)))
    print("  ".join("-" * w for w in WIDTHS))
    for row in rows:
        print(row)
    total = prompt_tokens + completion_tokens
    print()
    print(f"Tokens: {prompt_tokens:,} prompt · {completion_tokens:,} completion · {total:,} total")
    print(f"Time:   {elapsed:.1f}s (startup {startup:.1f}s)")


if __name__ == "__main__":
    main()
