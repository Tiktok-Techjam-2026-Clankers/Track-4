"""Session runner for custom evaluator 1.

The loop mirrors `evaluator.local_evaluator.evaluate` turn for turn: same turn
budget, same override gating, same disclosure rules, same scoring. The only
substitution is the persona that renders the customer's utterances, so a metric
gap between two personas isolates phrasing sensitivity.
"""

from __future__ import annotations

import random
import statistics
import time
from collections import Counter, defaultdict

from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES,
    MAX_TURNS,
    TOP_K,
    classify_constraint,
    coarse_category,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)

from evaluator.custom_evaluator_1_contract import check_response
from evaluator.custom_evaluator_1_personas import Persona

EMPTY_RESPONSE = {"message": "", "ask_attribute": None, "recommendations": []}


def session_rng(persona: Persona, sample_id: str, turn: int) -> random.Random:
    """Seeded per persona, sample and turn so wording never depends on run order."""
    return random.Random(f"{persona.name}\0{sample_id}\0{turn}")


def opening_message(
    persona: Persona, sample: dict, category: str, disclosed: set[str], rng: random.Random
) -> str:
    scenario = sample["scenario_type"]
    if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
        constraint = str(sample["intent_card"]["hard_constraints"][0])
        disclosed.add(constraint)
        return persona.opening_buying(category, constraint, rng)
    if scenario == "intent_override":
        return persona.opening_override(category, str(sample["behavior"]["override"]["old_value"]), rng)
    return persona.opening_browsing(category, rng)


def select_disclosures(sample: dict, attribute: str, disclosed: set[str]) -> list[str]:
    """The official simulator's choice of which constraints to reveal next."""
    constraints = [
        *[str(value) for value in sample["intent_card"].get("hard_constraints", [])],
        *[str(value) for value in sample["intent_card"].get("soft_preferences", [])],
    ]
    return [
        value for value in constraints
        if value not in disclosed and (attribute == "other" or classify_constraint(value) == attribute)
    ][:2]


def customer_reply(
    persona: Persona,
    sample: dict,
    ask_attribute: object,
    disclosed: set[str],
    boundary_used: bool,
    rng: random.Random,
) -> tuple[str, bool]:
    attribute = ask_attribute if isinstance(ask_attribute, str) else None
    if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
        return persona.boundary(attribute, rng), True
    if not attribute:
        return persona.nudge(rng), boundary_used
    if attribute not in ALLOWED_ATTRIBUTES:
        attribute = "other"
    matches = select_disclosures(sample, attribute, disclosed)
    if not matches:
        return persona.no_preference(attribute, rng), boundary_used
    disclosed.update(matches)
    return persona.reveal(matches, rng), boundary_used


def coerce(response: object) -> dict:
    """Apply the official evaluator's tolerance for malformed output."""
    if not isinstance(response, dict) or not isinstance(response.get("message"), str):
        return dict(EMPTY_RESPONSE)
    return response


def score_block(sessions: list[dict]) -> dict:
    overall = metric_summary(sessions)
    mttc = float(overall["mttc"]) if overall["mttc"] is not None else MAX_TURNS + 1
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    technical_score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
    }


def run_session(
    agent,
    persona: Persona,
    sample: dict,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    transcripts: bool,
    timings: list[float],
) -> tuple[dict, list[str], Counter]:
    sample_id = str(sample["sample_id"])
    session_id = f"custom_evaluator_1_{persona.name}_{sample_id}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    intent_card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": intent_card, "behavior": behavior}

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = opening_message(
        persona, effective, coarse_category(categories.get(target, [])), disclosed,
        session_rng(persona, sample_id, 0),
    )
    violations: Counter = Counter()
    tokens = Counter()
    turns: list[dict] = []
    hit_turn: int | None = None
    best_rank: int | None = None

    for turn in range(1, MAX_TURNS + 1):
        started = time.perf_counter()
        try:
            response = agent.respond(session_id, user_message, turn, TOP_K)
        except Exception as error:  # a raised exception counts as a miss, per the spec
            violations[f"exception:{type(error).__name__}"] += 1
            response = dict(EMPTY_RESPONSE)
        timings.append((time.perf_counter() - started) * 1000.0)
        violations.update(check_response(response, catalog_ids))
        response = coerce(response)
        usage = response.get("usage")
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens"):
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    tokens[key] += value
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        if transcripts:
            turns.append({
                "turn": turn,
                "user_message": user_message,
                "ask_attribute": response.get("ask_attribute"),
                "agent_message": response.get("message", ""),
                "top_recommendations": ranked[:3],
            })
        if override_applied and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn = turn
            break
        if turn == MAX_TURNS:
            break
        rng = session_rng(persona, sample_id, turn)
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            official = str(override.get("message", "Actually, please ignore my earlier preference."))
            user_message = persona.override(official, new_value, rng)
        else:
            user_message, boundary_used = customer_reply(
                persona, effective, response.get("ask_attribute"), disclosed, boundary_used, rng
            )

    session = {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
    }
    if transcripts:
        session["transcript"] = turns
    return session, violations, tokens


def run_persona(
    agent,
    persona: Persona,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    transcripts: bool = False,
    collect_timings: bool = False,
) -> dict:
    sessions: list[dict] = []
    violations: Counter = Counter()
    tokens: Counter = Counter()
    timings: list[float] = []
    for sample in samples:
        session, session_violations, session_tokens = run_session(
            agent, persona, sample, catalog_ids, categories, products, transcripts, timings
        )
        sessions.append(session)
        violations.update(session_violations)
        tokens.update(session_tokens)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    result = {
        "persona": persona.name,
        **score_block(sessions),
        "reported_token_usage": {
            "prompt_tokens": tokens["prompt_tokens"],
            "completion_tokens": tokens["completion_tokens"],
            "total_tokens": tokens["prompt_tokens"] + tokens["completion_tokens"],
        },
        "contract_violations": dict(sorted(violations.items())),
        "scenario_metrics": {name: metric_summary(grouped[name]) for name in sorted(grouped)},
        "sessions": sessions,
    }
    if collect_timings and timings:
        ordered = sorted(timings)
        result["timings_ms"] = {
            "turns": len(ordered),
            "mean": round(statistics.fmean(ordered), 3),
            "p50": round(ordered[len(ordered) // 2], 3),
            "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
            "max": round(ordered[-1], 3),
        }
    return result


def delta_against(control: dict, candidate: dict) -> dict:
    keys = ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")
    return {key: round(float(candidate[key]) - float(control[key]), 6) for key in keys}
