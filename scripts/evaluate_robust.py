"""Independent paraphrase/noise evaluator for generalization testing."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from starter.agent import Agent


MAX_TURNS = 10
TOP_K = 10
SYNONYMS = {
    "shoes": "footwear", "shoe": "trainer", "sneakers": "trainers",
    "boots": "bootwear", "comfortable": "comfy", "comfort": "cushioning",
    "waterproof": "rainproof", "water resistant": "weatherproof",
    "lightweight": "not heavy", "warm": "cosy", "formal": "dressy",
    "casual": "laid-back", "running": "jogging", "hiking": "trekking",
    "gray": "grey", "navy": "dark blue", "polyester": "synthetic fabric",
}
NO_PREFERENCE = (
    "Anything works for {attribute}; decide for me.",
    "I am flexible about {attribute}.",
    "That detail does not matter to me.",
    "No strong view on {attribute}; use your best judgement.",
)
NO_MORE = (
    "Nothing else comes to mind for {attribute}.",
    "I can be flexible there.",
    "No additional requirement for that.",
)


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _surface(value: str, style: int) -> str:
    if style % 3 == 0:
        return value
    result = value
    for source, target in sorted(SYNONYMS.items(), key=lambda item: len(item[0]), reverse=True):
        result = re.sub(r"\b" + re.escape(source) + r"\b", target, result, flags=re.I)
    if style % 3 == 2:
        result = re.sub(r"\b(?:features?|details?|department|style)\s*:\s*", "", result, flags=re.I)
    return result


def _constraint(attribute: str, value: str, style: int) -> str:
    if attribute == "budget":
        return f"roughly ${value}, with a little flexibility"
    surfaced = _surface(value, style)
    templates = (
        "{value}", "I would lean toward {value}", "please prioritize {value}",
        "something described as {value}", "{value} would be ideal", "I need {value}",
    )
    return templates[style % len(templates)].format(value=surfaced)


def _initial(case: dict) -> tuple[str, set[str]]:
    scenario = case["scenario_type"]
    style = int(case["language_style"])
    category = _surface(str(case["intent"]["category"]), style)
    attributes = list(case["intent"]["attributes"].items())
    revealed: set[str] = set()
    if scenario == "buying" and attributes:
        attribute, value = attributes[0]
        revealed.add(attribute)
        return (
            f"I need {category} soon. One non-negotiable is {_constraint(attribute, value, style)}.",
            revealed,
        )
    if scenario == "intent_override":
        decoy = case["decoy"]
        return (
            f"Could you help me find {category}? I was initially leaning toward "
            f"{_surface(str(decoy['value']), style)}.",
            revealed,
        )
    templates = (
        "I am browsing for {category} and could use some guidance.",
        "Show me a few directions for {category}; I have not decided on details.",
        "I need help narrowing down {category}.",
        "What might work in the {category} space?",
        "I am comparing options for {category}.",
        "Help me explore {category} without assuming too much.",
    )
    return templates[style % len(templates)].format(category=category), revealed


def _override(case: dict, revealed: set[str]) -> str:
    attributes = list(case["intent"]["attributes"].items())
    style = int(case["language_style"])
    selected = attributes[:2]
    revealed.update(attribute for attribute, _ in selected)
    requirements = " and ".join(
        _constraint(attribute, value, style + index)
        for index, (attribute, value) in enumerate(selected)
    ) or _surface(str(case["intent"]["category"]), style)
    old = _surface(str(case["decoy"]["value"]), style)
    return f"Scratch the earlier {old} idea. What I actually need is {requirements}."


def _reply(case: dict, ask: object, revealed: set[str], rng: random.Random) -> str:
    style = int(case["language_style"])
    attribute = ask if isinstance(ask, str) else "other"
    attributes = case["intent"]["attributes"]
    if attribute == "other":
        unseen = [(key, value) for key, value in attributes.items() if key not in revealed][:2]
    elif attribute in attributes and attribute not in revealed:
        unseen = [(attribute, attributes[attribute])]
    elif attribute == "category" and "category" not in revealed:
        unseen = [("category", case["intent"]["category"])]
    else:
        unseen = []
    if not unseen:
        return rng.choice(NO_MORE).format(attribute=attribute.replace("_", " "))
    revealed.update(key for key, _ in unseen)
    values = [
        _constraint(key, str(value), style + index)
        for index, (key, value) in enumerate(unseen)
    ]
    templates = (
        "The useful clues are: {values}.",
        "I care most about {values}.",
        "Please filter toward {values}.",
        "For me, {values} matters.",
        "Let us use {values} as the guide.",
        "My preference would be {values}.",
    )
    return templates[style % len(templates)].format(values="; ".join(values))


def _metric(sessions: list[dict]) -> dict:
    count = len(sessions)
    hit = sum(item["hit"] for item in sessions) / count
    mrr = statistics.fmean(item["reciprocal_rank"] for item in sessions)
    mttc = statistics.fmean(item["turn"] if item["turn"] else 11 for item in sessions)
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "sample_count": count,
        "hit_rate_at_10": round(hit, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(0.5 * hit + 0.3 * mrr + 0.2 * efficiency, 6),
    }


def evaluate(agent: Agent, cases: list[dict], catalog_ids: set[str]) -> dict:
    sessions = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    prompt_tokens = 0
    completion_tokens = 0
    for case_index, case in enumerate(cases):
        session_id = f"robust_{case_index:05d}"
        agent.reset(session_id, case["user_profile"])
        message, revealed = _initial(case)
        rng = random.Random(str(case["sample_id"]))
        boundary_used = False
        override_applied = case["scenario_type"] != "intent_override"
        target = str(case["ground_truth"]["parent_asin"])
        outcome = {"sample_id": case["sample_id"], "scenario_type": case["scenario_type"],
                   "hit": False, "turn": None, "rank": None, "reciprocal_rank": 0.0}
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, message, turn, TOP_K)
            usage = response.get("usage") if isinstance(response, dict) else None
            if isinstance(usage, dict):
                prompt_tokens += max(0, int(usage.get("prompt_tokens", 0)))
                completion_tokens += max(0, int(usage.get("completion_tokens", 0)))
            ranked = []
            for item in response.get("recommendations", []) if isinstance(response, dict) else []:
                asin = str(item.get("parent_asin", "")) if isinstance(item, dict) else ""
                if asin in catalog_ids and asin not in ranked:
                    ranked.append(asin)
                if len(ranked) == TOP_K:
                    break
            if override_applied and target in ranked:
                rank = ranked.index(target) + 1
                outcome.update(hit=True, turn=turn, rank=rank, reciprocal_rank=1.0 / rank)
                break
            if turn == MAX_TURNS:
                break
            if not override_applied and turn + 1 == int(case["override_turn"]):
                message = _override(case, revealed)
                override_applied = True
            elif case["scenario_type"] == "boundary" and not boundary_used:
                attribute = response.get("ask_attribute") or "that"
                message = rng.choice(NO_PREFERENCE).format(
                    attribute=str(attribute).replace("_", " ")
                )
                boundary_used = True
            else:
                message = _reply(case, response.get("ask_attribute"), revealed, rng)
        sessions.append(outcome)
        grouped[outcome["scenario_type"]].append(outcome)
    overall = _metric(sessions)
    overall["scenario_metrics"] = {
        name: _metric(values) for name, values in sorted(grouped.items())
    }
    overall["reported_token_usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    overall["sessions"] = sessions
    return overall


HEADERS = ("split", "n", "Hit@10", "MRR", "MTTC", "Eff", "Score")
WIDTHS = (16, 5, 8, 8, 8, 8, 8)
METRIC_KEYS = ("hit_rate_at_10", "mrr", "mttc", "efficiency", "technical_score")


def format_row(label: str, result: dict) -> str:
    cells = [
        label,
        str(result["sample_count"]),
        *(f"{float(result[k]):.4f}" for k in METRIC_KEYS),
    ]
    return "  ".join(c.rjust(w) for c, w in zip(cells, WIDTHS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "validation", "test", "stress"))
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--cases-dir", default="data/robust")
    parser.add_argument("--output-dir", default="benchmark-results/robust")
    parser.add_argument("--allow-locked-test", action="store_true")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable all LLM calls; run the deterministic pipeline only.",
    )
    args = parser.parse_args()
    if args.split == "test" and not args.allow_locked_test:
        raise SystemExit("robust test is locked; freeze configuration and pass --allow-locked-test")
    catalog_path = Path(args.catalog)
    catalog_ids = {
        str(row["parent_asin"]) for row in _load(catalog_path)
    }
    t0 = time.perf_counter()
    agent = Agent(catalog_path, use_llm=not args.no_llm)
    startup = time.perf_counter() - t0
    result = evaluate(agent, _load(Path(args.cases_dir) / f"{args.split}.jsonl"), catalog_ids)
    elapsed = time.perf_counter() - t0
    result["startup_seconds"] = round(startup, 6)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.split}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    print("Evaluator: evaluate-robust")
    print("=" * 65)
    print()
    print("  ".join(h.rjust(w) for h, w in zip(HEADERS, WIDTHS)))
    print("  ".join("-" * w for w in WIDTHS))
    print(format_row(args.split, result))
    for scenario, metrics in sorted(result.get("scenario_metrics", {}).items()):
        print(format_row(f"  {scenario}", metrics))
    usage = result.get("reported_token_usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    print()
    print(f"Tokens: {pt:,} prompt · {ct:,} completion · {pt + ct:,} total")
    print(f"Time:   {elapsed:.1f}s (startup {startup:.1f}s)")


if __name__ == "__main__":
    main()
