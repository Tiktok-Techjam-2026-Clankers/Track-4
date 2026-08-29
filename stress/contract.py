"""Contract checks for agent responses, driven by `docs/agent_api_contract.json`.

Violations are recorded for reporting only. Responses are still handed to the
official scoring path unchanged, so contract findings never move the metrics.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "docs" / "agent_api_contract.json"

MAX_RECOMMENDATIONS = 100
RESPONSE_KEYS = ("message", "ask_attribute", "recommendations", "usage")
REQUIRED_KEYS = ("message", "ask_attribute", "recommendations")
RECOMMENDATION_KEYS = ("parent_asin", "score")
USAGE_KEYS = ("prompt_tokens", "completion_tokens")


@lru_cache(maxsize=1)
def allowed_attributes() -> frozenset[str]:
    """Attribute enum as declared by the contract file, minus the null member."""
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    enum = contract["turn_response"]["properties"]["ask_attribute"]["enum"]
    return frozenset(value for value in enum if isinstance(value, str))


def _check_recommendations(payload: object, catalog_ids: set[str]) -> list[str]:
    if not isinstance(payload, list):
        return ["recommendations_not_list"]
    violations: list[str] = []
    if len(payload) > MAX_RECOMMENDATIONS:
        violations.append("recommendations_over_max_items")
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            violations.append("recommendation_not_object")
            continue
        for key in item:
            if key not in RECOMMENDATION_KEYS:
                violations.append(f"recommendation_unknown_key:{key}")
        parent_asin = item.get("parent_asin")
        if not isinstance(parent_asin, str) or not parent_asin:
            violations.append("recommendation_parent_asin_invalid")
            continue
        if "score" in item and not isinstance(item["score"], (int, float)):
            violations.append("recommendation_score_not_number")
        if parent_asin in seen:
            violations.append("recommendation_duplicate")
        seen.add(parent_asin)
        if parent_asin not in catalog_ids:
            violations.append("recommendation_not_in_catalog")
    return violations


def _check_usage(usage: object) -> list[str]:
    if not isinstance(usage, dict):
        return ["usage_not_object"]
    violations: list[str] = []
    for key in usage:
        if key not in USAGE_KEYS:
            violations.append(f"usage_unknown_key:{key}")
    for key in USAGE_KEYS:
        value = usage.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            violations.append(f"usage_{key}_not_integer")
        elif value < 0:
            violations.append(f"usage_{key}_negative")
    return violations


def check_response(response: object, catalog_ids: set[str]) -> list[str]:
    """Return contract violation codes for one `respond(...)` payload."""
    if not isinstance(response, dict):
        return ["response_not_object"]
    violations: list[str] = []
    for key in response:
        if key not in RESPONSE_KEYS:
            violations.append(f"response_unknown_key:{key}")
    for key in REQUIRED_KEYS:
        if key not in response:
            violations.append(f"response_missing_key:{key}")
    if "message" in response and not isinstance(response["message"], str):
        violations.append("message_not_string")
    if "ask_attribute" in response:
        attribute = response["ask_attribute"]
        if attribute is not None and attribute not in allowed_attributes():
            violations.append("ask_attribute_not_allowed")
    if "recommendations" in response:
        violations.extend(_check_recommendations(response["recommendations"], catalog_ids))
    if response.get("usage") is not None:
        violations.extend(_check_usage(response["usage"]))
    return violations
