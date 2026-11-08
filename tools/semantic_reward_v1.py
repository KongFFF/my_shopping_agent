"""Offline scoring prototype over reviewed requirements and verified comparisons.

This aggregates pass/fail/unknown evidence; it does NOT extract requirements,
verify arbitrary shopping facts, call an LLM, or integrate with Reward v3.
"""
from semantic_contract_v1 import DIMENSIONS, validate

POLICY = {
    "version": "shopping-semantic-reward-v1-draft",
    "category_budget_failure": -0.85,
    "partial_base": -0.30,
    "partial_scale": 0.55,
    "success_base": 0.55,
    "success_scale": 0.45,
    "dimension_weights": {name: 1.0 for name in DIMENSIONS},
    "target_identity_affects_reward": False,
}


def score(contract, comparisons, *, source_query):
    validate(contract, require_reviewed=True, source_query=source_query)
    requirements = [(dimension, r) for dimension in DIMENSIONS for r in contract[dimension]]
    ids = {r["id"] for _, r in requirements}
    if set(comparisons) != ids or any(value not in ("pass", "fail", "unknown") for value in comparisons.values()):
        raise ValueError("Exactly one pass/fail/unknown result is required per requirement")
    if not any(r["importance"] == "required" for _, r in requirements):
        raise ValueError("Review a nonempty mandatory task objective before scoring")

    def average(importance):
        per_dimension = {}
        for dimension in DIMENSIONS:
            items = [r for d, r in requirements if d == dimension and r["importance"] == importance]
            if items:
                per_dimension[dimension] = sum(comparisons[r["id"]] == "pass" for r in items) / len(items)
        return (sum(per_dimension.values()) / len(per_dimension) if per_dimension else None), per_dimension

    hard_failures = [r["id"] for dimension, r in requirements
                     if dimension in ("category", "budget") and r["importance"] == "required"
                     and comparisons[r["id"]] == "fail"]
    if hard_failures:
        return {"reward": -0.85, "semantic_success": False, "reward_valid": True,
                "sampling_invalid": False, "type": "category_budget_failure", "failed_ids": hard_failures}
    unknown = sorted(key for key, value in comparisons.items() if value == "unknown")
    if unknown:
        return {"reward": 0.0, "semantic_success": None, "reward_valid": False,
                "sampling_invalid": True, "type": "unverifiable", "unknown_ids": unknown}
    mandatory, mandatory_dimensions = average("required")
    preference, preference_dimensions = average("preference")
    all_mandatory = all(comparisons[r["id"]] == "pass" for _, r in requirements if r["importance"] == "required")
    if all_mandatory:
        reward = 1.0 if preference is None else POLICY["success_base"] + POLICY["success_scale"] * preference
        tier = "semantic_success"
    else:
        reward = POLICY["partial_base"] + POLICY["partial_scale"] * mandatory
        tier = "partial_failure"
    return {"reward": reward, "semantic_success": all_mandatory, "reward_valid": True,
            "sampling_invalid": False, "type": tier, "required_score": mandatory,
            "preference_score": preference, "required_dimension_scores": mandatory_dimensions,
            "preference_dimension_scores": preference_dimensions}
