"""Offline v2 aggregation only, over human-reviewed and fact-verified comparisons.

Same user-approved tier policy as v1; no SKU verifier or runtime integration.
"""
from semantic_contract_v2 import DIMENSIONS, validate
from semantic_reward_v1 import POLICY as V1_POLICY

POLICY = {**V1_POLICY, "version": "shopping-semantic-reward-v2-draft"}


def score(contract, comparisons, *, source_query):
    validate(contract, require_reviewed=True, source_query=source_query)
    requirements = [(d, r) for d in DIMENSIONS for r in contract[d]]
    ids = {r["id"] for _, r in requirements}
    if set(comparisons) != ids or any(v not in ("pass", "fail", "unknown") for v in comparisons.values()):
        raise ValueError("Exactly one pass/fail/unknown result per requirement is required")
    if not any(r["importance"] == "required" for _, r in requirements):
        raise ValueError("Nonempty mandatory objective required")
    hard = [r["id"] for d, r in requirements if d in ("category", "budget")
            and r["importance"] == "required" and comparisons[r["id"]] == "fail"]
    if hard:
        return {"reward": -0.85, "semantic_success": False, "reward_valid": True,
                "sampling_invalid": False, "type": "category_budget_failure", "failed_ids": hard}
    unknown = sorted(k for k, v in comparisons.items() if v == "unknown")
    if unknown:
        return {"reward": 0.0, "semantic_success": None, "reward_valid": False,
                "sampling_invalid": True, "type": "unverifiable", "unknown_ids": unknown}
    def average(importance):
        per_dimension = {}
        for d in DIMENSIONS:
            items = [r for dim, r in requirements if dim == d and r["importance"] == importance]
            if items:
                per_dimension[d] = sum(comparisons[r["id"]] == "pass" for r in items) / len(items)
        return (sum(per_dimension.values()) / len(per_dimension) if per_dimension else None), per_dimension
    mandatory, md = average("required")
    preference, pd = average("preference")
    success = all(comparisons[r["id"]] == "pass" for _, r in requirements if r["importance"] == "required")
    reward = (1 if preference is None else 0.55 + 0.45 * preference) if success else -0.30 + 0.55 * mandatory
    return {"reward": reward, "semantic_success": success, "reward_valid": True, "sampling_invalid": False,
            "type": "semantic_success" if success else "partial_failure", "required_score": mandatory,
            "preference_score": preference, "required_dimension_scores": md, "preference_dimension_scores": pd}
