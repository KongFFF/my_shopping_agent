"""Purchase-completeness guard using product metadata only, never task answers."""

VERSION = "purchase-guard-v1"


def inspect_purchase(product, selected_options):
    from web_agent_site.engine.variant_price import option_axes, resolve_variant_price

    price = resolve_variant_price(product, selected_options)
    missing = []
    if price.get("method") == "effective_price_axis_unselected":
        canonical = price["evidence"]["effective_price_axis"]
        axis = option_axes(product)["axes"].get(canonical)
        if axis is not None:
            missing.append(axis["source_axis"])
    return {"version": VERSION,
            "status": "missing_options" if missing else "ready" if price["status"] == "pass" else "metadata_unverifiable",
            "missing_axes": missing, "price_method": price["method"]}


def guard_purchase(server, session_id, *, enabled=True, max_attempts=3):
    """None means proceed with the original purchase, including metadata errors."""
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or not 1 <= max_attempts <= 10:
        raise ValueError("max_attempts must be an integer in [1,10]")
    session = server.user_sessions[session_id]
    if not enabled:
        return None
    product = server.product_item_dict[session["asin"]]
    readiness = inspect_purchase(product, session.get("options") or {})
    if readiness["status"] != "missing_options":
        return None
    count = session.get("purchase_guard_incomplete_attempts", 0) + 1
    session["purchase_guard_incomplete_attempts"] = count
    exhausted = count >= max_attempts
    feedback = {**readiness, "status": "exhausted" if exhausted else "rejected",
                "attempt": count, "max_attempts": max_attempts,
                "reason": "incomplete_purchase_limit" if exhausted else "missing_purchase_options",
                "message": ("缺少必要规格，购买未执行。请从当前页面补选：" + "、".join(readiness["missing_axes"])
                            + f"。不完整购买第{count}/{max_attempts}次。"
                            + ("已达到上限，本任务以失败结束。" if exhausted else "请补选后再购买；也可以返回搜索其他商品。"))}
    session["purchase_guard_feedback"] = feedback
    session.setdefault("purchase_guard_history", []).append(dict(feedback))
    if not exhausted:
        return {"feedback": feedback, "result": None}
    from web_agent_site.engine.comparators import FAIL, comparison
    from web_agent_site.engine.reward import DEFAULT_REWARDS, RewardResult

    reward = server.environment_config.get("reward", {}).get("wrong_purchase", DEFAULT_REWARDS["wrong_purchase"])
    result = RewardResult(reward=float(reward), reward_type="wrong_purchase", reward_valid=True,
                          termination_reason="wrong_purchase", target_asin_match=False,
                          hard_gates={"purchase_completeness": comparison(
                              FAIL, comparator=VERSION, required=readiness["missing_axes"], actual=dict(session.get("options") or {}),
                              source_field="public_product_options")}, weighted_score=0.0,
                          evidence={"purchase_guard": feedback, "target_identity_evaluated": False})
    return {"feedback": feedback, "result": result}
