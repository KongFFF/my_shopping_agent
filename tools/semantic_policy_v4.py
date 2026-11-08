"""Source-backed budget policy and conservative pending-requirement checks.

No automatic approval, subject inference, or deletion of arbitrary ambiguity.
"""
import copy
import re

NUM = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)"
MONEY = r"预算|价格|售价|价钱|价位|总价|单价|运费|定金|尾款|加工费|元|块"
SOFT = r"最好|优先|尽量|偏好|喜欢"
HARD = r"必须|务必|一定要|要求"


def structural_check(raw):
    """Check container types before the routing adapter can dereference them."""
    if not isinstance(raw, dict):
        raise ValueError("Root must be an object")
    for dim in ("category", "budget", "functions", "specifications", "brand", "model"):
        if dim not in raw:
            continue  # Required-field errors come from the strict schema.
        if not isinstance(raw[dim], list):
            raise ValueError(f"/{dim} must be an array")
        for i, r in enumerate(raw[dim]):
            if not isinstance(r, dict):
                raise ValueError(f"/{dim}/{i} must be an object")
            for name in ("expected", "subject"):
                if name in r and not isinstance(r[name], dict):
                    raise ValueError(f"/{dim}/{i}/{name} must be an object")
            for name in ("subtype", "attribute", "importance", "operator", "review_status"):
                if name in r and not isinstance(r[name], str):
                    raise ValueError(f"/{dim}/{i}/{name} must be text")
            value = r.get("expected", {})
            if value.get("type") == "components":
                items = value.get("items")
                if not isinstance(items, list) or any(not isinstance(x, dict) or not isinstance(x.get("name"), str) for x in items):
                    raise ValueError(f"/{dim}/{i}/expected/items must contain named component objects")
            if value.get("type") == "text" and not isinstance(value.get("value"), str):
                raise ValueError(f"/{dim}/{i}/expected/value must be text")
            if "evidence" in r and (not isinstance(r["evidence"], list) or any(not isinstance(e, dict) or not isinstance(e.get("text"), str) for e in r["evidence"])):
                raise ValueError(f"/{dim}/{i}/evidence must be an array of text objects")
    if "pending_mentions" in raw:
        if not isinstance(raw["pending_mentions"], list):
            raise ValueError("/pending_mentions must be an array")
        for i, p in enumerate(raw["pending_mentions"]):
            if not isinstance(p, dict):
                raise ValueError(f"/pending_mentions/{i} must be an object")
            e = p.get("evidence")
            if isinstance(e, list):
                if len(e) != 1 or not isinstance(e[0], dict):
                    raise ValueError(f"/pending_mentions/{i}/evidence: use ONE original context object, or separate independent pending items; do not combine multiple scopes")
                e = e[0]
            if not isinstance(e, dict) or not isinstance(e.get("text"), str):
                raise ValueError(f"/pending_mentions/{i}/evidence must be a single text object")


def budget_strength(clause):
    """A modifier must belong to the monetary phrase, not another requirement."""
    if re.search(r"如果|假如|随便|随意|不限|无所谓|可能|听说|市场价|参考价|行情|不要求|无需|不用|不需要|没有预算|不设预算", clause):
        return None
    has_soft = bool(re.search(SOFT, clause))
    has_hard = bool(re.search(HARD, clause))
    has_hope = "希望" in clause
    if has_soft and has_hard:
        return None
    target = rf"(?:价格|预算|售价|价钱|价位|总价|单价|运费|定金|尾款|加工费|控制|不超过|在|{NUM})"
    bound_soft = bool(re.search(rf"(?:{SOFT})(?:是|能|要|的)?{target}|(?:预算|价格|售价)(?:{SOFT})", clause))
    bound_hard = bool(re.search(rf"(?:{HARD})(?:是|能|要|的)?{target}|(?:预算|价格|售价)(?:{HARD})", clause))
    if has_soft:
        return "preference" if bound_soft else None
    if has_hard:
        return "required" if bound_hard else None
    if has_hope:
        return "uncertain"
    return "required"


def money_only(text):
    """Narrow grammar for deleting a redundant *monetary* pending flag only."""
    if not re.search(MONEY, text):
        return False
    rest = text
    words = ("不要超过", "不能超过", "不超过", "不低于", "不少于", "差不多", "大约", "大概", "左右", "上下", "以内", "以下", "以上",
             "出头", "至少", "最多", "不到", "小于", "大于", "超过", "低于", "预算", "价格", "售价", "价位", "价钱", "总价", "单价",
             "运费", "定金", "尾款", "加工费", "块钱", "每", "瓶", "袋", "盒", "套", "件", "本", "米", "公斤", "千克", "克", "毫升", "升",
             "最好", "优先", "尽量", "偏好", "喜欢", "必须", "务必", "一定要", "一共", "要求", "希望", "我的", "控制", "能搞定", "就行", "即可", "推荐", "能", "有", "吗", "元", "块", "钱", "多", "在", "的", "要", "了")
    rest = re.sub("|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)), "", rest)
    rest = re.sub(NUM, "", rest)
    return not rest.strip(" ，,。；;！？!?/\\-~～")


def resolved_pending(c, r, clause):
    if budget_strength(clause) not in ("required", "preference"):
        return []
    # A unique original clause anchors deletion; broad or mixed pending is kept.
    starts = [i for i in range(len(c["query"])) if c["query"].startswith(clause, i)]
    if len(starts) != 1:
        return []
    lo, hi = starts[0], starts[0] + len(clause)
    return [p for p in c["pending_mentions"] if p["reason"] in ("undefined_tolerance", "ambiguous_strength")
            and lo <= p["evidence"]["start"] < p["evidence"]["end"] <= hi and money_only(p["evidence"]["text"])]


def apply_budget_policy(c, r, clause, audit):
    target = budget_strength(clause)
    if target is not None and r["importance"] != target:
        before = r["importance"]
        r["importance"] = target
        audit.append({"kind": "source_budget_strength", "id": r["id"], "source": clause, "before": before, "after": target})
    remove = resolved_pending(c, r, clause)
    for p in remove:
        c["pending_mentions"].remove(p)
        audit.append({"kind": "resolved_budget_pending", "id": r["id"], "before": copy.deepcopy(p), "after": None, "source": clause})


def policy_flags(c, clause_parser):
    flags = []
    for i, r in enumerate(c["budget"]):
        if r["expected"]["type"] not in ("number", "range"):
            continue
        clauses = []
        for e in r["evidence"]:
            try:
                clauses.append(clause_parser(c["query"], e["text"]))
            except ValueError:
                continue
        targets = {budget_strength(clause) for clause in clauses}
        targets.discard(None)
        if len(targets) == 1 and r["importance"] != next(iter(targets)):
            flags.append({"code": "budget_strength_source", "location": f"/budget/{i}", "detail": "Apply confirmed budget policy and source modality; numeric approximation alone is not uncertainty"})
        if any(resolved_pending(c, r, clause) for clause in clauses):
            flags.append({"code": "resolved_budget_still_pending", "location": f"/budget/{i}", "detail": "Do not retain monetary tolerance/strength pending already resolved by the confirmed policy"})
    for i, p in enumerate(c["pending_mentions"]):
        t = p["evidence"]["text"]
        if p["reason"] == "ambiguous_meaning" and re.search(r"能用于|也能拿来做|需具备", t):
            covered = any(any(e["start"] <= p["evidence"]["start"] and e["end"] >= p["evidence"]["end"] for e in r["evidence"])
                          for r in c["functions"])
            if not covered:
                flags.append({"code": "objective_requirement_pending_only", "location": f"/pending_mentions/{i}", "detail": "Retain the explicit use/function as a requirement; pending alone must not hide it. Preserve real unresolved aspects separately."})
        if p["reason"] in ("ambiguous_meaning", "ambiguous_strength") and re.search(r"报考|准备结婚|检修|松动|不工作了|^急需$", t) and not re.search(r"想|要|需.+款|希望|能用于|适配|型号", t):
            flags.append({"code": "background_as_pending", "location": f"/pending_mentions/{i}", "detail": "Review whether this is background rather than an unresolved product requirement; do not automatically discard a genuinely unclear relation"})
    return flags
