"""Conservative, query-only repairs for the task/reward contract.

No task IDs, target ASINs or target prices enter the budget/brand rules.
Ambiguous price language is reviewed rather than assigned a convenient bound.
"""

import re


def budget_details_v1(instruction):
    import re
    import unicodedata
    from decimal import Decimal

    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(instruction or "")))
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = text.replace("块钱", "元").replace("块", "元")
    numeral = r"(?:\d+(?:\.\d+)?(?:万\d*千?|[千kK])?|[零〇一二两三四五六七八九十百千万]+)"

    def number(value):
        shorthand = re.fullmatch(r"(\d+(?:\.\d+)?)万(\d+)(千)?", value)
        if shorthand:
            if not shorthand[3] and len(shorthand[2]) == 2:
                return None
            suffix = Decimal(shorthand[2]) * (1000 if shorthand[3] or len(shorthand[2]) == 1 else 1)
            return Decimal(shorthand[1]) * 10000 + suffix
        arabic = re.fullmatch(r"(\d+(?:\.\d+)?)(万|千|[kK])?", value)
        if arabic:
            return Decimal(arabic[1]) * {None: 1, "万": 10000, "千": 1000, "k": 1000, "K": 1000}[arabic[2]]
        digits = dict(zip("零〇一二两三四五六七八九", [0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9]))
        if len(value) > 1 and any(a in digits and b in digits and a not in "零〇" for a, b in zip(value, value[1:])):
            return None  # e.g. 三百四五十 is a colloquial range, not 3450.
        # Colloquial trailing digits such as 两百五 are not silently read as 205.
        if re.search(r"[百千万][一二两三四五六七八九]$", value):
            return None
        total = section = current = 0
        for char in value:
            if char in digits:
                current = digits[char]
            elif char == "万":
                total += (section + current) * 10000
                section = current = 0
            else:
                section += (current or 1) * {"十": 10, "百": 100, "千": 1000}[char]
                current = 0
        return Decimal(total + section + current)

    records = []
    range_spans = []
    for match in re.finditer(rf"(?:预算|价格|售价)(?:区间)?(?:控制)?(?:在|为|是)?(?P<low>{numeral})元?(?:-|~|～|至|到)(?P<high>{numeral})元?", text):
        low, high = number(match["low"]), number(match["high"])
        if low is None or high is None or not 0 < low <= high:
            continue
        clause = re.split(r"[,，。;；!！?？]", text[:match.start()])[-1]
        records.append({"start": match.start(), "end": match.end(), "value": float(high), "upper": float(high),
                        "range_lower": float(low), "total_scope": False, "component": bool(re.search(r"(?:每米|每克|单价)", clause)),
                        "lower": False, "vague": False, "approximate": False, "quote": match[0]})
        range_spans.append(match.span())
    for match in re.finditer(rf"(?P<number>{numeral})(?P<currency>元|人民币)?", text):
        if any(start <= match.start() < end for start, end in range_spans):
            continue
        before, after = text[:match.start()], text[match.end():]
        clause = re.split(r"[,，。;；!！?？]", before)[-1]
        cue = re.search(r"(?:预算|价格|售价|价钱|总价|总预算|总金额|总价格|全款|总计|总费用)(?:控制)?(?:大概|大约|差不多|约)?(?:在|为|是)?$", clause)
        bound_cue = re.search(r"(?:不超过|不高于|不得超过|不得超出|别超|不要超过|不能超过|最高|最多|不低于|不少于|最低)(?:为|是)?$", clause)
        if not match["currency"] and not cue and not bound_cue:
            continue
        if not match["currency"] and after and not re.match(r"(?:以内|以下|内|左右|上下|出头|以上|多|起|[,，。;；!！?？]|的?预算)", after):
            continue
        # Currency is necessary but does not turn a bare monetary fact into a budget.
        if not cue and not bound_cue and not re.search(r"(?:预算|价格|售价|价钱|总价|全款|总计|总费用|花费|价值|工费|定金|订金|单价|运费|不超过|以内|以下)", clause):
            if not re.match(r"(?:以内|以下|左右|上下|出头|以上|元|多元|的?预算)", after):
                continue
        value = number(match["number"])
        if value is None or value <= 0:
            continue
        total_scope = bool(re.search(r"(?:总价|总预算|总金额|总价格|总费用|总计|全款)", clause))
        component = bool(re.search(r"(?:每米|每克|每平|每件|每套|每个|单价|工费|手工费|加工费|辅料|花边|定金|订金|运费)", clause) or re.match(r"[/每](?:米|克|平|件|套|个)", after))
        lower = bool(re.search(r"(?:不低于|不少于|最低)$", clause) or re.match(r"(?:以上|起)", after))
        approximate = bool(re.match(r"(?:左右|上下)", after) or re.search(r"(?:大概|大约|大致|差不多|约)(?:(?:预算|价格|售价|总价))?(?:控制)?(?:在|为|是)?$", clause))
        if re.match(r"(?:以内|以下|内)", after) or bound_cue and re.search(r"(?:不超过|不高于|不得超过|不得超出|别超|不要超过|不能超过|最高|最多)$", clause):
            approximate = False  # An explicit hard cap wins over hedging words.
        vague = bool(re.match(r"(?:出头|多(?:元)?|来元)", after))
        record = {"start": match.start(), "end": match.end(), "value": float(value),
                  "upper": float(value * Decimal("1.1")) if approximate else float(value),
                  "total_scope": total_scope, "component": component, "lower": lower,
                  "vague": vague, "approximate": approximate, "quote": text[max(0, match.start()-15):match.end()+8]}
        records.append(record)
    records.sort(key=lambda record: record["start"])
    # Recognize both endpoints; only the upper endpoint enters the existing upper-budget gate.
    for first, second in zip(records, records[1:]):
        between = text[first["end"]:second["start"]]
        if re.fullmatch(r"(?:-|~|～|至|到)", between):
            first["lower"] = True
            second["lower"] = False
            second["upper"] = second["value"]
            second["range_lower"] = first["value"]
    candidates = [r for r in records if not r["component"] and not r["lower"]]
    totals = [r for r in candidates if r["total_scope"]]
    if totals:
        candidates = totals
    if candidates and any(r["vague"] for r in candidates):
        return {"status": "review_required", "reason": "vague_price_bound", "upper": None, "evidence": records}
    if candidates:
        bounds = {r["upper"] for r in candidates}
        return {"status": "parsed", "reason": "explicit_total_upper", "upper": min(bounds), "evidence": records}
    if records:
        return {"status": "review_required", "reason": "component_or_lower_bound_only", "upper": None, "evidence": records}
    # If a price is mentioned but could not be parsed, do not manufacture a bound.
    if re.search(r"(?:预算|价格|售价|总价|价钱).{0,12}(?:\d|[一二三四五六七八九十百千万])|(?:不超过|不高于|不得超出).{0,12}元", text):
        return {"status": "review_required", "reason": "unparsed_price_language", "upper": None, "evidence": []}
    return {"status": "not_declared", "reason": "no_upper_budget", "upper": None, "evidence": []}


def filter_required_brands_v1(instruction, brands):
    import re
    import unicodedata
    from web_agent_site.engine.comparators import load_brand_aliases

    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(instruction))).casefold()
    aliases = load_brand_aliases()
    result = []
    for brand in brands:
        roles = []
        for alias, canonical in aliases.items():
            if canonical != brand or len(alias) < 2:
                continue
            for match in re.finditer(re.escape(alias), text):
                prefix = text[max(0, match.start()-16):match.start()]
                suffix = text[match.end():match.end()+8]
                auxiliary = bool(re.search(r"(?:适用(?:于)?|适配|兼容|支持|通用(?:于)?|代替|替代|类似(?:于)?|平替|接入|连接|用在|适合)$", prefix)
                                 or re.match(r"(?:专卖店|系统|生态|homekit)", suffix))
                roles.append(auxiliary)
        if not roles or not all(roles):
            result.append(brand)
    return result


def resolve_damaged_option_v1(value, product):
    """Return a unique match only when '?' replaces a known decorative glyph.

    This is annotation repair, not a permissive reward comparator. Digits,
    Chinese characters, colors and model names cannot be wildcard-matched.
    """
    from web_agent_site.engine.reward_features import normalize_option_text

    normalized = normalize_option_text(value)
    if "?" not in normalized:
        return None
    pattern = "".join("[❤♥⭐★☆➕+]" if char == "?" else re.escape(char) for char in normalized)
    matches = {(axis, entry["value"]) for axis, entries in (product.get("customization_options") or {}).items()
               for entry in entries or [] if re.fullmatch(pattern, normalize_option_text(entry.get("value")))}
    return next(iter(matches)) if len(matches) == 1 else None
