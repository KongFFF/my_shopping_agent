"""Query-only budget contract v2 and conservative annotation repair.

No task IDs, gold prices or target ASINs enter these runtime rules.
"""

import re
import unicodedata
from decimal import Decimal, InvalidOperation

from task_contract_v1 import budget_details_v1


MONEY_TOKEN_V2 = r"(?:\d+(?:\.\d+)?(?:万\d*千?|[千kK])?|[零〇一二两三四五六七八九十百千万]+)"


def money_number_v2(value):
    """Parse only supported money tokens, preserving explicit 零 distinctions."""
    value = str(value)
    arabic = re.fullmatch(r"(\d+(?:\.\d+)?)(万|千|[kK])?", value)
    if arabic:
        return Decimal(arabic[1]) * {None: 1, "万": 10000, "千": 1000, "k": 1000, "K": 1000}[arabic[2]]
    mixed = re.fullmatch(r"(\d+)万(\d+)(千)?", value)
    if mixed:
        if not mixed[3] and len(mixed[2]) == 2:
            return None
        return Decimal(mixed[1]) * 10000 + Decimal(mixed[2]) * (1000 if mixed[3] or len(mixed[2]) == 1 else 1)
    digits = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
    if not value or any(c not in digits and c not in "十百千万" for c in value):
        return None
    if any(a in digits and b in digits and a not in "零〇" for a, b in zip(value, value[1:])):
        return None  # Colloquial ranges are parsed separately, not as scalars.
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
    result = total + section + current
    short = re.search(r"([百千万])([一二两三四五六七八九])$", value)
    if short:
        digit = digits[short[2]]
        result += digit * ({"百": 10, "千": 100, "万": 1000}[short[1]] - 1)
    return Decimal(result)


def chutou_upper_v2(value):
    """User-approved: two digits +4; >=3 digits +(10**(digits-2)-1)."""
    value = Decimal(str(value))
    if value != value.to_integral_value() or value < 10:
        return None
    digits = len(str(int(value)))
    return value + (4 if digits == 2 else 10 ** (digits - 2) - 1)


def colloquial_range_v2(token):
    match = re.fullmatch(r"(?P<prefix>[零〇一二两三四五六七八九十百千万]*?)(?P<first>[一二两三四五六七八九])(?P<second>[一二两三四五六七八九])十", token)
    if not match:
        return None
    first, second = money_number_v2(match["first"]), money_number_v2(match["second"])
    prefix = money_number_v2(match["prefix"]) if match["prefix"] else Decimal(0)
    if prefix is None or second != first + 1:
        return None
    return prefix + first * 10, prefix + second * 10


def budget_details_v2(instruction):
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(instruction or "")))
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = text.replace("块钱", "元").replace("块", "元")
    records, spans = [], []
    errors = []

    def context(start, end):
        clause = re.split(r"[,，。;；!！?？]", text[:start])[-1]
        after = text[end:]
        component = bool(re.search(r"每米|每克|每平|每件|每套|每个|单价|工费|手工费|加工费|辅料|花边|定金|订金|运费", clause)
                         or re.match(r"[/每](?:米|克|平|件|套|个)", after))
        budget_intent = bool(re.search(r"预算|总价|总价格|总金额|总费用|能拿下", clause)
                             or re.match(r"(?:的)?预算|能拿下", after))
        return clause, after, component, budget_intent

    def record(start, end, lower, upper, *, lower_inclusive=True, upper_inclusive=True, kind="scalar", component=False, budget_intent=False):
        records.append({"start": start, "end": end, "quote": text[max(0, start - 18):end + 12],
                        "lower": float(lower) if lower is not None else None,
                        "upper": float(upper) if upper is not None else None,
                        "lower_inclusive": lower_inclusive, "upper_inclusive": upper_inclusive,
                        "kind": kind, "component": component, "budget_intent": budget_intent})

    range_pattern = rf"(?:预算|价格|售价|价钱|总价)(?:区间)?(?:控制)?(?:要)?(?:在|为|是)?(?P<low>{MONEY_TOKEN_V2})元?(?:-|~|～|至|到)(?P<high>{MONEY_TOKEN_V2})元?"
    for match in re.finditer(range_pattern, text):
        low, high = money_number_v2(match["low"]), money_number_v2(match["high"])
        if low is None or high is None or not 0 < low <= high:
            errors.append("unparsed_or_invalid_range")
            continue
        _, _, component, budget_intent = context(match.start(), match.end())
        # An explicit numeric interval is checked on both sides, even after 预算.
        record(match.start(), match.end(), low, high, kind="explicit_range", component=component, budget_intent=budget_intent)
        spans.append(match.span())

    for match in re.finditer(rf"(?P<number>{MONEY_TOKEN_V2})(?P<currency>元|人民币)?", text):
        if any(start <= match.start() < end for start, end in spans):
            continue
        clause, after, component, budget_intent = context(match.start(), match.end())
        cue = bool(re.search(r"(?:预算|价格|售价|价钱|总价|总预算|总金额|总价格|全款|总计|总费用)(?:控制)?(?:大概|大约|大致|差不多|约)?(?:在|为|是|要在)?$", clause))
        inherited_price_cue = bool(re.search(r"(?:预算|价格|售价|价钱)[^,，。;；!！?？]{0,8}[,，](?:大概|大约|大致|差不多|约)?$", text[:match.start()]))
        cue = cue or inherited_price_cue
        upper_cue = bool(re.search(r"(?:不(?:得|要|能)?(?:超过|超出|高于)|别(?:超过|超出|超)|最高|最多|不超|低于|小于|少于|不到)(?:为|是)?$", clause))
        lower_cue = bool(re.search(r"(?:不(?:能|得)?低于|不少于|最低|至少|高于|大于|超过)(?:为|是)?$", clause))
        if re.search(r"(?:不(?:能|得)?低于|不少于)$", clause):
            upper_cue = False
        elif upper_cue:
            lower_cue = False
        deferred_currency = bool(re.match(r"(?:多|几)(?:元|人民币)", after))
        implicit_spending = bool(re.fullmatch(r"(?:在|大概在|大约在)", clause)
                                 and re.match(r"(?:左右|上下)(?:的)?(?:都)?能接受", after))
        price_context = inherited_price_cue or bool(re.search(r"预算|价格|售价|价钱|总价|总价格|总金额|总费用|工费|单价|运费|定金|订金", clause))
        if not match["currency"] and not deferred_currency and not implicit_spending and not price_context:
            continue
        if not match["currency"] and not deferred_currency and not cue and not upper_cue and not lower_cue and not implicit_spending:
            continue
        if not match["currency"] and re.match(r"(?:毫米|厘米|公分|cm|mm|米|寸|克|斤|公斤|kg|g|瓦|W|伏|V|度|%|颗|码|号|个|升|毫升|ml|支|枝|包|年|月|天)", after, re.I):
            continue
        if not cue and not upper_cue and not lower_cue and not re.search(r"预算|价格|售价|价钱|总价|全款|总计|总费用|花费|价值|工费|定金|订金|单价|运费", clause):
            if not implicit_spending and not re.match(r"以内|以下|左右|上下|出头|以上|多元|多|几元", after):
                continue
        value = money_number_v2(match["number"])
        spoken = colloquial_range_v2(match["number"])
        if spoken is not None:
            low, high = spoken
            record(match.start(), match.end(), None if budget_intent else low, high, kind="colloquial_range", component=component, budget_intent=budget_intent)
            continue
        if value is None or value <= 0:
            errors.append("unparsed_price_language")
            continue
        hard_upper = upper_cue or bool(re.match(r"以内|以下|内", after))
        lower = lower_cue or bool(re.match(r"以上|起", after))
        if lower:
            strict = bool(re.search(r"(?:高于|大于|超过)$", clause))
            record(match.start(), match.end(), value, None, lower_inclusive=not strict, kind="explicit_lower", component=component, budget_intent=budget_intent)
        elif hard_upper:
            strict = bool(re.search(r"(?:低于|小于|少于|不到)$", clause))
            record(match.start(), match.end(), None, value, upper_inclusive=not strict, kind="hard_upper", component=component, budget_intent=budget_intent)
        elif re.match(r"出头", after):
            high = chutou_upper_v2(value)
            if high is None:
                errors.append("unsupported_chutou_magnitude")
            else:
                # Explicitly approved as an upper budget only, no minimum spend.
                record(match.start(), match.end(), None, high, kind="chutou", component=component, budget_intent=budget_intent)
        elif re.match(r"多", after):
            if value != value.to_integral_value() or int(value) % 10:
                errors.append("unsupported_duo_magnitude")
                continue
            trailing = len(str(int(value))) - len(str(int(value)).rstrip("0"))
            high = value + Decimal(10) ** trailing
            record(match.start(), match.end(), None if budget_intent else value, high,
                   lower_inclusive=False, upper_inclusive=False, kind="duo", component=component, budget_intent=budget_intent)
        elif re.match(r"几", after):
            if value % 10:
                errors.append("unsupported_ji_magnitude")
                continue
            record(match.start(), match.end(), None if budget_intent else value + 1, value + 9,
                   kind="ji", component=component, budget_intent=budget_intent)
        else:
            approximate = bool(re.match(r"左右|上下", after) or re.search(r"(?:大概|大约|大致|差不多|约)(?:(?:预算|价格|售价|总价))?(?:控制)?(?:在|为|是)?$", clause))
            record(match.start(), match.end(), None, value * (Decimal("1.1") if approximate else 1),
                   kind="approximate" if approximate else "scalar", component=component, budget_intent=budget_intent)

    base = {"version": "task-budget-v2", "lower": None, "upper": None, "lower_inclusive": True,
            "upper_inclusive": True, "unit_scope": None, "evidence": records}
    if errors:
        return {**base, "status": "review_required", "reason": ";".join(sorted(set(errors)))}
    standalone = [r for r in records if not r["component"]]
    components = [r for r in records if r["component"]]
    if re.search(r"定金|订金", text) and not standalone:
        return {**base, "status": "review_required", "reason": "deposit_transaction_unsupported"}
    if components and standalone:
        return {**base, "status": "review_required", "reason": "component_constraints_need_evidence"}
    unit_scope = None
    if components and not standalone:
        # Only the approved one-metre case is supported. Other per-unit or fee
        # compositions remain explicit evidence gaps, not invented totals.
        one_metre = bool(re.search(r"长度(?:为|是|在)?(?:一|1)米", text))
        conflicting_quantity = bool(re.search(r"(?:需要|要|买)(?:[2-9]|\d{2,}|[二两三四五六七八九十])(?:条|根|份|卷)", text))
        if len(components) == 1 and one_metre and "单价" in text and not conflicting_quantity:
            standalone = components
            unit_scope = {"kind": "length", "quantity": 1, "unit": "m"}
        else:
            return {**base, "status": "review_required", "reason": "component_price_evidence_missing"}
    if not standalone:
        legacy = budget_details_v1(instruction)
        # Preserve known supported v1 price language, not its unsupported/lower
        # cases. Runtime never treats unparsed declared budgets as no budget.
        if legacy["status"] == "parsed" and not records:
            return {**base, "status": "parsed", "reason": "v1_supported_language", "upper": legacy["upper"], "evidence": legacy["evidence"]}
        if legacy["status"] == "review_required":
            return {**base, "status": "review_required", "reason": legacy["reason"]}
        return {**base, "status": "not_declared", "reason": "no_price_constraint"}
    low_records = [r for r in standalone if r["lower"] is not None]
    high_records = [r for r in standalone if r["upper"] is not None]
    lower = max((r["lower"] for r in low_records), default=None)
    upper = min((r["upper"] for r in high_records), default=None)
    low_inc = all(r["lower_inclusive"] for r in low_records if r["lower"] == lower)
    high_inc = all(r["upper_inclusive"] for r in high_records if r["upper"] == upper)
    if lower is not None and upper is not None and (lower > upper or lower == upper and not (low_inc and high_inc)):
        return {**base, "status": "review_required", "reason": "contradictory_price_constraints"}
    return {**base, "status": "parsed", "reason": "query_price_contract", "lower": lower, "upper": upper,
            "lower_inclusive": low_inc, "upper_inclusive": high_inc, "unit_scope": unit_scope}


def price_gate_v2(price_resolution, goal, *, product=None, selected_options=None):
    from web_agent_site.engine.comparators import PASS, FAIL, UNVERIFIABLE, comparison

    contract = goal.get("budget_contract_v2")
    if not isinstance(contract, dict):
        baseline = globals().get("_baseline_price_gate_v2")
        if baseline is None:
            from web_agent_site.engine.reward import _price_gate as baseline
        return baseline(price_resolution, goal)
    actual = price_resolution.get("price") if isinstance(price_resolution, dict) else None
    required = {key: contract.get(key) for key in ("lower", "lower_inclusive", "upper", "upper_inclusive", "unit_scope")}
    evidence = {"price_resolution": price_resolution, "budget_contract": contract}
    status = PASS
    if contract["status"] == "review_required":
        status = UNVERIFIABLE
    elif contract["status"] == "not_declared":
        return comparison(PASS, comparator="budget_not_declared_v2", required=None, actual=actual, source_field="instruction", evidence=evidence)
    elif not isinstance(price_resolution, dict) or price_resolution.get("status") != PASS:
        status = UNVERIFIABLE
    else:
        try:
            price = Decimal(str(actual))
            if not price.is_finite() or price < 0:
                status = UNVERIFIABLE
            for name, inclusive, lower_bound in (("lower", "lower_inclusive", True), ("upper", "upper_inclusive", False)):
                if contract.get(name) is None:
                    continue
                limit = Decimal(str(contract[name]))
                if not limit.is_finite():
                    status = UNVERIFIABLE
                    break
                failed = price < limit if lower_bound else price > limit
                failed = failed or price == limit and not contract.get(inclusive, True)
                if failed and status != UNVERIFIABLE:
                    status = FAIL
        except (InvalidOperation, ValueError, TypeError):
            status = UNVERIFIABLE
        if contract.get("unit_scope"):
            # The selected, publicly listed option must provide the unit proof.
            values = selected_options.values() if isinstance(selected_options, dict) else []
            lengths = []
            for value in values:
                text = unicodedata.normalize("NFKC", str(value)).casefold()
                lengths.extend(Decimal(m[1]) for m in re.finditer(r"(?<![\d.])([0-9]+(?:\.[0-9]+)?)\s*(?:米|m(?![a-z0-9]))(?![²平方])", text))
            evidence["selected_lengths_m"] = [float(v) for v in lengths]
            if not lengths:
                status = UNVERIFIABLE
            elif any(v != Decimal(1) for v in lengths):
                status = FAIL
    return comparison(status, comparator="variant_price_budget_v2", required=required, actual=actual,
                      source_field="variant_price|query", evidence=evidence)


def resolve_damaged_option_v2(value, product, query):
    """Restore only unique full matches differing in approved glyphs/punctuation."""
    def normalized(text):
        return re.sub(r"\s+", "", unicodedata.normalize("NFC", str(text))).casefold().replace("/", "|")

    required = normalized(value)
    candidates = [(axis, str(entry["value"])) for axis, entries in (product.get("customization_options") or {}).items()
                  for entry in entries if entry.get("value") is not None]
    if any(normalized(candidate) == required for _, candidate in candidates):
        return None
    glyphs = "❤♥⭐★☆➕+✅✔❄\ufe0f"
    parts = []
    for index, char in enumerate(required):
        if char != "?":
            parts.append(re.escape(char))
            continue
        allowed = glyphs
        # ² is allowed solely in a marked square-metre expression corroborated
        # by query text. Never permit an arbitrary digit through '?'.
        prefix = required[:index]
        area = re.search(r"(\d+(?:\.\d+)?)m$", prefix)
        area_supported = bool(area and (re.search(re.escape(area[1]) + r"\s*(?:平方米|平方|平米|m²)", str(query), re.I)
                                       or area[1] == "1" and re.search(r"一平米|一平方米", str(query))))
        if area:
            if not area_supported:
                return None
            allowed = "²"
        parts.append("[" + re.escape(allowed) + "]")
    pattern = "".join(parts)
    matched = {(axis, candidate) for axis, candidate in candidates
               if ("?" in required and re.fullmatch(pattern, normalized(candidate)))
               or normalized(candidate).rstrip(";；") == required.rstrip(";；")}
    return next(iter(matched)) if len(matched) == 1 else None
