"""Pure, reproducible goal helpers for Environment v2."""

from __future__ import annotations

import hashlib
import math
import random
import re


CONSTRAINT_CONTRACT_VERSION = "shopping-task-constraints-v1"


def _clean_annotation_values(value):
    """Normalize task-authored annotations without inferring new constraints."""
    if not isinstance(value, list):
        return [], False
    cleaned = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned, True


def compile_task_constraint_contract(instruction_record):
    """Compile the task's existing annotations into a fail-closed contract.

    Environment v2 uses only fields already attached to the current natural
    language instruction. It deliberately does not infer brand/model or read
    extra attributes from the target product.
    """
    if not isinstance(instruction_record, dict):
        instruction_record = {}
    attributes, attributes_valid = _clean_annotation_values(
        instruction_record.get("attributes")
    )
    options, options_valid = _clean_annotation_values(
        instruction_record.get("instruction_options")
    )
    instruction_valid = bool(
        re.sub(r"\s+", "", str(instruction_record.get("instruction") or ""))
    )
    complete = instruction_valid and attributes_valid and options_valid
    return {
        "hard_constraints": {
            "complete": complete,
            "contract_version": CONSTRAINT_CONTRACT_VERSION,
            "annotation_source": "instruction.attributes",
            "core_functions": attributes,
            "brand": [],
            "model": [],
            "key_specs": [],
            # Options continue to be checked by Reward v2's key_options gate.
            "annotated_option_count": len(options),
        },
        # The current task data does not label hard versus soft preferences.
        # Do not guess that distinction from keywords.
        "weighted_preferences": [],
    }


def _baseline_explicit_budget_v1(instruction):
    """Extract a clearly stated upper budget; return None when ambiguous."""
    text = str(instruction or "").replace(",", "")

    def scaled(number, unit):
        value = float(number)
        normalized_unit = str(unit or "").casefold()
        if normalized_unit == "万":
            value *= 10000
        elif normalized_unit in {"千", "k"}:
            value *= 1000
        return value

    shorthand = re.search(
        r"预算(?:控制)?在?\s*(\d+)\s*万\s*(\d+)\s*(?:千)?\s*(以内|以下|内|左右)?",
        text,
    )
    if shorthand:
        value = float(shorthand.group(1)) * 10000 + float(shorthand.group(2)) * 1000
        if shorthand.group(3) == "左右":
            value *= 1.1
        return value

    # Ranges are explicit upper bounds. Parse them before single-value
    # patterns so “10-20” is never mistaken for an upper bound of 10.
    price_range = re.search(
        r"(?:预算|价格)(?:控制)?在?\s*"
        r"(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?\s*"
        r"(?:-|~|～|至|到)\s*"
        r"(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?"
        r"(?:之间|以内|以下|左右)?",
        text,
    )
    if price_range:
        low = scaled(price_range.group(1), price_range.group(2))
        high = scaled(price_range.group(3), price_range.group(4))
        if low > 0 and high >= low:
            return high

    # “4k+” describes a lower/open-ended price, not a maximum.
    if re.search(
        r"(?:预算|价格)(?:控制)?在?\s*\d+(?:\.\d+)?\s*[kK]\s*\+",
        text,
    ):
        return None

    patterns = (
        r"预算(?:控制)?在?\s*(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?(以内|以下|内|左右)?(?!\s*[-~～至到+kK])",
        r"价格(?:控制)?在?\s*(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?(以内|以下|内|左右)?(?!\s*[-~～至到+kK])",
        r"(?:不超过|不高于|最高)\s*(\d+(?:\.\d+)?)\s*(万|千)?\s*元",
        r"(\d+(?:\.\d+)?)\s*(万|千)?\s*元(以内|以下)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = scaled(match.group(1), match.group(2))
            qualifier = match.group(3) if match.lastindex and match.lastindex >= 3 else None
            # “左右” is not a strict upper bound. Environment v2 freezes a
            # small deterministic tolerance rather than letting target items
            # just above the round-number budget become impossible.
            if qualifier == "左右":
                value *= 1.1
            if value > 0:
                return value
    return None


def _price_range_above(price):
    if price <= 100:
        step = 3
    elif price <= 1000:
        step = 10
    elif price <= 5000:
        step = 50
    elif price <= 10000:
        step = 100
    else:
        step = 4
    base = math.ceil(price / 10) * 10
    return [base + index * 10 for index in range(step)]


def deterministic_price_upper(asin, instruction, price):
    explicit = explicit_budget_from_instruction(instruction)
    if explicit is not None:
        return explicit
    price_range = _price_range_above(float(price))
    if len(price_range) < 2:
        return 10000000
    digest = hashlib.sha256(
        f"{asin}\0{instruction}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    _, upper = sorted(rng.sample(price_range, 2))
    return upper


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


def _v1_explicit_budget_v2(instruction):
    details = budget_details_v1(instruction)
    # Ambiguity is not permission to silently remove an existing budget.
    if details["status"] == "review_required":
        return _baseline_explicit_budget_v1(instruction)
    return details["upper"]


import unicodedata
from decimal import Decimal, InvalidOperation
MONEY_TOKEN_V2 = '(?:\\d+(?:\\.\\d+)?(?:万\\d*千?|[千kK])?|[零〇一二两三四五六七八九十百千万]+)'

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


def explicit_budget_from_instruction(instruction):
    details = budget_details_v2(instruction)
    return details["upper"] if details["status"] != "review_required" else _v1_explicit_budget_v2(instruction)
