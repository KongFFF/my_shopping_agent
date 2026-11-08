"""Independent offline semantic contract v2. No training/runtime integration.

LLM output has evidence text and no IDs/offsets. Normalization is audited and
deterministic; semantic uncertainty is retained, never silently approved.
"""
import copy
import json
import math
import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from jsonschema import Draft202012Validator
from semantic_contract_v1 import ROOT, obj, tagged
from semantic_scope_v7 import approximation_evidence, direct_hope_scopes, split_flags, unit_source_flags

VERSION = "shopping-semantic-requirements-v2-r7-closeout-draft"
PACKAGE = ROOT / "experiments/semantic_contract_v2_r7_closeout"
DIMENSIONS = {
    "category": ["product_category"],
    "budget": ["total_price", "unit_price", "deposit", "balance", "processing_fee", "shipping_fee"],
    "functions": ["function", "purpose", "target_user", "use_environment"],
    "specifications": ["quantity", "attribute_count", "bundle", "components", "color", "size", "dimensions",
                       "material", "capacity", "net_content", "compatibility", "merchant", "delivery", "warranty",
                       "packaging", "origin", "pattern", "form_factor", "flavor", "condition", "craft",
                       "customization", "installation_service", "composition", "style", "content", "weight", "manufacturer", "transaction_terms"],
    "brand": ["brand"], "model": ["product_model", "version"],
}
PURCHASE_UNITS = ["piece", "bottle", "bag", "box", "pair", "set", "book", "sheet", "roll", "tube"]
UNITS = PURCHASE_UNITS + ["CNY", "count", "kg", "g", "jin", "ml", "l", "mm", "cm", "m", "inch",
                          "W", "V", "A", "mAh", "Ah", "Wh", "kWh", "degree", "day", "month", "year"]
ATTRIBUTES = {
    "quantity": ["purchase_count", "purchase_mass", "purchase_volume", "per_variant_count"],
    "attribute_count": ["gear_count", "preset_count", "effect_type_count", "tone_type_count", "face_count", "other_feature_count"],
    "capacity": ["liquid_volume", "energy_capacity", "electric_charge", "load_capacity", "storage_capacity"],
    "dimensions": ["length", "width", "height", "diameter", "angle", "dimensions", "thickness"],
    "weight": ["mass"],
    "size": ["size_label", "wearer_height", "wearer_weight_limit", "shoe_upper_height", "sleeve_length", "size"],
    "net_content": ["mass", "liquid_volume"],
    "function": ["function", "power", "voltage", "drive_structure"],
    "transaction_terms": ["billing_method", "settlement_adjustment", "return_shipping_reimbursement"],
    "bundle": ["promotion"], "components": ["included_components", "gift_presence"],
    "packaging": ["packaging", "pack_count"],
}
for subtypes in DIMENSIONS.values():
    for subtype in subtypes:
        ATTRIBUTES.setdefault(subtype, [subtype])

POLICY = {
    "version": "semantic-contract-v2-policy-20261107-r7-closeout",
    "confirmed_by_user": True,
    "base_policy_confirmed_by_user": True,
    "budget_approximate_upper_factor": 1.1,
    "budget_chutou": "two digits +4; >=3 digits +10**(digits-2)-1; upper only",
    "budget_round_hundred_duo": "100..900 in whole hundreds +99; upper only; other duo phrases pending",
    "bare_unit_price": "upper_bound",
    "non_budget_approximation": "preserve_original_and_subject; tolerance pending; never apply budget factor",
    "ambiguous_gift_or_merchant_strength": "uncertain_and_pending",
    "subjective_quality_style_latest": "preserve_pending_until_evidence_standard_is_reviewed",
    "hope_without_other_strength_marker": "uncertain_and_pending; not automatically preference or required",
    "normalization_revision": "audited source-supported routing and quantity roles; no automatic approval",
    "packaging_and_scope": "preserve ambiguous AND/OR wording pending; never infer either-or",
    "permitted_material_strength": "preserve permission and pending strength; never automatically reward as preference",
    "bare_feature_count_relation": "exact count; explicit at-least/reaches is lower bound; duo is strict greater with range pending",
    "unresolved_decisions": ["exhaustive_three_piece_set", "coordinated_at_least_scope"],
    "revision": "frozen r7 behavior plus separately confirmed source-count proofs; no task-ID bypass",
    "elided_one_object_clear": "confirmed by user: 加个/配个/有个 with a clear object means one",
    "r7_representation": "six dimensions unchanged; excludes operator; manufacturer role; total mass/volume and per-variant quantity attributes",
    "ambiguous_negation": "not-needed alone never automatically becomes exclusion; explicit user decisions remain traced",
    "optional_accessory_omission": "confirmed by user: not needed is no requirement by default; explicit exclusion forbids; capability requirements remain required",
    "confirmed_colloquial_budget": "十几<=19; 四十几<=49; 三四十<=49; upper only; no extra approximation factor",
    "review3_money": "source-only monetary spans; preserve denominator and explicit hard/double limits; unconfirmed colloquial intervals pending",
    "review_findings": "separate diagnostics; no guessed strength feedback; block reward approval",
}
REASONS = ["ambiguous_meaning", "ambiguous_strength", "unsupported_type", "unsupported_unit", "contradiction",
           "undefined_tolerance", "unverifiable_subjective", "missing_reference_time"]


@lru_cache(maxsize=2)
def schema(*, llm=False):
    # Pure versioned schema, shared read-only by offline validators/exporters.
    text = {"type": "string", "minLength": 1}
    num = {"type": "number", "minimum": 0}
    nullable_num = {"type": ["number", "null"], "minimum": 0}
    evidence = obj({"text": text} if llm else {"text": text, "start": {"type": "integer", "minimum": 0},
                                               "end": {"type": "integer", "minimum": 1}})
    pending_evidence = ({"oneOf": [{"$ref": "#/$defs/evidence"},
                         {"type": "array", "minItems": 1, "maxItems": 1, "items": {"$ref": "#/$defs/evidence"}}]}
                        if llm else {"$ref": "#/$defs/evidence"})
    subject = obj({"kind": {"enum": ["primary_product", "component", "gift", "compatible_target", "intended_user", "order"]},
                   "name": text})
    basis = {"oneOf": [{"type": "null"}, obj({"amount": {"type": "number", "exclusiveMinimum": 0},
                                               "unit": {"enum": PURCHASE_UNITS + ["m", "cm", "kg", "g", "l", "ml", "m2"]}})]}
    expected = {"oneOf": [tagged("text", {"value": text}),
        tagged("text_list", {"values": {"type": "array", "items": text, "minItems": 1, "uniqueItems": True}}),
        tagged("number", {"value": num, "unit": {"enum": UNITS}}),
        tagged("range", {"lower": nullable_num, "upper": nullable_num, "unit": {"enum": UNITS},
                         "lower_inclusive": {"type": "boolean"}, "upper_inclusive": {"type": "boolean"}}),
        tagged("bundle", {"paid_quantity": {"type": ["integer", "null"], "minimum": 1},
                          "gift_quantity": {"type": ["integer", "null"], "minimum": 0},
                          "received_quantity": {"type": ["integer", "null"], "minimum": 1},
                          "unit": {"enum": PURCHASE_UNITS}, "same_item": {"const": True}}),
        tagged("components", {"items": {"type": "array", "minItems": 1, "items": obj({"name": text,
                   "quantity": nullable_num, "unit": {"enum": PURCHASE_UNITS + [None]}})}})]}
    props = {"subtype": {"enum": sorted(ATTRIBUTES)}, "subject": {"$ref": "#/$defs/subject"},
             "attribute": {"enum": sorted({a for values in ATTRIBUTES.values() for a in values})},
             "importance": {"enum": ["required", "preference", "uncertain"]},
             "operator": {"enum": ["eq", "one_of", "contains_all", "excludes", "at_least", "at_most", "between", "matches_bundle"]},
             "expected": {"$ref": "#/$defs/expected"}, "price_basis": basis,
             "evidence": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/evidence"}},
             "review_status": {"enum": ["pending", "approved"]}}
    if not llm:
        props["id"] = text
    root = {"schema_version": {"const": VERSION}, "task_id": {"type": "integer", "minimum": 0}, "query": text}
    for dimension, subtypes in DIMENSIONS.items():
        conditions = [{"$ref": "#/$defs/requirement"}, {"properties": {"subtype": {"enum": subtypes}}}]
        if llm and dimension == "budget":
            conditions.append({"properties": {"operator": {"const": "eq"}, "expected": tagged("text", {"value": text})}})
        root[dimension] = {"type": "array", "items": {"allOf": conditions}}
    root["pending_mentions"] = {"type": "array", "items": obj({"evidence": pending_evidence,
                  "reason": {"enum": REASONS}, "importance": {"enum": ["required", "preference", "uncertain"]}})}
    result = {"$schema": "https://json-schema.org/draft/2020-12/schema", **obj(root),
              "$defs": {"evidence": evidence, "subject": subject, "expected": expected, "requirement": obj(props)}}
    Draft202012Validator.check_schema(result)
    return result


def chinese_number(value):
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return Decimal(value)
    digits = {c: i for i, c in enumerate("零一二三四五六七八九")}
    digits.update({"两": 2, "〇": 0})
    scales = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    if not value or any(c not in digits and c not in scales for c in value):
        raise ValueError("Unsupported or malformed amount")
    if re.search(r"[百千][一二两三四五六七八九]$", value):
        raise ValueError("Colloquial abbreviated amount needs review")
    total = section = number = 0
    for c in value:
        if c in digits:
            number = number * 10 + digits[c]
        elif scales[c] == 10000:
            total += (section + number) * 10000
            section = number = 0
        else:
            section += (number or 1) * scales[c]
            number = 0
    return Decimal(total + section + number)


AMOUNT = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)"


class SourceMismatch(ValueError):
    pass


def price_basis_from_phrase(phrase):
    units = {"瓶": "bottle", "袋": "bag", "盒": "box", "套": "set", "件": "piece", "本": "book",
             "米": "m", "厘米": "cm", "平米": "m2", "平方米": "m2", "公斤": "kg", "千克": "kg",
             "克": "g", "升": "l", "毫升": "ml"}
    unit_expr = "|".join(sorted(units, key=len, reverse=True))
    each = re.search(f"每\\s*({AMOUNT})?\\s*({unit_expr})", phrase)
    after_price = re.search(f"(?:元|块钱|块)\\s*(?:/)?\\s*({AMOUNT})?\\s*({unit_expr})", phrase)
    found = each or after_price
    if not found:
        raise ValueError("Unit price denominator not explicitly supported in source phrase")
    amount = float(chinese_number(found.group(1))) if found.group(1) else 1
    return {"amount": amount, "unit": units[found.group(2)]}


def free_shipping(phrase):
    return bool(re.search(r"包邮(?:费)?|免运费|运费全免", phrase)) and not re.search(
        r"不包邮|无需包邮|不用包邮|是否|[?？]|不免运费|如果|假如|满|不足|偏远|除外|达到", phrase)


def budget_for_role(phrase, subtype):
    if re.search(r"预估|接近", phrase):
        raise ValueError("Budget relation has no confirmed policy; preserve original wording pending review")
    if subtype == "shipping_fee" and free_shipping(phrase):
        return "at_most", {"type": "number", "value": 0, "unit": "CNY"}, "explicit_free_shipping"
    if subtype == "shipping_fee" and re.search(r"包邮|免运费|运费全免", phrase):
        raise ValueError("Conditional/negated shipping offer needs scope review, not a numeric shipping fee")
    if subtype != "shipping_fee" and free_shipping(phrase) and not re.search(AMOUNT, phrase):
        raise SourceMismatch("Free shipping is a shipping_fee requirement, not total order price")
    return budget_amount(phrase)


def _base_budget_amount(phrase):
    """Interpret a quoted budget clause only. Unknown phrasing is NOT guessed."""
    if re.search(r"美元|美金|欧元|日元|USD|EUR|JPY", phrase, re.I):
        raise ValueError("Currency outside confirmed CNY policy")
    if free_shipping(phrase) and not re.search(AMOUNT, phrase):
        return "at_most", {"type": "number", "value": 0, "unit": "CNY"}, "explicit_free_shipping"
    # The numeral in the modality 一定要 (or total marker 一共) is not
    # a second price. Keep original source positions and all real amounts.
    idioms = [(m.start(), m.end()) for m in re.finditer(r"一定要|一共", phrase)]
    matches = [m for m in re.finditer(AMOUNT, phrase)
               if not any(lo <= m.start() and m.end() <= hi for lo, hi in idioms)]
    if not matches:
        raise ValueError("Budget amount not found")
    # A unit-price denominator such as 一瓶 is not a second monetary amount.
    monetary = [m for m in matches if not (m.start() > 0 and phrase[m.start()-1] == "每")
                and not re.match(r"(?:瓶|袋|盒|套|件|米|公斤|千克|克|升|毫升|本)", phrase[m.end():])]
    if not monetary:
        raise ValueError("Budget amount is ambiguous")
    ranged = re.search(f"({AMOUNT})\\s*(?:元|块钱|块)?\\s*(?:至|到|[-~～])\\s*({AMOUNT})", phrase)
    if ranged:
        low, high = (float(chinese_number(x)) for x in ranged.groups())
        return "between", {"type": "range", "lower": low, "upper": high, "unit": "CNY",
                           "lower_inclusive": True, "upper_inclusive": True}, "explicit_range"
    if len(monetary) != 1:
        raise ValueError("Multiple monetary amounts need relational review")
    match = monetary[0]
    amount = chinese_number(match.group())
    tail = phrase[match.end():]
    if "出头" in tail:
        if amount != int(amount) or amount < 10:
            raise ValueError("Chutou outside confirmed policy")
        n = len(str(int(amount)))
        upper = amount + (4 if n == 2 else 10 ** (n - 2) - 1)
        rule = "chutou_upper"
    elif re.match(r"(?:元|块钱|块)?多", tail):
        if not 100 <= amount <= 900 or amount % 100:
            raise ValueError("Duo is confirmed only for round-hundred multiples")
        upper, rule = amount + 99, "round_hundred_duo_upper"
    elif re.search(r"不低于|不少于|至少|以上", phrase):
        return "at_least", {"type": "number", "value": float(amount), "unit": "CNY"}, "explicit_lower"
    elif re.search(r"大于|超过", phrase) and not re.search(r"不超过|别超|不要超过|不能超过", phrase):
        return "between", {"type": "range", "lower": float(amount), "upper": None, "unit": "CNY",
                           "lower_inclusive": False, "upper_inclusive": False}, "explicit_open_lower"
    elif re.search(r"小于|不到|低于", phrase):
        return "between", {"type": "range", "lower": None, "upper": float(amount), "unit": "CNY",
                           "lower_inclusive": False, "upper_inclusive": False}, "explicit_open_upper"
    elif re.search(r"不超过|不要超过|不能超过|别超|以内|以下|最多|上限", phrase):
        upper, rule = amount, "explicit_upper"
    elif re.search(r"左右|上下|大约|大概|差不多", phrase):
        upper, rule = amount * Decimal("1.1"), "approximate_budget_upper"
    else:
        upper, rule = amount, "bare_budget_or_unit_price_upper"
    return "at_most", {"type": "number", "value": float(upper), "unit": "CNY"}, rule


def budget_amount(phrase):
    from semantic_targeted_money import bounded_money
    from semantic_closeout_policy import budget_amount as confirmed_budget
    return confirmed_budget(phrase, chinese_number, lambda p: bounded_money(p, chinese_number, _base_budget_amount))


def _evidences(contract):
    for dim in DIMENSIONS:
        for index, r in enumerate(contract[dim]):
            for j, e in enumerate(r["evidence"]):
                yield f"/{dim}/{index}/evidence/{j}", e
    for index, mention in enumerate(contract["pending_mentions"]):
        yield f"/pending_mentions/{index}/evidence", mention["evidence"]


def _quote(r):
    return "，".join(e["text"] for e in r["evidence"])


def _legacy_budget_source_clause(query, phrase):
    hits = [i for i in range(len(query)) if query.startswith(phrase, i)]
    if len(hits) != 1:
        raise ValueError("Budget phrase needs unique original context")
    start, end = hits[0], hits[0] + len(phrase)
    for clause in re.finditer(r"[^，。；！？?;!\n]+[。！？?;!]?", query):
        if clause.start() <= start and end <= clause.end():
            return clause.group()
    raise ValueError("Budget crossing clauses needs relational review")


def budget_source_clause(query, phrase):
    from semantic_targeted_money import money_context
    return money_context(query, phrase, _legacy_budget_source_clause)


def lint(contract):
    """Conservative review flags, not an automatic claim of semantic correctness."""
    flags = []
    def add(code, location, detail):
        flags.append({"code": code, "location": location, "detail": detail})
    seen = {}
    for dim in DIMENSIONS:
        for i, r in enumerate(contract[dim]):
            location = f"/{dim}/{i}"
            value = r["expected"]
            quote = _quote(r)
            text_value = value.get("value", "") if value["type"] == "text" else ""
            if dim == "budget" and r["importance"] == "preference" and not re.search(r"最好|优先|尽量|偏好|喜欢|希望", quote):
                add("budget_strength", location, "Approximation does not justify optional budget; require source soft modifier or pending strength")
            if dim == "budget" and r["importance"] == "required" and re.search(r"最好|优先|偏好", quote):
                add("budget_strength", location, "Explicit soft budget modifier cannot silently become mandatory")
            if r["subtype"] == "color" and re.search(r"香味|口味|图案|造型|福字", str(text_value)):
                add("suspicious_color", location, "Possible flavor/pattern/form assigned to color; review, do not auto-remap")
            if r["subtype"] == "quantity" and re.search(r"档位|音色|预设|失真|棱面|承重", quote):
                add("purchase_count_scope", location, "Feature count or physical capability is not purchase quantity")
            if r["subtype"] == "components" and value["type"] == "components":
                if any(x["quantity"] is not None for x in value["items"]) and not re.search(r"\d|[一二两三四五六七八九十百]", quote):
                    add("invented_component_count", location, "No explicit component quantity in source; keep null instead of defaulting to one")
            if value["type"] == "bundle":
                if re.search(r"几|一些|若干", quote) and value["gift_quantity"] is not None:
                    add("invented_gift_count", location, "Unspecified gift count must stay null, never zero/one")
                promotion = re.search(f"买({AMOUNT})(?:套|袋|件|瓶)?(?:送|赠)({AMOUNT})", quote)
                if promotion:
                    paid, gift = (int(chinese_number(x)) for x in promotion.groups())
                    if value["paid_quantity"] != paid or value["gift_quantity"] != gift:
                        add("promotion_source_mismatch", location, "Paid/gift counts disagree with explicit source promotion")
            if value["type"] == "range":
                if re.search(r"大于|超过", quote) and not re.search(r"不超过|不要超过|不能超过", quote) and value["lower"] is not None and value["lower_inclusive"]:
                    add("open_boundary", location, "Strict lower bound cannot become inclusive")
                if re.search(r"小于|不到", quote) and value["upper"] is not None and value["upper_inclusive"]:
                    add("open_boundary", location, "Strict upper bound cannot become inclusive")
            if r["operator"] == "at_least" and re.search(r"大于|超过", quote) and not re.search(r"不超过|不要超过|不能超过", quote):
                add("open_boundary", location, "Use open range for strict greater-than")
            if r["operator"] == "at_most" and re.search(r"小于|不到", quote):
                add("open_boundary", location, "Use open range for strict less-than")
            if r["attribute"] == "load_capacity" and r["operator"] == "at_most" and not re.search(r"不超过|以内|至多|最多|上限|不能超过|不得超过", quote):
                add("load_direction", location, "Load-capacity upper bound needs explicit source; do not reverse capability requirement")
            if r["operator"] == "at_most" and re.search(r"达到|至少|不低于", quote):
                add("lower_bound_direction", location, "A lower-bound capability cannot become an upper bound")
            if value["type"] in ("number", "range") and value["unit"] == "mAh" and not re.search(r"毫安时|mAh", quote, re.I):
                add("invented_charge_unit", location, "Cannot turn 毫安 into 毫安时 without source evidence")
            if r["subtype"] == "merchant" and re.search(r"产自|生产|产地", str(text_value)) and not re.search(r"直接销售|直销|直营|直供", str(text_value)):
                add("origin_as_merchant", location, "Production origin is not merchant")
            if r["subtype"] == "delivery" and re.search(r"定制|组装", str(text_value)):
                add("service_as_delivery", location, "Customization/installation is not delivery")
            if r["importance"] != "uncertain" and re.search(r"可以赠送|听说.+(?:不错|好)", quote) and not re.search(r"(?:最好|优先|偏好|必须|务必|一定要|要求)[^，,。；;]{0,12}可以赠送", quote):
                add("weak_strength", location, "Ambiguous gift/shop language must retain pending strength")
            if r["importance"] != "uncertain" and re.search(r"质量要好|品质较好|百搭|新潮|最新|复古|地道", str(text_value)):
                add("subjective_standard", location, "Subjective or relative requirement needs reviewed evidence standard")
            if dim != "budget" and approximation_evidence(r) and r["importance"] != "uncertain":
                add("non_budget_tolerance", location, "Preserve object and original approximate requirement; tolerance pending")
            key = json.dumps([dim, r["subtype"], r["subject"], r["attribute"], r["price_basis"], r["expected"]], sort_keys=True, ensure_ascii=False)
            if key in seen:
                add("duplicate_fact", location, f"Same subject/attribute/value as {seen[key]}; no automatic merging")
            seen[key] = location
    from semantic_coverage_v7 import coverage_flags
    flags.extend(coverage_flags(contract, chinese_number))
    from semantic_relations_v7 import relation_flags
    flags.extend(relation_flags(contract, chinese_number))
    from semantic_policy_v4 import policy_flags
    flags.extend(policy_flags(contract, budget_source_clause))
    from semantic_repairs_v7 import extra_flags
    flags.extend(extra_flags(contract))
    from semantic_review_source_proofs import filter_flags
    from semantic_acceptance_fixes import filter_structure_flags
    return filter_structure_flags(contract, filter_flags(contract, flags, chinese_number), chinese_number)


def validate(contract, *, source_query, require_reviewed=False, extraction=False):
    Draft202012Validator(schema()).validate(contract)
    if contract["query"] != source_query:
        raise ValueError("Original query changed")
    if any(isinstance(n, float) and not math.isfinite(n) for n in _numbers(contract)):
        raise ValueError("Non-finite number")
    if not any(contract[d] for d in DIMENSIONS) and not contract["pending_mentions"]:
        raise ValueError("Empty extraction cannot count as a valid task contract")
    for _, e in _evidences(contract):
        if not 0 <= e["start"] < e["end"] <= len(source_query) or source_query[e["start"]:e["end"]] != e["text"]:
            raise ValueError("Evidence must match original character span")
    ids = set()
    for dim in DIMENSIONS:
        for r in contract[dim]:
            if r["id"] in ids:
                raise ValueError("Duplicate requirement ID")
            ids.add(r["id"])
            if r["attribute"] not in ATTRIBUTES[r["subtype"]]:
                raise ValueError("Attribute does not match subtype")
            if extraction and r["review_status"] != "pending":
                raise ValueError("Extraction cannot approve itself")
            if r["subject"]["kind"] in ("primary_product", "order"):
                canonical = "主商品" if r["subject"]["kind"] == "primary_product" else "本次订单"
                if r["subject"]["name"] != canonical:
                    raise ValueError("Use the canonical primary-product/order subject name")
            if r["subject"]["kind"] in ("component", "gift", "compatible_target", "intended_user") and r["subject"]["name"] not in _quote(r):
                raise ValueError("subject_evidence: quote the literal subject and attribute in separate evidence entries if needed; do not invent composite subject names")
            v, op = r["expected"], r["operator"]
            required_type = {"one_of": "text_list", "at_most": "number",
                             "at_least": "number", "between": "range", "matches_bundle": "bundle"}.get(op)
            if required_type and v["type"] != required_type:
                raise ValueError(f"{r['id']} ({dim}/{r['subtype']}): operator {op} requires {required_type}, got {v['type']}")
            if op == "contains_all" and v["type"] not in ("text_list", "components"):
                raise ValueError(f"{r['id']} ({dim}/{r['subtype']}): contains_all requires text_list or components, got {v['type']}")
            if op == "excludes":
                if v["type"] not in ("text_list", "components"):
                    raise ValueError("Exclusion requires named text_list/components, never an unscoped numeric negation")
                if not re.search(r"不需要|不要|不接受|不含|不得|禁止|排除|不带|无需", _quote(r)):
                    raise ValueError("Exclusion needs an explicit negative source; it cannot be inferred from absence")
                if v['type']=='components' and any(item['quantity'] is not None or item['unit'] is not None for item in v['items']):
                    raise ValueError('Excludes names disallowed items; bounded forbidden quantities require separate scope review')
            if v["type"] == "components" and r["subtype"] != "components":
                raise ValueError(f"{r['id']}: component value must use components subtype")
            if v["type"] == "range":
                a, b = v["lower"], v["upper"]
                if a is None and b is None or a is not None and b is not None and (a > b or a == b and not(v["lower_inclusive"] and v["upper_inclusive"])):
                    raise ValueError("Empty range")
            if v["type"] == "bundle":
                if r["subtype"] != "bundle" or op != "matches_bundle":
                    raise ValueError("Bundle must use bundle/matches_bundle")
                paid, gift, received = (v[k] for k in ("paid_quantity", "gift_quantity", "received_quantity"))
                if paid is None and gift is None and received is None:
                    raise ValueError("Empty promotion")
                if all(x is not None for x in (paid, gift, received)) and paid + gift != received:
                    raise ValueError("Paid plus gift must equal received quantity for same-item promotion")
            if r["subtype"] == "quantity" and v["type"] in ("number", "range"):
                allowed_quantity_units = {"purchase_count": PURCHASE_UNITS, "per_variant_count": PURCHASE_UNITS,
                                          "purchase_mass": ["kg", "g", "jin"], "purchase_volume": ["ml", "l"]}
                if v["unit"] not in allowed_quantity_units[r["attribute"]]:
                    raise ValueError("Quantity attribute/unit mismatch")
                if r["attribute"] == "per_variant_count" and not re.search(r"每(?:种|款)", _quote(r)):
                    raise ValueError("Per-variant quantity needs explicit 每种/每款 source")
            if r["attribute"] == "pack_count" and v["type"] in ("number", "range") and v["unit"] not in PURCHASE_UNITS:
                raise ValueError("Pack count must use a count unit, never a physical mass")
            if r["subtype"] == "attribute_count" and v["type"] in ("number", "range") and v["unit"] != "count":
                raise ValueError("Feature counts use count, never piece")
            allowed_units = {"power": ["W"], "voltage": ["V"], "energy_capacity": ["Wh", "kWh"],
                             "electric_charge": ["Ah", "mAh"], "load_capacity": ["kg", "jin"], "angle": ["degree"],
                             "mass": ["kg", "g", "jin"], "liquid_volume": ["ml", "l"],
                             "length": ["mm", "cm", "m", "inch"], "width": ["mm", "cm", "m", "inch"],
                             "height": ["mm", "cm", "m", "inch"], "diameter": ["mm", "cm", "m", "inch"],
                             "thickness": ["mm", "cm", "m", "inch"],
                             "wearer_height": ["cm", "m"], "wearer_weight_limit": ["kg", "jin"]}
            if v["type"] in ("number", "range") and r["attribute"] in allowed_units and v["unit"] not in allowed_units[r["attribute"]]:
                raise ValueError("Physical attribute/unit mismatch")
            if dim == "budget":
                if v["type"] in ("number", "range") and v["unit"] != "CNY":
                    raise ValueError("Money must use CNY")
                if v["type"] == "text" and r["importance"] != "uncertain":
                    raise ValueError("Unparsed budget must be uncertain")
                if r["subtype"] == "unit_price" and r["price_basis"] is None and r["importance"] != "uncertain":
                    raise ValueError("Unit price denominator is required")
                if v["type"] in ("number", "range"):
                    supported = False
                    for e in r["evidence"]:
                        try:
                            clause = budget_source_clause(source_query, e["text"])
                            source_op, source_value, _ = budget_for_role(clause, r["subtype"])
                            basis = price_basis_from_phrase(clause) if r["price_basis"] is not None else None
                        except ValueError:
                            continue
                        if source_op == op and source_value == v and basis == r["price_basis"]:
                            supported = True
                    if not supported:
                        raise SourceMismatch("Budget amount/operator/basis disagrees with original source and confirmed policy")
            if r["price_basis"] is not None and (dim != "budget" or r["subtype"] == "total_price"):
                raise ValueError("Price denominator belongs to a per-unit monetary requirement, not total order price")
            if require_reviewed and (r["review_status"] != "approved" or r["importance"] == "uncertain"):
                raise ValueError("Unreviewed or uncertain requirement")
    flags = lint(contract)
    flags, review_findings = split_flags(contract, flags)
    flags.extend(unit_source_flags(contract))
    if flags:
        raise ValueError("Semantic review flags: " + json.dumps(flags, ensure_ascii=False))
    if require_reviewed and review_findings:
        raise ValueError("Unresolved semantic review findings: " + json.dumps(review_findings, ensure_ascii=False))
    if require_reviewed and contract["pending_mentions"]:
        raise ValueError("Pending mentions cannot be ignored for reward")
    return contract


def _numbers(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _numbers(v)
    elif isinstance(value, list):
        for v in value:
            yield from _numbers(v)
    elif isinstance(value, (int, float)):
        yield value


def normalize(raw, task):
    from semantic_policy_v4 import structural_check, apply_budget_policy
    structural_check(raw)
    from semantic_relations_v7 import canonicalize
    raw, pre_audit = canonicalize(raw, task, DIMENSIONS, ATTRIBUTES, chinese_number, price_basis_from_phrase)
    from semantic_repairs_v7 import preprocess
    raw, repairs = preprocess(raw, task, chinese_number)
    pre_audit.extend(repairs)
    Draft202012Validator(schema(llm=True)).validate(raw)
    if raw["task_id"] != task["task_id"] or raw["query"] != task["query"]:
        raise ValueError("Original task ID/query changed")
    c, audit = copy.deepcopy(raw), list(pre_audit)
    query = task["query"]
    for path, e in _evidences(c):
        text = e["text"]
        hits = [i for i in range(len(query)) if query.startswith(text, i)]
        if len(hits) != 1:
            raise ValueError(f"{path}: quote must occur once (matches={len(hits)}); use exact contiguous source or multiple evidence entries; never paraphrase/join nonadjacent spans")
        e.update(start=hits[0], end=hits[0] + len(text))
        audit.append({"kind": "source_coordinates", "path": path, "after": copy.deepcopy(e)})
    entries = [(dim, i, r) for dim in DIMENSIONS for i, r in enumerate(c[dim])]
    for n, (dim, i, r) in enumerate(sorted(entries, key=lambda x: (min(e["start"] for e in x[2]["evidence"]), list(DIMENSIONS).index(x[0]), x[1])), 1):
        r["id"] = f"r{n:03d}"
        audit.append({"kind": "assigned_id", "path": f"/{dim}/{i}", "after": r["id"]})
    source_unit_errors = unit_source_flags(c)
    if source_unit_errors:
        raise ValueError("Semantic review flags: " + json.dumps(source_unit_errors, ensure_ascii=False))
    for dim, _, r in entries:
        quote = _quote(r)
        reason = None
        before = {"importance": r["importance"], "operator": r["operator"], "expected": copy.deepcopy(r["expected"])}
        approximate = approximation_evidence(r) if dim != "budget" else []
        if approximate:
            # Only target-bound numeric approximation can change a requirement.
            reason = "undefined_tolerance"
            r["operator"], r["expected"] = "eq", {"type": "text", "value": quote}
        elif re.search(r"可以赠送|听说.+(?:不错|好)", quote) and not re.search(r"最好|优先|偏好|喜欢|希望", quote):
            reason = "ambiguous_strength"
        elif re.search(r"最新", str(r["expected"].get("value", ""))):
            reason = "missing_reference_time"
        elif re.search(r"质量要好|品质较好|百搭|新潮|复古|地道", str(r["expected"].get("value", ""))):
            reason = "unverifiable_subjective"
        if reason:
            r["importance"] = "uncertain"
            mention = {"evidence": copy.deepcopy(approximate[0] if approximate else r["evidence"][0]), "reason": reason, "importance": "uncertain"}
            if mention not in c["pending_mentions"]:
                c["pending_mentions"].append(mention)
            audit.append({"kind": "pending_policy", "id": r["id"], "reason": reason, "before": before,
                          "after": {"importance": "uncertain", "expected": copy.deepcopy(r["expected"])}})
        if r["expected"]["type"] == "bundle":
            v = r["expected"]
            if v["paid_quantity"] is not None and v["gift_quantity"] is not None and v["received_quantity"] is None:
                v["received_quantity"] = v["paid_quantity"] + v["gift_quantity"]
                audit.append({"kind": "same_item_received_arithmetic", "id": r["id"], "after": v["received_quantity"]})
        for scope in direct_hope_scopes(r, query):
            before_strength = r["importance"]
            r["importance"] = "uncertain"
            evidence = {"text": scope["text"], "start": scope["start"], "end": scope["end"]}
            mention = {"evidence": evidence, "reason": "ambiguous_strength", "importance": "uncertain"}
            if mention not in c["pending_mentions"]:
                c["pending_mentions"].append(mention)
            audit.append({"kind": "hope_strength_pending", "id": r["id"], "before": before_strength,
                          "after": "uncertain", "source": copy.deepcopy(evidence)})
    for r in c["budget"]:
        phrase = r["expected"]["value"]
        if phrase not in query or not any(phrase in e["text"] for e in r["evidence"]):
            raise ValueError("Budget phrase must be an exact quoted source clause")
        before = copy.deepcopy(r["expected"])
        try:
            source_clause = budget_source_clause(query, phrase)
            if source_clause != phrase:
                start = query.index(source_clause)
                evidence = {"text": source_clause, "start": start, "end": start + len(source_clause)}
                if evidence not in r["evidence"]:
                    r["evidence"].append(evidence)
                audit.append({"kind": "budget_source_context", "id": r["id"], "before": phrase, "after": source_clause})
            try:
                basis = price_basis_from_phrase(source_clause)
            except ValueError:
                basis = None
            if r["subtype"] == "total_price" and basis is not None:
                raise SourceMismatch("Source is per-unit money, not total order price")
            if r["subtype"] == "unit_price" or basis is not None or r["price_basis"] is not None:
                if basis is None:
                    raise ValueError("Per-unit monetary requirement has no supported source denominator")
                if r["price_basis"] is not None and r["price_basis"] != basis:
                    raise SourceMismatch("Unit price denominator differs from original source phrase")
                r["price_basis"] = basis
                audit.append({"kind": "price_basis", "id": r["id"], "after": basis})
            r["operator"], r["expected"], rule = budget_for_role(source_clause, r["subtype"])
            audit.append({"kind": "budget_policy", "id": r["id"], "before": before,
                          "after": copy.deepcopy(r["expected"]), "rule": rule})
            apply_budget_policy(c, r, source_clause, audit)
        except SourceMismatch:
            raise
        except ValueError as exc:
            r["importance"] = "uncertain"
            c["pending_mentions"].append({"evidence": copy.deepcopy(r["evidence"][0]),
                                            "reason": "ambiguous_meaning", "importance": "uncertain"})
            audit.append({"kind": "unparsed_budget_pending", "id": r["id"], "reason": str(exc)})
    # Deduplicate byte-identical pending records only; different scopes/reasons
    # remain separate and nothing is automatically approved.
    unique = []
    for mention in c["pending_mentions"]:
        if mention not in unique:
            unique.append(mention)
    if len(unique) != len(c["pending_mentions"]):
        audit.append({"kind": "exact_pending_dedup", "before": len(c["pending_mentions"]), "after": len(unique)})
        c["pending_mentions"] = unique
    validate(c, source_query=query, extraction=True)
    _, findings = split_flags(c, lint(c))
    if findings:
        audit.append({"kind": "semantic_review_findings", "blocking_for_extraction": False,
                      "blocking_for_reward": True, "findings": findings})
    from semantic_review_source_proofs import certificates
    audit.extend(certificates(c, chinese_number))
    return c, audit


def write_schema():
    PACKAGE.mkdir(parents=True, exist_ok=True)
    raw = raw_example()
    contract, audit = normalize(raw, {"task_id": 0, "query": raw["query"]})
    for name, value in (("requirements.schema.json", schema()), ("llm_output.schema.json", schema(llm=True)), ("policy.json", POLICY),
                        ("example.synthetic.llm.json", raw), ("example.synthetic.contract.json", contract), ("example.synthetic.audit.json", audit)):
        (PACKAGE / name).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    covered_raw = coverage_example()
    covered_contract, covered_audit = normalize(covered_raw, {"task_id": covered_raw["task_id"], "query": covered_raw["query"]})
    for name, value in (("example.coverage.synthetic.llm.json", covered_raw),
                        ("example.coverage.synthetic.contract.json", covered_contract),
                        ("example.coverage.synthetic.audit.json", covered_audit)):
        (PACKAGE / name).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def raw_example():
    query = "要全钢压脚，买10送1，预算100元以内，最好黑色。"
    result = {"schema_version": VERSION, "task_id": 0, "query": query,
              **{d: [] for d in DIMENSIONS}, "pending_mentions": []}
    def add(d, subtype, attribute, value, quote, importance="required", operator="eq"):
        result[d].append({"subtype": subtype, "attribute": attribute,
                          "subject": {"kind": "order" if d == "budget" else "primary_product",
                                      "name": "本次订单" if d == "budget" else "主商品"},
                          "importance": importance, "operator": operator, "expected": value,
                          "price_basis": None, "evidence": [{"text": quote}], "review_status": "pending"})
    add("category", "product_category", "product_category", {"type": "text", "value": "压脚"}, "压脚")
    add("specifications", "material", "material", {"type": "text", "value": "全钢"}, "全钢")
    add("specifications", "bundle", "promotion", {"type": "bundle", "paid_quantity": 10,
        "gift_quantity": 1, "received_quantity": None, "unit": "piece", "same_item": True}, "买10送1", operator="matches_bundle")
    add("budget", "total_price", "total_price", {"type": "text", "value": "预算100元以内"}, "预算100元以内")
    add("specifications", "color", "color", {"type": "text", "value": "黑色"}, "最好黑色", "preference")
    return result


def coverage_example():
    query = "买两套文具，最好蓝色，必须每套自带一个收纳盒，预算100元以内。"
    c = {"schema_version": VERSION, "task_id": 0, "query": query, **{d: [] for d in DIMENSIONS}, "pending_mentions": []}
    def add(dim, subtype, attribute, expected, quote, importance="required", operator="eq"):
        c[dim].append({"subtype": subtype, "attribute": attribute, "subject": {
            "kind": "order" if dim == "budget" else "primary_product", "name": "本次订单" if dim == "budget" else "主商品"},
            "importance": importance, "operator": operator, "expected": expected, "price_basis": None,
            "evidence": [{"text": quote}], "review_status": "pending"})
    add("category", "product_category", "product_category", {"type": "text", "value": "文具"}, "文具")
    add("specifications", "quantity", "purchase_count", {"type": "number", "value": 2, "unit": "set"}, "买两套文具")
    add("specifications", "color", "color", {"type": "text", "value": "蓝色"}, "最好蓝色", "preference")
    add("specifications", "components", "included_components", {"type": "components", "items": [
        {"name": "收纳盒", "quantity": 1, "unit": "piece"}]}, "必须每套自带一个收纳盒", operator="contains_all")
    add("budget", "total_price", "total_price", {"type": "text", "value": "预算100元以内"}, "预算100元以内")
    return c


if __name__ == "__main__":
    write_schema()
    print("Wrote semantic-v2-r7-closeout schemas/examples; no API calls; contracts remain pending")
