"""Conservative, source-backed r3 corrections and semantic diagnostics.

Never infers a missing subject, resolves an ambiguous conjunction, or approves
a requirement. Routing is limited to a closed map; corrections are audited.
"""
import copy
import re

DIMS = ("category", "budget", "functions", "specifications", "brand", "model")
NUM = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)"


def quote(r):
    return "，".join(e.get("text", "") for e in r.get("evidence", []) if isinstance(e, dict))


def meanings(r):
    v = r["expected"]
    if v["type"] == "text":
        return v["value"]
    if v["type"] == "text_list":
        return "、".join(v["values"])
    if v["type"] == "components":
        return "、".join(i["name"] for i in v["items"])
    return ""


def canonicalize(raw, task, dimensions, attributes, number_parser, basis_parser):
    c, audit = copy.deepcopy(raw), []
    if not isinstance(c, dict) or c.get("query") != task["query"] or c.get("task_id") != task["task_id"]:
        raise ValueError("Original task ID/query changed")
    destinations = {s: d for d, subs in dimensions.items() for s in subs}
    entries = [(d, i, r) for d in dimensions if isinstance(c.get(d), list) for i, r in enumerate(c[d])]
    routed = {d: [] for d in dimensions}
    for dimension, index, r in entries:
        if not isinstance(r, dict):
            routed[dimension].append(r)
            continue
        before = copy.deepcopy(r)
        reasons = []
        # Only an explicitly nullable schema member gets a default. This does
        # not supply a missing price denominator or any other required field.
        if "price_basis" not in r:
            r["price_basis"] = None
            reasons.append("nullable_price_basis_default")
        q = quote(r)
        evidence_ok = bool(r.get("evidence")) and all(isinstance(e, dict) and isinstance(e.get("text"), str)
            and e["text"] and e["text"] in task["query"] for e in r["evidence"])
        subtype, attr = r.get("subtype"), r.get("attribute")
        # These are attributes with one unambiguous controlled parent subtype.
        if evidence_ok and subtype == attr and subtype in ("power", "voltage", "drive_structure"):
            r["subtype"] = "function"
            reasons.append("physical_attribute_parent")
        v = r.get("expected", {})
        text = v.get("value", "") if isinstance(v, dict) and v.get("type") == "text" else ""
        if evidence_ok:
            if r.get("subtype") == "components" and r.get("attribute") == "included_components" and isinstance(v, dict) and v.get("type") == "components":
                gifts = [s for s in re.split(r"[，,。；;！？!?\n]", q) if re.match(r"^(?:还|再|另外|并且)?(?:要)?(?:附赠|赠送)", s)]
                if any(all(item.get("name") and item["name"] in s for item in v.get("items", [])) for s in gifts) and v.get("items"):
                    r["attribute"] = "gift_presence"
                    reasons.append("explicit_gift_not_included_component")
            if r.get("subtype") == "condition" and text in ("防水", "防尘", "防摔", "保暖", "防静电") and text in q:
                r.update(subtype="function", attribute="function")
                reasons.append("explicit_function_not_condition")
            if r.get("subtype") == "origin" and re.search(r"从.+(?:邮寄|发货|寄出)", text) and text in q:
                r.update(subtype="delivery", attribute="delivery")
                reasons.append("shipping_source_not_production_origin")
            if r.get("subtype") in ("unit_price", "total_price") and isinstance(text, str) and text and text in q:
                # Include the original clause so a clipped quote cannot remove
                # a denominator or make a true unit price look like total cost.
                clauses = [m.group() for m in re.finditer(r"[^，,。；;！？!?\n]+", task["query"]) if text in m.group()]
                scope = clauses[0] if len(clauses) == 1 else text
                try:
                    basis = basis_parser(scope)
                except ValueError:
                    basis = None
                if r["price_basis"] is not None and basis is None:
                    raise ValueError("price_basis has no supported source denominator; do not hide an invented basis as ordinary uncertainty")
                if r["subtype"] == "unit_price" and r["price_basis"] is None and basis is None and r.get("subject", {}).get("kind") in ("primary_product", "order") and re.search(r"价格|售价|预算|卖", scope) and not re.search(r"单价|每|均价|一[瓶袋盒套件本米]", scope):
                    r.update(subtype="total_price", attribute="total_price")
                    r["subject"] = {"kind": "order", "name": "本次订单"}
                    reasons.append("ordinary_price_not_unit_price")
            if r.get("subtype") == "quantity" and isinstance(v, dict) and v.get("type") == "number":
                mass = re.search(rf"({NUM})(斤|千克|公斤|克)装", q)
                pack = re.search(rf"({NUM}|单)(支|个|袋|瓶|盒|件|本|套)装", q)
                book = re.search(rf"(?:购买|订购|买|要)({NUM})本", q)
                if mass and v.get("unit") in ("jin", "kg", "g"):
                    unit = {"斤": "jin", "千克": "kg", "公斤": "kg", "克": "g"}[mass[2]]
                    if v.get("value") == float(number_parser(mass[1])) and v["unit"] == unit:
                        r.update(subtype="net_content", attribute="mass")
                        reasons.append("pack_mass_not_purchase_count")
                elif pack and r.get("subject", {}).get("kind") == "primary_product":
                    n = 1 if pack[1] == "单" else float(number_parser(pack[1]))
                    unit = {"支": "piece", "个": "piece", "袋": "bag", "瓶": "bottle", "盒": "box", "件": "piece", "本": "book", "套": "set"}[pack[2]]
                    if v.get("value") == n and v.get("unit") == unit and r.get("operator") == "eq":
                        r.update(subtype="packaging", attribute="pack_count")
                        reasons.append("pack_count_not_order_count")
                elif book and v.get("unit") == "piece" and v.get("value") == float(number_parser(book[1])):
                    v["unit"] = "book"
                    reasons.append("literal_book_counter")
            if r.get("attribute") in ("energy_capacity", "load_capacity", "power") and isinstance(v, dict) and v.get("type") == "number" and r.get("operator") == "eq":
                lower = re.search(rf"(?<!未)(?<!不)达到({NUM})", q)
                if lower and v.get("value") == float(number_parser(lower[1])):
                    r["operator"] = "at_least"
                    reasons.append("explicit_reaches_lower_bound")
        new_dimension = destinations.get(r.get("subtype"), dimension)
        # Unknown fields/subtypes and invalid attribute combinations still go
        # through the strict schema/attribute checks below, never into a catchall.
        if new_dimension != dimension:
            reasons.append("controlled_dimension_route")
        routed[new_dimension].append(r)
        if before != r or new_dimension != dimension:
            audit.append({"kind": "source_supported_canonicalization", "source_path": f"/{dimension}/{index}",
                          "destination_dimension": new_dimension, "rules": reasons, "before": before, "after": copy.deepcopy(r)})
    for d in dimensions:
        if isinstance(c.get(d), list):
            c[d] = routed[d]
    # A single, explicitly preferred accessory has known strength. Remove only
    # that redundant strength flag, not numeric/logical/subject uncertainty.
    if isinstance(c.get("pending_mentions"), list):
        kept = []
        for index, p in enumerate(c["pending_mentions"]):
            evidence = p.get("evidence") if isinstance(p, dict) else None
            # A one-element wrapper carries exactly the same evidence. Record
            # the structural adapter; never merge multiple scopes or drop data.
            if isinstance(evidence, list) and len(evidence) == 1 and isinstance(evidence[0], dict):
                p["evidence"] = copy.deepcopy(evidence[0])
                audit.append({"kind": "singleton_pending_evidence_unwrap", "path": f"/pending_mentions/{index}/evidence",
                              "before": copy.deepcopy(evidence), "after": copy.deepcopy(p["evidence"])})
                evidence = p["evidence"]
            e = evidence.get("text", "") if isinstance(evidence, dict) else ""
            if not isinstance(e, str):
                e = ""  # The strict schema reports malformed evidence below.
            clear = isinstance(p, dict) and p.get("reason") == "ambiguous_strength" and re.match(r"^最好", e) and not re.search(r"\d|[一二两三四五六七八九十]|或者|或|和|但|必须|希望", e)
            matched = clear and any(isinstance(r, dict) and r.get("subtype") == "components" and r.get("importance") == "preference"
                and r.get("expected", {}).get("type") == "components" and len(r["expected"].get("items", [])) == 1
                and r["expected"]["items"][0].get("name") in e and e in task["query"]
                for r in routed["specifications"])
            if matched:
                audit.append({"kind": "redundant_explicit_preference_pending", "before": copy.deepcopy(p), "after": None})
            else:
                kept.append(p)
        c["pending_mentions"] = kept
    return c, audit


def feature_relation_flags(r, path, number_parser):
    """Bind counting language locally, so another clause's 至少 cannot leak."""
    keywords = {"gear_count": r"档位|档", "preset_count": r"预设", "effect_type_count": r"失真|效果",
                "tone_type_count": r"音色", "face_count": r"棱面", "other_feature_count": r"."}
    if r["attribute"] not in keywords:
        return []
    candidates = []
    for clause in re.split(r"[，,。；;！？!?\n]", quote(r)):
        if not re.search(keywords[r["attribute"]], clause):
            continue
        for m in re.finditer(rf"({NUM})(多)?(?:个|种|档|面)", clause):
            try:
                n = float(number_parser(m[1]))
            except ValueError:
                continue
            candidates.append((n, m[2], clause[max(0, m.start()-10):m.start()], clause[m.end():m.end()+4]))
    v = r["expected"]
    if len(candidates) > 1 and v["type"] == "number":
        candidates = [c for c in candidates if c[0] == v["value"]]
    if len(candidates) != 1:
        return []  # Ambiguous alignment is not resolved by guessing.
    n, duo, before, after = candidates[0]
    if re.search(r"不要|不能|不需要|不想|无需|如果|假如", before):
        return []  # Not an affirmative bare count; preserve its logical scope.
    code = None
    if duo:
        if v["type"] == "number" or v["type"] == "range" and not (v["lower"] == n and not v["lower_inclusive"]):
            code = "feature_duo_boundary"
    elif re.search(r"至少|达到|不少于|不低于", before) or re.match(r"以上", after):
        if r["operator"] != "at_least" or v["type"] != "number" or v["value"] != n:
            code = "explicit_feature_lower_bound"
    elif not re.search(r"最多|至多|不超过|小于|大于|超过|低于|不到|左右|上下|大概|大约|约|[<>≤≥]", before) and not re.match(r"左右|上下|以下|以内", after):
        if r["operator"] != "eq" or v["type"] != "number" or v["value"] != n:
            code = "bare_feature_exact_count"
    return ([{"code": code, "location": path,
              "detail": "Bare feature counts are exact; explicit 至少/达到 is a lower bound; 20多 is strict >20 with range pending. Bind each count to its own clause."}] if code else [])


def relation_flags(contract, number_parser):
    flags = []
    def add(code, path, detail):
        flags.append({"code": code, "location": path, "detail": detail})
    reqs = [(d, i, r) for d in DIMS for i, r in enumerate(contract[d])]
    for d, i, r in reqs:
        path, q, text = f"/{d}/{i}", quote(r), meanings(r)
        v, sub, attr = r["expected"], r["subtype"], r["attribute"]
        if sub == "attribute_count":
            flags.extend(feature_relation_flags(r, path, number_parser))
        if d == "category" and re.search(r"超轻|中筒|圆领|地暖用|(?:\d+|[一二两三四五六七八九十]+)寸", text):
            add("category_scope", path, "Keep product kind in category; express weight/height/neckline/dimensions/use separately, never as a hard category failure")
        if sub == "quantity" and re.search(r"(?:\d+|[一二两三四五六七八九十]+|单)(?:个|支|袋|瓶|斤|件|盒|套)装", q):
            add("pack_as_purchase", path, "Pack contents are not order count: use packaging/pack_count or net_content/mass; keep ambiguous conjunctions pending")
        if sub == "quantity" and r["subject"]["kind"] in ("component", "gift"):
            add("accessory_as_purchase", path, "Accessory count belongs to the named component/gift; preserve unclear quantities pending, not as purchase_count")
        if sub == "origin" and re.search(r"邮寄|发货|寄出", text):
            add("shipping_as_origin", path, "Shipping source does not establish manufacturing origin")
        if sub == "capacity" and attr == "energy_capacity" and re.search(r"毫安(?!时)", q):
            add("charge_as_energy", path, "Raw 毫安 does not establish energy capacity; preserve unsupported unit/meaning without inventing mAh")
        if attr == "wearer_weight_limit" and re.search(r"能穿到\s*" + NUM + "斤", q) and not re.search(r"左右|大约|大概|上下", q) and v["type"] == "text":
            add("wearer_limit_relation", path, "Preserve the stated wearer-weight coverage numerically; not an undefined approximation")
        if sub == "attribute_count" and re.search(NUM + r"多(?:个|种)", q):
            pending = any(p["reason"] in ("ambiguous_meaning", "undefined_tolerance") and re.search(NUM+r"多", p["evidence"]["text"]) for p in contract["pending_mentions"])
            if r["importance"] != "uncertain" or not pending:
                add("feature_duo_pending", path, "20多 is not >=20; preserve strict >20 information and unresolved range/upper interpretation pending")
        if attr in ("energy_capacity", "load_capacity", "power") and re.search(r"(?<!未)(?<!不)达到", q) and r["operator"] == "eq":
            add("reaches_as_equality", path, "达到 states a capability lower bound, not exact equality")
        if sub == "components" and v["type"] == "components" and any(re.fullmatch(r"[一二三四五六七八九十\d]+件套", x["name"]) for x in v["items"]):
            add("set_as_component", path, "A three-piece set is a pack structure, not a component literally named 三件套; retain unknown composition pending")
        if sub == "material" and (re.search(r"胡桃色|黑色|金色|红色", text) and re.search(r"实木|金属|塑料", text)):
            add("material_color_atomicity", path, "Split material and color, keeping both bound to the same component")
        if sub == "color" and re.search(r"耐脏|防水|耐磨", text):
            add("color_function_atomicity", path, "Split color and performance, retaining unresolved evidence standards separately")
        if sub == "composition" and "自然" in text and "无添加" in text:
            add("composition_atomicity", path, "Separate subjective 自然 from the explicit additive constraint; do not conflate evidence standards")
        if attr == "drive_structure" and re.search(r"气轮|充气轮|实心轮", text):
            add("drive_wheel_atomicity", path, "Split drive structure from wheel/tire construction; do not grade both through one drive attribute")
        if sub == "warranty" and "最好" in q and re.search(r"质保久|长时间质保", q):
            add("warranty_strength_scope", path, "Split base warranty requirement/undefined duration and preferred three-year duration")
        if sub == "product_model" and r["subject"]["kind"] == "primary_product" and re.search(r"适配|配件|泵头", contract["query"]) and re.search(r"型号(?:是|为)", q) and "泵头" in contract["query"]:
            add("model_target_scope", path, "Confirm whether the model belongs to the compatible target or the purchased part; preserve source scope pending instead of guessing")
        if sub in ("product_model", "version") and re.search(r"\d+级", text) and re.search(r"课本|课程|教程", q):
            add("course_level_as_version", path, "Course level is content/level, not necessarily product model or generation")
        if sub == "material" and re.search(r"可以是", q) and r["importance"] != "uncertain":
            add("permitted_material_strength", path, "Permitted material does not establish a preference; retain source and pending strength until policy confirmed")
        if sub == "flavor" and r["subject"]["kind"] == "component" and text in (r["subject"]["name"]+"味", r["subject"]["name"]+"香味"):
            add("flavor_as_subject", path, "Name the flavor's carrier, not the flavor itself; retain the full source pending if the carrier is unknown")
    if re.search(r"(?:用完|使用后).{0,8}不会红肿", contract["query"]) and not any(d == "functions" and "红肿" in meanings(r) for d, _, r in reqs):
        add("explicit_function_coverage", "/query", "Keep the explicitly requested non-redness effect as a function; an ambiguous_strength pending alone does not preserve its role")
    # Check known explicit functions against the extracted meaning, rather than
    # merely finding the words anywhere in an evidence quote or pending note.
    for m in re.finditer(r"[^，,。；;！？!?\n]+", contract["query"]):
        clause = m.group()
        if not re.search(r"想|买|要|需|具备|能够|内层|外层|支持|兼容", clause):
            continue
        for token in ("保暖", "防水", "防尘", "防摔", "不容易起静电"):
            if token not in clause or re.search(r"不(?:需要|要求|用).{0,6}" + token, clause):
                continue
            appropriate = [r for d, _, r in reqs if d == "functions" and token in meanings(r)]
            if "内层" in clause:
                appropriate = [r for r in appropriate if r["subject"]["name"] == "内层"]
            if not appropriate:
                add("explicit_function_coverage", "/query", f"Retain explicit function {token} with correct subject; an unrelated category/evidence/pending-only match is insufficient")
        if "交流" in clause and "转换" in clause and not any("交流" in meanings(r) and "转换" in meanings(r) for _, _, r in reqs):
            add("conversion_relation_coverage", "/query", "Keep the AC conversion relation, not just a list of voltage labels")
    return flags
