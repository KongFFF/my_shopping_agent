"""Target-bound numeric scope and non-destructive semantic review diagnostics.

No task IDs, product lexicons, missing facts, or inferred requirement strength.
High-confidence contradictions block extraction; scope guesses go to review.
"""
import re
from semantic_coverage_v7 import clauses

NUM = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)"
UNIT = r"(?:千克|公斤|毫升|毫米|厘米|平米|平方(?:米)?|小时|分钟|个月|mm|cm|ml|mAh|kWh|Wh|kg|[gmlWVA]|克|斤|升|米|寸|档|种|个|支|瓶|包|袋|盒|套|双|条|年|月|日|天|%|％)"
MEASURE = rf"{NUM}\s*(?:{UNIT})?"
# Suffixes need a numeric antecedent and a phrase boundary. This deliberately
# does not match directional 左右自动摇头 / 上下调节, or 左右各一个.
APPROX = re.compile(rf"(?:约为|大约|大概|差不多|约)(?:为|在|有|是)?\s*{MEASURE}"
                    rf"|{MEASURE}\s*(?:左右|上下)(?=$|[，,。；;、！？!?\s]|的|就|即可|为|宽|高|长|厚)", re.I)
NUMERIC_SUBTYPES = {"quantity", "attribute_count", "dimensions", "capacity", "net_content", "weight", "size"}
LINK = r"(?:(?:是|要|能|能够|选|选择|为|有|采用|具备|达到)\s*){0,2}"
CUE = r"最好|优先|尽量|偏好|喜欢|必须|务必|一定要|要求|希望"
STRENGTH_CODES = {"soft_clause_mandatory", "explicit_hard_clause_optional", "mixed_strength",
                  "hope_strength_pending", "hard_clause_after_soft_uncovered", "soft_clause_in_mixed_uncovered"}
REVIEW_ONLY = {"background_as_pending", "objective_requirement_pending_only"}


def literals(r):
    v = r["expected"]
    if v["type"] == "text":
        return [v["value"]]
    if v["type"] == "text_list":
        return v["values"]
    return []


def direct_scopes(r, query):
    """Recognize only a local modifier followed directly by the literal target.

    Categories are deliberately not weakened from a hobby/preference sentence.
    Cross-clause and nested meanings remain human review, never a guessed label.
    """
    if r["subtype"] == "product_category":
        return []
    out = []
    for clause in clauses(query):
        if not any(e["start"] < clause["end"] and clause["start"] < e["end"] for e in r["evidence"]):
            continue
        for value in literals(r):
            # In 喜欢尝试不同风味的咖啡爱好者, 喜欢 describes the
            # intended users. It is not an optionality marker on the purchase.
            nested_likes = r['subtype']=='target_user' and value.startswith('喜欢')
            if not nested_likes:
                value = re.sub(rf"^(?:{CUE}){LINK}", "", value).strip()
            if not value:
                continue
            pattern = rf"(?P<cue>{CUE}){LINK}{re.escape(value)}"
            for m in re.finditer(pattern, clause["text"]):
                # 希望必须/最好 has an explicit inner cue; do not let 希望
                # override it. A cue in a negated statement is not affirmative.
                if re.search(r"不|别|无需|不用", clause["text"][max(0,m.start()-2):m.start()]):
                    continue
                out.append({**clause, "cue": m["cue"]})
    return out


def direct_hope_scopes(r, query):
    direct = direct_scopes(r, query)
    if direct and all(x["cue"] == "希望" for x in direct):
        return list({(x["start"], x["end"]): x for x in direct}.values())
    return []


def approximation_evidence(r):
    """Return source evidence where approximation binds this numeric target.

    Multiple subject/value evidence entries are checked separately. A number in
    another attribute's broad quote does not make a textual function approximate.
    """
    value = r["expected"]
    if value["type"] in ("components", "bundle"):
        return []  # Never flatten a structured composition into approximate text.
    targets = literals(r)
    found = []
    for e in r["evidence"]:
        for m in APPROX.finditer(e["text"]):
            numbers = re.findall(NUM, m.group())
            bound_text = any(any(n in text for n in numbers) for text in targets)
            bound_numeric = (value["type"] in ("number", "range")
                             and r["subtype"] in NUMERIC_SUBTYPES
                             and len(list(APPROX.finditer(e["text"]))) == 1)
            if bound_text or bound_numeric:
                found.append(e)
                break
    return found


def _requirement(c, location):
    parts = location.strip("/").split("/")
    if len(parts) == 2 and parts[0] in c and parts[1].isdigit():
        return c[parts[0]][int(parts[1])]
    return None


def split_flags(c, flags):
    """Scope heuristics are visible findings, not retry demands to weaken facts."""
    blocking, review = [], []
    for original in flags:
        f = dict(original)
        code = f["code"]
        if (code in {'component_count_source_unproven','recommendation_quantity_scope',
                     'component_vs_order_quantity_scope','component_bound_scope','axis_scope_unproven',
                     'unresolved_count_interpretation','gift_vs_included_scope','selection_quantity_scope'}
                or (code in {'purchase_quantity_coverage','quantity_scope_pending'} and f.get('source',{}).get('source','').startswith('推荐'))
                or (code=='component_quantity_coverage' and f.get('source',{}).get('source','').startswith('配置'))):
            f.update(severity='review', action='Preserve source and review scope; never invent a quantity or relation merely to pass')
            review.append(f)
            continue
        if code in STRENGTH_CODES:
            r = _requirement(c, f.get("location", ""))
            direct = direct_scopes(r, c["query"]) if r else []
            # Only directly bound soft/hard modifiers justify blocking feedback.
            cues = {x["cue"] for x in direct}
            proven = ((code == "soft_clause_mandatory" and bool(cues & {"最好","优先","尽量","偏好","喜欢"}) and not cues & {"必须","务必","一定要","要求"})
                      or (code == "explicit_hard_clause_optional" and bool(cues & {"必须","务必","一定要","要求"}) and not cues & {"最好","优先","尽量","偏好","喜欢"})
                      or (code == "hope_strength_pending" and cues == {"希望"}))
            if not proven:
                f.update(severity="review", action="Review modifier scope; do not infer or downgrade requirement strength to satisfy this heuristic")
                review.append(f)
                continue
        if code in REVIEW_ONLY:
            f.update(severity="review", action="Retain source; review coverage/background distinction without deleting or guessing facts")
            review.append(f)
        else:
            blocking.append(f)
    # Lack of a recognized cue is an uncertainty in the checker, not proof of a
    # mandatory label. Expose optional/uncertain items without rewriting them.
    for dim in ("functions", "specifications", "brand", "model"):
        for i, r in enumerate(c[dim]):
            if r["importance"] == "uncertain":
                explained = any(p["evidence"]["start"] < e["end"] and e["start"] < p["evidence"]["end"]
                                for p in c["pending_mentions"] for e in r["evidence"])
                if not explained:
                    review.append({"code":"uncertainty_without_reason", "location":f"/{dim}/{i}",
                                   "severity":"review", "detail":"Uncertain requirement has no overlapping pending explanation; inspect scope without guessing required/preference"})
            if r["importance"] != "preference" or direct_scopes(r, c["query"]):
                continue
            review.append({"code":"preference_scope_unproven", "location":f"/{dim}/{i}",
                           "severity":"review", "detail":"No directly bound preference marker established; review original scope, never promote automatically"})
    return blocking, review


def unit_source_flags(c):
    flags = []
    for dim in ("category", "budget", "functions", "specifications", "brand", "model"):
        for i, r in enumerate(c[dim]):
            v = r["expected"]
            if v["type"] not in ("number", "range"):
                continue
            quote = "，".join(e["text"] for e in r["evidence"])
            values = [v["value"]] if v["type"] == "number" else [v["lower"], v["upper"]]
            percentages = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*[%％]", quote)]
            if v["unit"] == "count" and any(x in percentages for x in values if x is not None):
                flags.append({"code":"percentage_as_count", "location":f"/{dim}/{i}",
                              "detail":"Percent is not count. Schema has no percent unit: retain the literal source as text with unsupported_unit pending; do not invent a legal unit."})
            if r["attribute"] == "size_label" and v["unit"] == "count":
                flags.append({"code":"size_label_as_count", "location":f"/{dim}/{i}",
                              "detail":"A size label is not a feature count; preserve the source size label as text."})
    return flags
