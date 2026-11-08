"""Conservative source coverage/strength checks, not a general Chinese parser.

Only explicit buying/accessory counts with recognized counters are checked.
No automatic creation of requirements or assignment of mandatory strength.
"""
import re

DIMS = ("category", "budget", "functions", "specifications", "brand", "model")
NUM = r"(?:\d+|[零〇一二两三四五六七八九十百千万]+)"
UNITS = {"套": "set", "支": "piece", "双": "pair", "对": "pair", "袋": "bag", "包": "bag",
         "根": "piece", "本": "book", "瓶": "bottle", "件": "piece", "个": "piece", "盒": "box",
         "片": "piece", "张": "sheet", "卷": "roll", "条": "piece", "把": "piece"}
COUNTERS = "|".join(UNITS)
MAIN = re.compile(r"(?P<prefix>购买|订购|购入|采购|选购|挑选|只需|需要|筛选|买|找|要)"
                  + rf"(?P<approx>大约|大概|约)?(?P<num>{NUM})(?P<extra>多|余|来)?(?P<unit>{COUNTERS})(?P<post>左右|上下)?")
PART = re.compile(r"(?P<prefix>自带|带有|配有|配备|配置|附带|附赠|赠送|另配|再配|赠|送)"
                  + rf"(?P<approx>大约|大概|约)?(?P<num>{NUM})(?P<extra>多|余|来)?(?P<unit>{COUNTERS})(?P<post>左右|上下)?"
                  + r"(?P<name>[^，,。；;！？!?\n]{1,24})")
FEATURE = re.compile(r"(?:不同的?|个不同的?)?(?:档位|档|音色|预设|效果|功能|模式|颜色|规格|尺寸|电压|电流|功率|棱面|孔|接口|通道|引脚|月|年|天|星期|小时|分钟)")
EXPLICIT_SOFT = re.compile(r"最好|优先|尽量|喜欢|偏好")
EXPLICIT_HARD = re.compile(r"必须|务必|一定要|要求|不得|不许|禁止")
HARD = re.compile(r"必须|务必|一定要|要求|需要|得是|不许|禁止|不得|不应|不要|不能|别(?!的)|至少|不低于|"
                  r"(?:^|颜色|尺码|预算|数量|材质)要(?:有|是|能|在)?")


def clauses(query):
    result, sentence = [], 0
    for match in re.finditer(r"([^，,。；;！？!?\n]+)([，,。；;！？!?\n]?)", query):
        original = match.group(1)
        # Clear new modality after a connector also creates a scope boundary;
        # this does not split a soft negation such as 最好不要红色.
        cursor, spans = 0, []
        for boundary in re.finditer(r"(?:但是|但|而且|并且|同时|另外)(?=必须|务必|一定要|要求|需要|得是|不得|不许|禁止|最好|优先|希望|尽量|喜欢|偏好)", original):
            spans.append((cursor, boundary.start()))
            cursor = boundary.end()
        spans.append((cursor, len(original)))
        for start, end in spans:
            text = original[start:end]
            if not text:
                continue
            hope_uncertain = "希望" in text and not EXPLICIT_SOFT.search(text) and not EXPLICIT_HARD.search(text)
            result.append({"start": match.start(1)+start, "end": match.start(1)+end, "text": text, "sentence": sentence,
                           "soft": bool(EXPLICIT_SOFT.search(text)) or hope_uncertain,
                           "hope_uncertain": bool(hope_uncertain), "hard": bool(HARD.search(text)) and not hope_uncertain})
        if match.group(2) and match.group(2) not in ("，", ","):
            sentence += 1
    return result


def _covers(evidence, start, end):
    return evidence["start"] <= start and end <= evidence["end"]


def _pending(contract, start, end, reason=None):
    return any(_covers(p["evidence"], start, end) and (reason is None or p["reason"] == reason)
               for p in contract["pending_mentions"])


def explicit_counts(query, number_parser):
    """Return high-confidence literal counts and explicit ambiguity candidates."""
    found = []
    for role, pattern in (("purchase", MAIN), ("component", PART)):
        for m in pattern.finditer(query):
            before = query[max(0, m.start()-4):m.start()]
            if re.search(r"不|别|不要|无需|不用", before):
                continue
            name = m.groupdict().get("name", "")
            tail = query[m.end():m.end()+16] if role == "purchase" else name
            feature_suffix = re.match(r"^[^，,。；;]{0,8}(?:接口|插孔|通道|引脚|档位|音色|预设|棱面)(?:的|能|可|用|来|[，,。；;]|$)", tail)
            if FEATURE.match(tail) or feature_suffix or tail.startswith("套") and m["unit"] == "件":
                continue  # 30个档位/三件套 are not a purchase count.
            # A bare '送1' in a promotion has no named accessory here.
            if role == "component" and not name.strip():
                continue
            try:
                n = number_parser(m["num"])
                if n != int(n):
                    continue
            except ValueError:
                continue
            count_start = m.start("approx") if m["approx"] else m.start("num")
            count_end = m.end("post") if m["post"] else m.end("unit")
            actual_role = "gift" if role == "component" and m["prefix"] in ("附赠", "赠送", "赠", "送") else role
            ambiguous = bool(m["approx"] or m["extra"] or m["post"])
            if re.search(r"如果|假如|倘若", before):
                ambiguous = True
            context = re.split(r"[，,。；;！？!?\n]", query[max(0, m.start()-8):m.start("num")])[-1]
            operator = "at_least" if re.search(r"至少|不少于|不低于", context) else "at_most" if re.search(r"最多|至多|不超过", context) else "eq"
            if role == "purchase" and tail.startswith("装"):
                ambiguous = True  # 两支装 may be pack contents, not ordered quantity.
            sentence = next((s for s in re.finditer(r"[^。；;！？!?\n]+", query) if s.start() <= m.start() < s.end()), None)
            if role == "purchase" and sentence and re.search(rf"买{NUM}(?:送|赠){NUM}", sentence.group()):
                if not re.search(r"付费|付款|到手|含赠|不含赠", sentence.group()):
                    ambiguous = True
            found.append({"role": actual_role, "count": int(n), "unit": UNITS[m["unit"]], "name_context": name,
                          "operator": operator,
                          "start": count_start, "end": count_end,
                          "text": query[count_start:count_end], "source": m.group(), "ambiguous": ambiguous})
    return found


def _numeric_matches(value, unit, candidate):
    if value is None or unit is None:
        return False
    if unit == candidate["unit"] and value == candidate["count"]:
        return True
    return candidate["unit"] == "pair" and unit == "piece" and value == candidate["count"] * 2


def quantity_flags(contract, number_parser):
    flags = []
    requirements = [(d, i, r) for d in DIMS for i, r in enumerate(contract[d])]
    for candidate in explicit_counts(contract["query"], number_parser):
        if candidate["ambiguous"]:
            if not _pending(contract, candidate["start"], candidate["end"]):
                flags.append({"code": "quantity_scope_pending", "location": "/query",
                              "detail": f"Quantity/pack/promotion scope must remain pending: {candidate['source']}", "source": candidate})
            continue
        covered = False
        for d, i, r in requirements:
            if not any(_covers(e, candidate["start"], candidate["end"]) for e in r["evidence"]):
                continue
            value = r["expected"]
            if candidate["role"] == "purchase":
                if r["subtype"] == "quantity" and r["subject"]["kind"] in ("primary_product", "order") and r["operator"] == candidate["operator"] and value["type"] == "number":
                    covered |= _numeric_matches(value["value"], value["unit"], candidate)
            elif r["subtype"] == "components" and value["type"] == "components":
                wanted = "gift_presence" if candidate["role"] == "gift" else "included_components"
                if r["attribute"] != wanted:
                    continue
                for item in value["items"]:
                    if len(item["name"]) >= 2 and item["name"] in candidate["name_context"]:
                        prefix = candidate["name_context"].split(item["name"], 1)[0]
                        # Do not lend the first accessory's count to a later
                        # conjunct merely because both occur in a broad quote.
                        if not re.search(r"和|以及|及|与|并", prefix):
                            covered |= _numeric_matches(item["quantity"], item["unit"], candidate)
            elif candidate["role"] == "gift" and r["subtype"] == "bundle" and value["type"] == "bundle":
                covered |= _numeric_matches(value["gift_quantity"], value["unit"], candidate)
        if not covered:
            code = "purchase_quantity_coverage" if candidate["role"] == "purchase" else "component_quantity_coverage"
            flags.append({"code": code, "location": "/query", "source": candidate,
                          "detail": f"Retain explicit {candidate['role']} count {candidate['count']} {candidate['unit']} from {candidate['source']}; do not leave missing/null/text-only or bind to another object"})
    return flags


def _focus(r, source_clauses):
    touched = [i for i, c in enumerate(source_clauses) if any(e["start"] < c["end"] and c["start"] < e["end"] for e in r["evidence"])]
    v = r["expected"]
    values = [v["value"]] if v["type"] == "text" else v.get("values", []) if v["type"] == "text_list" else [x["name"] for x in v["items"]] if v["type"] == "components" else []
    # Exact literals can identify a narrow target even when quotation is broad.
    targets = {i for value in values for i in touched if len(value) >= 2 and value in source_clauses[i]["text"]}
    return targets if values and all(any(len(v) >= 2 and v in source_clauses[i]["text"] for i in touched) for v in values) else set(touched)


def uncertain_hope_scopes(requirement, query):
    source = clauses(query)
    target = _focus(requirement, source)
    # A mixed requirement must be split by the extractor, not automatically
    # hidden by making the whole soft/hard group uncertain.
    if target and all(source[i]["hope_uncertain"] for i in target):
        return [source[i] for i in sorted(target)]
    return []


def strength_flags(contract):
    source = clauses(contract["query"])
    requirements = [(f"/{d}/{i}", r) for d in DIMS for i, r in enumerate(contract[d])]
    flags = []
    focuses = [(location, r, _focus(r, source)) for location, r in requirements]
    for location, r, target in focuses:
        soft = [i for i in target if source[i]["soft"]]
        hard = [i for i in target if source[i]["hard"] and not source[i]["soft"]]
        hope = [i for i in target if source[i]["hope_uncertain"]]
        if hope and len(hope) == len(target):
            if r["importance"] != "uncertain" or any(not _pending(contract, source[i]["start"], source[i]["end"], "ambiguous_strength") for i in hope):
                flags.append({"code": "hope_strength_pending", "location": location,
                              "detail": "希望 alone does not determine strength; preserve uncertain with contextual ambiguous_strength pending"})
        if soft and hard:
            lo, hi = min(source[i]["start"] for i in soft+hard), max(source[i]["end"] for i in soft+hard)
            if r["importance"] != "uncertain" or not _pending(contract, lo, hi, "ambiguous_strength"):
                flags.append({"code": "mixed_strength", "location": location,
                              "detail": "Split soft and separately explicit hard clauses into distinct requirements, or preserve the entire scope as uncertain with ambiguous_strength pending; never downgrade the whole group",
                              "source": contract["query"][lo:hi]})
        elif soft and not hard and r["importance"] == "required":
            flags.append({"code": "soft_clause_mandatory", "location": location,
                          "detail": "Source clause has a soft modifier; retain preference or uncertain strength, not mandatory",
                          "source": [source[i]["text"] for i in soft]})
        elif hard and not soft and r["importance"] == "preference" and any(EXPLICIT_HARD.search(source[i]["text"]) for i in hard):
            flags.append({"code": "explicit_hard_clause_optional", "location": location,
                          "detail": "An explicit mandatory modifier cannot be downgraded merely because 希望 also occurs",
                          "source": [source[i]["text"] for i in hard]})
    # A later hard clause cannot disappear behind an earlier preferred clause.
    sentences_with_soft, mixed_sentences = set(), set()
    for i, clause in enumerate(source):
        if clause["soft"]:
            sentences_with_soft.add(clause["sentence"])
        elif clause["hard"] and clause["sentence"] in sentences_with_soft:
            mixed_sentences.add(clause["sentence"])
            covered = any(i in focus and r["importance"] == "required" for _, r, focus in focuses)
            pending = _pending(contract, clause["start"], clause["end"])
            if not covered and not pending:
                flags.append({"code": "hard_clause_after_soft_uncovered", "location": "/query",
                              "detail": "Later explicit directive needs its own mandatory requirement or an explicit pending-strength decision",
                              "source": clause["text"]})
    for i, clause in enumerate(source):
        if clause["soft"] and clause["sentence"] in mixed_sentences:
            covered = any(i in focus and r["importance"] == "preference" for _, r, focus in focuses)
            if not covered and not _pending(contract, clause["start"], clause["end"]):
                flags.append({"code": "soft_clause_in_mixed_uncovered", "location": "/query",
                              "detail": "Do not delete the preferred part when splitting a mixed-strength sentence",
                              "source": clause["text"]})
    return flags


def coverage_flags(contract, number_parser):
    return quantity_flags(contract, number_parser) + strength_flags(contract)
