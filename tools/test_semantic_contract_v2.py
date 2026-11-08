import copy
import json
import unittest
from jsonschema import Draft202012Validator
from semantic_contract_v2 import (ATTRIBUTES, DIMENSIONS, VERSION, budget_amount, chinese_number,
                                   lint, normalize, price_basis_from_phrase, raw_example, schema, validate)
from semantic_reward_v2 import score


def single(query, subtype, attribute, expected, *, dimension="specifications", operator="eq",
           importance="required", subject=None, basis=None, quote=None):
    return {"schema_version": VERSION, "task_id": 1, "query": query, **{d: [] for d in DIMENSIONS},
            "pending_mentions": [], dimension: [{"subtype": subtype, "attribute": attribute,
                "subject": subject or {"kind": "primary_product", "name": "主商品"},
                "importance": importance, "operator": operator, "expected": expected, "price_basis": basis,
                "evidence": [{"text": quote or query}], "review_status": "pending"}]}


def normalized(raw):
    return normalize(raw, {"task_id": raw["task_id"], "query": raw["query"]})[0]


class ContractV2Test(unittest.TestCase):
    def test_both_schemas_valid_and_program_assigns_ids_evidence_and_budget(self):
        for llm in (False, True):
            Draft202012Validator.check_schema(schema(llm=llm))
        raw = raw_example()
        before = copy.deepcopy(raw)
        c, audit = normalize(raw, {"task_id": 0, "query": raw["query"]})
        self.assertEqual(raw, before)
        self.assertEqual(c["budget"][0]["expected"]["value"], 100)
        ids = [r["id"] for d in DIMENSIONS for r in c[d]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(c["specifications"][1]["expected"]["received_quantity"], 11)
        self.assertTrue(any(a["kind"] == "same_item_received_arithmetic" for a in audit))
        validate(c, source_query=raw["query"], extraction=True)

    def test_budget_policies(self):
        cases = {"170元的预算": 170, "预算20元左右": 22, "20元出头": 24, "100元出头": 109,
                 "1000块钱出头": 1099, "五十块钱出头": 54, "九百多块钱": 999,
                 "一百多元": 199, "价格差不多在1700元以内": 1700, "价格不能超过20元": 20,
                 "价格大概50元": 55, "20元一瓶装的": 20, "每米价格在50元左右": 55}
        for phrase, want in cases.items():
            with self.subTest(phrase=phrase):
                op, value, _ = budget_amount(phrase)
                self.assertEqual(op, "at_most")
                self.assertAlmostEqual(value["value"], want)
        op, value, _ = budget_amount("预算不到100元")
        self.assertEqual(op, "between")
        self.assertFalse(value["upper_inclusive"])
        self.assertEqual(budget_amount("价格不低于340元")[0], "at_least")
        self.assertEqual(budget_amount("价格100至200元")[1]["lower"], 100)

    def test_unknown_amounts_are_not_extrapolated(self):
        for phrase in ("一千多块钱", "五十多块钱", "大概一100包", "预算1万", "预算一百五"):
            with self.subTest(phrase=phrase), self.assertRaises(ValueError):
                budget_amount(phrase)

    def test_unit_price_denominator_and_bare_unit_price_upper(self):
        self.assertEqual(price_basis_from_phrase("20元一瓶装的"), {"amount": 1, "unit": "bottle"})
        self.assertEqual(price_basis_from_phrase("每米价格50元"), {"amount": 1, "unit": "m"})
        self.assertEqual(price_basis_from_phrase("每两瓶50元"), {"amount": 2, "unit": "bottle"})
        raw = single("20元一瓶装的", "unit_price", "unit_price", {"type": "text", "value": "20元一瓶装的"}, dimension="budget")
        c = normalized(raw)
        self.assertEqual(c["budget"][0]["price_basis"], {"amount": 1, "unit": "bottle"})
        self.assertEqual(c["budget"][0]["operator"], "at_most")
        raw["budget"][0]["price_basis"] = {"amount": 1, "unit": "m"}
        with self.assertRaisesRegex(ValueError, "denominator"):
            normalized(raw)
        fee = single("每米加工费20元", "processing_fee", "processing_fee", {"type": "text", "value": "每米加工费20元"}, dimension="budget")
        self.assertEqual(normalized(fee)["budget"][0]["price_basis"], {"amount": 1, "unit": "m"})
        total = single("每米价格20元", "total_price", "total_price", {"type": "text", "value": "每米价格20元"}, dimension="budget")
        with self.assertRaisesRegex(ValueError, "not total"):
            normalized(total)

    def test_feature_count_is_not_purchase_quantity(self):
        raw = single("要30个档位", "quantity", "purchase_count", {"type": "number", "value": 30, "unit": "piece"})
        with self.assertRaisesRegex(ValueError, "purchase_count_scope"):
            normalized(raw)
        raw["specifications"][0].update(subtype="attribute_count", attribute="gear_count")
        raw["specifications"][0]["expected"]["unit"] = "count"
        self.assertEqual(normalized(raw)["specifications"][0]["attribute"], "gear_count")

    def test_budgets_do_not_become_preferences_due_to_approximation(self):
        raw = single("价格在100元左右", "total_price", "total_price", {"type": "text", "value": "价格在100元左右"}, dimension="budget", importance="preference")
        with self.assertRaisesRegex(ValueError, "budget_strength"):
            normalized(raw)
        raw["query"] = "最好价格在100元左右"
        raw["budget"][0]["evidence"][0]["text"] = raw["query"]
        raw["budget"][0]["expected"]["value"] = raw["query"]
        self.assertEqual(normalized(raw)["budget"][0]["importance"], "preference")

    def test_budget_qualifiers_cannot_be_lost_by_quoting_amount_only(self):
        raw = single("预算不低于340元", "total_price", "total_price", {"type": "text", "value": "340元"}, dimension="budget", quote="340元")
        r = normalized(raw)["budget"][0]
        self.assertEqual(r["operator"], "at_least")
        self.assertEqual(r["expected"]["value"], 340)
        raw = single("最好预算100元左右", "total_price", "total_price", {"type": "text", "value": "100元"}, dimension="budget", quote="100元")
        with self.assertRaisesRegex(ValueError, "budget_strength"):
            normalized(raw)
        c = normalized(single("价格在九百多块钱", "total_price", "total_price", {"type": "text", "value": "价格在九百多块钱"}, dimension="budget"))
        c["budget"][0]["expected"]["value"] = 909
        with self.assertRaisesRegex(ValueError, "confirmed policy"):
            validate(c, source_query=c["query"])

    def test_nonbudget_tolerance_retains_original_and_blocks_reward(self):
        raw = single("承重范围大约200公斤", "capacity", "load_capacity", {"type": "number", "value": 220, "unit": "kg"}, operator="at_most")
        c = normalized(raw)
        r = c["specifications"][0]
        self.assertEqual(r["expected"], {"type": "text", "value": raw["query"]})
        self.assertEqual(r["importance"], "uncertain")
        self.assertEqual(c["pending_mentions"][0]["reason"], "undefined_tolerance")
        r["review_status"] = "approved"
        with self.assertRaisesRegex(ValueError, "uncertain"):
            validate(c, source_query=raw["query"], require_reviewed=True)

    def test_ambiguous_gift_and_shop_are_not_guessed(self):
        for query, subtype, attribute in (("可以赠送耳机", "components", "gift_presence"), ("听说甲店不错", "merchant", "merchant")):
            raw = single(query, subtype, attribute, {"type": "text", "value": query})
            c = normalized(raw)
            self.assertEqual(c["specifications"][0]["importance"], "uncertain")
            self.assertEqual(c["pending_mentions"][0]["reason"], "ambiguous_strength")

    def test_component_colors_have_distinct_subjects(self):
        raw = single("绳子红色，牌子金色", "color", "color", {"type": "text", "value": "红色"}, subject={"kind": "component", "name": "绳子"}, quote="绳子红色")
        second = copy.deepcopy(raw["specifications"][0])
        second.update(subject={"kind": "component", "name": "牌子"}, expected={"type": "text", "value": "金色"}, evidence=[{"text": "牌子金色"}])
        raw["specifications"].append(second)
        c = normalized(raw)
        self.assertEqual([r["subject"]["name"] for r in c["specifications"]], ["绳子", "牌子"])
        raw["specifications"][1]["subject"]["name"] = "框架"
        with self.assertRaisesRegex(ValueError, "Subject"):
            normalized(raw)

    def test_wrong_types_rejected_without_automatic_remapping(self):
        for text in ("薄荷香味", "小狗造型", "福字图案"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "suspicious_color"):
                normalized(single(text, "color", "color", {"type": "text", "value": text}))
        with self.assertRaisesRegex(ValueError, "origin_as_merchant"):
            normalized(single("韩国生产", "merchant", "merchant", {"type": "text", "value": "韩国生产"}))

    def test_strict_comparison_and_physical_units(self):
        raw = single("刀尖大于55度", "dimensions", "angle", {"type": "number", "value": 55, "unit": "degree"}, operator="at_least")
        with self.assertRaisesRegex(ValueError, "open_boundary"):
            normalized(raw)
        raw["specifications"][0].update(operator="between", expected={"type": "range", "lower": 55, "upper": None, "unit": "degree", "lower_inclusive": False, "upper_inclusive": False})
        normalized(raw)
        with self.assertRaisesRegex(ValueError, "unit mismatch"):
            normalized(single("1800瓦功率", "function", "power", {"type": "number", "value": 1800, "unit": "kWh"}, dimension="functions"))
        with self.assertRaisesRegex(ValueError, "invented_charge_unit"):
            normalized(single("容量一万毫安", "capacity", "electric_charge", {"type": "number", "value": 10000, "unit": "mAh"}))
        with self.assertRaisesRegex(ValueError, "load_direction"):
            normalized(single("承重200公斤", "capacity", "load_capacity", {"type": "number", "value": 200, "unit": "kg"}, operator="at_most"))
        with self.assertRaisesRegex(ValueError, "lower_bound_direction"):
            normalized(single("容量达到1度电", "capacity", "energy_capacity", {"type": "number", "value": 1, "unit": "kWh"}, operator="at_most"))

    def test_duplicate_facts_and_unknown_component_quantity(self):
        raw = raw_example()
        raw["specifications"].append(copy.deepcopy(raw["specifications"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate_fact"):
            normalized(raw)
        raw = single("由主机和手柄组成", "components", "included_components", {"type": "components", "items": [{"name": "主机", "quantity": None, "unit": None}, {"name": "手柄", "quantity": None, "unit": None}]})
        normalized(raw)
        raw["specifications"][0]["expected"]["items"][0].update(quantity=1, unit="piece")
        with self.assertRaisesRegex(ValueError, "invented_component_count"):
            normalized(raw)

    def test_components_contains_all_and_actionable_type_errors(self):
        raw = single("自带一个小压力表", "components", "included_components",
                     {"type": "components", "items": [{"name": "小压力表", "quantity": 1, "unit": "piece"}]},
                     operator="contains_all")
        c = normalized(raw)
        self.assertEqual(c["specifications"][0]["operator"], "contains_all")
        raw["specifications"][0]["expected"] = {"type": "text", "value": "小压力表"}
        with self.assertRaisesRegex(ValueError, "contains_all requires text_list or components"):
            normalized(raw)

    def test_bundle_arithmetic_and_unknown_quantity(self):
        raw = raw_example()
        raw["specifications"][1]["expected"]["received_quantity"] = 12
        with self.assertRaisesRegex(ValueError, "Paid plus gift"):
            normalized(raw)
        raw["specifications"][1]["expected"].update(gift_quantity=None, received_quantity=None)
        raw["query"] = raw["query"].replace("买10送1", "买10赠送若干")
        raw["specifications"][1]["evidence"] = [{"text": "买10赠送若干"}]
        c = normalized(raw)
        self.assertIsNone(c["specifications"][1]["expected"]["gift_quantity"])

    def test_all_50_reviewed_query_budget_regressions(self):
        from semantic_contract_v2 import PACKAGE
        from build_semantic_pilot50 import TARGET
        cases = json.loads((PACKAGE / "pilot50_regression_cases.json").read_text(encoding="utf-8"))["cases"]
        frozen = {r["task_id"]: r["query"] for line in (TARGET / "tasks.jsonl").read_text(encoding="utf-8").splitlines() for r in [json.loads(line)]}
        self.assertEqual(len(cases), 50)
        self.assertEqual({r["task_id"] for r in cases}, set(frozen))
        count = 0
        for case in cases:
            self.assertEqual(case["query"], frozen[case["task_id"]])
            for expected in case["budget_assertions"]:
                with self.subTest(task_id=case["task_id"]):
                    quote = expected["source_phrase"]
                    # This fixture tests budget policy, not an incomplete full
                    # extraction which deliberately omits all other demands.
                    op, value, _ = budget_amount(quote)
                    self.assertEqual(op, expected["expected_operator"])
                    self.assertAlmostEqual(value["value"], expected["expected_value"])
                    self.assertEqual(expected["expected_importance"], "required")
                    if expected["price_basis"]:
                        self.assertEqual(price_basis_from_phrase(quote), expected["price_basis"])
                    count += 1
        self.assertEqual(count, 48)

    def test_extraction_runner_supports_new_profile_and_both_scales(self):
        from extract_semantic_requirements import snapshot, contract_from_response, summarize, dump
        import tempfile
        from pathlib import Path
        tasks50, requests50, files50, _ = snapshot("50", "semantic-v2")
        tasks1000, requests1000, files1000, _ = snapshot("1000", "semantic-v2")
        self.assertEqual(len(tasks50), 50)
        self.assertEqual(len(tasks1000), 1000)
        full = {r["task_id"]: r["messages"] for r in requests1000}
        for r in requests50:
            self.assertEqual(r["messages"], full[r["task_id"]])
        self.assertEqual(files50["output.schema.json"], files1000["output.schema.json"])
        self.assertIn("llm_output.schema.json", files50)
        self.assertIn("normalizer_source.py", files50)
        self.assertIn("coverage_rules_source.py", files50)
        raw = raw_example()
        response = dump({"choices": [{"finish_reason": "stop", "message": {"content": dump(raw)}}]})
        task = {"task_id": raw["task_id"], "query": raw["query"]}
        c = contract_from_response(response, task, schema())
        self.assertIn("id", c["category"][0])
        with tempfile.TemporaryDirectory() as directory:
            _, report = summarize(Path(directory), [task], schema(), [{"task_id": task["task_id"], "http_status": 200, "accepted": True, "raw_response": response}])
            self.assertEqual(report["responses_file"], "responses.normalized.jsonl")
            self.assertEqual(report["normalization_pipeline"], "semantic-v2-programmatic-ids-evidence-budget")

    def test_subjective_and_latest_stay_pending(self):
        for text, subtype, reason in (("质量要好", "condition", "unverifiable_subjective"), ("最新款", "version", "missing_reference_time")):
            dim = "model" if subtype == "version" else "specifications"
            c = normalized(single(text, subtype, subtype, {"type": "text", "value": text}, dimension=dim))
            self.assertEqual(c["pending_mentions"][0]["reason"], reason)

    def test_query_identity_evidence_and_self_approval_gates(self):
        raw = raw_example()
        with self.assertRaisesRegex(ValueError, "Original"):
            normalize(raw, {"task_id": 0, "query": "other"})
        raw["category"][0]["evidence"][0]["text"] = "不存在的商品"
        with self.assertRaisesRegex(ValueError, "quote"):
            normalized(raw)
        raw = raw_example()
        raw["category"][0]["review_status"] = "approved"
        with self.assertRaisesRegex(ValueError, "approve"):
            normalized(raw)

    def test_empty_extraction_cannot_pass_by_omitting_all_requirements(self):
        raw = raw_example()
        for d in DIMENSIONS:
            raw[d] = []
        with self.assertRaisesRegex(ValueError, "Empty extraction"):
            normalized(raw)

    def test_reward_tiers_and_pending_contract_gate(self):
        c = normalized(raw_example())
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            score(c, {}, source_query=c["query"])
        for d in DIMENSIONS:
            for r in c[d]:
                r["review_status"] = "approved"
        comparisons = {r["id"]: "pass" for d in DIMENSIONS for r in c[d]}
        self.assertEqual(score(c, comparisons, source_query=c["query"])["reward"], 1)
        preference = c["specifications"][-1]["id"]
        comparisons[preference] = "fail"
        self.assertEqual(score(c, comparisons, source_query=c["query"])["reward"], 0.55)
        bundle = c["specifications"][1]["id"]
        comparisons[bundle] = "fail"
        self.assertLessEqual(score(c, comparisons, source_query=c["query"])["reward"], 0.25)
        comparisons[c["budget"][0]["id"]] = "fail"
        self.assertEqual(score(c, comparisons, source_query=c["query"])["reward"], -0.85)


if __name__ == "__main__":
    unittest.main()
