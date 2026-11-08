"""Behavioral checks of approved price semantics and real runtime gates."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_task_contract_v2 as builder
import task_contract_v2 as rules


class BudgetSemanticsTest(unittest.TestCase):
    def assert_bounds(self, query, lower, upper, low_inc=True, high_inc=True):
        details = rules.budget_details_v2(query)
        self.assertEqual(details["status"], "parsed", details)
        self.assertEqual((details["lower"], details["upper"]), (lower, upper), query)
        self.assertEqual((details["lower_inclusive"], details["upper_inclusive"]), (low_inc, high_inc), query)
        return details

    def test_user_final_chutou_policy(self):
        for amount, upper in [(10, 14), (20, 24), (65, 69), (99, 103), (100, 109), (420, 429),
                              (600, 609), (1000, 1099), (2000, 2099), (10000, 10999), (100000, 109999)]:
            with self.subTest(amount=amount):
                self.assert_bounds(f"预算{amount}元出头", None, upper)
        self.assert_bounds("售价420元出头", None, 429)
        self.assert_bounds("预算大概一千出头", None, 1099)
        self.assert_bounds("价格在九十元出头", None, 94)
        self.assertEqual(rules.budget_details_v2("预算5元出头")["status"], "review_required")
        self.assertEqual(rules.budget_details_v2("预算20.5元出头")["status"], "review_required")

    def test_chinese_shorthand_and_zero(self):
        for token, amount in [("两百五", 250), ("两百零五", 205), ("一百六", 160), ("一百零六", 106),
                              ("一千七", 1700), ("一千零七", 1007), ("六千二", 6200), ("六千零二", 6002),
                              ("一万三", 13000), ("一万零三", 10003), ("一千三百五", 1350)]:
            self.assertEqual(rules.money_number_v2(token), amount, token)
        self.assertIsNone(rules.money_number_v2("三百四五十"))
        self.assertIsNone(rules.money_number_v2("1万30"))
        self.assert_bounds("预算两百五元左右", None, 275)
        self.assert_bounds("价格别太贵，大概四百五左右", None, 495)
        self.assert_bounds("预算六千二以内", None, 6200)

    def test_lower_and_combined_bounds(self):
        for query in ("价格在50元以上", "价格不低于50元", "价格不能低于50元", "价格至少50元"):
            self.assert_bounds(query, 50, None)
        self.assert_bounds("价格高于250元，不超300元", 250, 300, False, True)
        self.assert_bounds("价格在960元以上，不超过1000元", 960, 1000)
        self.assert_bounds("价格要在60到70元之间", 60, 70)
        self.assert_bounds("预算在800-820元之间", 800, 820)

    def test_negative_upper_cues_and_hard_caps(self):
        for query in ("价格别超过200元", "价格尽量别超过200元", "价格不能高于200元", "价格不得超出200元", "价格大概不超过200元"):
            self.assert_bounds(query, None, 200)
        self.assert_bounds("价格差不多在1700元以内", None, 1700)
        self.assert_bounds("价格要低于三百元", None, 300, True, False)
        self.assert_bounds("价格不超过20的产品", None, 20)

    def test_spoken_ranges_without_stacked_tolerance(self):
        self.assert_bounds("预算五六十块钱吧", None, 60)
        self.assert_bounds("价格三四十块左右", 30, 40)
        self.assert_bounds("价格三百四五十 左右", 340, 350)
        self.assert_bounds("十几块钱左右", 11, 19)
        self.assert_bounds("预算四十几块", None, 49)

    def test_duo_budget_has_no_minimum_spend(self):
        self.assert_bounds("预算80多块", None, 90, True, False)
        self.assert_bounds("预算一百三十多元左右", None, 140, True, False)
        self.assert_bounds("价格在九百多块钱", 900, 1000, False, False)
        self.assert_bounds("预算在250多元", None, 260, True, False)

    def test_non_money_numbers_do_not_create_price_bounds(self):
        examples = [
            ("每厘米长度内的孔数大于1500，价格为5元左右", 5.5),
            ("最多5支就够了，价格不超过50元", 50),
            ("9头蝴蝶兰不超过3枝，价格低于300元", 300),
            ("每次投放不超过2包，预算900元左右", 990),
            ("克重不超过0.3g，在280左右的都能接受", 308),
            ("长度不超过35公分，角度不超过55度，纯度不低于99%，价格不超过2000块", 2000),
        ]
        for query, upper in examples:
            result = rules.budget_details_v2(query)
            self.assertEqual(result["status"], "parsed", result)
            self.assertIsNone(result["lower"])
            self.assertEqual(result["upper"], upper)
            self.assertEqual(len(result["evidence"]), 1, result)
        self.assertEqual(rules.budget_details_v2("克重不超过0.3g，颜色黑色")["status"], "not_declared")

    def test_ordinary_approximation_unchanged(self):
        for query, upper in [("价格在80元左右", 88), ("大致预算是40元", 44), ("每瓶价格要在10元左右", 11),
                              ("价格在500元左右", 550), ("170元的预算", 170), ("预算1,300元以内", 1300)]:
            self.assert_bounds(query, None, upper)

    def test_unverified_units_and_deposits_are_not_invented_totals(self):
        for query in ("每米价格60元左右，总价不超过120元", "手工费不超过200元，总价1600元左右", "定金价格40000块左右", "每米价格50元左右"):
            self.assertEqual(rules.budget_details_v2(query)["status"], "review_required", query)
        result = self.assert_bounds("长度为一米，单价在20块钱左右", None, 22)
        self.assertEqual(result["unit_scope"], {"kind": "length", "quantity": 1, "unit": "m"})


class ActualRuntimeGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = builder.load_runtime(builder.runtime_sources())
        cls.reward = cls.modules["reward.py"]
        cls.features = cls.modules["reward_features.py"]

    def goal(self, query):
        product = {"asin": "synthetic", "category": "配件", "title": "测试商品", "attribute": [], "customization_options": {}}
        instruction = {"instruction": query, "attributes": [], "instruction_options": []}
        return product, {"asin": "synthetic", **self.features.compile_reward_features(instruction, product)}

    def gate(self, query, price, selected=None):
        product, goal = self.goal(query)
        result = self.reward.evaluate_purchase(product, goal, selected_options=selected or {}, price=price)
        return result.hard_gates["budget"]["status"], result

    def test_actual_purchase_rejects_below_lower_bound(self):
        self.assertEqual(self.gate("价格50元以上", 47)[0], "fail")
        self.assertEqual(self.gate("价格50元以上", 50)[0], "pass")
        self.assertEqual(self.gate("价格50元以上", 55)[0], "pass")
        self.assertEqual(self.gate("价格50元以上", 47)[1].reward_type, "wrong_purchase")

    def test_open_and_closed_interval_endpoints(self):
        for price, expected in [(249.99, "fail"), (250, "fail"), (250.01, "pass"), (299, "pass"), (300, "pass"), (300.01, "fail")]:
            self.assertEqual(self.gate("价格高于250元，不超300元", price)[0], expected, price)
        self.assertEqual(self.gate("价格低于300元", 300)[0], "fail")

    def test_user_chutou_endpoints_and_no_floor(self):
        for query, cap in [("预算20元出头", 24), ("预算100元出头", 109), ("预算1000元出头", 1099)]:
            self.assertEqual(self.gate(query, cap)[0], "pass")
            self.assertEqual(self.gate(query, cap + 0.01)[0], "fail")
            self.assertEqual(self.gate(query, 1)[0], "pass")

    def test_vague_and_approximate_boundary_behavior(self):
        self.assertEqual(self.gate("预算四十几块", 38)[0], "pass")
        self.assertEqual(self.gate("预算80多块", 89.99)[0], "pass")
        self.assertEqual(self.gate("预算80多块", 90)[0], "fail")
        self.assertEqual(self.gate("价格80多块", 80)[0], "fail")
        self.assertEqual(self.gate("价格在500元左右", 556)[0], "fail")
        self.assertEqual(self.gate("预算120元能拿下吗", 125)[0], "fail")

    def test_eligibility_uses_same_lower_gate_as_purchase(self):
        product, goal = self.goal("价格不低于50元")
        product["customization_options"] = {"颜色": [{"value": "默认", "price": 47}]}
        result = self.reward.evaluate_candidate_eligibility(product, goal)
        self.assertEqual(result["hard_gates"]["budget"]["status"], "fail")
        self.assertFalse(result["known_acceptable"])

    def test_unknown_price_for_lower_bound_is_unverifiable(self):
        product, goal = self.goal("价格不低于50元")
        result = self.reward.evaluate_purchase(product, goal, selected_options={}, price_resolution={"status": "unverifiable", "price": None})
        self.assertEqual(result.reward_type, "reward_unverifiable")
        self.assertFalse(result.reward_valid)

    def test_per_metre_comparison_requires_selected_unit_proof(self):
        query = "长度为一米，单价在20块钱左右"
        self.assertEqual(self.gate(query, 22, {"规格": "φ0.9mm*1米"})[0], "pass")
        self.assertEqual(self.gate(query, 22.01, {"规格": "φ0.9mm*1米"})[0], "fail")
        self.assertEqual(self.gate(query, 22, {"规格": "φ0.9mm*2米"})[0], "fail")
        self.assertEqual(self.gate(query, 22, {"规格": "φ0.9mm"})[0], "unverifiable")
        self.assertEqual(self.gate(query, 22, {"规格": "1m²"})[0], "unverifiable")

    def test_unsupported_declared_budget_fails_closed(self):
        result = self.gate("定金价格40000元左右", 40000)[1]
        self.assertEqual(result.reward_type, "reward_unverifiable")
        self.assertFalse(result.reward_valid)


class AnnotationRepairTest(unittest.TestCase):
    def product(self, *values):
        return {"customization_options": {"颜色": [{"value": value} for value in values]}}

    def test_unique_symbol_and_terminal_punctuation(self):
        self.assertEqual(rules.resolve_damaged_option_v2("【?推荐】5斤", self.product("【✅推荐】5斤"), "要5斤"), ("颜色", "【✅推荐】5斤"))
        self.assertEqual(rules.resolve_damaged_option_v2("深灰??科技布", self.product("深灰❤️科技布"), "科技布"), ("颜色", "深灰❤️科技布"))
        self.assertEqual(rules.resolve_damaged_option_v2("12色60ml带盒", self.product("12色60ml带盒;"), "12色60ml带盒"), ("颜色", "12色60ml带盒;"))

    def test_semantic_characters_and_ambiguous_matches_cannot_be_guessed(self):
        self.assertIsNone(rules.resolve_damaged_option_v2("红?", self.product("红色"), "红色"))
        self.assertIsNone(rules.resolve_damaged_option_v2("A?0", self.product("A20"), "A20"))
        self.assertIsNone(rules.resolve_damaged_option_v2("A?", self.product("A❤", "A♥"), "A"))

    def test_square_metre_requires_matching_query_area(self):
        self.assertEqual(rules.resolve_damaged_option_v2("1m?", self.product("1m²"), "整好一平米"), ("颜色", "1m²"))
        self.assertEqual(rules.resolve_damaged_option_v2("120m?降温", self.product("120m²降温"), "120平方"), ("颜色", "120m²降温"))
        self.assertIsNone(rules.resolve_damaged_option_v2("120m?降温", self.product("120m²降温"), "130平方"))
        self.assertIsNone(rules.resolve_damaged_option_v2("1m?", self.product("1m²"), "长度1米"))
        self.assertIsNone(rules.resolve_damaged_option_v2("1m?", self.product("1m❤"), "整好一平米"))


class PackageInvariantsTest(unittest.TestCase):
    def test_v1_unchanged_and_v2_split_bound_to_runtime(self):
        builder.verify_v1()
        manifest = json.loads((builder.PACKAGE / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((builder.PACKAGE / name).read_bytes()).hexdigest(), expected, name)
        self.assertEqual(manifest["retained_count"] + manifest["quarantined_count"], 1000)
        self.assertFalse(set(manifest["retained_task_ids"]) & set(manifest["validation_task_ids_unchanged"]))
        holds, _ = builder.manual_holds()
        self.assertFalse(set(holds) & set(manifest["retained_task_ids"]))
        for name, expected in manifest["source_runtime_normalized_sha256"].items():
            self.assertEqual(hashlib.sha256((builder.ENGINE / name).read_text(encoding="utf-8").encode()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
