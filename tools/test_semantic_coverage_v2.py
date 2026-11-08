import copy
import json
import unittest
from semantic_contract_v2 import chinese_number, normalize
from semantic_coverage_v2 import explicit_counts
from test_semantic_contract_v2 import single, normalized


def add(raw, other, dimension="specifications"):
    raw[dimension].extend(copy.deepcopy(other[dimension]))
    return raw


class CoverageV2Test(unittest.TestCase):
    def test_purchase_count_missing_text_only_wrong_unit_and_correct_count(self):
        q = "想购买一套咖啡液"
        raw = single(q, "product_category", "product_category", {"type": "text", "value": "咖啡液"}, dimension="category", quote="咖啡液")
        with self.assertRaisesRegex(ValueError, "purchase_quantity_coverage"):
            normalized(raw)
        count = single(q, "quantity", "purchase_count", {"type": "text", "value": "一套"}, quote="一套")
        add(raw, count)
        with self.assertRaisesRegex(ValueError, "purchase_quantity_coverage"):
            normalized(raw)
        raw["specifications"][0]["expected"] = {"type": "number", "value": 1, "unit": "piece"}
        with self.assertRaisesRegex(ValueError, "purchase_quantity_coverage"):
            normalized(raw)
        raw["specifications"][0]["expected"]["unit"] = "set"
        normalized(raw)

    def test_component_explicit_one_cannot_be_null_or_wrong_object(self):
        q = "自带一个小压力表"
        raw = single(q, "components", "included_components", {"type": "components", "items": [{"name": "压力表", "quantity": None, "unit": None}]}, operator="contains_all")
        with self.assertRaisesRegex(ValueError, "component_quantity_coverage"):
            normalized(raw)
        item = raw["specifications"][0]["expected"]["items"][0]
        item.update(quantity=1, unit="piece")
        normalized(raw)
        item["name"] = "温度表"
        with self.assertRaisesRegex(ValueError, "component_quantity_coverage"):
            normalized(raw)
        raw = single("自带一个压力表和两个温度表", "components", "included_components", {"type": "components", "items": [{"name": "温度表", "quantity": 1, "unit": "piece"}]})
        with self.assertRaisesRegex(ValueError, "component_quantity_coverage"):
            normalized(raw)

    def test_unknown_accessory_count_stays_unknown_without_invention(self):
        raw = single("附带压力表", "components", "included_components", {"type": "components", "items": [{"name": "压力表", "quantity": None, "unit": None}]})
        normalized(raw)

    def test_purchase_and_gift_counts_cannot_substitute_for_each_other(self):
        q = "买两套文具，附赠一个收纳盒"
        raw = single(q, "quantity", "purchase_count", {"type": "number", "value": 2, "unit": "set"}, quote="买两套文具")
        gift = single(q, "components", "gift_presence", {"type": "components", "items": [{"name": "收纳盒", "quantity": 1, "unit": "piece"}]}, quote="附赠一个收纳盒")
        add(raw, gift)
        normalized(raw)
        raw["specifications"][1]["attribute"] = "included_components"
        with self.assertRaisesRegex(ValueError, "component_quantity_coverage"):
            normalized(raw)

    def test_no_purchase_count_from_feature_pack_unit_price_or_negation(self):
        for q in ("需要30个档位", "买三件套化妆品", "买一款点痣笔，有30个档位", "20元一瓶", "不买两套", "不要附赠一个收纳袋", "需要一个USB接口", "需要两个电压", "要两个颜色"):
            with self.subTest(query=q):
                self.assertEqual(explicit_counts(q, chinese_number), [])

    def test_pair_conversion_and_lower_bound_are_preserved(self):
        raw = single("需要一对火箭筒", "quantity", "purchase_count", {"type": "number", "value": 2, "unit": "piece"})
        normalized(raw)
        raw = single("至少买两套文具", "quantity", "purchase_count", {"type": "number", "value": 2, "unit": "set"}, operator="at_least")
        normalized(raw)
        raw["specifications"][0]["operator"] = "eq"
        with self.assertRaisesRegex(ValueError, "purchase_quantity_coverage"):
            normalized(raw)

    def test_promotion_count_scope_needs_pending_not_a_paid_count_guess(self):
        q = "要两袋，买一送一"
        raw = single(q, "quantity", "purchase_count", {"type": "number", "value": 2, "unit": "bag"}, quote="要两袋")
        with self.assertRaisesRegex(ValueError, "quantity_scope_pending"):
            normalized(raw)
        raw["specifications"][0]["importance"] = "uncertain"
        raw["pending_mentions"] = [{"evidence": {"text": q}, "reason": "ambiguous_meaning", "importance": "uncertain"}]
        normalized(raw)

    def test_mixed_strength_cannot_be_a_single_preference(self):
        q = "希望水流大一些，别卡水、憋水"
        raw = single(q, "function", "function", {"type": "text", "value": "水流大一些，别卡水、憋水"}, dimension="functions", importance="preference")
        with self.assertRaisesRegex(ValueError, "mixed_strength"):
            normalized(raw)
        raw = single("最好蓝色但必须防水", "function", "function", {"type": "text", "value": "蓝色但必须防水"}, dimension="functions", importance="preference")
        with self.assertRaisesRegex(ValueError, "mixed_strength"):
            normalized(raw)

    def test_split_strength_or_local_uncertainty_passes_without_auto_decision(self):
        q = "希望水流大一些，别卡水、憋水"
        raw = single(q, "function", "function", {"type": "text", "value": "水流大一些"}, dimension="functions", importance="preference", quote="希望水流大一些")
        hard = single(q, "function", "function", {"type": "text", "value": "别卡水、憋水"}, dimension="functions", quote="别卡水、憋水")
        add(raw, hard, "functions")
        normalized(raw)
        raw["functions"][1]["importance"] = "uncertain"
        raw["pending_mentions"] = [{"evidence": {"text": "别卡水、憋水"}, "reason": "ambiguous_strength", "importance": "uncertain"}]
        normalized(raw)

    def test_same_clause_soft_negation_is_not_forced_mandatory(self):
        raw = single("最好不要红色", "color", "color", {"type": "text", "value": "不要红色"}, importance="preference")
        normalized(raw)
        raw["specifications"][0]["importance"] = "required"
        with self.assertRaisesRegex(ValueError, "soft_clause_mandatory"):
            normalized(raw)

    def test_clipped_soft_modifier_and_dropped_hard_clause_are_detected(self):
        raw = single("最好防水", "function", "function", {"type": "text", "value": "防水"}, dimension="functions", quote="防水")
        with self.assertRaisesRegex(ValueError, "soft_clause_mandatory"):
            normalized(raw)
        q = "最好蓝色，必须防水"
        raw = single(q, "color", "color", {"type": "text", "value": "蓝色"}, importance="preference", quote="最好蓝色")
        with self.assertRaisesRegex(ValueError, "hard_clause_after_soft_uncovered"):
            normalized(raw)

    def test_broad_context_is_safe_if_each_target_is_separate(self):
        q = "最好蓝色，必须防水"
        raw = single(q, "color", "color", {"type": "text", "value": "蓝色"}, importance="preference", quote=q)
        hard = single(q, "function", "function", {"type": "text", "value": "防水"}, dimension="functions", quote=q)
        add(raw, hard, "functions")
        normalized(raw)
        raw["specifications"] = []
        with self.assertRaisesRegex(ValueError, "soft_clause_in_mixed_uncovered"):
            normalized(raw)
        q = "最好蓝色但必须防水"
        raw = single(q, "color", "color", {"type": "text", "value": "蓝色"}, importance="preference", quote="最好蓝色")
        add(raw, single(q, "function", "function", {"type": "text", "value": "防水"}, dimension="functions", quote="必须防水"), "functions")
        normalized(raw)

    def test_real_first_batch_failures_and_manual_repair_fixtures(self):
        from semantic_contract_v2 import PACKAGE
        cases = json.loads((PACKAGE / "coverage_known_cases.r2.json").read_text(encoding="utf-8"))["cases"]
        self.assertEqual({c["task_id"] for c in cases}, {2547, 3499})
        for case in cases:
            with self.subTest(task_id=case["task_id"]):
                with self.assertRaises(ValueError) as caught:
                    normalized(case["negative_raw"])
                for code in case["expected_flag_codes"]:
                    self.assertIn(code, str(caught.exception))
                self.assertFalse(case["fixture_is_llm_output"])
                self.assertFalse(case["human_approved"])
                normalized(case["manually_repaired_raw_fixture"])

    def test_hope_alone_is_pending_for_both_original_strengths(self):
        for importance in ("required", "preference"):
            raw = single("希望防水", "function", "function", {"type": "text", "value": "防水"},
                         dimension="functions", importance=importance, quote="防水")
            c = normalized(raw)
            self.assertEqual(c["functions"][0]["importance"], "uncertain")
            self.assertEqual(c["pending_mentions"][0]["evidence"]["text"], "希望防水")
        raw = single("希望最好防水", "function", "function", {"type": "text", "value": "防水"}, dimension="functions", importance="preference")
        self.assertEqual(normalized(raw)["functions"][0]["importance"], "preference")
        raw = single("希望必须防水", "function", "function", {"type": "text", "value": "防水"}, dimension="functions")
        self.assertEqual(normalized(raw)["functions"][0]["importance"], "required")
        raw["functions"][0]["importance"] = "preference"
        with self.assertRaisesRegex(ValueError, "explicit_hard_clause_optional"):
            normalized(raw)
        raw = single("希望价格100元左右", "total_price", "total_price", {"type": "text", "value": "价格100元左右"},
                     dimension="budget", quote="价格100元左右", importance="preference")
        c = normalized(raw)
        self.assertEqual(c["budget"][0]["importance"], "uncertain")
        self.assertEqual(c["budget"][0]["expected"]["value"], 110)


if __name__ == "__main__":
    unittest.main()
