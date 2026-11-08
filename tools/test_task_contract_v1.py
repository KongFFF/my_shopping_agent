"""Behavioral regressions for query semantics and conservative data repairs."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_contract_v1 as rules
import build_task_contract_v1 as builder


class BudgetContractTest(unittest.TestCase):
    def test_equivalent_currency_and_approximation(self):
        for text, expected in [
            ("预算在20块左右", 22), ("价格在20元左右", 22),
            ("预算85块钱左右", 93.5), ("价格70元上下", 77),
            ("预算在三百三十元", 330), ("价格别超三百", 300),
            ("170元的预算，想买咖啡", 170),
            ("总价大概1万3左右，工费单价不超过220元/克", 14300),
            ("手工费不超过200元，总价在1600元左右", 1760),
            ("每米价格60元左右，总价不超过120元", 120),
            ("价格在960元以上，不超过1000元", 1000),
            ("预算4000以内，价格3900元以内", 3900),
            ("预算在800-820元之间", 820), ("价格区间在20-30之间", 30),
            ("预算1,300元以内", 1300),
            ("大致预算是40元", 44), ("预算1万3000元以内", 13000),
        ]:
            with self.subTest(text=text):
                result = rules.budget_details_v1(text)
                self.assertEqual(result["status"], "parsed")
                self.assertEqual(result["upper"], expected)

    def test_explicit_hard_caps_are_not_relaxed(self):
        for text, expected in [("价格差不多在1700元以内", 1700),
                               ("预算大概不超过200元", 200),
                               ("不得超出160元", 160)]:
            self.assertEqual(rules.budget_details_v1(text)["upper"], expected)

    def test_dimensions_and_percentages_are_not_prices(self):
        result = rules.budget_details_v1("剑刃长度不超过35公分，角度不超过55度，纯度不低于99%，价格不超过2000块")
        self.assertEqual(result["upper"], 2000)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(rules.budget_details_v1("长度不超过35公分，型号A20，无预算要求")["status"], "not_declared")

    def test_ambiguity_and_unsupported_lower_or_unit_prices_need_review(self):
        for text in ["价格80多块", "预算600元出头", "价格50元以上", "每米价格135元，花边每米25元", "预算五六十块", "预算两百五元左右", "预算一万三元", "预算1万30元"]:
            with self.subTest(text=text):
                self.assertEqual(rules.budget_details_v1(text)["status"], "review_required")


class BrandAndAnnotationTest(unittest.TestCase):
    def test_compatibility_does_not_assert_manufacturer(self):
        for text, brand in [("适合飞利浦电动牙刷的替换刷头", "philips"),
                            ("类似于苹果蓝牙跟踪设备", "apple"),
                            ("可以只支持苹果系统", "apple"),
                            ("铝板适合苹果专卖店作外墙", "apple")]:
            self.assertEqual(rules.filter_required_brands_v1(text, [brand]), [])
        self.assertEqual(rules.filter_required_brands_v1("我要飞利浦品牌的牙刷", ["philips"]), ["philips"])
        self.assertEqual(rules.filter_required_brands_v1("兼容苹果系统，但品牌要飞利浦", ["apple", "philips"]), ["philips"])

    def test_unique_decorative_glyph_repair_only(self):
        product = {"customization_options": {"颜色": [{"value": "A❤赠品"}, {"value": "B黑色"}]}}
        self.assertEqual(rules.resolve_damaged_option_v1("A?赠品", product), ("颜色", "A❤赠品"))
        self.assertIsNone(rules.resolve_damaged_option_v1("B?色", product))
        self.assertIsNone(rules.resolve_damaged_option_v1("A?赠品", {"customization_options": {"颜色": [{"value": "A❤赠品"}, {"value": "A♥赠品"}]}}))

    def test_runtime_replacements_preserve_legacy_on_ambiguity(self):
        sources = builder.runtime_sources()
        fixed = builder.load_module(sources["constraints.py"][1], "test_fixed_constraints")
        self.assertEqual(fixed.explicit_budget_from_instruction("预算20块左右"), 22)
        self.assertEqual(fixed.explicit_budget_from_instruction("价格80多块"), builder.baseline_budget("价格80多块"))
        self.assertEqual(fixed.explicit_budget_from_instruction("总价1600元左右，手工费不超过200元"), 1760)


class GeneratedPackageTest(unittest.TestCase):
    def test_manifest_and_split_invariants(self):
        manifest = json.loads((builder.PACKAGE / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((builder.PACKAGE / name).read_bytes()).hexdigest(), expected, name)
        self.assertEqual(manifest["retained_count"] + manifest["quarantined_count"], 1000)
        self.assertEqual(manifest["original_90_recovered_count"] + manifest["original_90_quarantined_count"], 90)
        self.assertFalse(set(manifest["retained_task_ids"]) & set(manifest["validation_task_ids_unchanged"]))
        audits = [json.loads(line) for line in (builder.PACKAGE / "all_train_audit.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(a["new_reward_type"] == "gold_purchase" for a in audits if a["status"] == "retained"))
        for name, expected in manifest["source_runtime_normalized_sha256"].items():
            content = (builder.ENGINE / name).read_text(encoding="utf-8").encode()
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected, "baseline mutated")


if __name__ == "__main__":
    unittest.main()
