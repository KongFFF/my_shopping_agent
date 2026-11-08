import unittest
from jsonschema import ValidationError
from semantic_contract_v1 import DIMENSIONS, example, validate


class ContractTest(unittest.TestCase):
    def test_fixed_empty_dimensions_and_bundle_are_preserved(self):
        contract = example()
        self.assertTrue(all(key in contract for key in DIMENSIONS))
        self.assertEqual(contract["model"], [])
        self.assertEqual(contract["specifications"][1]["expected"]["gift_quantity"], 1)

    def test_cannot_add_gold_or_new_scoring_dimension(self):
        for key in ("target_asin", "reward", "new_dimension"):
            contract = example()
            contract[key] = "hidden"
            with self.assertRaises(ValidationError):
                validate(contract)

    def test_must_preserve_original_evidence_span(self):
        contract = example()
        contract["category"][0]["evidence"][0]["text"] = "用户未说的内容"
        with self.assertRaisesRegex(ValueError, "Evidence"):
            validate(contract)

    def test_pending_contract_cannot_be_used_as_reviewed(self):
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            validate(example(), require_reviewed=True, source_query=example()["query"])

    def test_reviewed_contract_can_pass_but_semantics_not_proven(self):
        contract = example()
        for dimension in DIMENSIONS:
            for r in contract[dimension]:
                r["review_status"] = "approved"
        validate(contract, require_reviewed=True, source_query=contract["query"])

    def test_unresolved_meaning_is_not_silently_ignored(self):
        contract = example()
        for dimension in DIMENSIONS:
            for r in contract[dimension]:
                r["review_status"] = "approved"
        contract["specifications"][0]["importance"] = "uncertain"
        with self.assertRaisesRegex(ValueError, "uncertain"):
            validate(contract, require_reviewed=True, source_query=contract["query"])

    def test_query_cannot_be_replaced_by_extractor(self):
        with self.assertRaisesRegex(ValueError, "original user input"):
            validate(example(), source_query="另一条原始需求")

    def test_duplicate_ids_across_dimensions_rejected(self):
        contract = example()
        contract["budget"][0]["id"] = "r1"
        with self.assertRaisesRegex(ValueError, "unique"):
            validate(contract)

    def test_type_unit_and_operator_checks(self):
        for change in ("unit", "operator", "nonfinite"):
            contract = example()
            item = contract["budget"][0]
            if change == "unit":
                item["expected"]["unit"] = "kg"
            elif change == "operator":
                item["operator"] = "matches_bundle"
            else:
                item["expected"]["value"] = float("inf")
            with self.assertRaises(ValueError):
                validate(contract)

    def test_empty_and_inverted_ranges_rejected(self):
        for lower, upper in ((None, None), (200, 100)):
            contract = example()
            item = contract["budget"][0]
            item["operator"] = "between"
            item["expected"] = {"type": "range", "lower": lower, "upper": upper, "unit": "CNY",
                                "lower_inclusive": True, "upper_inclusive": True}
            with self.assertRaisesRegex(ValueError, "range"):
                validate(contract)


if __name__ == "__main__":
    unittest.main()
