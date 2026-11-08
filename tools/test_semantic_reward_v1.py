import unittest
from semantic_contract_v1 import DIMENSIONS, example
from semantic_reward_v1 import score


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.contract = example()
        for dimension in DIMENSIONS:
            for r in self.contract[dimension]:
                r["review_status"] = "approved"
        self.results = {f"r{i}": "pass" for i in range(1, 6)}

    def run_score(self):
        return score(self.contract, self.results, source_query=self.contract["query"])

    def test_all_satisfied_is_one(self):
        self.assertEqual(self.run_score()["reward"], 1)

    def test_preference_missing_is_still_success(self):
        self.results["r5"] = "fail"
        result = self.run_score()
        self.assertTrue(result["semantic_success"])
        self.assertEqual(result["reward"], 0.55)

    def test_absent_preference_not_penalized(self):
        self.contract["specifications"].pop()
        self.results.pop("r5")
        self.assertEqual(self.run_score()["reward"], 1)

    def test_wrong_bundle_cannot_be_offset_by_correct_preferences(self):
        self.results["r3"] = "fail"
        result = self.run_score()
        self.assertFalse(result["semantic_success"])
        self.assertLessEqual(result["reward"], 0.25)
        self.results["r5"] = "fail"
        self.assertEqual(result["reward"], self.run_score()["reward"])

    def test_category_and_budget_failures_are_valid_negatives(self):
        for key in ("r1", "r4"):
            self.results = {f"r{i}": "pass" for i in range(1, 6)}
            self.results[key] = "fail"
            self.assertEqual(self.run_score()["reward"], -0.85)
            self.assertTrue(self.run_score()["reward_valid"])

    def test_unknown_is_not_silently_treated_as_pass_or_fail(self):
        self.results["r3"] = "unknown"
        self.assertIsNone(self.run_score()["semantic_success"])
        self.assertTrue(self.run_score()["sampling_invalid"])
        self.results["r4"] = "fail"
        self.assertEqual(self.run_score()["reward"], -0.85)

    def test_dimension_equal_weight_not_requirement_count_weight(self):
        self.results["r2"] = "fail"
        self.results["r3"] = "fail"
        # category=1, budget=1, specifications=0: average=2/3, not 2/4.
        self.assertAlmostEqual(self.run_score()["required_score"], 2 / 3)

    def test_missing_or_extra_comparison_rejected(self):
        self.results.pop("r3")
        with self.assertRaises(ValueError):
            self.run_score()


if __name__ == "__main__":
    unittest.main()
