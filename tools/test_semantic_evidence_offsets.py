import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from build_semantic_pilot50 import compact_schema
from check_semantic_pilot50 import validate_response
from extract_semantic_requirements import contract_from_response, dump, execute, journal_read, summarize
from semantic_contract_v1 import example
from semantic_evidence_offsets import normalize_evidence


class EvidenceOffsetsTest(unittest.TestCase):
    def setUp(self):
        self.contract = example()
        self.schema = compact_schema()
        self.task = {"task_id": self.contract["task_id"], "query": self.contract["query"]}

    def raw(self, contract):
        return dump({"choices": [{"finish_reason": "stop", "message": {"content": dump(contract)}}],
                     "usage": {"total_tokens": 7}})

    def test_exact_unique_offset_only_and_no_input_mutation(self):
        damaged = copy.deepcopy(self.contract)
        e = damaged["category"][0]["evidence"][0]
        e["end"] -= 1
        before = copy.deepcopy(damaged)
        normalized, changes = normalize_evidence(damaged, self.task, self.schema)
        self.assertEqual(damaged, before)
        self.assertEqual(normalized, self.contract)
        self.assertEqual(len(changes), 1)
        validate_response(normalized, {self.task["task_id"]: self.task["query"]}, self.schema)

    def test_pending_mentions_also_normalized(self):
        damaged = copy.deepcopy(self.contract)
        text = self.task["query"]
        damaged["pending_mentions"] = [{"reason": "unsupported_type", "evidence": {"start": 0, "end": 1, "text": text}}]
        normalized, changes = normalize_evidence(damaged, self.task, self.schema)
        self.assertEqual(normalized["pending_mentions"][0]["evidence"]["end"], len(text))
        self.assertEqual(changes[0]["path"], "/pending_mentions/0/evidence")

    def test_no_fuzzy_match_and_no_guess_on_repeated_text(self):
        for query, text in (("苹果苹果", "苹果"), ("苹果", "苹 果")):
            damaged = copy.deepcopy(self.contract)
            damaged["query"] = query
            damaged["category"][0]["evidence"] = [{"start": 0, "end": 1, "text": text}]
            with self.assertRaisesRegex(ValueError, "exactly one"):
                normalize_evidence(damaged, {"task_id": damaged["task_id"], "query": query}, self.schema)

    def test_query_and_schema_cannot_be_repaired(self):
        with self.assertRaises(ValueError):
            normalize_evidence(self.contract, dict(self.task, query="other"), self.schema)
        invalid = copy.deepcopy(self.contract)
        invalid["category"][0]["subtype"] = "packaging"
        with self.assertRaises(Exception):
            normalize_evidence(invalid, self.task, self.schema)

    def test_strict_still_rejects_but_opt_in_generates_audited_derived_output(self):
        damaged = copy.deepcopy(self.contract)
        damaged["category"][0]["evidence"][0]["end"] -= 1
        raw = self.raw(damaged)
        with self.assertRaises(ValueError):
            contract_from_response(raw, self.task, self.schema)
        config = {"model": "mock", "endpoint": "https://example.com/v1/chat/completions",
                  "max_tokens": 8192, "thinking": "omit", "response_format": "json_object", "evidence_offsets": "exact-unique"}
        row = {"custom_id": "test", "messages": []}
        calls = []
        def transport(*args):
            calls.append(args)
            return 200, raw
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertEqual(execute(folder, [self.task], [row], self.schema, config, "test-secret", transport=transport), 0)
            self.assertEqual(execute(folder, [self.task], [row], self.schema, config, "test-secret", transport=transport), 0)
            self.assertEqual(len(calls), 1)
            self.assertFalse((folder / "responses.jsonl").exists())
            self.assertEqual(json.loads((folder / "responses.normalized.jsonl").read_text(encoding="utf-8")), self.contract)
            self.assertEqual(journal_read(folder / "attempts.jsonl")[0]["raw_response"], raw)
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["normalized_tasks"], 1)
            self.assertEqual(report["raw_validated_tasks"], 0)
            self.assertEqual(len(journal_read(folder / "normalizations.jsonl")), 1)

    def test_historical_failed_retries_select_first_valid_normalized_attempt(self):
        damaged = copy.deepcopy(self.contract)
        damaged["category"][0]["evidence"][0]["end"] -= 1
        record = {"task_id": self.task["task_id"], "http_status": 200, "accepted": False, "raw_response": self.raw(damaged)}
        with tempfile.TemporaryDirectory() as directory:
            _, report = summarize(Path(directory), [self.task], self.schema, [record, record], "exact-unique")
            self.assertEqual(report["validated_tasks"], 1)
            self.assertEqual(report["attempts"], 2)
            self.assertEqual(report["usage_all_attempts"]["total_tokens"], 14)


if __name__ == "__main__":
    unittest.main()
