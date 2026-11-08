import copy
import json
import tempfile
import unittest
from collections import Counter
from jsonschema import Draft202012Validator
from build_semantic_pilot50 import STRATA, artifacts, compact_schema, select, source_tasks, verify
from check_semantic_pilot50 import reject_duplicate_keys, validate_response
from semantic_contract_v1 import example, schema


class PilotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks, _ = source_tasks()
        cls.selected, cls.audit = select(cls.tasks)
        cls.files = artifacts()

    def test_exact_50_original_train_queries_and_quotas(self):
        originals = {t["task_id"]: t["query"] for t in self.tasks}
        self.assertEqual(len(self.selected), 50)
        self.assertEqual(len({t["query"] for t in self.selected}), 50)
        for t in self.selected:
            self.assertEqual(t["query"], originals[t["task_id"]])
        self.assertEqual(Counter(r["primary_stratum"] for r in self.audit), {n: q for n, q, _ in STRATA})

    def test_selection_does_not_depend_on_input_order(self):
        self.assertEqual(select(list(reversed(self.tasks))), (self.selected, self.audit))

    def test_requests_contain_only_original_task_and_fixed_system(self):
        requests = [json.loads(line) for line in self.files["requests.jsonl"].decode().splitlines()]
        self.assertEqual(len(requests), 50)
        system = self.files["system_prompt.txt"].decode()
        for row, task in zip(requests, self.selected):
            self.assertEqual(row["messages"][0], {"role": "system", "content": system})
            user = row["messages"][1]["content"]
            payload = json.loads(user[user.index("{"):])
            self.assertEqual(payload, task)
            self.assertEqual(set(payload), {"task_id", "query"})

    def test_compact_schema_agrees_on_examples_and_invalid_subtypes(self):
        a, b = Draft202012Validator(schema()), Draft202012Validator(compact_schema())
        valid = example()
        a.validate(valid)
        b.validate(valid)
        for field, value in (("subtype", "merchant"), ("subtype", "unrecognized"), ("importance", "invented")):
            invalid = copy.deepcopy(valid)
            invalid["category"][0][field] = value
            self.assertEqual(a.is_valid(invalid), b.is_valid(invalid))
            self.assertFalse(b.is_valid(invalid))

    def test_original_query_and_review_gate_are_enforced(self):
        contract = example()
        tasks = {0: contract["query"]}
        validate_response(contract, tasks, compact_schema())
        with self.assertRaises(ValueError):
            validate_response(contract, {0: "别的原文"}, compact_schema())
        contract["category"][0]["review_status"] = "approved"
        with self.assertRaisesRegex(ValueError, "approve"):
            validate_response(contract, tasks, compact_schema())

    def test_unknown_task_and_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "not in frozen"):
            validate_response(example(), {}, compact_schema())
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            json.loads('{"task_id":1,"task_id":2}', object_pairs_hook=reject_duplicate_keys)

    def test_frozen_snapshot_matches_exact_rebuild(self):
        verify()

    def test_prompt_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            from pathlib import Path
            target = Path(directory)
            for name, data in self.files.items():
                (target / name).write_bytes(data)
            with (target / "system_prompt.txt").open("ab") as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(AssertionError, "Frozen artifact changed"):
                verify(target)


if __name__ == "__main__":
    unittest.main()
