"""No paid calls: exercise persistence, rejection, retries and exact data sources."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import os
import sys
from unittest.mock import patch

from build_semantic_pilot50 import compact_schema, TARGET
from extract_semantic_requirements import (HTTPFailure, contract_from_response, dump, endpoint,
                                           execute, payload, prepare, reuse, snapshot, summarize)
from semantic_contract_v1 import example


CONFIG = {"endpoint": "https://api.deepseek.com/chat/completions", "model": "test-model",
          "thinking": "disabled", "response_format": "json_object", "temperature": 0, "max_tokens": 8192}


def response(contract, finish="stop"):
    return dump({"model": "test-snapshot", "choices": [{"finish_reason": finish,
                 "message": {"content": dump(contract)}}],
                 "usage": {"prompt_tokens": 5000, "completion_tokens": 2000, "total_tokens": 7000}})


class ExtractionTest(unittest.TestCase):
    def setUp(self):
        self.contract = example()
        self.task = {"task_id": 0, "query": self.contract["query"]}
        self.row = {"task_id": 0, "custom_id": "test-0", "messages": []}
        self.schema = compact_schema()

    def test_50_1000_sources_and_identical_messages(self):
        small, requests, files, _ = snapshot("50")
        full, full_requests, _, _ = snapshot("1000")
        self.assertEqual(len(small), 50)
        self.assertEqual(len(full), 1000)
        mapping = {r["task_id"]: r for r in full_requests}
        frozen = [json.loads(line) for line in (TARGET / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        for row, original in zip(requests, frozen):
            self.assertEqual(row["messages"], original["messages"])
            self.assertEqual(row["messages"], mapping[row["task_id"]]["messages"])
        self.assertEqual(files["system_prompt.txt"], (TARGET / "system_prompt.txt").read_bytes())

    def test_prepare_resume_and_reject_model_or_input_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / ".run.lock").touch()
            prepare(folder, "50", CONFIG)
            prepare(folder, "50", CONFIG)
            changed = dict(CONFIG, model="another-model")
            with self.assertRaisesRegex(ValueError, "configuration"):
                prepare(folder, "50", changed)
            (folder / "tasks.jsonl").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                prepare(folder, "50", CONFIG)

    def test_no_silent_repair_of_bad_output(self):
        for raw in [response(self.contract, "length"), response({}),
                    response(dict(self.contract, query="changed"))]:
            with self.assertRaises(Exception):
                contract_from_response(raw, self.task, self.schema)
        duplicate = json.loads(response(self.contract))
        duplicate["choices"][0]["message"]["content"] = dump(self.contract).replace('"task_id":0', '"task_id":0,"task_id":0')
        with self.assertRaises(ValueError):
            contract_from_response(dump(duplicate), self.task, self.schema)
        altered = copy.deepcopy(self.contract)
        altered["category"][0]["review_status"] = "approved"
        with self.assertRaises(Exception):
            contract_from_response(response(altered), self.task, self.schema)

    def test_failed_structure_retry_then_resume_skips_valid_and_counts_both_calls(self):
        calls = []
        def transport(*args):
            calls.append(args)
            return 200, response({} if len(calls) == 1 else self.contract)
        with tempfile.TemporaryDirectory() as directory, patch("extract_semantic_requirements.time.sleep"):
            folder = Path(directory)
            self.assertEqual(execute(folder, [self.task], [self.row], self.schema, CONFIG, "test-secret", transport=transport), 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual(execute(folder, [self.task], [self.row], self.schema, CONFIG, "test-secret", transport=transport), 0)
            self.assertEqual(len(calls), 2)
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["usage_all_attempts"]["total_tokens"], 14000)
            self.assertEqual(report["returned_models"], ["test-snapshot"])
            self.assertEqual(len((folder / "responses.jsonl").read_text(encoding="utf-8").splitlines()), 1)

    def test_auth_error_stops_immediately_and_redacts_key(self):
        calls = []
        def transport(*args):
            calls.append(args)
            raise HTTPFailure(401, "bad test-secret")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            other = dict(self.task, task_id=1)
            status = execute(folder, [self.task, other], [self.row, self.row], self.schema,
                             CONFIG, "test-secret", transport=transport)
            self.assertEqual(status, 2)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("test-secret", (folder / "attempts.jsonl").read_text(encoding="utf-8"))

    def test_interrupt_preserves_completed_task(self):
        calls = []
        def transport(*args):
            calls.append(args)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return 200, response(self.contract)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(KeyboardInterrupt):
                execute(folder, [self.task, dict(self.task, task_id=1)], [self.row, self.row],
                        self.schema, CONFIG, "test-secret", transport=transport)
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["validated_tasks"], 1)
            self.assertEqual(report["missing_task_ids"], [1])

    def test_bad_journal_is_not_silently_discarded(self):
        from extract_semantic_requirements import journal_read
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.jsonl"
            path.write_text('{"task_id":', encoding="utf-8")
            with self.assertRaises(ValueError):
                journal_read(path)

    def test_reuse_does_not_call_api_or_double_count_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            donor, target = Path(directory) / "pilot", Path(directory) / "full"
            donor.mkdir()
            target.mkdir()
            tasks, _, schema = prepare(donor, "50", CONFIG)
            full, _, _ = prepare(target, "1000", CONFIG)
            task = tasks[0]
            contract = dict(self.contract, task_id=task["task_id"], query=task["query"])
            for dimension in ("category", "budget", "functions", "specifications", "brand", "model"):
                contract[dimension] = []
            raw = response(contract)
            record = {"task_id": task["task_id"], "http_status": 200, "accepted": True, "raw_response": raw}
            (donor / "attempts.jsonl").write_text(dump(record) + "\n", encoding="utf-8")
            reuse(target, donor, full, schema, CONFIG)
            reuse(target, donor, full, schema, CONFIG)
            from extract_semantic_requirements import journal_read
            _, report = summarize(target, full, schema, journal_read(target / "attempts.jsonl"))
            self.assertEqual(report["validated_tasks"], 1)
            self.assertEqual(report["reused_tasks"], 1)
            self.assertEqual(report["usage_all_attempts"], {})
            with self.assertRaisesRegex(ValueError, "identical"):
                reuse(target, donor, full, schema, dict(CONFIG, model="other"))

    def test_limit_then_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            tasks = [self.task, dict(self.task, task_id=1)]
            rows = [self.row, dict(self.row, task_id=1)]
            def transport(url, key, body, timeout):
                return 200, response(dict(self.contract, task_id=body["messages"][0]["id"]))
            rows = [dict(r, messages=[{"id": t["task_id"]}]) for t, r in zip(tasks, rows)]
            self.assertEqual(execute(folder, tasks, rows, self.schema, CONFIG, "test-secret", limit=1, transport=transport), 2)
            self.assertEqual(execute(folder, tasks, rows, self.schema, CONFIG, "test-secret", transport=transport), 0)

    def test_request_configuration_and_endpoint(self):
        self.assertEqual(endpoint("https://api.deepseek.com/v1/"), "https://api.deepseek.com/v1/chat/completions")
        for url in ["http://example.com", "https://name:secret@example.com", "https://example.com?key=x"]:
            with self.assertRaises(ValueError):
                endpoint(url)
        body = payload(self.row, CONFIG)
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertNotIn("thinking", payload(self.row, dict(CONFIG, thinking="omit")))

    def test_school_http_requires_explicit_flag_and_exact_base(self):
        school = "http://aigw.dlut.edu.cn/v1"
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            endpoint(school)
        self.assertEqual(endpoint(school, allow_school_http=True), school + "/chat/completions")
        self.assertEqual(endpoint(school + "/", allow_school_http=True), school + "/chat/completions")
        for base in ("http://aigw.dlut.edu.cn", "http://aigw.dlut.edu.cn/v2",
                     "http://aigw.dlut.edu.cn/v1/chat/completions", "http://aigw.dlut.edu.cn:80/v1",
                     "http://aigw.dlut.edu.cn.evil.example/v1", "http://evil.example/v1",
                     "http://aigw.dlut.edu.cn/v1/../v2"):
            with self.subTest(base=base), self.assertRaises(ValueError):
                endpoint(base, allow_school_http=True)

    def test_guided_profile_preserves_query_and_adds_programmed_indexes(self):
        tasks, requests, files, provenance = snapshot("50", "guided-v2")
        self.assertEqual(provenance["profile"], "guided-v2")
        self.assertEqual(files["output.schema.json"], (TARGET / "output.schema.json").read_bytes())
        for task, request in zip(tasks, requests):
            user = request["messages"][1]["content"]
            lines = user.splitlines()
            self.assertEqual(json.loads(lines[1]), task)
            indexes = json.loads(lines[-1])
            self.assertEqual(indexes, [[i, c] for i, c in enumerate(task["query"])])
        self.assertIn("packaging", requests[0]["messages"][0]["content"])
        self.assertIn("170", requests[0]["messages"][0]["content"])

    def test_guided_retry_records_actual_feedback_without_changing_raw_response(self):
        calls = []
        invalid = response({})
        def transport(url, key, body, timeout):
            calls.append(copy.deepcopy(body))
            return 200, invalid if len(calls) == 1 else response(self.contract)
        config = dict(CONFIG, extraction_profile="guided-v2")
        with tempfile.TemporaryDirectory() as directory, patch("extract_semantic_requirements.time.sleep"):
            folder = Path(directory)
            self.assertEqual(execute(folder, [self.task], [self.row], self.schema,
                                     config, "test-secret", transport=transport), 0)
            self.assertEqual(calls[0]["messages"], [])
            self.assertIn("schema_version", calls[1]["messages"][-1]["content"])
            self.assertEqual(self.row["messages"], [])
            from extract_semantic_requirements import journal_read
            records = journal_read(folder / "attempts.jsonl")
            self.assertEqual(records[0]["raw_response"], invalid)
            self.assertEqual(records[1]["request_messages"], calls[1]["messages"])

    def test_query_template_tokens_are_not_replaced_inside_original_query(self):
        from extract_semantic_requirements import render_user
        task = {"task_id": 1, "query": "原文{character_index}和{task_json}"}
        rendered = render_user("{task_json}\n{character_index}", task)
        self.assertEqual(json.loads(rendered.splitlines()[0]), task)

    def test_local_preparation_allows_http_but_never_unsafe_url_fields(self):
        self.assertEqual(endpoint("http://example.com/v1", local_only=True),
                         "http://example.com/v1/chat/completions")
        for base in ("http://user:password@aigw.dlut.edu.cn/v1", "http://@aigw.dlut.edu.cn/v1",
                     "http://aigw.dlut.edu.cn/v1?key=x", "http://aigw.dlut.edu.cn/v1#fragment",
                     "http://aigw.dlut.edu.cn/v1?", "http://aigw.dlut.edu.cn/v1#",
                     "http://aigw.dlut.edu.cn/v1\n"):
            for local in (True, False):
                with self.subTest(base=base, local=local), self.assertRaises(ValueError):
                    endpoint(base, local_only=local, allow_school_http=True)

    def test_prepare_only_does_not_require_key_or_call_api_for_either_mode(self):
        from extract_semantic_requirements import main
        for mode in ("50", "1000"):
            with tempfile.TemporaryDirectory() as directory:
                argv = ["extract", "--mode", mode, "--output", directory, "--prepare-only",
                        "--base-url", "http://aigw.dlut.edu.cn/v1", "--model", "school-model"]
                with patch.object(sys, "argv", argv), patch.dict(os.environ, {}, clear=True), \
                        patch("extract_semantic_requirements.prepare", return_value=([], [], self.schema)) as prepared, \
                        patch("extract_semantic_requirements.execute") as executed, \
                        patch("urllib.request.build_opener") as network:
                    self.assertEqual(main(), 0)
                    self.assertEqual(prepared.call_args.args[2]["endpoint"],
                                     "http://aigw.dlut.edu.cn/v1/chat/completions")
                    executed.assert_not_called()
                    network.assert_not_called()

    def test_school_actual_call_flag_checked_before_any_request(self):
        from extract_semantic_requirements import main
        with tempfile.TemporaryDirectory() as directory:
            argv = ["extract", "--mode", "50", "--output", directory, "--thinking", "omit",
                    "--base-url", "http://aigw.dlut.edu.cn/v1", "--model", "school-model"]
            with patch.dict(os.environ, {"LLM_API_KEY": "test-secret"}), \
                    patch("extract_semantic_requirements.prepare", return_value=([], [], self.schema)), \
                    patch("extract_semantic_requirements.execute", return_value=0) as executed:
                with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "allow-school-http"):
                    main()
                executed.assert_not_called()
                with patch.object(sys, "argv", argv + ["--allow-school-http"]):
                    self.assertEqual(main(), 0)
                self.assertEqual(executed.call_args.args[4]["endpoint"],
                                 "http://aigw.dlut.edu.cn/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
