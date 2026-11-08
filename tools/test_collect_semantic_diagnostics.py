import copy
import functools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import collect_semantic_diagnostics as collector
from extract_semantic_requirements import HTTPFailure, dump, execute, journal_read, prepare
from semantic_contract_v1 import example, schema

CONFIG = {"endpoint": "https://example.com/v1/chat/completions", "model": "test-model",
          "thinking": "omit", "response_format": "json_object", "max_tokens": 8192, "temperature": 0}


def response(tid, valid=True):
    c = example()
    c["task_id"] = tid
    return dump({"choices": [{"finish_reason": "stop", "message": {"content": dump(c) if valid else "not-json"}}],
                 "usage": {"total_tokens": 7}})


def setup(output, count=5, seed=None, attempts=1):
    seed = seed or []
    (output / "seed_attempts.jsonl").write_text("".join(dump(r) + "\n" for r in seed), encoding="utf-8")
    tasks = [{"task_id": i, "query": example()["query"]} for i in range(count)]
    requests = [{"custom_id": str(i), "messages": [{"role": "user", "content": str(i)}]} for i in range(count)]
    manifest = {"api": CONFIG, "controls": {"attempts_per_new_task": attempts, "timeout": 1}, "seed_attempts": len(seed)}
    return tasks, requests, manifest


class CollectionTest(unittest.TestCase):
    def test_validation_failures_do_not_stop_collection_and_do_not_pass_validation(self):
        calls = []
        def transport(url, key, body, timeout):
            tid = int(body["messages"][0]["content"])
            calls.append(tid)
            return 200, response(tid, tid == 4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            tasks, requests, manifest = setup(output)
            status = collector.collect(output, tasks, requests, schema(), manifest, "test-key",
                                       runner=functools.partial(execute, transport=transport))
            self.assertEqual(status, 0)
            self.assertEqual(calls, list(range(5)))
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["collection_complete"])
            self.assertFalse(report["complete"])
            self.assertEqual(report["validated_tasks"], 1)
            self.assertEqual(report["collected_validation_failed_task_ids"], [0, 1, 2, 3])
            self.assertEqual(len(journal_read(output / "collected_failures.jsonl")), 4)
            with patch.object(collector, "execute", side_effect=AssertionError("No repeat calls")) as unused:
                collector.collect(output, tasks, requests, schema(), manifest, "test-key", runner=unused)
                unused.assert_not_called()

    def test_seed_success_and_failure_are_both_skipped_without_rebilling(self):
        seed = [{"task_id": i, "http_status": 200, "accepted": bool(i), "raw_response": response(i, bool(i))} for i in range(2)]
        calls = []
        def transport(url, key, body, timeout):
            tid = int(body["messages"][0]["content"])
            calls.append(tid)
            return 200, response(tid)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            tasks, requests, manifest = setup(output, 3, seed)
            before = (output / "seed_attempts.jsonl").read_bytes()
            collector.collect(output, tasks, requests, schema(), manifest, "test-key", runner=functools.partial(execute, transport=transport))
            self.assertEqual(calls, [2])
            self.assertEqual((output / "seed_attempts.jsonl").read_bytes(), before)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["seed_api_attempts"], 2)
            self.assertEqual(report["new_api_attempts"], 1)
            self.assertEqual(report["usage_all_attempts"]["total_tokens"], 21)

    def test_transport_circuit_breaker_and_resume(self):
        calls = []
        def unavailable(*args):
            calls.append(1)
            raise TimeoutError("unavailable")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            tasks, requests, manifest = setup(output)
            self.assertEqual(collector.collect(output, tasks, requests, schema(), manifest, "test-key", runner=functools.partial(execute, transport=unavailable)), 2)
            self.assertEqual(len(calls), 3)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["collected_tasks"], 0)
            self.assertIn("transport", report["stop_reason"])
            def available(url, key, body, timeout):
                tid = int(body["messages"][0]["content"])
                return 200, response(tid)
            self.assertEqual(collector.collect(output, tasks, requests, schema(), manifest, "test-key", runner=functools.partial(execute, transport=available)), 0)
            self.assertEqual(len(journal_read(output / "attempts.jsonl")), 8)

    def test_authentication_stops_even_after_a_model_validation_failure(self):
        calls = []
        def transport(*args):
            calls.append(1)
            if len(calls) == 1:
                return 200, response(0, False)
            raise HTTPFailure(401, "expired test-key")
        with tempfile.TemporaryDirectory() as directory, patch("extract_semantic_requirements.time.sleep"):
            output = Path(directory)
            tasks, requests, manifest = setup(output, attempts=2)
            self.assertEqual(collector.collect(output, tasks, requests, schema(), manifest, "test-key", runner=functools.partial(execute, transport=transport)), 2)
            self.assertEqual(len(calls), 2)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertIn("401", report["stop_reason"])
            self.assertNotIn("test-key", (output / "attempts.jsonl").read_text(encoding="utf-8"))

    def test_interruption_retains_completed_task_and_releases_no_evidence(self):
        def transport(url, key, body, timeout):
            tid = int(body["messages"][0]["content"])
            if tid:
                raise KeyboardInterrupt()
            return 200, response(tid)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            tasks, requests, manifest = setup(output, 2)
            with self.assertRaises(KeyboardInterrupt):
                collector.collect(output, tasks, requests, schema(), manifest, "test-key", runner=functools.partial(execute, transport=transport))
            self.assertEqual(len(journal_read(output / "attempts.jsonl")), 1)
            self.assertEqual(json.loads((output / "report.json").read_text(encoding="utf-8"))["collected_tasks"], 1)

    def test_prepare_only_without_key_or_http_exception_calls_no_api(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "out"
            source.mkdir()
            config = dict(CONFIG, endpoint="http://aigw.dlut.edu.cn/v1/chat/completions")
            prepare(source, "50", config)
            # A new prepare-only snapshot has no source journal at all.
            self.assertFalse((source / "attempts.jsonl").exists())
            original = (source / "run_manifest.json").read_bytes()
            argv = ["collect", "--source-run", str(source), "--output", str(output), "--prepare-only"]
            with patch.object(sys, "argv", argv), patch.dict(os.environ, {}, clear=True), patch("urllib.request.build_opener") as network:
                self.assertEqual(collector.main(), 0)
                network.assert_not_called()
            self.assertEqual((output / "source_run_manifest.json").read_bytes(), original)
            self.assertEqual((source / "run_manifest.json").read_bytes(), original)
            self.assertFalse((output / ".run.lock").exists())
            with patch.object(sys, "argv", argv[:-1]), self.assertRaisesRegex(ValueError, "allow-school-http"):
                collector.main()

    def test_changed_protocol_or_controls_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "out"
            source.mkdir()
            output.mkdir()
            prepare(source, "50", CONFIG)
            (source / "attempts.jsonl").write_bytes(b"")
            collector.prepare_collection(source, output)
            with self.assertRaisesRegex(ValueError, "configuration"):
                collector.prepare_collection(source, output, attempts=1)
            with (source / "system_prompt.txt").open("ab") as f:
                f.write(b"changed")
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                collector.prepare_collection(source, output)

    def test_model_envelope_not_just_http_200(self):
        self.assertFalse(collector.model_reply({"http_status": 200, "raw_response": '{"error":"quota"}'}))
        self.assertFalse(collector.model_reply({"http_status": 500, "raw_response": response(0)}))
        self.assertTrue(collector.model_reply({"http_status": 200, "raw_response": response(0, False)}))


if __name__ == "__main__":
    unittest.main()
