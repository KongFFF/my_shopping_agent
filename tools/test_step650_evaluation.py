"""CPU-light checks for frozen evaluation protocols and isolated export prep."""
import copy
from pathlib import Path
import tempfile
import runpy
import sys
import unittest
from unittest.mock import patch

import step650_evaluation as tool
from prepare_bounded_grpo_resume import RUN, digest, read, write

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def fixture(self, project):
        run = project / RUN
        runtime = project / "isolated"
        source = project / "scripts/evaluate_shop_benchmark.py"
        isolated_source = runtime / "scripts/evaluate_shop_benchmark.py"
        for path in (source, isolated_source):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# identical frozen evaluation\n", encoding="utf-8")
        reference_dir = run / "preparation/posttrain_step500_v1"
        reference_dir.mkdir(parents=True)
        old_output = run / "evaluation/step500_independent_v1"
        (old_output / "final200_clean").mkdir(parents=True)
        write(old_output / "final200_clean/summary.json", {"strict_successes": 137, "expected_tasks": 200})
        reference = {"project": str(project), "evaluation_root": str(old_output),
                     "source_sha256": {str(source): digest(source)}, "commands": {},
                     "benchmarks": {}, "protocols": {}}
        import json
        for suite, size in (("smoke3", 3), ("dev50", 50), ("final200_clean", 200)):
            benchmark = project / f"{suite}.jsonl"
            benchmark.write_text("".join(json.dumps({"task_id": i}) + "\n" for i in range(size)), encoding="utf-8")
            reference["benchmarks"][suite] = {"path": str(benchmark), "sha256": digest(benchmark), "task_ids": list(range(size))}
            reference["commands"][suite] = ["python", str(source), "--benchmark", str(benchmark),
                "--model", "step500", "--output", str(old_output / suite / "trajectories.jsonl"),
                "--summary", str(old_output / suite / "summary.json"), "--temperature", "0", "--max-steps", "35"]
            reference["protocols"][suite] = {"temperature": 0.0, "max_steps": 35}
        write(reference_dir / "evaluation_spec.json", reference)
        (reference_dir / "run_eval.py").write_text(
            "import subprocess\ncallback=lambda: subprocess.run(['/usr/bin/shutdown'],check=True)\n", encoding="utf-8")
        parent = run / "preparation/formal_resume600_to650_v1"
        parent.mkdir()
        write(parent / "exit_status.json", {"exit_code": 0})
        write(parent / "launch_spec.json", {"project": str(runtime), "resume_step": 600, "total_updates": 650,
                                          "source_sha256": {str(isolated_source): digest(isolated_source)}})
        actor = run / "formal_resume600_to650_v1/global_step_650/actor"
        actor.mkdir(parents=True)
        write(actor / "lora_train_meta.json", {"r": 16, "lora_alpha": 32})
        write(actor / "fsdp_config.json", {"world_size": 1})
        return reference, runtime, run

    def test_protocols_tasks_and_nonidentity_cli_options_unchanged(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            reference, runtime, run = self.fixture(project)
            original = copy.deepcopy(reference)
            result = tool.derive_reference(reference, project, runtime, run)
            self.assertEqual(reference, original)
            self.assertEqual(result["protocols"], original["protocols"])
            self.assertEqual(result["benchmarks"], original["benchmarks"])
            for suite, command in result["commands"].items():
                old = original["commands"][suite]
                for option in ("--benchmark", "--temperature", "--max-steps"):
                    self.assertEqual(command[command.index(option)+1], old[old.index(option)+1])
            self.assertIn("global_step_650", result["checkpoint"])

    def test_prepare_does_not_export_or_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            reference, _, run = self.fixture(project)
            with patch.object(tool, "FINAL_HASH", reference["benchmarks"]["final200_clean"]["sha256"]):
                tool.prepare(project)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    tool.prepare(project)
            post = run / "preparation/posttrain_step650_v1"
            spec = read(post / "evaluation_spec.json")
            self.assertFalse(Path(spec["raw_export"]).exists())
            self.assertFalse(Path(spec["evaluation_root"]).exists())
            self.assertIn('["/bin/bash","-c","/usr/bin/shutdown"]', (post / "run_eval.py").read_text())
            self.assertFalse(read(post / "preparation_receipt.json")["evaluation_started"])

    def test_changed_benchmark_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            reference, _, _ = self.fixture(project)
            Path(reference["benchmarks"]["final200_clean"]["path"]).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "benchmark changed"):
                tool.prepare(project)

    def test_changed_isolated_source_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            reference, runtime, run = self.fixture(project)
            (runtime / "scripts/evaluate_shop_benchmark.py").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source differs"):
                tool.derive_reference(reference, project, runtime, run)

    def test_unknown_shutdown_runner_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown frozen runner"):
            tool.patch_runner("# unrelated runner\n")

    def test_background_launch_is_exclusive_and_does_not_load_tensors(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            reference, _, run = self.fixture(project)
            with patch.object(tool, "FINAL_HASH", reference["benchmarks"]["final200_clean"]["sha256"]):
                tool.prepare(project)
            argv = [str(Path(tool.__file__)), "export", "--launch", "--project", str(project)]
            with patch.object(sys, "argv", argv), patch("subprocess.Popen") as popen:
                popen.return_value.pid = 12345
                runpy.run_path(tool.__file__, run_name="__main__")
                self.assertEqual(popen.call_args.kwargs["start_new_session"], True)
                self.assertNotIn("--launch", popen.call_args.args[0])
                with self.assertRaises(FileExistsError):
                    runpy.run_path(tool.__file__, run_name="__main__")
                self.assertEqual(popen.call_count, 1)
            post = run / "preparation/posttrain_step650_v1"
            self.assertEqual(read(post / "export_launcher.json")["pid"], 12345)
            self.assertFalse((post / "export_receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
