"""Ensure preflight refuses unsafe sessions before touching CUDA/services."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import check_bounded_grpo_session as checker
from prepare_bounded_grpo_resume import RUN, write

ROOT = Path(__file__).resolve().parents[1]


class PreflightGuards(unittest.TestCase):
    def fixture(self, project):
        prep = project / RUN / "preparation/formal_resume600_to650_v1"
        prep.mkdir(parents=True)
        spec = {"resume_step": 600, "total_updates": 650, "project": str(project),
                "source_sha256": {}, "env": {}, "required_free_bytes": 10**30}
        write(prep / "launch_spec.json", spec)
        return prep, spec

    def test_wrong_bounds_rejected_before_gpu(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td)
            prep, spec = self.fixture(project)
            spec["total_updates"] = 700
            write(prep / "launch_spec.json", spec)
            with patch.object(checker.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "bounds"):
                    checker.check(project, 600)
                run.assert_not_called()

    def test_existing_training_rejected_before_gpu(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td)
            prep, _ = self.fixture(project)
            (prep / "train.pid").write_text("123", encoding="utf-8")
            with patch.object(checker.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "training records"):
                    checker.check(project, 600)
                run.assert_not_called()

    def test_low_disk_clears_old_passes_without_starting_services(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td)
            prep, _ = self.fixture(project)
            for name in ("resource_check.json", "runtime_preflight.json", "environment_preflight.json"):
                write(prep / name, {"passed": True})
            with patch.object(checker.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "disk budget"):
                    checker.check(project, 600)
                run.assert_not_called()
            self.assertFalse((prep / "environment_preflight.json").exists())
            self.assertFalse((prep / "environment_process.json").exists())

    def test_650_to800_routes_to_new_preparation_before_gpu(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            prep = project / RUN / "preparation/formal_resume650_to800_v1"
            prep.mkdir(parents=True)
            write(prep / "launch_spec.json", {"resume_step": 650, "total_updates": 800,
                "project": str(project), "source_sha256": {}, "env": {}, "required_free_bytes": 10**30})
            with patch.object(checker.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "disk budget"):
                    checker.check(project, 650, 800)
                run.assert_not_called()

    def test_unsupported_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            checker.check(Path("/unused"), 650, 650)


if __name__ == "__main__":
    unittest.main()
