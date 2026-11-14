"""CPU-only tests of continuation boundaries; no Torch, models or services."""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch
import zipfile

from omegaconf import OmegaConf

import prepare_bounded_grpo_resume as resume

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "analysis/grpo_baseline_20261022_202816/configs/formal_resume350_v1.yaml"


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.before = OmegaConf.to_container(OmegaConf.load(HISTORY), resolve=True)
        self.report = {"project": "/example/project", "checkpoint": "/example/step500"}

    def test_configuration_changes_are_bounded(self):
        original = copy.deepcopy(self.before)
        cfg, changed, env, allowed = resume.derive_config(self.before, self.report)
        self.assertEqual(self.before, original)
        self.assertEqual(cfg.trainer.total_training_steps, 650)
        self.assertEqual(cfg.trainer.resume_from_path, self.report["checkpoint"])
        self.assertFalse(cfg.trainer.del_local_ckpt_after_load)
        self.assertFalse(cfg.trainer.val_before_train)
        self.assertEqual(cfg.actor_rollout_ref.actor.optim, OmegaConf.create(original["actor_rollout_ref"]["actor"]["optim"]))
        self.assertLessEqual(set(changed), set(allowed))
        self.assertEqual(cfg.ray_kwargs.ray_init.runtime_env.env_vars.GRPO_OUTPUT_DIR, env["GRPO_OUTPUT_DIR"])
        self.assertEqual(cfg.ray_kwargs.ray_init.runtime_env.env_vars.SHOPPING_GRPO_DIAGNOSTICS_PATH,
                         env["SHOPPING_GRPO_DIAGNOSTICS_PATH"])

    def test_nonconstant_scheduler_rejected(self):
        self.before["actor_rollout_ref"]["actor"]["optim"]["lr_scheduler_type"] = "cosine"
        with self.assertRaisesRegex(ValueError, "lr_scheduler_type"):
            resume.derive_config(self.before, self.report)

    def test_missing_optimizer_restore_rejected(self):
        self.before["actor_rollout_ref"]["actor"]["checkpoint"]["load_contents"] = ["model"]
        with self.assertRaisesRegex(ValueError, "load_contents"):
            resume.derive_config(self.before, self.report)

    def test_duplicate_overrides_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicated"):
            resume.replace_overrides(["trainer.total_training_steps=500", "++trainer.total_training_steps=500"],
                                     {"trainer.total_training_steps": 650})

    def test_shutdown_monitor_fix_only_changes_invocation_and_is_idempotent(self):
        source = 'import subprocess\nshutdown="/usr/bin/shutdown"\ncallback=lambda: subprocess.run([shutdown], check=True)\n'
        fixed = resume.compatible_shutdown_monitor(source)
        self.assertEqual(fixed, source.replace('subprocess.run([shutdown], check=True)',
            'subprocess.run(["/bin/bash", "-c", shutdown], check=True)'))
        self.assertEqual(resume.compatible_shutdown_monitor(fixed), fixed)
        with self.assertRaisesRegex(ValueError, "Unknown shutdown"):
            resume.compatible_shutdown_monitor("# unknown monitor\n")

    def fixture(self, project):
        parent = project / resume.RUN / "preparation/formal_resume350_v1"
        parent.mkdir(parents=True)
        checkpoint = project / resume.RUN / "formal_resume350_v1/global_step_500"
        for name in resume.REQUIRED:
            path = checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".pt":
                with zipfile.ZipFile(path, "w") as f:
                    f.writestr("archive/data.pkl", b"fixture; do not deserialize")
            else:
                path.write_text("{}", encoding="utf-8")
        source = project / "source.py"
        source.write_text("# frozen original engine\n", encoding="utf-8")
        data = project / "train.parquet"
        data.write_bytes(b"frozen data")
        model = project / "sft"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        resume.write(parent.parent / "baseline_plan.json", {"data_sha256": {"train.parquet": resume.digest(data)},
                     "model": str(model), "model_sha256": {"config.json": resume.digest(model / "config.json")}})
        spec = {"project": str(project), "output": str(checkpoint.parent),
                "source_sha256": {str(source): resume.digest(source)},
                "command": ["python", "train.py", "--output", "old", "--experiment-name", "old", "--"],
                "preflight": ["python", "preflight.py"], "env": {"GRPO_CONFIG_NAME": "grpo"}}
        resume.write(parent / "launch_spec.json", spec)
        resume.write(parent / "exit_status.json", {"exit_code": 0})
        resume.write(parent / "math_probe.json", {"passed": True, "engine_file": str(source),
                     "engine_sha256": resume.digest(source)})
        (parent / "launch_formal.py").write_text("from pathlib import Path\nprep = Path(__file__).resolve().parent\n", encoding="utf-8")
        config = OmegaConf.create(self.before)
        (project / "configs").mkdir()
        for path in (parent / "composed_config.yaml", project / "configs/grpo.yaml"):
            path.write_text(OmegaConf.to_yaml(config), encoding="utf-8")
        return parent, checkpoint, source

    def test_audit_and_prepare_without_launching_or_overwriting(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            parent, checkpoint, source = self.fixture(project)
            report = resume.audit(project)
            self.assertTrue(report["passed"], report["issues"])
            with patch.dict("os.environ", {}, clear=False):
                resume.prepare(report)
            prep = parent.parent / "formal_resume500_to650_v1"
            spec = resume.read(prep / "launch_spec.json")
            self.assertEqual(spec["checkpoint_steps"], [550, 600, 650])
            self.assertIn("++trainer.total_training_steps=650", spec["command"])
            self.assertTrue(checkpoint.is_dir())
            self.assertFalse(Path(spec["output"]).exists())
            self.assertFalse((prep / "train.pid").exists())
            self.assertFalse((prep / "exit_status.json").exists())
            with self.assertRaisesRegex(ValueError, "already exists"):
                resume.prepare(report)

    def test_source_drift_blocks_preparation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            parent, _, source = self.fixture(project)
            source.write_text("# semantic experiment instead\n", encoding="utf-8")
            report = resume.audit(project)
            self.assertFalse(report["passed"])
            self.assertTrue(any(item["kind"] == "source" for item in report["issues"]))
            with self.assertRaisesRegex(ValueError, "Audit failed"):
                resume.prepare(report)
            self.assertFalse((parent.parent / "formal_resume500_to650_v1").exists())

    def test_missing_optimizer_blocks_preparation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            _, checkpoint, _ = self.fixture(project)
            (checkpoint / "actor/optim_world_size_1_rank_0.pt").unlink()
            report = resume.audit(project)
            self.assertFalse(report["passed"])
            self.assertTrue(any(item["kind"] == "checkpoint" for item in report["issues"]))

    def test_isolated_paths_preserve_training_and_asset_bindings(self):
        report = dict(self.report, runtime_project="/isolated/runtime")
        cfg, changed, env, allowed = resume.derive_config(self.before, report)
        self.assertEqual(cfg.actor_rollout_ref.model.path, self.before["actor_rollout_ref"]["model"]["path"])
        self.assertEqual(cfg.data.train_files, self.before["data"]["train_files"])
        self.assertEqual(cfg.actor_rollout_ref.actor.optim.lr, 1e-6)
        self.assertEqual(env["PYTHONPATH"], str(Path("/isolated/runtime/src")))
        self.assertEqual(cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path,
                         str(Path("/isolated/runtime/configs/agent_loop.yaml")))
        self.assertEqual(cfg.ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH, env["PYTHONPATH"])
        self.assertLessEqual(set(changed), set(allowed))

    def test_isolation_cannot_waive_missing_optimizer_or_unknown_source(self):
        for kind, name, detail in [
            ("checkpoint", "actor/optim.pt", "missing"),
            ("source", "src/unknown.py", {"expected": "bad", "actual": "changed"}),
            ("source", "configs/grpo.yaml", "missing"),
        ]:
            report = dict(self.report, issues=[{"kind": kind,
                          "path": str(Path(self.report["project"]) / name), "detail": detail}])
            with self.assertRaisesRegex(ValueError, "cannot resolve"):
                resume.isolate_runtime(report)

    def test_real_git_rebuild_matches_all_five_cloud_expected_hashes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            stage = Path(td) / "code"
            stage.mkdir()
            resume.export_pinned_code(ROOT / "upstream/shopping-grpo-longhorizon", stage, Path(td) / "code.tar")
            self.assertEqual({n: resume.digest(stage / n) for n in resume.PINNED_REPAIRS}, resume.PINNED_REPAIRS)
            self.assertFalse((stage / "data").exists())
            self.assertFalse((stage / ".venv").exists())

    def test_full_isolation_keeps_original_files_and_reuses_verified_copy(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            parent, _, _ = self.fixture(project)
            for name in (".venv", "data", "environments"):
                (project / name).mkdir()
            # Confirm symlink support before doing the integration build.
            probe = project / "link_probe"
            try:
                probe.symlink_to(project / "data", target_is_directory=True)
            except OSError:
                self.skipTest("This Windows account cannot create directory symlinks; run this case on Linux")
            probe.unlink()
            spec = resume.read(parent / "launch_spec.json")
            for name, expected in resume.PINNED_REPAIRS.items():
                path = project / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# later experiment\n", encoding="utf-8")
                spec["source_sha256"][str(path)] = expected
            resume.write(parent / "launch_spec.json", spec)
            original_hashes = {name: resume.digest(project / name) for name in resume.PINNED_REPAIRS}
            report = resume.audit(project)
            self.assertEqual(len(report["issues"]), 5)
            run = subprocess.run
            def use_real_git(command, **kwargs):
                if command[:2] == ["git", "-C"]:
                    command = list(command)
                    command[2] = str(ROOT / "upstream/shopping-grpo-longhorizon")
                return run(command, **kwargs)
            with patch.object(resume.subprocess, "run", side_effect=use_real_git):
                effective = resume.isolate_runtime(report)
            self.assertTrue(effective["passed"])
            self.assertEqual(original_hashes, {name: resume.digest(project / name) for name in resume.PINNED_REPAIRS})
            self.assertFalse((project / resume.RUN / "formal_resume500_to650_v1").exists())
            reused = resume.isolate_runtime(report)
            self.assertEqual(reused["runtime_project"], effective["runtime_project"])
            isolated = Path(effective["runtime_project"])
            (isolated / "configs/grpo.yaml").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file hashes differ"):
                resume.isolate_runtime(report)


if __name__ == "__main__":
    unittest.main()
