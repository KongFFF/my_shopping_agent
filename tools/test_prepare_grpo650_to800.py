"""CPU-only checks of the650->800 recipe and retained-file requirements."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import prepare_grpo650_to800 as tool
import prepare_bounded_grpo_resume as base
import prepare_bounded_grpo_recovery as recovery
import test_prepare_bounded_grpo_recovery as fixtures

ROOT = fixtures.ROOT


class ContinuationTests(unittest.TestCase):
    def fixture(self, project):
        old_parent, old_checkpoint, before = fixtures.RecoveryTests().fixture(project)
        run = project / base.RUN
        parent = run / "preparation/formal_resume600_to650_v1"
        parent.mkdir()
        previous_out = run / "formal_resume600_to650_v1"
        checkpoint = previous_out / "global_step_650"
        shutil.copytree(old_checkpoint, checkpoint)
        base.write(checkpoint / "actor/fsdp_config.json", {"world_size": 1})
        base.write(checkpoint / "actor/lora_train_meta.json", {"r": 16, "lora_alpha": 32})
        cfg, _, env, _ = recovery.derive(before, old_checkpoint, previous_out)
        for path in (parent / "composed_config.yaml", project / "configs/grpo.yaml"):
            path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
        spec = base.read(old_parent / "launch_spec.json")
        spec.update(output=str(previous_out), resume_step=600, total_updates=650,
                    env=dict(spec["env"], **env))
        base.write(parent / "launch_spec.json", spec)
        base.write(parent / "exit_status.json", {"exit_code": 0})
        for name in ("launch_formal.py", "math_probe.json", "auto_shutdown.py"):
            shutil.copy2(old_parent / name, parent / name)
        import json
        with (previous_out / "training_diagnostics.jsonl").open("w") as stream:
            for step in range(601, 651):
                stream.write(json.dumps({"event": "optimizer_step", "global_step": step,
                                        "metrics": {"training/optimizer_updated": 1}}) + "\n")
        return parent, checkpoint, OmegaConf.to_container(cfg, resolve=True)

    def test_only_bounds_and_record_bindings_change(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            _, checkpoint, before = self.fixture(project)
            cfg, changes, env, changed = tool.derive(before, checkpoint, project / "new")
            self.assertEqual(cfg.trainer.total_training_steps, 800)
            self.assertEqual(cfg.actor_rollout_ref.actor.optim, OmegaConf.create(before["actor_rollout_ref"]["actor"]["optim"]))
            self.assertEqual(cfg.data.train_files, before["data"]["train_files"])
            self.assertEqual(cfg.data.seed, before["data"]["seed"])
            self.assertEqual(cfg.ray_kwargs.ray_init.runtime_env.env_vars.GRPO_OUTPUT_DIR, env["GRPO_OUTPUT_DIR"])
            self.assertLessEqual(set(changed), set(changes))

    def test_staging_after_old_checkpoints_are_archived(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            parent, _, _ = self.fixture(project)
            # Move only fixture directories within this verified temporary
            # workspace, simulating completed cloud archival of old weights.
            for old in (project / base.RUN / "formal_resume350_v1/global_step_500",
                        project / base.RUN / "formal_resume500_to650_v1/global_step_600"):
                assert old.resolve().is_relative_to(project)
                old.rename(project / ("archived_" + old.name))
            old_spec_hash = base.digest(parent / "launch_spec.json")
            with patch.dict("os.environ", {}, clear=False):
                tool.prepare(project)
            prep = parent.parent / "formal_resume650_to800_v1"
            spec = base.read(prep / "launch_spec.json")
            self.assertEqual(spec["checkpoint_steps"], [700, 750, 800])
            self.assertEqual(spec["remaining_updates"], 150)
            self.assertTrue(spec["resume_from"].endswith("global_step_650"))
            self.assertEqual(old_spec_hash, base.digest(parent / "launch_spec.json"))
            self.assertIn('["/bin/bash", "-c", shutdown]', (prep / "auto_shutdown.py").read_text())
            self.assertFalse((prep / "resource_check.json").exists())
            self.assertFalse(Path(spec["output"]).exists())
            with self.assertRaisesRegex(ValueError, "already exists"):
                tool.prepare(project)

    def test_missing_optimizer_blocks_preparation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            _, checkpoint, _ = self.fixture(project)
            (checkpoint / "actor/optim_world_size_1_rank_0.pt").unlink()
            with self.assertRaisesRegex(ValueError, "checkpoint file missing"):
                tool.prepare(project)

    def test_space_below_budget_blocks_preparation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            self.fixture(project)
            from collections import namedtuple
            usage = namedtuple("Usage", "total used free")(100, 99, 1)
            with patch.object(tool.shutil, "disk_usage", return_value=usage):
                with self.assertRaisesRegex(ValueError, "GiB free"):
                    tool.prepare(project)

    def test_runtime_drift_blocks_preparation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            self.fixture(project)
            (project / "source.py").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source missing or changed"):
                tool.prepare(project)


if __name__ == "__main__":
    unittest.main()
