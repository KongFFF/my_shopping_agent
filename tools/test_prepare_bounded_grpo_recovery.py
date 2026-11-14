"""CPU-only recovery boundaries, with no real training or tensor loading."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import prepare_bounded_grpo_recovery as recovery
import prepare_bounded_grpo_resume as base
import test_prepare_bounded_grpo_resume as fixtures

ROOT = fixtures.ROOT


class RecoveryTests(unittest.TestCase):
    def fixture(self, project):
        helper = fixtures.ResumeTests()
        helper.setUp()
        old_parent, old_ckpt, _ = helper.fixture(project)
        parent = old_parent.parent / "formal_resume500_to650_v1"
        parent.mkdir()
        checkpoint = project / base.RUN / "formal_resume500_to650_v1/global_step_600"
        import shutil
        shutil.copytree(old_ckpt, checkpoint)
        before, _, env, _ = base.derive_config(helper.before, {
            "project": str(project), "checkpoint": str(old_ckpt),
        })
        # Compose against a complete fixture config; the production run uses
        # the actual already-verified isolated source and installed veRL.
        config = OmegaConf.to_yaml(before, resolve=True)
        (parent / "composed_config.yaml").write_text(config, encoding="utf-8")
        (project / "configs/grpo.yaml").write_text(config, encoding="utf-8")
        spec = base.read(old_parent / "launch_spec.json")
        spec.update(output=str(checkpoint.parent), env=dict(spec["env"], **env), resume_step=500, total_updates=650)
        base.write(parent / "launch_spec.json", spec)
        for name in ("launch_formal.py", "math_probe.json"):
            shutil.copy2(old_parent / name, parent / name)
        (parent / "auto_shutdown.py").write_text(
            'import subprocess\nshutdown="/usr/bin/shutdown"\n'
            'callback=lambda: subprocess.run([shutdown], check=True)\n', encoding="utf-8")
        with (checkpoint.parent / "training_diagnostics.jsonl").open("w") as stream:
            import json
            for step in range(501, 624):
                stream.write(json.dumps({"event": "optimizer_step", "global_step": step,
                                        "metrics": {"training/optimizer_updated": 1}}) + "\n")
        return parent, checkpoint, OmegaConf.to_container(before, resolve=True)

    def test_recovery_preserves_optimizer_settings_and_total_bound(self):
        helper = fixtures.ResumeTests()
        helper.setUp()
        before, _, _, _ = base.derive_config(helper.before, helper.report)
        original = OmegaConf.to_container(before, resolve=True)
        cfg, changes, env, changed = recovery.derive(original, Path("/step600"), Path("/new"))
        self.assertEqual(cfg.trainer.total_training_steps, 650)
        self.assertEqual(cfg.actor_rollout_ref.actor.optim, before.actor_rollout_ref.actor.optim)
        self.assertEqual(cfg.trainer.resume_from_path, str(Path("/step600")))
        self.assertEqual(cfg.ray_kwargs.ray_init.runtime_env.env_vars.GRPO_OUTPUT_DIR, env["GRPO_OUTPUT_DIR"])
        self.assertLessEqual(set(changed), set(changes))

    def test_complete_recovery_staging_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            parent, _, _ = self.fixture(project)
            original_log = (Path(base.read(parent / "launch_spec.json")["output"]) / "training_diagnostics.jsonl")
            original_hash = base.digest(original_log)
            with patch.dict("os.environ", {}, clear=False):
                recovery.prepare(project)
            prep = parent.parent / "formal_resume600_to650_v1"
            record = base.read(prep / "launch_spec.json")
            receipt = base.read(prep / "resume_receipt.json")
            self.assertEqual(record["remaining_updates"], 50)
            self.assertEqual(record["checkpoint_steps"], [650])
            self.assertEqual(receipt["discarded_parent_steps"], list(range(601, 624)))
            self.assertEqual(original_hash, base.digest(original_log))
            self.assertFalse(Path(record["output"]).exists())
            self.assertFalse((prep / "resource_check.json").exists())
            self.assertFalse((prep / "train.pid").exists())
            self.assertIn('["/bin/bash", "-c", shutdown]', (prep / "auto_shutdown.py").read_text())
            self.assertIn('subprocess.run([shutdown]', (parent / "auto_shutdown.py").read_text())
            with self.assertRaisesRegex(ValueError, "already exists"):
                recovery.prepare(project)

    def test_missing_data_state_blocks_recovery(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            _, checkpoint, _ = self.fixture(project)
            (checkpoint / "data.pt").unlink()
            with self.assertRaisesRegex(ValueError, "Missing recovery checkpoint"):
                recovery.prepare(project)

    def test_source_drift_blocks_recovery(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            self.fixture(project)
            (project / "source.py").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source/package changed"):
                recovery.prepare(project)

    def test_insufficient_disk_blocks_recovery(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as td:
            project = Path(td).resolve()
            self.fixture(project)
            from collections import namedtuple
            usage = namedtuple("Usage", "total used free")(100, 99, 1)
            with patch.object(recovery.shutil, "disk_usage", return_value=usage):
                with self.assertRaisesRegex(ValueError, "Insufficient free disk"):
                    recovery.prepare(project)


if __name__ == "__main__":
    unittest.main()
