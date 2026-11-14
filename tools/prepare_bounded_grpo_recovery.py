"""Stage a separate step600->650 recovery after the confirmed billing shutdown.

No Torch imports, tensor deserialization, services, or training. Preserve the
interrupted run; only its committed prefix through step600 belongs to the
eventual effective learning chain.
"""
import argparse
import json
import math
import numbers
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import zipfile

from prepare_bounded_grpo_resume import (
    DEFAULT_PROJECT, GIB, REQUIRED, RUN, differences, digest, read,
    replace_overrides, write,
)


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value, numbers.Real) else True


def derive(before, checkpoint, out):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(before)
    for key, expected in {
        "trainer.total_training_steps": 650, "trainer.save_freq": 50,
        "trainer.test_freq": 50, "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
    }.items():
        if OmegaConf.select(cfg, key) != expected:
            raise ValueError(f"Parent configuration changed: {key}")
    if set(cfg.actor_rollout_ref.actor.checkpoint.load_contents) != {"model", "optimizer", "extra"}:
        raise ValueError("Full model/optimizer/extra restore is required")
    env = {"GRPO_OUTPUT_DIR": str(out),
           "SHOPPING_GRPO_DIAGNOSTICS_PATH": str(out / "training_diagnostics.jsonl")}
    changes = {
        "trainer.resume_mode": "resume_path", "trainer.resume_from_path": str(checkpoint),
        "trainer.del_local_ckpt_after_load": False, "trainer.val_before_train": False,
        "trainer.default_local_dir": str(out), "trainer.validation_data_dir": str(out / "validation"),
        "trainer.experiment_name": "shopping-grpo-resume600-to650-v1",
        "actor_rollout_ref.rollout.trace.experiment_name": "shopping-grpo-resume600-to650-v1",
    }
    changes.update({"ray_kwargs.ray_init.runtime_env.env_vars." + k: v for k, v in env.items()})
    for key, value in changes.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    changed = differences(before, OmegaConf.to_container(cfg, resolve=True))
    if not set(changed) <= set(changes):
        raise ValueError(f"Unexpected changes: {changed}")
    return cfg, changes, env, changed


def prepare(project):
    from omegaconf import OmegaConf
    project = Path(project).resolve()
    run = project / RUN
    parent = run / "preparation/formal_resume500_to650_v1"
    prep = run / "preparation/formal_resume600_to650_v1"
    out = run / "formal_resume600_to650_v1"
    if prep.exists() or out.exists():
        raise ValueError("Recovery preparation/output already exists; refusing to overwrite or restart")
    old = read(parent / "launch_spec.json")
    previous_out = Path(old["output"])
    if previous_out.resolve() != (run / "formal_resume500_to650_v1").resolve():
        raise ValueError("Unexpected interrupted parent run")
    if old.get("resume_step") != 500 or old.get("total_updates") != 650:
        raise ValueError("Unexpected parent training bounds")
    # Never kill processes or modify the original run. PID reuse after reboot
    # is handled by matching command arguments, rather than PID existence.
    proc = Path("/proc")
    if proc.is_dir():
        for folder in proc.iterdir():
            if not folder.name.isdigit():
                continue
            try:
                args = (folder / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if any(arg in (str(parent / "launch_formal.py").encode(),
                           str(parent / "auto_shutdown.py").encode()) for arg in args):
                raise ValueError(f"Parent launcher/monitor still running: PID={folder.name}")
    steps = []
    with (previous_out / "training_diagnostics.jsonl").open() as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") != "optimizer_step":
                continue
            if not finite(event["metrics"]) or event["metrics"].get("training/optimizer_updated") != 1:
                raise ValueError("Invalid parent optimizer metrics")
            steps.append(int(event["global_step"]))
    if not steps or steps != list(range(501, steps[-1] + 1)) or not 600 <= steps[-1] < 650:
        raise ValueError("Parent update chain is not a continuous incomplete 501..last chain covering 600")
    if (parent / "exit_status.json").exists() and read(parent / "exit_status.json").get("exit_code") == 0:
        raise ValueError("Parent recorded a successful exit; inspect before treating it as an interrupted run")
    print(f"Parent completed through {steps[-1]}; preserve 501..600, replay 601..{steps[-1]}", flush=True)
    for name, expected in old["source_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Historical isolated source/package changed: {name}")
    baseline = read(run / "preparation/baseline_plan.json")
    for name, expected in baseline["data_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError(f"Data changed: {name}")
    for name, expected in baseline["model_sha256"].items():
        if digest(Path(baseline["model"]) / name) != expected:
            raise ValueError(f"SFT base changed: {name}")
    checkpoint = previous_out / "global_step_600"
    for name in REQUIRED:
        path = checkpoint / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing recovery checkpoint file: {path}")
        if path.suffix == ".pt":
            with zipfile.ZipFile(path) as archive:
                if not archive.namelist():
                    raise ValueError(f"Empty checkpoint archive: {path}")
    size = sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
    weights = (checkpoint / REQUIRED[0]).stat().st_size
    training = size + 15 * GIB
    required = training + 2 * weights + 5 * GIB
    free = shutil.disk_usage(project).free
    print(f"Available: {free/GIB:.2f} GiB; training: {training/GIB:.2f} GiB; with one export: {required/GIB:.2f} GiB")
    if free < required:
        raise ValueError("Insufficient free disk for one new checkpoint plus one export")
    before = OmegaConf.to_container(OmegaConf.load(parent / "composed_config.yaml"), resolve=True)
    cfg, changes, env_updates, changed = derive(before, checkpoint, out)
    env = dict(old["env"], **env_updates)
    if env.get("SHOPPING_EVIDENCE_MANIFEST") or env.get("SHOPPING_REWARD_TOOLS_DIR"):
        raise ValueError("Unexpected semantic reward bindings")
    command = list(old["command"])
    pos = command.index("--")
    prefix = command[:pos]
    prefix[0] = sys.executable
    for key, value in (("--output", str(out)), ("--experiment-name", "shopping-grpo-resume600-to650-v1")):
        prefix[prefix.index(key) + 1] = value
    command = prefix + ["--"] + replace_overrides(command[pos+1:], changes)
    preflight = [sys.executable, old["preflight"][1], *replace_overrides(old["preflight"][2:], changes)]
    os.environ.update(env)
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(Path(old["project"]) / "configs")):
        actual = compose(config_name=env.get("GRPO_CONFIG_NAME", "grpo"), overrides=preflight[2:])
    mismatch = differences(OmegaConf.to_container(cfg, resolve=True), OmegaConf.to_container(actual, resolve=True))
    if mismatch:
        raise ValueError(f"Recovery CLI differs from intended frozen configuration: {mismatch}")
    record = dict(old)
    record.update(command=command, preflight=preflight, env=env, output=str(out),
                  required_free_bytes=required, resume_from=str(checkpoint), resume_step=600,
                  total_updates=650, remaining_updates=50, checkpoint_steps=[650],
                  parent_preparation=str(parent), initial_policy="Restore complete GRPO step600 training state")
    receipt = {
        "reason": "User confirmed insufficient balance; subsequently recharged",
        "parent_preparation": str(parent), "parent_last_completed_step": steps[-1],
        "checkpoint": str(checkpoint), "resume_step": 600, "target_step": 650,
        "effective_parent_steps": [501, 600], "new_effective_steps": [601, 650],
        "discarded_parent_steps": list(range(601, steps[-1] + 1)),
        "discarded_cost_is_still_counted": True, "config_differences": changed,
        "parent_spec_sha256": digest(parent / "launch_spec.json"),
        "parent_config_sha256": digest(parent / "composed_config.yaml"),
        "preparation_tool_sha256": digest(__file__), "required_free_bytes": required,
        "training_started": False, "checkpoint_tensor_load_test_pending": True,
    }
    with tempfile.TemporaryDirectory(prefix="recovery600_", dir=parent.parent) as td:
        stage = Path(td) / "prepared"
        stage.mkdir()
        for name in ("launch_formal.py", "math_probe.json", "auto_shutdown.py"):
            shutil.copy2(parent / name, stage / name)
        for name in ("composed_config.yaml", "continuation.yaml"):
            (stage / name).write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
        bindings = dict(env, PROJECT=old["project"], ASSET_PROJECT=str(project),
                        GRPO_RUN=str(run), FORMAL_PREP=str(prep), FORMAL_OUT=str(out))
        (stage / "session.env").write_text("".join(
            f"export {key}={shlex.quote(str(value))}\n" for key, value in sorted(bindings.items())
        ), encoding="utf-8")
        (stage / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        write(stage / "resume_receipt.json", receipt)
        write(stage / "launch_spec.json", record)
        stage.rename(prep)
    print("PREPARED:", prep)
    print("PASS: recovery configuration only; no GPU, services, or training started")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(DEFAULT_PROJECT))
    args = parser.parse_args()
    try:
        prepare(args.project)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"Recovery preparation blocked: {exc}") from None
