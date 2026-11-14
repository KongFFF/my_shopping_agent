"""Stage the authorized step650->800 continuation with unchanged training recipe.

CPU-light file checks and Hydra composition only: no models, training or services.
Old checkpoints below650 may have been archived; they are not required here.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import zipfile

from prepare_bounded_grpo_resume import (
    DEFAULT_PROJECT, GIB, REQUIRED, RUN, compatible_shutdown_monitor,
    differences, digest, read, replace_overrides, write,
)
from prepare_bounded_grpo_recovery import finite


def derive(before, checkpoint, out):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(before)
    expected = {
        "trainer.total_training_steps": 650, "trainer.save_freq": 50, "trainer.test_freq": 50,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "data.train_batch_size": 2, "actor_rollout_ref.rollout.n": 4,
        "algorithm.norm_adv_by_std_in_grpo": False,
    }
    for key, value in expected.items():
        if OmegaConf.select(cfg, key) != value:
            raise ValueError(f"Unexpected original recipe: {key}")
    if set(cfg.actor_rollout_ref.actor.checkpoint.load_contents) != {"model", "optimizer", "extra"}:
        raise ValueError("Full model/optimizer/extra restoration is required")
    env = {"GRPO_OUTPUT_DIR": str(out),
           "SHOPPING_GRPO_DIAGNOSTICS_PATH": str(out / "training_diagnostics.jsonl")}
    changes = {
        "trainer.total_training_steps": 800,
        "trainer.resume_mode": "resume_path", "trainer.resume_from_path": str(checkpoint),
        "trainer.del_local_ckpt_after_load": False, "trainer.val_before_train": False,
        "trainer.default_local_dir": str(out), "trainer.validation_data_dir": str(out / "validation"),
        "trainer.experiment_name": "shopping-grpo-resume650-to800-v1",
        "actor_rollout_ref.rollout.trace.experiment_name": "shopping-grpo-resume650-to800-v1",
    }
    changes.update({"ray_kwargs.ray_init.runtime_env.env_vars." + k: v for k, v in env.items()})
    for key, value in changes.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    changed = differences(before, OmegaConf.to_container(cfg, resolve=True))
    if not set(changed) <= set(changes):
        raise ValueError(f"Unexpected configuration differences: {changed}")
    return cfg, changes, env, changed


def prepare(project):
    from omegaconf import OmegaConf
    project = Path(project).resolve()
    run = project / RUN
    parent = run / "preparation/formal_resume600_to650_v1"
    prep = run / "preparation/formal_resume650_to800_v1"
    out = run / "formal_resume650_to800_v1"
    if prep.exists() or out.exists():
        raise ValueError("New session already exists; refusing to overwrite or restart")
    old = read(parent / "launch_spec.json")
    previous_out = run / "formal_resume600_to650_v1"
    if Path(old["output"]).resolve() != previous_out.resolve() or old.get("total_updates") != 650:
        raise ValueError("Unexpected step650 parent")
    if read(parent / "exit_status.json").get("exit_code") != 0:
        raise ValueError("Parent training did not complete normally")
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
                raise ValueError(f"Parent launcher/monitor still active: PID={folder.name}")
    steps = []
    with (previous_out / "training_diagnostics.jsonl").open() as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") != "optimizer_step":
                continue
            if event["metrics"].get("training/optimizer_updated") != 1 or not finite(event["metrics"]):
                raise ValueError("Invalid completed-parent optimizer metrics")
            steps.append(int(event["global_step"]))
    if steps != list(range(601, 651)):
        raise ValueError("Parent 601..650 update chain incomplete")
    print("Checking retained runtime, data and SFT base hashes (no tensor loading)...", flush=True)
    for name, expected in old["source_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Runtime/package source missing or changed after cleanup: {name}")
    baseline = read(run / "preparation/baseline_plan.json")
    for name, expected in baseline["data_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError(f"Original data missing or changed: {name}")
    for name, expected in baseline["model_sha256"].items():
        if digest(Path(baseline["model"]) / name) != expected:
            raise ValueError(f"SFT base missing or changed: {name}")
    checkpoint = previous_out / "global_step_650"
    for name in (*REQUIRED, "actor/lora_train_meta.json"):
        path = checkpoint / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Retained step650 checkpoint file missing: {path}")
        if path.suffix == ".pt":
            with zipfile.ZipFile(path) as archive:
                if not archive.namelist():
                    raise ValueError(f"Empty checkpoint archive: {path}")
    meta, fsdp = read(checkpoint / "actor/lora_train_meta.json"), read(checkpoint / "actor/fsdp_config.json")
    if int(meta["r"]) != 16 or int(meta["lora_alpha"]) != 32 or int(fsdp["world_size"]) != 1:
        raise ValueError("Unexpected step650 LoRA/FSDP metadata")
    size = sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
    training = 3 * size + 15 * GIB
    required = training + 2 * (checkpoint / REQUIRED[0]).stat().st_size + 5 * GIB
    available = shutil.disk_usage(project).free
    print(f"Available: {available/GIB:.2f} GiB; training: {training/GIB:.2f} GiB; with one export: {required/GIB:.2f} GiB", flush=True)
    if available < required:
        raise ValueError(f"Need another {(required-available)/GIB:.2f} GiB free; no configuration staged")
    before = OmegaConf.to_container(OmegaConf.load(parent / "composed_config.yaml"), resolve=True)
    cfg, changes, env_updates, changed = derive(before, checkpoint, out)
    env = dict(old["env"], **env_updates)
    if env.get("SHOPPING_EVIDENCE_MANIFEST") or env.get("SHOPPING_REWARD_TOOLS_DIR"):
        raise ValueError("Unexpected semantic reward backend")
    command = list(old["command"])
    pos = command.index("--")
    prefix = command[:pos]
    prefix[0] = sys.executable
    for key, value in (("--output", str(out)), ("--experiment-name", "shopping-grpo-resume650-to800-v1")):
        prefix[prefix.index(key)+1] = value
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
        raise ValueError(f"Staged CLI differs from intended recipe: {mismatch}")
    record = dict(old)
    record.update(command=command, preflight=preflight, env=env, output=str(out),
                  required_free_bytes=required, resume_from=str(checkpoint), resume_step=650,
                  total_updates=800, remaining_updates=150, checkpoint_steps=[700, 750, 800],
                  parent_preparation=str(parent), initial_policy="Restore complete GRPO step650 training state")
    receipt = {"resume_step": 650, "target_step": 800, "checkpoint": str(checkpoint),
        "new_checkpoint_steps": [700, 750, 800], "config_differences": changed,
        "parent_spec_sha256": digest(parent / "launch_spec.json"),
        "parent_config_sha256": digest(parent / "composed_config.yaml"),
        "preparation_tool_sha256": digest(__file__), "checkpoint_bytes": size,
        "training_required_bytes": training, "required_free_bytes": required,
        "available_bytes": available, "training_started": False,
        "checkpoint_tensor_load_test_pending": True,
        "shutdown_invocation": "Fixed /usr/bin/shutdown via /bin/bash -c; historical files unchanged"}
    with tempfile.TemporaryDirectory(prefix="continue650_", dir=parent.parent) as td:
        stage = Path(td) / "prepared"
        stage.mkdir()
        for name in ("launch_formal.py", "math_probe.json"):
            shutil.copy2(parent / name, stage / name)
        monitor = compatible_shutdown_monitor((parent / "auto_shutdown.py").read_text(encoding="utf-8-sig"))
        (stage / "auto_shutdown.py").write_text(monitor, encoding="utf-8")
        for name in ("composed_config.yaml", "continuation.yaml"):
            (stage / name).write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
        bindings = dict(env, PROJECT=old["project"], ASSET_PROJECT=str(project),
                        GRPO_RUN=str(run), FORMAL_PREP=str(prep), FORMAL_OUT=str(out))
        (stage / "session.env").write_text("".join(
            f"export {key}={shlex.quote(str(value))}\n" for key, value in sorted(bindings.items())
        ), encoding="utf-8")
        (stage / "command.txt").write_text(shlex.join(command)+"\n", encoding="utf-8")
        write(stage / "resume_receipt.json", receipt)
        write(stage / "launch_spec.json", record)
        stage.rename(prep)
    print("PREPARED:", prep)
    print("PASS: step650->800 staged; same recipe; no models, services or training started")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(DEFAULT_PROJECT))
    args = parser.parse_args()
    try:
        prepare(args.project)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"Continuation preparation blocked: {exc}") from None
