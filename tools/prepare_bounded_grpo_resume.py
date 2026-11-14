"""Audit the historical GRPO run and stage step500->650; never launch training.

Run on AutoDL with the original project's .venv Python. Audit uses only stdlib;
--prepare additionally requires the original Hydra/OmegaConf environment.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

GIB = 2**30
BASE = "a583dba4c149be09622ce58ad6f7a160db77bb41"
PINNED_REPAIRS = {
    "configs/agent_loop.yaml": "64e31ca42a06cfd3b40f19ccae2881ed2a6d55c03c811717f46b1011fbdc3a2d",
    "configs/grpo.yaml": "8a59aa536ee95e70de9c55e3ea7bf2d15819dd17bfcda385bcf126b8475c6508",
    "pyproject.toml": "8b9651716f091a0451119b0e393073920993055d4ba1632e2245b40db2a59f06",
    "scripts/check_grpo_runtime.py": "cd25a56adca8c54f44461acb029de85c73dda9aa4bb3d2b57dfe36b0476b21ed",
    "scripts/train_grpo.py": "c181815a57bb8210994245b2d335d37d2488e6ae141d1f25f9bffb9167d3f159",
}
RUN = "outputs/runs/grpo_baseline_20261022_202816"
DEFAULT_PROJECT = "/root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon"
REQUIRED = (
    "actor/model_world_size_1_rank_0.pt", "actor/optim_world_size_1_rank_0.pt",
    "actor/extra_state_world_size_1_rank_0.pt", "actor/fsdp_config.json", "data.pt",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compatible_shutdown_monitor(source):
    """Use a shell for AutoDL's command, which can be a no-shebang script.

    This only transforms source; it never calls the shutdown command. Existing
    monitor logs and historical files remain unchanged.
    """
    old = "subprocess.run([shutdown], check=True)"
    new = 'subprocess.run(["/bin/bash", "-c", shutdown], check=True)'
    if source.count(old) == 1 and new not in source:
        source = source.replace(old, new)
    elif old in source or source.count(new) != 1:
        raise ValueError("Unknown shutdown monitor implementation; refusing a broad replacement")
    ast.parse(source)
    return source


def audit(project, source_overrides=None):
    project = Path(project).resolve()
    run = project / RUN
    parent = run / "preparation/formal_resume350_v1"
    checkpoint = run / "formal_resume350_v1/global_step_500"
    issues, checked = [], []

    def problem(kind, path, detail):
        issues.append({"kind": kind, "path": str(path), "detail": detail})

    def check_hash(path, expected, category):
        path = Path(path)
        if not path.is_file():
            problem(category, path, "missing")
            return
        actual = digest(path)
        checked.append({"path": str(path), "sha256": actual, "category": category})
        if actual != expected:
            problem(category, path, {"expected": expected, "actual": actual})

    spec = read(parent / "launch_spec.json")
    baseline = read(run / "preparation/baseline_plan.json")
    if Path(spec["project"]).resolve() != project:
        problem("project", project, "Historical absolute project path differs; do not silently relocate")
    if Path(spec["output"]).resolve() != checkpoint.parent:
        problem("parent_run", parent, "launch_spec output does not identify the step500 run")
    if read(parent / "exit_status.json").get("exit_code") != 0:
        problem("parent_exit", parent, "Historical run has no successful completion")
    if not spec.get("source_sha256"):
        problem("source", parent, "Missing original source hash bindings")
    for name, expected in spec.get("source_sha256", {}).items():
        check_hash((source_overrides or {}).get(name, name), expected, "source")
    for name, expected in baseline["data_sha256"].items():
        check_hash(project / name, expected, "data")
    for name, expected in baseline["model_sha256"].items():
        check_hash(Path(baseline["model"]) / name, expected, "sft_model")
    for name in REQUIRED:
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            problem("checkpoint", path, "missing or empty")
        elif path.suffix == ".pt":
            try:
                with zipfile.ZipFile(path) as archive:
                    if not archive.namelist():
                        raise ValueError("empty archive")
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                problem("checkpoint", path, str(exc))
    # Check structural validity only: never deserialize multi-GB tensors on a
    # no-GPU instance with a small RAM quota.
    for name in ("composed_config.yaml", "launch_formal.py", "math_probe.json"):
        if not (parent / name).is_file():
            problem("preparation", parent / name, "missing")
    probe = read(parent / "math_probe.json")
    if not probe.get("passed") or probe.get("engine_sha256") != spec["source_sha256"].get(probe.get("engine_file")):
        problem("math_probe", parent, "Original numerical probe does not bind the original engine")
    # We only allow the old, CUDA-free parent launcher, not a modified one.
    driver = (parent / "launch_formal.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(driver)
    if any(isinstance(n, ast.Import) and any(a.name == "torch" for a in n.names) for n in tree.body):
        problem("launcher", parent, "Parent launcher imports torch in its long-lived process")
    if "Path(__file__).resolve().parent" not in driver:
        problem("launcher", parent, "Parent launcher is not bound to its own preparation directory")
    active = []
    proc = Path("/proc")
    if proc.is_dir():
        for folder in proc.iterdir():
            if not folder.name.isdigit():
                continue
            try:
                args = (folder / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if any(str(run / "preparation").encode() in arg and
                   (arg.endswith(b"launch_formal.py") or arg.endswith(b"auto_shutdown.py")) for arg in args):
                active.append(folder.name)
    if active:
        problem("active_old_process", run, {"pids": active})
    size = sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
    weights = checkpoint / REQUIRED[0]
    weight_size = weights.stat().st_size if weights.is_file() else 0
    free = shutil.disk_usage(project).free
    training = 3 * size + 15 * GIB
    with_export = training + 2 * weight_size + 5 * GIB
    if free < with_export:
        problem("disk", project, {"required_bytes": with_export, "free_bytes": free})
    return {
        "schema": "bounded-grpo-resume-v1", "project": str(project),
        "parent_preparation": str(parent), "checkpoint": str(checkpoint),
        "resume_step": 500, "target_step": 650, "new_checkpoint_steps": [550, 600, 650],
        "checkpoint_bytes": size, "free_bytes": free, "training_required_bytes": training,
        "training_and_one_export_required_bytes": with_export,
        "checked": checked, "issues": issues, "passed": not issues,
        "checkpoint_tensor_load_test_pending": True, "training_started": False,
    }


def export_pinned_code(original, stage, archive):
    with Path(archive).open("wb") as f:
        subprocess.run(["git", "-C", str(original), "archive", BASE,
                        "src", "configs", "scripts", "pyproject.toml"], stdout=f, check=True)
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            if member.issym() or member.islnk() or not (stage / member.name).resolve().is_relative_to(stage):
                raise ValueError("Unsafe pinned Git archive")
        tf.extractall(stage, filter="data")
    launcher = stage / "scripts/train_grpo.py"
    raw = launcher.read_bytes()
    anchor = (b'        str(ROOT / "scripts/check_grpo_runtime.py"),\n'
              b'        *overrides,\n        *extra,\n')
    if raw.count(anchor) != 1:
        raise ValueError("Original launcher argument-forwarding anchor differs")
    launcher.write_bytes(raw.replace(anchor,
        b'        str(ROOT / "scripts/check_grpo_runtime.py"),\n        *command[5:],\n'))
    for name, expected in PINNED_REPAIRS.items():
        if digest(stage / name) != expected:
            raise ValueError(f"Pinned reconstruction differs from original hash: {name}")


def isolate_runtime(report):
    """Export pinned code plus the exact old launcher fix; share large assets.

    Original source files and frozen expected hashes are never changed.
    Installed packages are shared only after the original audit matches them.
    """
    original = Path(report["project"])
    for issue in report["issues"]:
        try:
            relative = Path(issue["path"]).relative_to(original).as_posix()
        except ValueError:
            relative = ""
        if (issue["kind"] != "source" or relative not in PINNED_REPAIRS or
                not isinstance(issue["detail"], dict) or
                issue["detail"].get("expected") != PINNED_REPAIRS[relative]):
            raise ValueError(f"Isolation cannot resolve this audit failure: {issue['kind']} {issue['path']}")
    spec = read(Path(report["parent_preparation"]) / "launch_spec.json")
    destination = original / "outputs/bounded_resume500_to650/runtime"
    links = {name: str(original / name) for name in (".venv", "data", "environments")}
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="isolated_resume_", dir=destination.parent) as td:
            stage = Path(td) / "runtime"
            stage.mkdir()
            archive = Path(td) / "code.tar"
            export_pinned_code(original, stage, archive)
            files = {p.relative_to(stage).as_posix(): digest(p) for p in stage.rglob("*") if p.is_file()}
            for name, target in links.items():
                if not Path(target).is_dir():
                    raise ValueError(f"Shared asset directory missing: {target}")
                (stage / name).symlink_to(target, target_is_directory=True)
            write(stage / "runtime_receipt.json", {
                "base": BASE, "files": files, "shared_directories": links,
                "repair": "Original unique preflight argument-forwarding fix, SHA256 verified",
                "original_files_modified": False,
            })
            stage.rename(destination)
    receipt = read(destination / "runtime_receipt.json")
    if receipt.get("base") != BASE or receipt.get("shared_directories") != links:
        raise ValueError("Existing isolated runtime identity differs")
    actual_files = {p.relative_to(destination).as_posix(): digest(p)
                    for name in ("src", "configs", "scripts")
                    for p in (destination / name).rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    actual_files["pyproject.toml"] = digest(destination / "pyproject.toml")
    if actual_files != receipt["files"]:
        raise ValueError("Existing isolated runtime file hashes differ")
    for name, target in links.items():
        if not (destination / name).is_symlink() or (destination / name).resolve() != Path(target).resolve():
            raise ValueError(f"Isolated asset link changed: {name}")
    bindings = {}
    for name, expected in spec["source_sha256"].items():
        source = Path(name)
        if source.is_relative_to(original):
            relative = source.relative_to(original)
            if relative.parts[0] in ("src", "configs", "scripts") or relative.as_posix() == "pyproject.toml":
                target = destination / relative
                if not target.is_file() or digest(target) != expected:
                    raise ValueError(f"Isolated source does not match historical binding: {relative}")
                bindings[name] = str(target)
    effective = audit(original, source_overrides=bindings)
    effective.update(runtime_project=str(destination), source_overrides=bindings,
                     original_source_issues=report["issues"],
                     runtime_receipt_sha256=digest(destination / "runtime_receipt.json"))
    if not effective["passed"]:
        raise ValueError(f"Isolated effective audit failed: {effective['issues']}")
    print("PASS: isolated pinned runtime, original expected hashes unchanged; large assets shared")
    return effective


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            name = f"{prefix}.{key}" if prefix else key
            result.extend([name] if key not in a or key not in b else differences(a[key], b[key], name))
        return result
    return [] if a == b else [prefix]


def derive_config(before, report):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(before)
    expected = {
        "trainer.total_training_steps": 500,
        "trainer.save_freq": 50, "trainer.test_freq": 50,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "data.train_batch_size": 2, "actor_rollout_ref.rollout.n": 4,
        "algorithm.norm_adv_by_std_in_grpo": False,
    }
    for name, value in expected.items():
        if OmegaConf.select(cfg, name) != value:
            raise ValueError(f"Unexpected original configuration: {name}")
    contents = OmegaConf.select(cfg, "actor_rollout_ref.actor.checkpoint.load_contents")
    if contents is not None and set(contents) != {"model", "optimizer", "extra"}:
        raise ValueError("Original checkpoint load_contents does not restore all training state")
    project = Path(report["project"])
    out = project / RUN / "formal_resume500_to650_v1"
    changes = {
        "trainer.total_training_steps": 650,
        "trainer.resume_mode": "resume_path",
        "trainer.resume_from_path": report["checkpoint"],
        "trainer.del_local_ckpt_after_load": False,
        "trainer.val_before_train": False,
        "trainer.default_local_dir": str(out),
        "trainer.validation_data_dir": str(out / "validation"),
        "trainer.experiment_name": "shopping-grpo-resume500-to650-v1",
        "actor_rollout_ref.rollout.trace.experiment_name": "shopping-grpo-resume500-to650-v1",
    }
    for name, value in changes.items():
        OmegaConf.update(cfg, name, value, force_add=True)
    env_updates = {"GRPO_OUTPUT_DIR": str(out),
                   "SHOPPING_GRPO_DIAGNOSTICS_PATH": str(out / "training_diagnostics.jsonl")}
    if report.get("runtime_project"):
        runtime = Path(report["runtime_project"])
        path_changes = {
            "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(runtime / "configs/agent_loop.yaml"),
            "actor_rollout_ref.rollout.multi_turn.tool_config_path": str(runtime / "configs/tools.json"),
            "data.tool_config_path": str(runtime / "configs/tools.json"),
        }
        for name, value in path_changes.items():
            if OmegaConf.select(cfg, name) is not None:
                changes[name] = value
                OmegaConf.update(cfg, name, value)
        env_updates.update(PYTHONPATH=str(runtime / "src"), SHOPPING_GRPO_ROOT=str(runtime),
                           SHOPPING_ENV_MANIFEST=str(runtime / "data/environment.json"),
                           SHOPPING_AGENT_LOOP_CONFIG=str(runtime / "configs/agent_loop.yaml"),
                           SHOPPING_TOOL_CONFIG=str(runtime / "configs/tools.json"))
    # Resolved Hydra config includes literal Ray worker values. Update both
    # driver and worker bindings, so resumed rollouts cannot write old logs.
    for name, value in env_updates.items():
        key = "ray_kwargs.ray_init.runtime_env.env_vars." + name
        changes[key] = value
        OmegaConf.update(cfg, key, value, force_add=True)
    after = OmegaConf.to_container(cfg, resolve=True)
    changed = differences(before, after)
    if not set(changed) <= set(changes):
        raise ValueError(f"Unexpected resolved configuration changes: {changed}")
    return cfg, changed, env_updates, changes


def replace_overrides(overrides, changes):
    """Replace explicit Hydra keys, rejecting duplicated old bindings."""
    from hydra.core.override_parser.types import Quote, QuotedString
    result, seen = [], set()
    for value in overrides:
        key = value.lstrip("+").split("=", 1)[0]
        if key in changes:
            if key in seen:
                raise ValueError(f"Duplicated original override: {key}")
            seen.add(key)
        else:
            result.append(value)
    for key, value in changes.items():
        literal = QuotedString(value, Quote.double).with_quotes() if isinstance(value, str) else json.dumps(value)
        result.append("++" + key + "=" + literal)
    return result


def prepare(report):
    if not report["passed"]:
        raise ValueError("Audit failed; no continuation configuration was generated")
    from omegaconf import OmegaConf
    project = Path(report["project"])
    runtime = Path(report.get("runtime_project", project))
    run = project / RUN
    parent = Path(report["parent_preparation"])
    prep = run / "preparation/formal_resume500_to650_v1"
    out = run / "formal_resume500_to650_v1"
    if prep.exists() or out.exists():
        raise ValueError("New preparation/output already exists; refusing to overwrite or restart")
    old = read(parent / "launch_spec.json")
    before = OmegaConf.to_container(OmegaConf.load(parent / "composed_config.yaml"), resolve=True)
    cfg, changed, env_updates, changes = derive_config(before, report)
    # Preserve the historical wrapper and preflight invocation. Both their
    # source files and the resulting fully resolved configuration are checked.
    command = list(old["command"])
    split = command.index("--")
    prefix = command[:split]
    prefix[0] = sys.executable
    if runtime != project:
        prefix[1] = str(runtime / "scripts/train_grpo.py")
        prefix[prefix.index("--config") + 1] = str(runtime / "configs/grpo.yaml")
    for name, value in (("--output", str(out)),
                        ("--experiment-name", "shopping-grpo-resume500-to650-v1")):
        prefix[prefix.index(name) + 1] = value
    command = prefix + ["--"] + replace_overrides(command[split+1:], changes)
    preflight = list(old["preflight"])
    preflight[0] = sys.executable
    if runtime != project:
        preflight[1] = str(runtime / "scripts/check_grpo_runtime.py")
    preflight = preflight[:2] + replace_overrides(preflight[2:], changes)
    env = dict(old["env"], **env_updates)
    for key in ("SHOPPING_EVIDENCE_MANIFEST", "SHOPPING_REWARD_TOOLS_DIR"):
        if env.get(key):
            raise ValueError(f"Original environment unexpectedly enables {key}")
    # Prevent ambient semantic-experiment bindings from entering a new shell.
    env.update(SHOPPING_EVIDENCE_MANIFEST="", SHOPPING_REWARD_TOOLS_DIR="")
    os.environ.update(env)
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(runtime / "configs")):
        actual = compose(config_name=env.get("GRPO_CONFIG_NAME", "grpo"), overrides=preflight[2:])
    cli_diff = differences(OmegaConf.to_container(cfg, resolve=True), OmegaConf.to_container(actual, resolve=True))
    if cli_diff:
        raise ValueError(f"New CLI preflight differs from intended configuration: {cli_diff}")
    spec = dict(old)
    spec.update(command=command, env=env, output=str(out), project=str(runtime),
                asset_project=str(project), source_sha256={
                    report.get("source_overrides", {}).get(name, name): expected
                    for name, expected in old["source_sha256"].items()},
                required_free_bytes=report["training_and_one_export_required_bytes"],
                preflight=preflight,
                resume_from=report["checkpoint"], resume_step=500,
                total_updates=650, remaining_updates=150, checkpoint_steps=[550, 600, 650],
                parent_preparation=str(parent), validation_every=50, checkpoint_every=50,
                checkpoint_selection="Dev50 only; Final excluded",
                initial_policy="Restore complete original GRPO step500 training state")
    # Stage atomically: partial preparation is never mistaken for readiness.
    with tempfile.TemporaryDirectory(prefix="bounded_resume_", dir=parent.parent) as td:
        stage = Path(td) / "prepared"
        stage.mkdir()
        for name in ("launch_formal.py", "math_probe.json"):
            shutil.copy2(parent / name, stage / name)
        # Copying the monitor does not start it. It is supplied for later use.
        if (parent / "auto_shutdown.py").is_file():
            monitor = compatible_shutdown_monitor((parent / "auto_shutdown.py").read_text(encoding="utf-8-sig"))
            (stage / "auto_shutdown.py").write_text(monitor, encoding="utf-8")
        for name in ("continuation.yaml", "composed_config.yaml"):
            (stage / name).write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
        write(stage / "resume_receipt.json", dict(report, config_differences=changed,
              inherited_constant_lr=1e-6, parent_config_sha256=digest(parent / "composed_config.yaml"),
              parent_spec_sha256=digest(parent / "launch_spec.json"),
              preparation_tool_sha256=digest(__file__)))
        (stage / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        bindings = dict(env, PROJECT=str(runtime), ASSET_PROJECT=str(project), GRPO_RUN=str(run),
                        FORMAL_PREP=str(prep), FORMAL_OUT=str(out))
        # Export only the historical declared bindings, never inherited secrets.
        (stage / "session.env").write_text("".join(
            f"export {k}={shlex.quote(str(v))}\n" for k, v in sorted(bindings.items())
        ), encoding="utf-8")
        write(stage / "launch_spec.json", spec)
        stage.rename(prep)
    print("PREPARED:", prep)
    print("No environment service, Ray, GPU process, or training was started.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(DEFAULT_PROJECT))
    parser.add_argument("--report", type=Path, required=True, help="New JSON report path; never overwritten")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--isolate-runtime", action="store_true",
                        help="Rebuild exact original code separately; keep current experiment files intact")
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("Report already exists; choose a new report filename")
    print("Checking historical files and streaming model hashes; no tensor loading or GPU use.", flush=True)
    try:
        report = audit(args.project)
    except (OSError, KeyError, ValueError) as exc:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write(args.report, {"passed": False, "training_started": False,
                           "project": str(args.project), "audit_input_error": str(exc)})
        raise SystemExit(f"Historical audit input missing or invalid: {exc}. Report: {args.report}") from None
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.isolate_runtime:
        try:
            report = isolate_runtime(report)
        except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
            report.update(isolation_error=str(exc), passed=False)
            write(args.report, report)
            raise SystemExit(f"Isolated preparation failed: {exc}. Report: {args.report}") from None
    write(args.report, report)
    for label, key in (("Available", "free_bytes"), ("Training", "training_required_bytes"),
                       ("Training + one export", "training_and_one_export_required_bytes")):
        print(f"{label}: {report[key]/GIB:.2f} GiB")
    print(f"Hash-checked files: {len(report['checked'])}")
    if report["issues"]:
        for item in report["issues"]:
            print(f"FAIL [{item['kind']}] {item['path']}: {item['detail']}")
        raise SystemExit("Audit failed; original files untouched. See " + str(args.report))
    print("PASS: historical hashes, checkpoint structure, parent receipts and disk budget")
    if args.prepare:
        prepare(report)
    else:
        print("Audit only; no continuation configuration generated.")


if __name__ == "__main__":
    main()
