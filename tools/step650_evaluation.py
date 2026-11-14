"""Prepare/export step650 using the frozen step500 independent evaluation.

prepare is CPU-light; export needs the original high-RAM instance. Neither
starts model serving, evaluation, training, or shutdown.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from prepare_bounded_grpo_resume import DEFAULT_PROJECT, GIB, RUN, digest, read, write

FINAL_HASH = "d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5"


def replace_option(argv, key, value):
    if argv.count(key) != 1:
        raise ValueError(f"Expected one {key} in frozen command")
    result = list(argv)
    result[result.index(key) + 1] = str(value)
    return result


def patch_runner(source):
    old = "subprocess.run(['/usr/bin/shutdown'],check=True)"
    new = 'subprocess.run(["/bin/bash","-c","/usr/bin/shutdown"],check=True)'
    if source.count(old) == 1 and new not in source:
        source = source.replace(old, new)
    elif old in source or source.count(new) != 1:
        raise ValueError("Unknown frozen runner shutdown call; refusing broad edits")
    ast.parse(source)
    return source


def derive_reference(reference, project, runtime, run):
    post = run / "preparation/posttrain_step650_v1"
    export = run / "export/step650_v1"
    evaluation = run / "evaluation/step650_independent_v1"
    tag = "shopping-agent-grpo-step650"
    result = dict(reference)
    commands = {}
    for suite, argv in reference["commands"].items():
        if suite not in ("smoke3", "dev50", "final200_clean"):
            raise ValueError(f"Unknown frozen evaluation suite: {suite}")
        command = list(argv)
        command[0] = sys.executable
        command[1] = str(runtime / "scripts/evaluate_shop_benchmark.py")
        for key, value in (("--model", tag),
                           ("--output", evaluation / suite / "trajectories.jsonl"),
                           ("--summary", evaluation / suite / "summary.json")):
            command = replace_option(command, key, value)
        commands[suite] = command
    if set(commands) != {"smoke3", "dev50", "final200_clean"}:
        raise ValueError("Frozen smoke/dev/final commands are not complete")
    sources = {}
    for name, expected in reference["source_sha256"].items():
        path = Path(name)
        if path.is_relative_to(project):
            relative = path.relative_to(project)
            if relative.parts[0] in ("src", "scripts", "configs"):
                path = runtime / relative
        if digest(path) != expected:
            raise ValueError(f"Step500 evaluation source differs at its isolated binding: {path}")
        sources[str(path)] = expected
    result.update(project=str(runtime), asset_project=str(project), post=str(post),
                  checkpoint=str(run / "formal_resume600_to650_v1/global_step_650"),
                  actor=str(run / "formal_resume600_to650_v1/global_step_650/actor"),
                  raw_export=str(export / "hf_unmerged"), merged_model=str(export / "merged"),
                  model_tag=tag, evaluation_root=str(evaluation), commands=commands,
                  source_sha256=sources, reference_step=500, diagnostic_endpoint_step=650,
                  purpose="One predeclared continuation-endpoint Final evaluation; not checkpoint selection")
    return result


def prepare(project):
    project = Path(project).resolve()
    run = project / RUN
    post = run / "preparation/posttrain_step650_v1"
    if post.exists() or (run / "export/step650_v1").exists() or (run / "evaluation/step650_independent_v1").exists():
        raise ValueError("Step650 postprocessing already exists; inspect rather than overwrite")
    parent = run / "preparation/formal_resume600_to650_v1"
    if read(parent / "exit_status.json").get("exit_code") != 0:
        raise ValueError("Step650 training did not record successful completion")
    training = read(parent / "launch_spec.json")
    runtime = Path(training["project"])
    if training.get("resume_step") != 600 or training.get("total_updates") != 650:
        raise ValueError("Unexpected completed training bounds")
    checkpoint = run / "formal_resume600_to650_v1/global_step_650"
    actor = checkpoint / "actor"
    meta = read(actor / "lora_train_meta.json")
    fsdp = read(actor / "fsdp_config.json")
    if int(meta["r"]) != 16 or int(meta["lora_alpha"]) != 32 or int(fsdp["world_size"]) != 1:
        raise ValueError("Unexpected LoRA/FSDP export metadata")
    for name, expected in training["source_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Completed training source changed: {name}")
    reference_dir = run / "preparation/posttrain_step500_v1"
    reference = read(reference_dir / "evaluation_spec.json")
    old_summary = read(Path(reference["evaluation_root"]) / "final200_clean/summary.json")
    if old_summary["strict_successes"] != 137 or old_summary["expected_tasks"] != 200:
        raise ValueError("Frozen step500 Final reference is not 137/200")
    for suite, n in (("smoke3", 3), ("dev50", 50), ("final200_clean", 200)):
        benchmark = reference["benchmarks"][suite]
        if digest(benchmark["path"]) != benchmark["sha256"]:
            raise ValueError(f"Frozen benchmark changed: {suite}")
        rows = [json.loads(line) for line in Path(benchmark["path"]).read_text().splitlines() if line.strip()]
        ids = [int(row["task_id"]) for row in rows]
        if len(ids) != n or len(set(ids)) != n or ids != benchmark["task_ids"]:
            raise ValueError(f"Frozen benchmark identity/order differs: {suite}")
    if reference["benchmarks"]["final200_clean"]["sha256"] != FINAL_HASH:
        raise ValueError("Unexpected Final Clean task manifest")
    spec = derive_reference(reference, project, runtime, run)
    runner = patch_runner((reference_dir / "run_eval.py").read_text(encoding="utf-8-sig"))
    with tempfile.TemporaryDirectory(prefix="step650_eval_", dir=post.parent) as td:
        stage = Path(td) / "prepared"
        stage.mkdir()
        (stage / "run_eval.py").write_text(runner, encoding="utf-8")
        write(stage / "evaluation_spec.json", spec)
        write(stage / "preparation_receipt.json", {
            "reference_spec_sha256": digest(reference_dir / "evaluation_spec.json"),
            "reference_runner_sha256": digest(reference_dir / "run_eval.py"),
            "runner_change": "Only replace direct shutdown exec with shell invocation",
            "training_status": read(parent / "exit_status.json"),
            "protocols_unchanged": spec["protocols"] == reference["protocols"],
            "benchmarks_unchanged": spec["benchmarks"] == reference["benchmarks"],
            "export_started": False, "evaluation_started": False,
        })
        bindings = {
            "PROJECT": str(runtime), "ASSET_PROJECT": str(project), "GRPO_RUN": str(run),
            "POST": str(post), "ACTOR": spec["actor"], "RAW_EXPORT": spec["raw_export"],
            "GRPO_MERGED": spec["merged_model"], "EVAL_ROOT": spec["evaluation_root"],
            "MODEL_TAG": spec["model_tag"], "PYTHONPATH": str(runtime / "src"),
            "OMP_NUM_THREADS": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "SERVED_MODEL_NAME": spec["model_tag"], "LLM_PORT": "8000",
            "SHOPSIM_BASE_URL": "http://127.0.0.1:5700",
        }
        (stage / "session.env").write_text("".join(
            f"export {k}={shlex.quote(v)}\n" for k, v in bindings.items()), encoding="utf-8")
        stage.rename(post)
    print("PREPARED:", post)
    print("PASS: frozen step500 protocols/tasks reused; no export, model serving, evaluation or shutdown started")


def export_model(project):
    project = Path(project).resolve()
    post = project / RUN / "preparation/posttrain_step650_v1"
    spec = read(post / "evaluation_spec.json")
    actor, raw, merged = (Path(spec[key]) for key in ("actor", "raw_export", "merged_model"))
    for name, expected in spec["source_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Export/evaluation source changed: {name}")
    receipt_path = post / "export_receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["checkpoint"] != spec["checkpoint"]:
            raise ValueError("Existing export identifies another checkpoint")
        for name, expected in receipt["merged_weights_sha256"].items():
            if digest(merged / name) != expected:
                raise ValueError(f"Existing export changed: {name}")
        print("PASS: existing verified step650 export reused:", merged)
        return
    limit = Path("/sys/fs/cgroup/memory.max")
    if limit.exists() and limit.read_text().strip() != "max" and int(limit.read_text()) < 110 * GIB:
        raise ValueError("Export requires original high-RAM instance (at least110 GiB); no-GPU2 GiB instance is insufficient")
    required = 2 * (actor / "model_world_size_1_rank_0.pt").stat().st_size + 5 * GIB
    free = shutil.disk_usage(actor).free
    print(f"Export disk available: {free/GIB:.2f} GiB; required: {required/GIB:.2f} GiB", flush=True)
    if free < required:
        raise ValueError("Insufficient export disk space")
    if not (post / "fsdp_export_done.json").exists():
        if raw.exists() and any(raw.iterdir()):
            raise ValueError("Unfinished raw export exists; inspect fsdp_export.log, do not overwrite")
        code = '''import runpy,sys,tempfile
from pathlib import Path
import torch.distributed as dist
with tempfile.TemporaryDirectory() as td:
    dist.init_process_group('gloo',init_method=Path(td,'init').as_uri(),rank=0,world_size=1)
    try:
        sys.argv=['verl.model_merger','merge','--backend','fsdp','--local_dir',sys.argv[1],
                  '--target_dir',sys.argv[2],'--trust-remote-code','--use_cpu_initialization']
        runpy.run_module('verl.model_merger',run_name='__main__')
    finally:
        dist.destroy_process_group()
'''
        with (post / "fsdp_export.log").open("w") as log:
            subprocess.run([sys.executable, "-u", "-c", code, str(actor), str(raw)],
                           cwd=spec["project"], stdout=log, stderr=subprocess.STDOUT, check=True)
        config = read(raw / "lora_adapter/adapter_config.json")
        if int(config["r"]) != 16 or int(config["lora_alpha"]) != 32 or not (raw / "config.json").is_file():
            raise ValueError("Raw export metadata differs")
        write(post / "fsdp_export_done.json", {"raw_export": str(raw), "rank": 16, "alpha": 32})
    adapter = raw / "lora_adapter"
    import torch
    from safetensors import safe_open
    count, nonzero_b = 0, 0
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite adapter tensor: {key}")
            count += 1
            if ".lora_B." in key:
                nonzero_b += bool(torch.count_nonzero(tensor))
    if not count or not nonzero_b:
        raise ValueError("Exported adapter has no nonzero LoRA B parameters")
    if merged.exists() and any(merged.iterdir()):
        raise ValueError("Unverified merged output exists; inspect adapter_merge.log, do not overwrite")
    with (post / "adapter_merge.log").open("w") as log:
        subprocess.run([sys.executable, "-u", "scripts/merge_lora_adapter.py", "--base-model", str(raw),
                        "--adapter", str(adapter), "--output", str(merged), "--bf16"],
                       cwd=spec["project"], stdout=log, stderr=subprocess.STDOUT, check=True)
    files = sorted(merged.glob("*.safetensors"))
    if not files or not (merged / "config.json").is_file():
        raise ValueError("Merged model output incomplete")
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if any("lora_" in key for key in handle.keys()):
                raise ValueError("Merged output still contains adapter tensors")
    write(receipt_path, {"operation": "GRPO FSDP export followed by PEFT merge_and_unload",
          "checkpoint": spec["checkpoint"], "raw_export": str(raw), "adapter": str(adapter),
          "merged_model": str(merged), "lora_rank": 16, "lora_alpha": 32,
          "adapter_tensor_count": count, "actor_model_sha256": digest(actor / "model_world_size_1_rank_0.pt"),
          "merged_weights_sha256": {p.name: digest(p) for p in files}})
    print("EXPORTED:", merged)
    print("PASS: step650 export and LoRA merge verified; no model serving or evaluation started")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "export"))
    parser.add_argument("--project", type=Path, default=Path(DEFAULT_PROJECT))
    parser.add_argument("--launch", action="store_true", help="Run export in an independent background process")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.launch:
            raise SystemExit("--launch is only supported for export")
        prepare(args.project)
    elif args.launch:
        post = args.project.resolve() / RUN / "preparation/posttrain_step650_v1"
        read(post / "evaluation_spec.json")
        with (post / "export_driver.log").open("x") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                "export", "--project", str(args.project.resolve())],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(post / "export_launcher.json", {"pid": process.pid, "started_unix": time.time()})
        print(f"Export launched: PID={process.pid}; log={post/'export_driver.log'}")
    else:
        import fcntl
        post = args.project.resolve() / RUN / "preparation/posttrain_step650_v1"
        with (post / "export.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            started, code = time.time(), 1
            try:
                export_model(args.project)
                code = 0
            finally:
                write(post / "export_exit_status.json", {"exit_code": code,
                      "started_unix": started, "ended_unix": time.time()})
