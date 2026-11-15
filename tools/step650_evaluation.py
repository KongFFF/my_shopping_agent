"""Evaluate step650 or Dev-selected step750 with frozen step500 protocols.

prepare is CPU-light and starts no services. pipeline exports the model,
starts verified services, then runs smoke3, dev50 and final200_clean.
"""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
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


def derive_reference(reference, project, runtime, run, step=650):
    post = run / f"preparation/posttrain_step{step}_v1"
    export = run / f"export/step{step}_v1"
    evaluation = run / f"evaluation/step{step}_independent_v1"
    tag = f"shopping-agent-grpo-step{step}"
    training_folder = "formal_resume600_to650_v1" if step == 650 else "formal_resume650_to800_v1"
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
                  checkpoint=str(run / training_folder / f"global_step_{step}"),
                  actor=str(run / training_folder / f"global_step_{step}/actor"),
                  raw_export=str(export / "hf_unmerged"), merged_model=str(export / "merged"),
                  model_tag=tag, evaluation_root=str(evaluation), commands=commands,
                  source_sha256=sources, reference_step=500, evaluated_step=step,
                  purpose=("One predeclared continuation-endpoint Final evaluation" if step == 650
                           else "Step750 fixed by training Dev selection before Final evaluation"))
    result["diagnostic_endpoint_step" if step == 650 else "dev_selected_step"] = step
    return result


def dev_selection(folder):
    scores = {700: [], 750: [], 800: []}
    fingerprints = {}
    for path in sorted((folder / "validation").glob("*.jsonl")):
        fingerprints[str(path)] = digest(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            step, score = int(row["step"]), float(row["score"])
            if step not in scores or not math.isfinite(score):
                raise ValueError("Unexpected training Dev record")
            scores[step].append(score)
    if any(len(values) != 50 for values in scores.values()):
        raise ValueError("Training Dev must contain exactly50 records at700/750/800")
    rows = [{"step": step, "gold": sum(math.isclose(x, 1, abs_tol=1e-6, rel_tol=0) for x in values),
             "mean_reward": math.fsum(values)/50} for step, values in scores.items()]
    chosen = sorted(rows, key=lambda row: (-row["gold"], -row["mean_reward"], row["step"]))[0]
    if chosen["step"] != 750:
        raise ValueError(f"Frozen Dev ranking did not select750: {rows}")
    return {"rows": rows, "selected_step": 750, "source_sha256": fingerprints,
            "ranking": "gold descending, mean reward descending, step ascending"}


def prepare(project, step=650):
    project = Path(project).resolve()
    run = project / RUN
    post = run / f"preparation/posttrain_step{step}_v1"
    if post.exists() or (run / f"export/step{step}_v1").exists() or (run / f"evaluation/step{step}_independent_v1").exists():
        raise ValueError(f"Step{step} postprocessing already exists; inspect rather than overwrite")
    training_folder = "formal_resume600_to650_v1" if step == 650 else "formal_resume650_to800_v1"
    parent = run / "preparation" / training_folder
    if read(parent / "exit_status.json").get("exit_code") != 0:
        raise ValueError("Parent training did not record successful completion")
    training = read(parent / "launch_spec.json")
    runtime = Path(training["project"])
    expected_bounds = (600, 650) if step == 650 else (650, 800)
    if (training.get("resume_step"), training.get("total_updates")) != expected_bounds:
        raise ValueError("Unexpected completed training bounds")
    selection = dev_selection(run / training_folder) if step == 750 else None
    checkpoint = run / training_folder / f"global_step_{step}"
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
    spec = derive_reference(reference, project, runtime, run, step)
    runner = patch_runner((reference_dir / "run_eval.py").read_text(encoding="utf-8-sig"))
    with tempfile.TemporaryDirectory(prefix=f"step{step}_eval_", dir=post.parent) as td:
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
            "selection": selection,
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


def export_model(project, step=650):
    project = Path(project).resolve()
    post = project / RUN / f"preparation/posttrain_step{step}_v1"
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
        print(f"PASS: existing verified step{step} export reused:", merged)
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
    print(f"PASS: step{step} export and LoRA merge verified; no model serving or evaluation started")


def serve(project, step=650):
    """Start/reuse verified local services only; no task evaluation."""
    from contextlib import ExitStack
    from urllib.request import urlopen
    post = Path(project).resolve() / RUN / f"preparation/posttrain_step{step}_v1"
    spec = read(post / "evaluation_spec.json")
    runtime = Path(spec["project"])
    if read(post / "export_exit_status.json").get("exit_code") != 0:
        raise ValueError("Export has not exited successfully")
    receipt = read(post / "export_receipt.json")
    if receipt["checkpoint"] != spec["checkpoint"] or receipt["merged_model"] != spec["merged_model"]:
        raise ValueError("Export receipt identity differs")
    for name, expected in spec["source_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    env = dict(os.environ, OMP_NUM_THREADS="1", VLLM_USE_FLASHINFER_SAMPLER="0",
               PYTHONPATH=str(runtime / "src"), SERVED_MODEL_NAME=spec["model_tag"],
               LLM_PORT="8000", SHOPSIM_PORT="5700", SHOPSIM_ENV_SLOTS="8", SHOP_MAX_STEPS="35")
    os.environ.update({"PYTHONPATH": str(runtime / "src"), "OMP_NUM_THREADS": "1"})
    sys.path.insert(0, str(runtime / "src"))
    def listening(port):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            return False
    if not listening(5700):
        path = post / f"environment_{time.time_ns()}.log"
        with path.open("x") as log:
            process = subprocess.Popen(["bash", "scripts/start_environment.sh"], cwd=runtime, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Environment PID={process.pid}; log={path}", flush=True)
        deadline = time.monotonic()+120
        while not listening(5700):
            if process.poll() is not None:
                raise ValueError(path.read_text()[-12000:])
            if time.monotonic() >= deadline:
                raise ValueError(f"Environment startup timeout: {path}")
            time.sleep(2)
    from shopping_grpo.environment.client import ShopAgentEnv
    smoke = Path(spec["benchmarks"]["smoke3"]["path"])
    probe_task = int(json.loads(next(line for line in smoke.read_text().splitlines() if line.strip()))["task_id"])
    with ExitStack() as stack:
        slots = []
        for _ in range(8):
            client = stack.enter_context(ShopAgentEnv("http://127.0.0.1:5700"))
            if client.reset(probe_task)["environment_version"] != "shopsimulator-environment-v2.1":
                raise ValueError("Environment version differs")
            slots.append(client.env_idx)
        if len(set(slots)) != 8:
            raise ValueError("Expected eight distinct environment slots")
    print("PASS: eight environment slots checked and released", flush=True)
    model_process = None
    if not listening(8000):
        subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(); "
            "free,_=torch.cuda.mem_get_info(); print(f'GPU free: {free/2**30:.2f} GiB'); "
            "assert free>=90*2**30, 'GPU free below90 GiB'"], check=True, env=env)
        with (post / "vllm.log").open("x") as log:
            model_process = subprocess.Popen(["bash", "scripts/serve_model.sh", spec["merged_model"]],
                cwd=runtime, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        write(post / "vllm_launcher.json", {"pid": model_process.pid})
        print(f"Model PID={model_process.pid}; waiting for service readiness", flush=True)
    deadline = time.monotonic()+300
    while True:
        try:
            with urlopen("http://127.0.0.1:8000/v1/models", timeout=10) as response:
                models = json.load(response)["data"]
            break
        except (OSError, ValueError):
            if model_process is not None and model_process.poll() is not None:
                raise ValueError((post / "vllm.log").read_text()[-12000:])
            if time.monotonic() >= deadline:
                raise ValueError(f"Model startup timeout; inspect {post/'vllm.log'}")
            time.sleep(3)
    found = [model for model in models if model["id"] == spec["model_tag"]]
    if (len(found) != 1 or Path(found[0]["root"]).resolve() != Path(spec["merged_model"]).resolve()
            or int(found[0]["max_model_len"]) != 24576):
        raise ValueError(f"Port8000 model identity differs; no process was killed: {models}")
    write(post / "services_ready.json", {"passed": True, "step": step, "model": found[0]})
    print(f"PASS: step{step} service identity verified; no evaluation started")


def pipeline(project, step, shutdown_after_exit=False):
    """Export in a child, then smoke/dev/final; stop on integrity failure."""
    import fcntl
    project = Path(project).resolve()
    post = project / RUN / f"preparation/posttrain_step{step}_v1"
    spec = read(post / "evaluation_spec.json")
    started, code = time.time(), 1
    with (post / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (post / "export_receipt.json").exists():
                export_model(project, step)
            else:
                with (post / "export_driver.log").open("x") as log:
                    subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "export",
                        "--project", str(project), "--step", str(step)],
                        stdout=log, stderr=subprocess.STDOUT, check=True)
            serve(project, step)
            env = dict(os.environ, PYTHONPATH=str(Path(spec["project"]) / "src"), OMP_NUM_THREADS="1")
            results = {}
            for suite in ("smoke3", "dev50", "final200_clean"):
                print(f"START: {suite}", flush=True)
                subprocess.run([sys.executable, str(post / "run_eval.py"), suite, "--launch"],
                    cwd=spec["project"], env=env, check=True)
                out = Path(spec["evaluation_root"]) / suite
                while True:
                    status = out / "exit_status.json"
                    if status.exists():
                        try:
                            completed = read(status)
                            break
                        except json.JSONDecodeError:
                            pass
                    pid = int((out / "launcher.pid").read_text())
                    try:
                        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                        alive = str(post / "run_eval.py").encode() in args and suite.encode() in args
                    except OSError:
                        alive = False
                    if not alive:
                        # Allow an exit-status write already in progress.
                        time.sleep(1)
                        if not status.exists():
                            raise ValueError(f"{suite} process ended without exit status; inspect {out/'driver.log'}")
                    time.sleep(2)
                if completed.get("exit_code") != 0 or not read(out / "passed.json").get("passed"):
                    raise ValueError(f"{suite} did not pass; inspect {out/'driver.log'}")
                summary = read(out / "summary.json")
                results[suite] = {"strict_successes": summary["strict_successes"],
                    "expected_tasks": summary["expected_tasks"], "mean_final_reward": summary["mean_final_reward"]}
                print(f"PASS: {suite}: {results[suite]}", flush=True)
            write(post / "pipeline_result.json", results)
            code = 0
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
        finally:
            write(post / "pipeline_exit_status.json", {"exit_code": code,
                  "started_unix": started, "ended_unix": time.time()})
            print(f"PIPELINE_EXIT_CODE={code}", flush=True)
            if shutdown_after_exit:
                cancel = post / "pipeline_auto_shutdown.cancel"
                print("Pipeline exited; shutdown in60 seconds unless pipeline_auto_shutdown.cancel exists", flush=True)
                for _ in range(12):
                    if cancel.exists():
                        break
                    time.sleep(5)
                else:
                    if not cancel.exists():
                        write(post / "pipeline_shutdown_request.json", {"requested_unix": time.time(), "pipeline_exit_code": code})
                        os.sync()
                        try:
                            subprocess.run(["/bin/bash", "-c", "/usr/bin/shutdown"], check=True)
                        except (OSError, subprocess.CalledProcessError) as exc:
                            write(post / "pipeline_shutdown_error.json", {"error": str(exc)})
                            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "export", "serve", "pipeline"))
    parser.add_argument("--step", type=int, choices=(650, 750), default=650)
    parser.add_argument("--project", type=Path, default=Path(DEFAULT_PROJECT))
    parser.add_argument("--launch", action="store_true", help="Run export or pipeline in an independent background process")
    parser.add_argument("--shutdown-after-exit", action="store_true", help="For pipeline only: request shutdown60 seconds after exit")
    args = parser.parse_args()
    if args.shutdown_after_exit and args.action != "pipeline":
        raise SystemExit("--shutdown-after-exit is only supported for pipeline")
    if args.action == "prepare":
        if args.launch:
            raise SystemExit("--launch is only supported for export or pipeline")
        prepare(args.project, args.step)
    elif args.action == "serve":
        if args.launch:
            raise SystemExit("--launch is only supported for export or pipeline")
        serve(args.project, args.step)
    elif args.action == "pipeline" and not args.launch:
        pipeline(args.project, args.step, args.shutdown_after_exit)
    elif args.launch:
        post = args.project.resolve() / RUN / f"preparation/posttrain_step{args.step}_v1"
        read(post / "evaluation_spec.json")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), args.action,
                   "--project", str(args.project.resolve()), "--step", str(args.step)]
        if args.shutdown_after_exit:
            command.append("--shutdown-after-exit")
        with (post / f"{args.action}_driver.log").open("x") as log:
            process = subprocess.Popen(command,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(post / f"{args.action}_launcher.json", {"pid": process.pid, "started_unix": time.time()})
        print(f"{args.action} launched: PID={process.pid}; log={post/f'{args.action}_driver.log'}")
    else:
        import fcntl
        post = args.project.resolve() / RUN / f"preparation/posttrain_step{args.step}_v1"
        with (post / "export.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            started, code = time.time(), 1
            try:
                export_model(args.project, args.step)
                code = 0
            finally:
                write(post / "export_exit_status.json", {"exit_code": code,
                      "started_unix": started, "ended_unix": time.time()})
