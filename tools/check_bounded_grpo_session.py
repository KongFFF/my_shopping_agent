"""GPU/resource, environment-slot and Ray checks; never launch model training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from contextlib import ExitStack

from prepare_bounded_grpo_resume import DEFAULT_PROJECT, RUN, read, write


def listening():
    try:
        with socket.create_connection(("127.0.0.1", 5700), timeout=2):
            return True
    except OSError:
        return False


def check(project, step, target=650):
    if (step, target) not in ((500, 650), (600, 650), (650, 800)):
        raise ValueError("Unsupported continuation bounds")
    project = Path(project).resolve()
    prep = project / RUN / f"preparation/formal_resume{step}_to{target}_v1"
    spec = read(prep / "launch_spec.json")
    runtime = Path(spec["project"])
    if spec["resume_step"] != step or spec["total_updates"] != target:
        raise ValueError("Unexpected recovery bounds")
    for name in ("driver.log", "train.log", "train.pid", "exit_status.json"):
        if (prep / name).exists():
            raise ValueError(f"Session already has training records; inspect instead of checking/restarting: {name}")
    for name in ("resource_check.json", "runtime_preflight.json", "environment_preflight.json"):
        (prep / name).unlink(missing_ok=True)
    for name, expected in spec["source_sha256"].items():
        h = hashlib.sha256()
        with Path(name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError(f"Source changed: {name}")
    env = dict(os.environ, **spec["env"])
    os.environ.update(spec["env"])
    sys.path.insert(0, str(runtime / "src"))
    free = shutil.disk_usage(runtime).free
    print(f"Disk available: {free/2**30:.2f} GiB", flush=True)
    if free < spec["required_free_bytes"]:
        raise ValueError("Insufficient disk budget")
    limit = Path("/sys/fs/cgroup/memory.max")
    if limit.exists() and limit.read_text().strip() != "max":
        memory = int(limit.read_text())
        print(f"CPU memory quota: {memory/2**30:.2f} GiB", flush=True)
        if memory < 110 * 2**30:
            raise ValueError("CPU memory quota below the original 110 GiB gate")
    subprocess.run([sys.executable, "-c",
        "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "free,total=torch.cuda.mem_get_info(); "
        "print(f'GPU available/total: {free/2**30:.2f}/{total/2**30:.2f} GiB'); "
        "assert free>=90*2**30, 'GPU available memory below 90 GiB'"], check=True, env=env)
    write(prep / "resource_check.json", {"passed": True, "resume_step": step, "target_step": target})
    log_path = prep / f"runtime_preflight_{time.time_ns()}.log"
    with log_path.open("x") as log:
        result = subprocess.run(spec["preflight"], cwd=runtime, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    print(log_path.read_text(), flush=True)
    if result.returncode:
        raise ValueError(f"Runtime preflight failed: {result.returncode}; log={log_path}")
    write(prep / "runtime_preflight.json", {"passed": True, "resume_step": step, "target_step": target})
    url = env["SHOPSIM_BASE_URL"]
    if url != "http://127.0.0.1:5700":
        raise ValueError(f"Unexpected environment URL: {url}")
    receipt = prep / "environment_process.json"
    if listening():
        if not receipt.is_file():
            raise ValueError("Port5700 is occupied by a service not recorded by this session; not reusing or killing it")
        pid = int(read(receipt)["pid"])
        expected_cwd = (runtime / "environments/ShopSimulator/shop_env/shop_env").resolve()
        if Path(f"/proc/{pid}/cwd").resolve() != expected_cwd or b"pack_api.py" not in Path(f"/proc/{pid}/cmdline").read_bytes():
            raise ValueError("Recorded service PID does not identify this environment")
        print(f"Reusing this session's environment: PID={pid}", flush=True)
    else:
        log_path = prep / f"environment_{time.time_ns()}.log"
        environment_env = dict(env, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
                               SHOPSIM_PORT="5700", SHOPSIM_ENV_SLOTS="8", SHOP_MAX_STEPS="35")
        with log_path.open("x") as log:
            process = subprocess.Popen(["bash", "scripts/start_environment.sh"], cwd=runtime,
                                       env=environment_env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(receipt, {"pid": process.pid, "log": str(log_path)})
        print(f"Environment started: PID={process.pid}; log={log_path}", flush=True)
        deadline = time.monotonic() + 120
        while not listening():
            if process.poll() is not None:
                raise ValueError(f"Environment exited: {log_path.read_text()}")
            if time.monotonic() >= deadline:
                raise ValueError(f"Environment startup timed out: {log_path}")
            time.sleep(2)
    from shopping_grpo.environment.client import ShopAgentEnv
    task = int(json.loads((runtime / "data/grpo/train.jsonl").read_text().splitlines()[0])["task_id"])
    with ExitStack() as stack:
        slots = []
        for _ in range(8):
            client = stack.enter_context(ShopAgentEnv(url))
            response = client.reset(task)
            if response["environment_version"] != "shopsimulator-environment-v2.1":
                raise ValueError("Environment version differs")
            slots.append(client.env_idx)
        if len(set(slots)) != 8:
            raise ValueError(f"Expected eight distinct slots: {slots}")
    print("PASS: eight environment slots checked and released", flush=True)
    import ray
    ray.init(address="local", num_cpus=1, num_gpus=0, object_store_memory=128 * 1024**2,
             include_dashboard=False, runtime_env={"env_vars": spec["env"]})
    try:
        @ray.remote(num_cpus=0.1)
        def probe(task_id, source):
            from pathlib import Path
            import os
            import shopping_grpo.environment.client as module
            if not Path(module.__file__).resolve().is_relative_to(Path(source).resolve()):
                raise ValueError(f"Worker imported a different runtime: {module.__file__}")
            with module.ShopAgentEnv(os.environ["SHOPSIM_BASE_URL"]) as client:
                result = client.reset(task_id)
                assert result["environment_version"] == "shopsimulator-environment-v2.1"
            assert client.env_idx is None
            return "PASS: Ray worker uses isolated source; reset and release succeeded"
        print(ray.get(probe.remote(task, str(runtime / "src")), timeout=120), flush=True)
    finally:
        ray.shutdown()
    write(prep / "environment_preflight.json", {"passed": True, "resume_step": step, "target_step": target,
          "runtime_project": str(runtime), "base_url": url, "checked_slots": 8})
    print(f"PASS: step{step}->{target} session preflight complete; environment retained; no training started")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(DEFAULT_PROJECT))
    parser.add_argument("--resume-step", type=int, choices=(500, 600, 650), required=True)
    parser.add_argument("--target-step", type=int, choices=(650, 800), default=650)
    args = parser.parse_args()
    check(args.project, args.resume_step, args.target_step)
