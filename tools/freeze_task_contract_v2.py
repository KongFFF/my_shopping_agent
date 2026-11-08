"""Seal/check the 928-task v2 contract; a different contract needs a new version."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "repairs/task_contract_v2"
LOCK_NAME = "freeze.lock.json"
FREEZE_ID = "task-contract-v2-928-20261029"
SHOP_SOURCE = ROOT / "upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env"


def digest(path, *, normalized=False):
    content = path.read_text(encoding="utf-8").encode("utf-8") if normalized else path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def local_path(root, name):
    root = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Frozen file escapes its root: {name}")
    return path


def verify_lock(package=PACKAGE):
    package = Path(package).resolve()
    lock = json.loads((package / LOCK_NAME).read_text(encoding="utf-8"))
    if lock.get("freeze_id") != FREEZE_ID or lock.get("status") != "FROZEN":
        raise ValueError("Unrecognized v2 freeze lock")
    for name, expected in lock["contract_files_sha256"].items():
        if digest(local_path(package, name)) != expected:
            raise ValueError(f"Frozen v2 artifact changed: {name}; use a new contract version")
    for name, expected in lock["rebuild_inputs_sha256"].items():
        if digest(local_path(ROOT, name)) != expected:
            raise ValueError(f"Frozen v2 rebuild input changed: {name}; use a new contract version")
    for name, expected in lock["normalized_sources_sha256"].items():
        if digest(local_path(ROOT, name), normalized=True) != expected:
            raise ValueError(f"Frozen v2 source changed: {name}; use a new contract version")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    train = [int(json.loads(line)["task_id"]) for line in (package / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    quarantined = [json.loads(line)["task_id"] for line in (package / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()]
    validation = [int(json.loads(line)["task_id"]) for line in (package / "validation.jsonl").read_text(encoding="utf-8").splitlines()]
    if not (manifest["retained_count"] == len(train) == len(set(train)) == lock["train_count"] == 928
            and manifest["quarantined_count"] == len(quarantined) == len(set(quarantined)) == lock["quarantine_count"] == 72
            and len(validation) == len(set(validation)) == lock["validation_count"] == 50
            and set(train) == set(manifest["retained_task_ids"])
            and set(quarantined) == set(manifest["quarantined_task_ids"])
            and not set(train) & (set(quarantined) | set(validation))):
        raise ValueError("Frozen v2 split is inconsistent")
    return lock


def verify_bundle_lock(bundle, lock):
    runtime = Path(bundle).resolve() / "runtime"
    for name, expected in lock["runtime_files_normalized_sha256"].items():
        if digest(local_path(runtime, name), normalized=True) != expected:
            raise ValueError(f"Frozen scoring/environment dependency changed: {name}")


def check_candidate(artifacts, lock):
    """Reject a changed rebuild BEFORE any frozen package file is written."""
    if set(artifacts) != set(lock["contract_files_sha256"]):
        raise ValueError("Rebuild changes the frozen artifact set; use a new version")
    for name, content in artifacts.items():
        if hashlib.sha256(content).hexdigest() != lock["contract_files_sha256"][name]:
            raise ValueError(f"Rebuild would replace frozen v2 artifact {name}; use a new version")


def create_lock(package=PACKAGE, *, bundle):
    package = Path(package).resolve()
    if (package / LOCK_NAME).exists():
        return verify_lock(package)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["repair_version"], manifest["retained_count"], manifest["quarantined_count"]) != ("task-contract-v2", 928, 72):
        raise ValueError("Only the approved 928/72 v2 candidate can be sealed")
    contract = {"manifest.json": digest(package / "manifest.json"), **manifest["artifact_sha256"]}
    for name, expected in contract.items():
        if digest(local_path(package, name)) != expected:
            raise ValueError(f"Cannot seal corrupt artifact: {name}")
    v1 = ROOT / "repairs/task_contract_v1"
    prior = json.loads((v1 / "manifest.json").read_text(encoding="utf-8"))
    inputs = ["repairs/task_contract_v1/manifest.json", "repairs/task_contract_v1/first_pass_review.jsonl",
              *["repairs/task_contract_v1/" + name for name in prior["artifact_sha256"]],
              "upstream/shopping-grpo-longhorizon/data/grpo/train.parquet",
              "upstream/shopping-grpo-longhorizon/data/grpo/validation.parquet",
              "upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"]
    runtime_paths = sorted({path.relative_to(SHOP_SOURCE).as_posix()
                            for path in (SHOP_SOURCE / "web_agent_site/engine").glob("*.py")}
                           | {path.relative_to(SHOP_SOURCE).as_posix() for path in (SHOP_SOURCE / "configs").rglob("*.json")}
                           | {"shop_env/pack_api.py", "shop_env/slot_lease_pool.py", "web_agent_site/envs/web_agent_text_env.py"})
    sources = ["tools/build_task_contract_v1.py", "tools/task_contract_v1.py", "tools/build_task_contract_v2.py",
               "tools/task_contract_v2.py", "tools/verify_task_contract_v2.py", "tools/freeze_task_contract_v2.py",
               "upstream/shopping-grpo-longhorizon/data/grpo/train.jsonl",
               "upstream/shopping-grpo-longhorizon/data/grpo/validation.jsonl",
               "upstream/shopping-grpo-longhorizon/data/evaluation/tasks.jsonl",
               *[(SHOP_SOURCE / name).relative_to(ROOT).as_posix() for name in runtime_paths]]
    lock = {"freeze_id": FREEZE_ID, "status": "FROZEN", "frozen_on": "2026-10-29", "timezone": "Asia/Shanghai",
            "repair_version": "task-contract-v2", "train_count": 928, "quarantine_count": 72, "validation_count": 50,
            "pinned_upstream_base": manifest["pinned_upstream"], "policy": manifest["policy"],
            "contract_files_sha256": contract,
            "rebuild_inputs_sha256": {name: digest(local_path(ROOT, name)) for name in inputs},
            "normalized_sources_sha256": {name: digest(local_path(ROOT, name), normalized=True) for name in sources},
            "runtime_files_normalized_sha256": {name: digest(local_path(Path(bundle) / "runtime", name), normalized=True) for name in runtime_paths},
            "original_catalog_uncompressed_sha256": manifest["original_catalog_sha256"],
            "repaired_catalog_sha256": manifest["repaired_catalog_sha256"],
            "reference_bundle": "outputs/task_contract_v2_frozen",
            "future_changes": "Keep this dataset/reward contract fixed for sampling/algorithm experiments. Changed splits, annotations or scoring require a new contract version.",
            "validation": "Original Dev50 and Final IDs stay fixed. Frozen status is not an AutoDL deployment or a training run."}
    with (package / LOCK_NAME).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(lock, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    verify_bundle_lock(bundle, lock)
    return verify_lock(package)


def verify_bundle(bundle):
    command = [sys.executable, str(ROOT / "tools/verify_task_contract_v2.py"), str(bundle)]
    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify an existing lock; write nothing")
    parser.add_argument("--bundle", type=Path, help="Also verify/replay an isolated materialized bundle")
    args = parser.parse_args()
    if not args.check and not args.bundle:
        parser.error("Creating a freeze requires --bundle for full isolated-runtime verification")
    if (PACKAGE / LOCK_NAME).exists():
        verify_lock()
    verification = verify_bundle(args.bundle) if args.bundle else None
    lock = verify_lock() if args.check else create_lock(bundle=args.bundle)
    if args.bundle:
        verify_bundle_lock(args.bundle, lock)
    print(json.dumps({"freeze_id": lock["freeze_id"], "status": lock["status"], "train_tasks": lock["train_count"],
                      "quarantined_tasks": lock["quarantine_count"], "validation_tasks": lock["validation_count"],
                      "lock_sha256": digest(PACKAGE / LOCK_NAME), "bundle_verification": verification}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
