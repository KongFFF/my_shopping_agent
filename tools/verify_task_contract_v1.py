"""Verify an isolated repair bundle without loading models or starting servers."""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(bundle):
    import pyarrow.parquet as pq
    bundle = bundle.resolve()
    repair = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((bundle / "materialization_receipt.json").read_text(encoding="utf-8"))
    env = json.loads((bundle / "environment.json").read_text(encoding="utf-8"))
    runtime = bundle / "runtime"
    assert repair["repair_version"] == env["task_contract_repair"]["version"] == "task-contract-v1"
    assert digest(bundle / "manifest.json") == receipt["repair_manifest_sha256"]
    assert digest(runtime / "data/items_eval_train.json") == env["product_data_sha256"] == repair["repaired_catalog_sha256"]
    for rel, expected in env["task_contract_repair"]["runtime_sha256"].items():
        assert digest(runtime / rel) == expected, rel
    runtime_paths = {"observation.py": "web_agent_site/engine/observation.py", "pack_api.py": "shop_env/pack_api.py",
                     "reward.py": "web_agent_site/engine/reward.py", "slot_lease_pool.py": "shop_env/slot_lease_pool.py",
                     "web_agent_text_env.py": "web_agent_site/envs/web_agent_text_env.py"}
    for name, rel in runtime_paths.items():
        assert digest(runtime / rel) == env["runtime_files_sha256"][name], rel
    for name, expected in (("train", repair["retained_task_ids"]), ("validation", repair["validation_task_ids_unchanged"])):
        path = bundle / f"{name}.parquet"
        assert digest(path) == receipt[f"{name}_parquet_sha256"]
        table = pq.read_table(path)
        ids = [int(x["task_id"]) for x in table["extra_info"].to_pylist()]
        assert len(ids) == len(set(ids)) == len(expected) and set(ids) == set(expected)
    index_path = runtime / "search_engine/products.sqlite3"
    with sqlite3.connect(f"{index_path.as_uri()}?mode=ro", uri=True) as connection:
        metadata = json.loads(connection.execute("SELECT payload FROM manifest").fetchone()[0])
        assert metadata["product_data_sha256"] == env["product_data_sha256"]
        assert metadata["product_count"] == 23421
    return {"bundle": str(bundle), "train_tasks": repair["retained_count"], "validation_tasks": 50,
            "runtime_and_catalog_hashes": "pass", "index_data_binding": "pass", "models_or_servers_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle), ensure_ascii=False, indent=2))
