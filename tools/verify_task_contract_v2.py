"""Check v2 artifact binding and replay retained gold with the isolated runtime."""

import argparse
import gzip
import hashlib
import json
import sqlite3
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(bundle):
    import pyarrow.parquet as pq

    root = Path(__file__).resolve().parents[1]
    package = root / "repairs/task_contract_v2"
    bundle = bundle.resolve()
    if (package / "freeze.lock.json").exists():
        from freeze_task_contract_v2 import verify_bundle_lock, verify_lock
        verify_bundle_lock(bundle, verify_lock(package))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["repair_version"] == "task-contract-v2"
    assert digest(bundle / "manifest.json") == digest(package / "manifest.json")
    for name, expected in manifest["artifact_sha256"].items():
        assert digest(package / name) == expected, name
    for name, expected in manifest["source_tools_sha256"].items():
        assert digest(root / "tools" / name) == expected, name
    receipt = json.loads((bundle / "materialization_receipt.json").read_text(encoding="utf-8"))
    env = json.loads((bundle / "environment.json").read_text(encoding="utf-8"))
    runtime = bundle / "runtime"
    assert env["task_contract_repair"]["version"] == manifest["repair_version"]
    assert digest(bundle / "manifest.json") == receipt["repair_manifest_sha256"]
    assert digest(runtime / "data/items_eval_train.json") == env["product_data_sha256"] == manifest["repaired_catalog_sha256"]
    for rel, expected in env["task_contract_repair"]["runtime_sha256"].items():
        assert digest(runtime / rel) == expected == manifest["artifact_sha256"]["runtime/" + Path(rel).name], rel
    paths = {"observation.py": "web_agent_site/engine/observation.py", "pack_api.py": "shop_env/pack_api.py",
             "reward.py": "web_agent_site/engine/reward.py", "slot_lease_pool.py": "shop_env/slot_lease_pool.py",
             "web_agent_text_env.py": "web_agent_site/envs/web_agent_text_env.py"}
    for name, rel in paths.items():
        assert digest(runtime / rel) == env["runtime_files_sha256"][name], rel
    original = root / "upstream/shopping-grpo-longhorizon/data/grpo"
    original_train = pq.read_table(original / "train.parquet")
    source_rows = {int(row["extra_info"]["task_id"]): row for row in original_train.to_pylist()}
    for name, ids in (("train", manifest["retained_task_ids"]), ("validation", manifest["validation_task_ids_unchanged"])):
        path = bundle / f"{name}.parquet"
        assert digest(path) == receipt[f"{name}_parquet_sha256"]
        table = pq.read_table(path)
        rows = table.to_pylist()
        actual = [int(row["extra_info"]["task_id"]) for row in rows]
        assert len(actual) == len(set(actual)) == len(ids) and set(actual) == set(ids)
        if name == "train":
            assert table.schema == original_train.schema
            assert rows == [source_rows[tid] for tid in actual], "An original training row was rewritten"
        else:
            assert digest(path) == digest(original / "validation.parquet")
    with sqlite3.connect(f"{(runtime / 'search_engine/products.sqlite3').as_uri()}?mode=ro", uri=True) as connection:
        metadata = json.loads(connection.execute("SELECT payload FROM manifest").fetchone()[0])
        assert metadata["product_count"] == 23421
        assert metadata["product_data_sha256"] == env["product_data_sha256"]
    # Fresh-process imports are deliberate: do not replay using builder-injected
    # modules, which could hide a broken materialized import chain.
    if any(name in sys.modules for name in ("web_agent_site.engine.reward", "web_agent_site.engine.reward_features")):
        raise RuntimeError("Run the verifier in a fresh Python process")
    sys.path.insert(0, str(runtime))
    from web_agent_site.engine.reward_features import compile_reward_features
    from web_agent_site.engine.reward import evaluate_purchase
    catalog = json.loads((runtime / "data/items_eval_train.json").read_text(encoding="utf-8"))
    with gzip.open(root / "upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz", "rb") as handle:
        original_bytes = handle.read()
    assert hashlib.sha256(original_bytes).hexdigest() == manifest["original_catalog_sha256"]
    original_catalog = json.loads(original_bytes)
    assert len(original_catalog) == len(catalog) == 23421
    for old, new in zip(original_catalog, catalog):
        assert {key: value for key, value in old.items() if key != "instructions"} == {key: value for key, value in new.items() if key != "instructions"}
        assert len(old.get("instructions", [])) == len(new.get("instructions", []))
        for old_instruction, new_instruction in zip(old.get("instructions", []), new.get("instructions", [])):
            assert {key: value for key, value in old_instruction.items() if key != "instruction_options"} == {key: value for key, value in new_instruction.items() if key != "instruction_options"}
    original_goals = [(p, i) for p in original_catalog for i in p.get("instructions", []) if i.get("attributes")]
    overrides = {item["task_id"]: item for item in json.loads((package / "annotation_overrides.json").read_text(encoding="utf-8"))}
    goals = [(p, i) for p in catalog for i in p.get("instructions", []) if i.get("attributes")]
    assert len(goals) == 23421
    for tid, ((_, old), (_, new)) in enumerate(zip(original_goals, goals)):
        if tid in overrides:
            assert old["instruction_options"] == overrides[tid]["instruction_options_before"]
            assert new["instruction_options"] == overrides[tid]["instruction_options_after"]
            assert tid in set(manifest["retained_task_ids"]) | set(manifest["quarantined_task_ids"])
        else:
            assert old["instruction_options"] == new["instruction_options"]
    passed = 0
    for tid in manifest["retained_task_ids"]:
        product, instruction = goals[tid]
        goal = {"asin": product["asin"], **compile_reward_features(instruction, product)}
        selected = {v["source_axis"]: v["value"] for v in goal["required_options_by_key"].values()}
        result = evaluate_purchase(product, goal, selected_options=selected)
        assert result.reward_type == "gold_purchase" and result.reward_valid, (tid, result.to_dict())
        passed += 1
    for query, price, expected in (("价格50元以上", 47, "fail"), ("价格50元以上", 50, "pass"),
                                   ("价格高于250元，不超300元", 250, "fail"),
                                   ("价格高于250元，不超300元", 300, "pass"),
                                   ("预算100元出头", 109, "pass"), ("预算100元出头", 109.01, "fail")):
        product = {"asin": "probe", "category": "测试", "title": "测试", "attribute": [], "customization_options": {}}
        instruction = {"instruction": query, "attributes": [], "instruction_options": []}
        goal = {"asin": "probe", **compile_reward_features(instruction, product)}
        result = evaluate_purchase(product, goal, selected_options={}, price=price)
        assert result.hard_gates["budget"]["status"] == expected, (query, price)
    return {"bundle": str(bundle), "train_tasks": passed, "validation_tasks_unchanged": 50,
            "original_rows_and_schema": "pass", "runtime_and_catalog_hashes": "pass", "index_data_binding": "pass",
            "only_approved_private_annotations_changed": "pass",
            "actual_runtime_gold_replay": f"{passed}/{passed}", "lower_and_chutou_counterexamples": "pass",
            "models_or_servers_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(verify(args.bundle), ensure_ascii=False, indent=2))
