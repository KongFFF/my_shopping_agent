"""Build the approved v2 contract without modifying v1 or the upstream checkout."""

import argparse
import ast
import copy
import difflib
import gzip
import inspect
import json
import shutil
import sys
import types
from collections import Counter
from pathlib import Path

import build_task_contract_v1 as base
import task_contract_v2 as rules

ROOT, UP, ENGINE = base.ROOT, base.UP, base.ENGINE
PACKAGE = ROOT / "repairs/task_contract_v2"
V1 = ROOT / "repairs/task_contract_v1"
VERSION = "task-contract-v2"


def rename_definition(source, before, after):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == before]
    assert len(nodes) == 1, before
    lines = source.splitlines(keepends=True)
    line = nodes[0].lineno - 1
    lines[line] = lines[line].replace(before, after, 1)
    return "".join(lines)


def runtime_sources():
    v1 = base.runtime_sources()
    approved_v1 = json.loads((V1 / "manifest.json").read_text(encoding="utf-8"))
    for name, (_, source) in v1.items():
        assert base.sha(source.encode("utf-8")) == approved_v1["artifact_sha256"][f"runtime/{name}"], f"v1 source drift: {name}"
    constraints = v1["constraints.py"][1]
    constraints = rename_definition(constraints, "explicit_budget_from_instruction", "_v1_explicit_budget_v2")
    constraints += "\n\nimport unicodedata\nfrom decimal import Decimal, InvalidOperation\n"
    constraints += f"MONEY_TOKEN_V2 = {rules.MONEY_TOKEN_V2!r}\n\n"
    for function in (rules.money_number_v2, rules.chutou_upper_v2, rules.colloquial_range_v2, rules.budget_details_v2):
        constraints += inspect.getsource(function) + "\n\n"
    constraints += '''def explicit_budget_from_instruction(instruction):
    details = budget_details_v2(instruction)
    return details["upper"] if details["status"] != "review_required" else _v1_explicit_budget_v2(instruction)
'''
    features = rename_definition(v1["reward_features.py"][1], "compile_reward_features", "_baseline_compile_reward_features_v2")
    features += '''

def compile_reward_features(instruction_record, target_product):
    from web_agent_site.engine.constraints import budget_details_v2
    result = _baseline_compile_reward_features_v2(instruction_record, target_product)
    instruction = instruction_record if isinstance(instruction_record, dict) else {}
    details = budget_details_v2(instruction.get("instruction", ""))
    result.update({"budget_contract_v2": details, "price_lower": details["lower"], "price_upper": details["upper"]})
    return result
'''
    old_reward = (ENGINE / "reward.py").read_text(encoding="utf-8")
    reward = rename_definition(old_reward, "_price_gate", "_baseline_price_gate_v2")
    needle = '"budget": _price_gate(price_resolution, goal),'
    assert reward.count(needle) == 1
    reward = reward.replace(needle, '"budget": _price_gate(price_resolution, goal, product=product, selected_options=selected_options),')
    reward += "\n\nimport re\nimport unicodedata\nfrom decimal import Decimal, InvalidOperation\n\n"
    reward += inspect.getsource(rules.price_gate_v2)
    reward += '''

def _price_gate(price_resolution, goal, *, product=None, selected_options=None):
    return price_gate_v2(price_resolution, goal, product=product, selected_options=selected_options)
'''
    sources = {"constraints.py": (v1["constraints.py"][0], constraints),
               "reward_features.py": (v1["reward_features.py"][0], features), "reward.py": (old_reward, reward)}
    for name, (_, source) in sources.items():
        compile(source, name, "exec")
    return sources


def load_runtime(sources):
    loaded = {}
    for name in ("constraints.py", "reward_features.py", "reward.py"):
        canonical = "web_agent_site.engine." + name.removesuffix(".py")
        module = types.ModuleType(canonical)
        sys.modules[canonical] = module
        exec(compile(sources[name][1], canonical, "exec"), module.__dict__)
        loaded[name] = module
    return loaded


def verify_v1():
    manifest = json.loads((V1 / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifact_sha256"].items():
        assert base.sha((V1 / name).read_bytes()) == expected, f"v1 changed: {name}"
    return manifest


def manual_holds():
    """Review-based data quarantine, never runtime scoring by task ID."""
    records = [json.loads(line) for line in (V1 / "first_pass_review.jsonl").read_text(encoding="utf-8").splitlines()]
    holds = {}
    for row in records:
        if row["primary_category"] == "price_or_spec_metadata_needed":
            holds[row["task_id"]] = "approved_direct_isolation_missing_metadata"
    assert len(holds) == 35
    holds.update({7714: "ready_to_use_vs_unprocessed_unverified", 21254: "red_query_vs_purple_annotation",
                  11055: "deposit_transaction_unsupported"})
    # Secondary evidence gaps explicitly found in the first-pass cards remain
    # unresolved even if a new budget or glyph rule would make gold pass.
    holds.update({3597: "selected_axis_prices_125_vs_145_unverified", 7920: "selected_axis_prices_25_vs_36_unverified",
                  20808: "selected_axis_prices_256_vs_150_unverified", 21105: "capacity_and_price_roles_unverified",
                  22407: "selected_axis_prices_399_vs_1439_unverified", 19581: "selected_axis_prices_29_vs_79_unverified",
                  8181: "high_speed_bur_package_inclusion_unverified", 22300: "root_measurement_definition_unverified"})
    return holds, records


def materialize(target, artifacts, catalog, catalog_bytes, sources, retained, search_sha):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if int(pa.__version__.split(".")[0]) < 25:
        raise SystemExit("Materialization requires PyArrow >=25; the small repair package was generated.")
    target = target.resolve()
    if target.exists():
        raise SystemExit(f"Refusing to overwrite {target}")
    table = pq.read_table(UP / "data/grpo/train.parquet")
    ids = [int(item["task_id"]) for item in table["extra_info"].to_pylist()]
    assert len(ids) == len(set(ids)) == 1000
    selected = [index for index, tid in enumerate(ids) if tid in set(retained)]
    filtered = table.take(pa.array(selected, type=pa.int64()))
    target.mkdir(parents=True)
    pq.write_table(filtered, target / "train.parquet")
    (target / "validation.parquet").write_bytes((UP / "data/grpo/validation.parquet").read_bytes())
    (target / "manifest.json").write_bytes(artifacts["manifest.json"])
    runtime = target / "runtime"
    shutil.copytree(UP / "environments/ShopSimulator/shop_env", runtime,
                    ignore=shutil.ignore_patterns("data", "search_engine", "__pycache__", ".ipynb_checkpoints", ".pytest_cache", "*.log", ".venv*", ".git", "models", "checkpoints", "outputs"))
    (runtime / "data").mkdir(exist_ok=True)
    (runtime / "data/items_eval_train.json").write_bytes(catalog_bytes)
    for name, (_, source) in sources.items():
        (runtime / "web_agent_site/engine" / name).write_bytes(source.encode("utf-8"))
    index = base.build_index(catalog, runtime / "search_engine/products.sqlite3", product_data_sha256=base.sha(catalog_bytes))
    (target / "index_manifest.json").write_bytes(base.json_bytes(index))
    env = json.loads((UP / "data/environment.json").read_text(encoding="utf-8"))
    env["product_data_sha256"] = base.sha(catalog_bytes)
    env["task_contract_repair"] = {"version": VERSION, "runtime_sha256": {
        f"web_agent_site/engine/{name}": base.sha(source.encode("utf-8")) for name, (_, source) in sources.items()},
        "search_documents_sha256_unchanged": search_sha,
        "annotation_overrides_sha256": base.sha(artifacts["annotation_overrides.json"])}
    paths = {"observation.py": "web_agent_site/engine/observation.py", "pack_api.py": "shop_env/pack_api.py",
             "reward.py": "web_agent_site/engine/reward.py", "slot_lease_pool.py": "shop_env/slot_lease_pool.py",
             "web_agent_text_env.py": "web_agent_site/envs/web_agent_text_env.py"}
    env["runtime_files_sha256"] = {name: base.sha((runtime / path).read_bytes()) for name, path in paths.items()}
    (target / "environment.json").write_bytes(base.json_bytes(env))
    receipt = {"train_rows": len(filtered), "validation_rows": 50,
               "train_parquet_sha256": base.sha((target / "train.parquet").read_bytes()),
               "validation_parquet_sha256": base.sha((target / "validation.parquet").read_bytes()),
               "product_data_sha256": base.sha(catalog_bytes), "repair_manifest_sha256": base.sha(artifacts["manifest.json"])}
    assert pq.read_table(target / "train.parquet").schema == table.schema
    assert pq.read_table(target / "train.parquet").to_pylist() == filtered.to_pylist()
    (target / "materialization_receipt.json").write_bytes(base.json_bytes(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize", type=Path, help="Create a NEW isolated bundle; requires PyArrow >=25")
    args = parser.parse_args()
    from freeze_task_contract_v2 import LOCK_NAME, check_candidate, verify_lock
    frozen = verify_lock(PACKAGE) if (PACKAGE / LOCK_NAME).exists() else None
    v1_manifest = verify_v1()
    holds, review_records = manual_holds()
    prior = {r["task_id"]: r for r in (json.loads(line) for line in (V1 / "all_train_audit.jsonl").read_text(encoding="utf-8").splitlines())}
    sources = runtime_sources()
    runtime = load_runtime(sources)
    features, reward = runtime["reward_features.py"], runtime["reward.py"]
    with gzip.open(UP / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz", "rb") as handle:
        original_bytes = handle.read()
    assert base.sha(original_bytes) == v1_manifest["original_catalog_sha256"]
    catalog = json.loads(original_bytes)
    goals = [(p, i) for p in catalog for i in p.get("instructions", []) if i.get("attributes")]
    assert len(goals) == len(catalog) == 23421
    train_ids = base.read_ids(UP / "data/grpo/train.jsonl")
    val_ids = base.read_ids(UP / "data/grpo/validation.jsonl")
    final_ids = base.read_ids(UP / "data/evaluation/tasks.jsonl")  # Split IDs only.
    assert set(train_ids) == set(prior) and not set(train_ids) & (set(val_ids) | set(final_ids))
    public_before = base.sha(json.dumps([base.product_fields(p) for p in catalog], ensure_ascii=False, sort_keys=True).encode())
    overrides, audits = [], []

    def evaluate(product, instruction):
        goal = {"asin": product["asin"], **features.compile_reward_features(instruction, product)}
        selected = {v["source_axis"]: v["value"] for v in goal["required_options_by_key"].values()}
        return goal, reward.evaluate_purchase(product, goal, selected_options=selected), selected

    for tid in train_ids:
        product, instruction = goals[tid]
        assert instruction["instruction"] == prior[tid]["query"] and product["asin"] == prior[tid]["asin"]
        before = copy.deepcopy(instruction)
        changed = []
        for index, value in enumerate(instruction.get("instruction_options") or []):
            resolved = rules.resolve_damaged_option_v2(value, product, instruction["instruction"])
            if resolved:
                axis, replacement = resolved
                instruction["instruction_options"][index] = replacement
                changed.append({"index": index, "before": value, "after": replacement, "source_axis": axis})
        if changed:
            overrides.append({"task_id": tid, "asin": product["asin"], "instruction": instruction["instruction"], "changes": changed,
                              "instruction_options_before": before["instruction_options"], "instruction_options_after": instruction["instruction_options"]})
        goal, result, selected = evaluate(product, instruction)
        details = goal["budget_contract_v2"]
        reasons = []
        if tid in holds:
            reasons.append(holds[tid])
        if details["status"] == "review_required":
            reasons.append(details["reason"])
        if goal["unresolved_option_requirements"]:
            reasons.append("unresolved_option_annotation")
        if result.reward_type != "gold_purchase":
            if result.reward_type == "reward_unverifiable":
                price = result.evidence["price_resolution"]
                reasons.append("budget_contract_unverifiable" if price.get("status") == "pass" else price["method"])
            else:
                reasons.append(result.reward_type)
        reasons = list(dict.fromkeys(reasons))
        keep = not reasons
        audits.append({"task_id": tid, "asin": product["asin"], "query": instruction["instruction"],
                       "was_original_90": prior[tid]["was_original_90"], "v1_status": prior[tid]["status"],
                       "status": "retained" if keep else "quarantined", "reasons": reasons,
                       "old_reward_type": prior[tid]["old_reward_type"], "v1_reward_type": prior[tid]["new_reward_type"],
                       "new_reward_type": result.reward_type, "old_budget": prior[tid]["old_budget"],
                       "v1_budget": prior[tid]["new_budget"], "new_budget": goal["price_upper"], "new_lower": goal["price_lower"],
                       "budget_parse": details, "annotation_changes": changed, "new_reward_evidence": result.to_dict(),
                       "selected_gold_options": selected})
    retained = [a["task_id"] for a in audits if a["status"] == "retained"]
    quarantined = [a["task_id"] for a in audits if a["status"] == "quarantined"]
    assert len(retained) + len(quarantined) == 1000 and not set(retained) & set(quarantined)
    for row in review_records:
        if row["primary_category"] in {"frozen_tolerance_boundary", "target_budget_conflict"}:
            item = next(a for a in audits if a["task_id"] == row["task_id"])
            assert item["new_reward_evidence"]["hard_gates"]["budget"]["status"] == "fail", item
    public_after = base.sha(json.dumps([base.product_fields(p) for p in catalog], ensure_ascii=False, sort_keys=True).encode())
    assert public_before == public_after
    validation = []
    for tid in val_ids:
        product, instruction = goals[tid]
        goal, result, _ = evaluate(product, instruction)
        validation.append({"task_id": tid, "new_reward_type": result.reward_type, "budget_parse": goal["budget_contract_v2"],
                           "new_reward_evidence": result.to_dict()})
    catalog_bytes = json.dumps(catalog, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    artifacts = {"train.jsonl": "".join(json.dumps({"task_id": tid}) + "\n" for tid in retained).encode(),
                 "validation.jsonl": (UP / "data/grpo/validation.jsonl").read_bytes(),
                 "annotation_overrides.json": base.json_bytes(overrides), "validation_audit.json": base.json_bytes(validation),
                 "manual_holds.json": base.json_bytes(holds),
                 "all_train_audit.jsonl": "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audits).encode("utf-8"),
                 "quarantine.jsonl": "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audits if a["status"] == "quarantined").encode("utf-8")}
    patch = []
    for name, (old, new) in sources.items():
        artifacts[f"runtime/{name}"] = new.encode("utf-8")
        relative = f"environments/ShopSimulator/shop_env/web_agent_site/engine/{name}"
        patch.extend(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True), fromfile=f"a/{relative}", tofile=f"b/{relative}"))
    artifacts["runtime.patch"] = "".join(patch).encode("utf-8")
    recovered = [a["task_id"] for a in audits if a["v1_status"] == "quarantined" and a["status"] == "retained"]
    newly_held = [a["task_id"] for a in audits if a["v1_status"] == "retained" and a["status"] == "quarantined"]
    manifest = {"repair_version": VERSION, "pinned_upstream": v1_manifest["pinned_upstream"], "original_train_count": 1000,
                "retained_count": len(retained), "quarantined_count": len(quarantined), "v1_retained_count": v1_manifest["retained_count"],
                "recovered_from_v1_count": len(recovered), "recovered_from_v1_task_ids": recovered,
                "newly_quarantined_from_v1_count": len(newly_held), "newly_quarantined_from_v1_task_ids": newly_held,
                "original_90_recovered_count": sum(a["was_original_90"] and a["status"] == "retained" for a in audits),
                "retained_task_ids": retained, "quarantined_task_ids": quarantined,
                "annotation_repaired_tasks": len(overrides), "quarantine_reason_counts": dict(Counter(r for a in audits if a["status"] == "quarantined" for r in a["reasons"])),
                "validation_static_gold_count": sum(row["new_reward_type"] == "gold_purchase" for row in validation),
                "policy": {"chutou": "integer CNY: two digits +4; d>=3 digits +(10^(d-2)-1); upper inclusive; no minimum spend",
                           "chutou_examples": {"20": 24, "65": 69, "100": 109, "420": 429, "1000": 1099, "2000": 2099, "10000": 10999},
                           "ordinary_approximate": "10% upper allowance; explicit hard caps win", "spoken_ranges": "no stacked approximate tolerance; budget intent has no inferred minimum",
                           "missing_metadata": "direct quarantine; no repricing", "deposit": "quarantine"},
                "source_v1_manifest_sha256": base.sha((V1 / "manifest.json").read_bytes()),
                "source_first_pass_sha256": base.sha((V1 / "first_pass_review.jsonl").read_bytes()),
                "source_tools_sha256": {path.name: base.sha(path.read_bytes()) for path in (Path(__file__), Path(rules.__file__))},
                "source_runtime_normalized_sha256": {name: base.sha(old.encode()) for name, (old, _) in sources.items()},
                "original_catalog_sha256": base.sha(original_bytes), "repaired_catalog_sha256": base.sha(catalog_bytes),
                "search_documents_sha256_unchanged": public_before, "validation_task_ids_unchanged": val_ids, "final_ids_overlap": [],
                "artifact_sha256": {name: base.sha(content) for name, content in artifacts.items()},
                "limits": ["Static gold feasibility is not a complete semantic audit.", "Review-required declared budgets fail closed in the v2 runtime.",
                           "Retained IDs, private annotations and all three runtime replacements form one versioned contract.", "No models, training or Final evaluation started."]}
    artifacts["manifest.json"] = base.json_bytes(manifest)
    if frozen:
        check_candidate(artifacts, frozen)
    else:
        PACKAGE.mkdir(parents=True, exist_ok=True)
        for name, content in artifacts.items():
            path = PACKAGE / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    if args.materialize:
        materialize(args.materialize, artifacts, catalog, catalog_bytes, sources, retained, public_before)
    verify_v1()
    print(json.dumps({k: manifest[k] for k in ("retained_count", "quarantined_count", "recovered_from_v1_count", "newly_quarantined_from_v1_count", "original_90_recovered_count", "annotation_repaired_tasks", "quarantine_reason_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
