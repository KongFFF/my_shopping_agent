"""Build a versioned task-contract repair; never mutate the pinned baseline.

CPU only. Emits auditable runtime replacements, annotation overrides and task
lists. --materialize additionally creates a catalog and filtered parquet in a
NEW output directory; it does not launch an environment or training.
"""

import argparse
import ast
import copy
import difflib
import gzip
import hashlib
import inspect
import json
import math
import shutil
import sys
import types
from collections import Counter
from pathlib import Path

import task_contract_v1 as rules

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream/shopping-grpo-longhorizon"
PACKAGE = ROOT / "repairs/task_contract_v1"
ENGINE = UP / "environments/ShopSimulator/shop_env/web_agent_site/engine"
sys.dont_write_bytecode = True
sys.path.insert(0, str(UP / "environments/ShopSimulator/shop_env"))
from web_agent_site.engine import reward_features as baseline_features
from web_agent_site.engine.constraints import explicit_budget_from_instruction as baseline_budget
from web_agent_site.engine.reward import evaluate_purchase
from web_agent_site.engine.search import build_index, product_fields


def sha(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def read_ids(path):
    return [int(json.loads(line)["task_id"]) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def load_module(source, name):
    module = types.ModuleType(name)
    exec(compile(source, name, "exec"), module.__dict__)
    return module


def runtime_sources():
    old_constraints = (ENGINE / "constraints.py").read_text(encoding="utf-8")
    tree = ast.parse(old_constraints)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "explicit_budget_from_instruction")
    lines = old_constraints.splitlines(keepends=True)
    lines[fn.lineno - 1] = lines[fn.lineno - 1].replace("explicit_budget_from_instruction", "_baseline_explicit_budget_v1", 1)
    constraints = "".join(lines) + "\n\n" + inspect.getsource(rules.budget_details_v1)
    constraints += '''\n\ndef explicit_budget_from_instruction(instruction):
    details = budget_details_v1(instruction)
    # Ambiguity is not permission to silently remove an existing budget.
    if details["status"] == "review_required":
        return _baseline_explicit_budget_v1(instruction)
    return details["upper"]
'''
    old_features = (ENGINE / "reward_features.py").read_text(encoding="utf-8")
    assert old_features.count("    return sorted(matches)\n") == 1
    features = old_features.replace("    return sorted(matches)\n", "    return filter_required_brands_v1(instruction, sorted(matches))\n")
    features += "\n\n" + inspect.getsource(rules.filter_required_brands_v1)
    for text in (constraints, features):
        compile(text, "repaired-runtime", "exec")
    return {"constraints.py": (old_constraints, constraints), "reward_features.py": (old_features, features)}


def case_table(audits):
    labels = {"multiple_effective_price_axes": "多个价格轴无组合定价", "vague_price_bound": "模糊价格需审阅",
              "unparsed_price_language": "预算表达未可靠解析", "wrong_purchase": "目标与预算冲突",
              "component_or_lower_bound_only": "单价/下限契约不支持", "unresolved_option_annotation": "规格标注未唯一匹配",
              "effective_price_axis_unselected": "标注未定位必要价格轴"}
    lines = ["# 原90题逐题处理表", "", "task_id不变；隔离不是删除原始数据。完整Query与证据见 original_90_review.jsonl。", "",
             "| task_id | 处理 | 原预算 → 新预算 | 确定修复 / 隔离原因 |", "| --- | --- | --- | --- |"]
    for item in audits:
        if not item["was_original_90"]:
            continue
        changes = []
        if budget_changed(item["old_budget"], item["new_budget"]):
            changes.append("预算解析")
        if item["old_expected_brand"] != item["new_expected_brand"]:
            changes.append("兼容对象不作为品牌")
        if item["annotation_changes"]:
            changes.append("规格字符恢复")
        changes.extend(labels.get(reason, reason) for reason in item["reasons"])
        fmt = lambda value: "未声明" if value is None else f"{value:g}"
        status = "修复并保留" if item["status"] == "retained" else "隔离待审"
        lines.append(f"| {item['task_id']} | {status} | {fmt(item['old_budget'])} → {fmt(item['new_budget'])} | {'；'.join(changes)} |")
    return ("\n".join(lines) + "\n").encode("utf-8")


def budget_changed(before, after):
    if before is None or after is None:
        return before != after
    return not math.isclose(before, after, rel_tol=1e-12, abs_tol=1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize", type=Path, help="NEW output directory for catalog JSON and parquet (PyArrow 25+)")
    args = parser.parse_args()
    sources = runtime_sources()
    fixed_constraints = load_module(sources["constraints.py"][1], "repaired_constraints")
    fixed_features = load_module(sources["reward_features.py"][1], "repaired_features")
    archive = UP / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
    with gzip.open(archive, "rb") as handle:
        original_catalog_bytes = handle.read()
    catalog = json.loads(original_catalog_bytes)
    search_before = sha(json.dumps([product_fields(p) for p in catalog], ensure_ascii=False, sort_keys=True).encode())
    goals = [(p, i) for p in catalog for i in p.get("instructions", []) if i.get("attributes")]
    assert len(goals) == len(catalog) == 23421
    train_ids = read_ids(UP / "data/grpo/train.jsonl")
    val_ids = read_ids(UP / "data/grpo/validation.jsonl")
    final_ids = read_ids(UP / "data/evaluation/tasks.jsonl")  # IDs only, never test queries/results.
    assert len(train_ids) == len(set(train_ids)) == 1000
    assert len(val_ids) == len(set(val_ids)) == 50
    assert not set(train_ids) & (set(val_ids) | set(final_ids))
    original_audit_path = ROOT / "analysis/grpo_baseline_20261022_202816/derived/gold_annotation_audit.json"
    original_audit = json.loads(original_audit_path.read_text(encoding="utf-8"))
    original_bad = {r["task_id"] for r in original_audit if r["split"] == "train" and not r["variant_search"]["gold_found"]}
    assert len(original_bad) == 90
    overrides, audits, retained, quarantined = [], [], [], []

    def goal_and_result(product, instruction, budget, compile_features):
        goal = {"asin": product["asin"], "category": product["category"], "price_upper": budget,
                **compile_features(instruction, product)}
        selected = {v["source_axis"]: v["value"] for v in goal["required_options_by_key"].values()}
        return goal, evaluate_purchase(product, goal, selected_options=selected)

    for task_id in train_ids:
        product, instruction = goals[task_id]
        before = copy.deepcopy(instruction)
        changed = []
        for index, value in enumerate(instruction.get("instruction_options") or []):
            resolved = rules.resolve_damaged_option_v1(value, product)
            if resolved is not None:
                axis, replacement = resolved
                instruction["instruction_options"][index] = replacement
                changed.append({"index": index, "before": value, "after": replacement, "source_axis": axis})
        if changed:
            overrides.append({"task_id": task_id, "asin": product["asin"], "instruction": instruction["instruction"],
                              "changes": changed, "instruction_options_before": before["instruction_options"],
                              "instruction_options_after": instruction["instruction_options"]})
        old_goal, old = goal_and_result(product, before, baseline_budget(before["instruction"]), baseline_features.compile_reward_features)
        details = rules.budget_details_v1(instruction["instruction"])
        new_goal, new = goal_and_result(product, instruction, fixed_constraints.explicit_budget_from_instruction(instruction["instruction"]), fixed_features.compile_reward_features)
        reasons = []
        if details["status"] == "review_required":
            reasons.append(details["reason"])
        if new_goal["unresolved_option_requirements"]:
            reasons.append("unresolved_option_annotation")
        if new.reward_type != "gold_purchase":
            reasons.append(new.evidence["price_resolution"]["method"] if new.reward_type == "reward_unverifiable" else new.reward_type)
        keep = not reasons
        (retained if keep else quarantined).append(task_id)
        audit = {"task_id": task_id, "asin": product["asin"], "query": instruction["instruction"],
                 "was_original_90": task_id in original_bad, "status": "retained" if keep else "quarantined",
                 "reasons": reasons, "old_reward_type": old.reward_type, "new_reward_type": new.reward_type,
                 "old_budget": old_goal["price_upper"], "new_budget": new_goal["price_upper"],
                 "budget_parse": details, "old_expected_brand": old_goal["expected_brand"],
                 "new_expected_brand": new_goal["expected_brand"], "annotation_changes": changed,
                 "new_reward_evidence": new.to_dict(), "selected_gold_options": {
                     v["source_axis"]: v["value"] for v in new_goal["required_options_by_key"].values()}}
        audits.append(audit)
    # Reproduce the old anomaly list rather than assuming the input audit still applies.
    assert {a["task_id"] for a in audits if a["old_reward_type"] != "gold_purchase"} == original_bad
    assert len(retained) + len(quarantined) == 1000 and not set(retained) & set(quarantined)
    assert all(a["new_reward_type"] == "gold_purchase" for a in audits if a["status"] == "retained")
    # The catalog order, natural-language queries, products and prices are never rewritten.
    assert all(a["task_id"] in train_ids for a in overrides)
    search_after = sha(json.dumps([product_fields(p) for p in catalog], ensure_ascii=False, sort_keys=True).encode())
    assert search_before == search_after, "annotation repair changed retrieval documents"
    validation_audit = []
    for task_id in val_ids:
        product, instruction = goals[task_id]
        old_goal, old = goal_and_result(product, instruction, baseline_budget(instruction["instruction"]), baseline_features.compile_reward_features)
        new_goal, new = goal_and_result(product, instruction, fixed_constraints.explicit_budget_from_instruction(instruction["instruction"]), fixed_features.compile_reward_features)
        validation_audit.append({"task_id": task_id, "old_reward_type": old.reward_type,
                                 "new_reward_type": new.reward_type, "old_budget": old_goal["price_upper"],
                                 "new_budget": new_goal["price_upper"], "budget_parse": rules.budget_details_v1(instruction["instruction"])})
    catalog_bytes = json.dumps(catalog, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    artifacts = {}
    artifacts["train.jsonl"] = "".join(json.dumps({"task_id": t}) + "\n" for t in retained).encode()
    artifacts["validation.jsonl"] = (UP / "data/grpo/validation.jsonl").read_bytes()
    artifacts["annotation_overrides.json"] = json_bytes(overrides)
    artifacts["validation_audit.json"] = json_bytes(validation_audit)
    artifacts["all_train_audit.jsonl"] = "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audits).encode("utf-8")
    artifacts["original_90_review.jsonl"] = "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audits if a["was_original_90"]).encode("utf-8")
    artifacts["original_90_review.md"] = case_table(audits)
    artifacts["quarantine.jsonl"] = "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audits if a["status"] == "quarantined").encode("utf-8")
    patch = []
    for name, (old, new) in sources.items():
        artifacts[f"runtime/{name}"] = new.encode("utf-8")
        relative = f"environments/ShopSimulator/shop_env/web_agent_site/engine/{name}"
        patch.extend(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True), fromfile=f"a/{relative}", tofile=f"b/{relative}"))
    artifacts["runtime.patch"] = "".join(patch).encode("utf-8")
    before = {name: sha(old.encode()) for name, (old, _) in sources.items()}
    original90 = [a for a in audits if a["was_original_90"]]
    manifest = {"repair_version": "task-contract-v1", "pinned_upstream": "a583dba4c149be09622ce58ad6f7a160db77bb41",
                "original_train_count": 1000, "retained_count": len(retained), "quarantined_count": len(quarantined),
                "original_90_recovered_count": sum(a["status"] == "retained" for a in original90),
                "original_90_quarantined_count": sum(a["status"] == "quarantined" for a in original90),
                "recovered_90_change_counts_overlapping": {
                    "budget": sum(budget_changed(a["old_budget"], a["new_budget"]) for a in original90 if a["status"] == "retained"),
                    "brand_role": sum(a["old_expected_brand"] != a["new_expected_brand"] for a in original90 if a["status"] == "retained"),
                    "annotation_characters": sum(bool(a["annotation_changes"]) for a in original90 if a["status"] == "retained")},
                "newly_quarantined_originally_feasible_count": sum(not a["was_original_90"] and a["status"] == "quarantined" for a in audits),
                "annotation_repaired_tasks": len(overrides), "retained_task_ids": retained, "quarantined_task_ids": quarantined,
                "quarantine_reason_counts": dict(Counter(reason for a in audits if a["status"] == "quarantined" for reason in a["reasons"])),
                "source_runtime_normalized_sha256": before,
                "original_catalog_sha256": sha(original_catalog_bytes), "repaired_catalog_sha256": sha(catalog_bytes),
                "search_documents_sha256_unchanged": search_before,
                "validation_task_ids_unchanged": val_ids, "final_ids_overlap": [],
                "artifact_sha256": {name: sha(content) for name, content in artifacts.items()},
                "limits": ["Retained tasks pass static gold feasibility and supported budget parsing, not a complete human semantic audit.",
                           "Unknown prices are not guessed; multi-axis price ambiguity is quarantined.",
                           "Ambiguous budgets keep legacy runtime behavior and are excluded from repaired training.",
                           "Do not train on this ID list under the unpatched runtime/catalog; they form a single versioned contract."]}
    artifacts["manifest.json"] = json_bytes(manifest)
    PACKAGE.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        path = PACKAGE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if args.materialize:
        import pyarrow as pa
        import pyarrow.parquet as pq
        if int(pa.__version__.split(".")[0]) < 25:
            raise SystemExit("Materialization requires PyArrow >=25 for the pinned parquet; repair/audit package was generated.")
        target = args.materialize.resolve()
        if target.exists():
            raise SystemExit(f"Refusing to overwrite {target}")
        table = pq.read_table(UP / "data/grpo/train.parquet")
        task_ids = [int(item["task_id"]) for item in table["extra_info"].to_pylist()]
        assert len(task_ids) == 1000 and set(task_ids) == set(train_ids)
        selected_rows = [i for i, task_id in enumerate(task_ids) if task_id in set(retained)]
        filtered = table.take(pa.array(selected_rows, type=pa.int64()))
        target.mkdir(parents=True)
        pq.write_table(filtered, target / "train.parquet")
        (target / "validation.parquet").write_bytes((UP / "data/grpo/validation.parquet").read_bytes())
        (target / "manifest.json").write_bytes(artifacts["manifest.json"])
        runtime = target / "runtime"
        shutil.copytree(UP / "environments/ShopSimulator/shop_env", runtime,
                        ignore=shutil.ignore_patterns("data", "search_engine", "__pycache__", ".ipynb_checkpoints", ".pytest_cache", "*.log", ".venv*", ".git", "models", "checkpoints", "outputs"))
        (runtime / "data").mkdir(exist_ok=True)
        product_path = runtime / "data/items_eval_train.json"
        product_path.write_bytes(catalog_bytes)
        for name, (_, source) in sources.items():
            (runtime / "web_agent_site/engine" / name).write_bytes(source.encode("utf-8"))
        index_manifest = build_index(catalog, runtime / "search_engine/products.sqlite3", product_data_sha256=sha(catalog_bytes))
        (target / "index_manifest.json").write_bytes(json_bytes(index_manifest))
        environment_manifest = json.loads((UP / "data/environment.json").read_text(encoding="utf-8"))
        environment_manifest["product_data_sha256"] = sha(catalog_bytes)
        environment_manifest["task_contract_repair"] = {
            "version": "task-contract-v1", "runtime_sha256": {
                f"web_agent_site/engine/{name}": sha(source.encode()) for name, (_, source) in sources.items()},
            "search_documents_sha256_unchanged": search_before,
            "annotation_overrides_sha256": sha(artifacts["annotation_overrides.json"]),
        }
        # The five upstream preflight files are copied unchanged from the selected checkout.
        runtime_paths = {"observation.py": "web_agent_site/engine/observation.py", "pack_api.py": "shop_env/pack_api.py",
                         "reward.py": "web_agent_site/engine/reward.py", "slot_lease_pool.py": "shop_env/slot_lease_pool.py",
                         "web_agent_text_env.py": "web_agent_site/envs/web_agent_text_env.py"}
        environment_manifest["runtime_files_sha256"] = {name: sha((runtime / rel).read_bytes()) for name, rel in runtime_paths.items()}
        (target / "environment.json").write_bytes(json_bytes(environment_manifest))
        receipt = {"train_rows": len(filtered), "validation_rows": 50,
                   "train_parquet_sha256": sha((target / "train.parquet").read_bytes()),
                   "validation_parquet_sha256": sha((target / "validation.parquet").read_bytes()),
                   "product_data_sha256": sha(catalog_bytes), "repair_manifest_sha256": sha(artifacts["manifest.json"])}
        reloaded = pq.read_table(target / "train.parquet")
        assert {int(item["task_id"]) for item in reloaded["extra_info"].to_pylist()} == set(retained)
        assert reloaded.schema == table.schema
        (target / "materialization_receipt.json").write_bytes(json_bytes(receipt))
    print(json.dumps({k: manifest[k] for k in ("retained_count", "quarantined_count", "original_90_recovered_count", "original_90_quarantined_count", "newly_quarantined_originally_feasible_count", "annotation_repaired_tasks", "quarantine_reason_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
