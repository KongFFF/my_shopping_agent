"""CPU-only paired audit of SFT/old GRPO on purchase-guard v1.1.

Hidden task annotations are used ONLY for offline diagnosis/counterfactuals.
No policy input, training data, reward implementation or source snapshot changes.
"""

import csv
import gzip
import hashlib
import itertools
import json
import math
import runpy
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis/purchase_guard_v1_1_dev50_20261031"
OUT = BASE / "derived"
UP = ROOT / "upstream/shopping-grpo-longhorizon"
sys.path[:0] = [str(UP / "src"), str(UP / "environments/ShopSimulator/shop_env")]
from shopping_grpo.environment.actions import action_reject_reason, clickable_buttons
from web_agent_site.engine.constraints import explicit_budget_from_instruction
from web_agent_site.engine.reward import evaluate_purchase
from web_agent_site.engine.reward_features import compile_reward_features
from web_agent_site.engine.variant_price import option_axes


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def variants(product, goal):
    """An oracle upper bound under ORIGINAL scoring, never a policy suggestion."""
    axes = option_axes(product)
    if axes["collisions"]:
        return {"complete": False, "reason": "axis_collision"}
    records = list(axes["axes"].values())
    choices = [[None, *[v["value"] for v in axis["values"].values()]] for axis in records]
    count = math.prod(map(len, choices))
    if count > 10000:
        return {"complete": False, "combinations": count}
    best, gold = None, None
    for values in itertools.product(*choices):
        selected = {axis["source_axis"]: value for axis, value in zip(records, values) if value is not None}
        result = evaluate_purchase(product, goal, selected_options=selected)
        if best is None or result.reward > best["reward"]:
            best = {"reward": result.reward, "type": result.reward_type, "options": selected}
        if result.reward_type == "gold_purchase" and gold is None:
            gold = selected
    return {"complete": True, "combinations": count, "gold_found": gold is not None,
            "gold_options_oracle_only": gold, "best": best}


def main():
    manifest = load(BASE / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        data = (BASE / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected, name
    evaluator = runpy.run_path(str(UP / "src/shopping_grpo/evaluation/summary.py"))
    success = evaluator["_is_strict_success"]
    models, summaries, specs = {}, {}, {}
    for label in ("sft", "grpo"):
        directory = BASE / f"{label}_dev50"
        rows = [json.loads(line) for line in (directory / "trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        models[label] = {int(row["task_id"]): row for row in rows}
        assert len(rows) == len(models[label]) == 50
        summaries[label] = load(directory / "summary.json")
        assert evaluator["summarize_trajectories"](list(models[label]), rows) == {
            k: v for k, v in summaries[label].items() if k != "protocol"}
        specs[label] = load(directory / "evaluation_spec.json")
    assert models["sft"].keys() == models["grpo"].keys()
    assert specs["sft"]["protocol"] == specs["grpo"]["protocol"]
    assert specs["sft"]["benchmark_sha256"] == specs["grpo"]["benchmark_sha256"]
    assert specs["sft"]["bundle_manifest"] == specs["grpo"]["bundle_manifest"]
    archive = UP / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
    with gzip.open(archive, "rb") as handle:
        data = handle.read()
    assert hashlib.sha256(data).hexdigest() == specs["sft"]["bundle_manifest"]["original_catalog_sha256"]
    products = json.loads(data)
    by_asin = {str(p["asin"]): p for p in products}
    indexed = [(p, inst) for p in products for inst in p.get("instructions", []) if inst.get("attributes")]
    reviews, pairs = [], []
    cache = {}
    replayed = blocked_replayed = selections_verified = 0
    for task_id in sorted(models["sft"]):
        target, instruction = indexed[task_id]
        query = instruction["instruction"]
        goal = {"asin": str(target["asin"]), "category": target["category"],
                "price_upper": explicit_budget_from_instruction(query), "instruction_text": query,
                **compile_reward_features(instruction, target)}
        status = {label: success(models[label][task_id]) for label in models}
        category = "both_success" if all(status.values()) else "grpo_only" if status["grpo"] else "sft_only" if status["sft"] else "both_fail"
        pair = {"task_id": task_id, "category": category}
        for label, trajectories in models.items():
            row = trajectories[task_id]
            assert row["initial_result"]["instruction"].removeprefix("Instruction: ") == query
            assert not row.get("error") and not row.get("release_error")
            terminal = row["terminal_result"]
            detail = terminal["reward_detail"]
            purchase = terminal.get("purchase") or {}
            steps = row["steps"]
            asin = str(purchase.get("asin") or "")
            if asin:
                result = evaluate_purchase(by_asin[asin], terminal["goal"], selected_options=purchase.get("options", {}))
                assert result.reward_type == detail["reward_type"]
                assert math.isclose(result.reward, row["final_reward"], abs_tol=1e-9)
                replayed += 1
            for blocked in row.get("blocked_tool_calls", []):
                n = blocked["step_index"]
                assert n > 0, "Initial guard requires explicit initial-state renderer"
                function = blocked["tool_call"]["function"]
                reason = action_reject_reason(function["name"], json.loads(function["arguments"]), steps[n - 1]["observation"])
                assert reason == blocked["reason"], (task_id, label, reason)
                blocked_replayed += 1
            dimensions = (detail.get("evidence") or {}).get("preference_scoring", {}).get("dimensions", {})
            failed = {name: {"score": rec["score"], "comparisons": rec.get("results", [])}
                      for name, rec in dimensions.items() if rec.get("active") and rec["score"] < 1}
            states = [(s.get("result") or {}).get("observation_state") or {} for s in steps]
            for step, state in zip(steps, states):
                if step["tool_name"] == "select_option":
                    value = step["parameters"]["value"].casefold()
                    assert value in {str(v).casefold() for v in (state.get("selected_options") or {}).values()}, (label, task_id, step["parameters"])
                    selections_verified += 1
            visible = [i + 1 for i, s in enumerate(steps) if str(target["asin"]) in s["observation"]]
            opened = [i + 1 for i, s in enumerate(steps) if s["tool_name"] == "open_product" and s["parameters"].get("asin") == str(target["asin"])]
            cf = None
            if not status[label] and asin:
                key = (task_id, asin)
                if key not in cache:
                    cache[key] = variants(by_asin[asin], terminal["goal"])
                cf = cache[key]
            if task_id not in cache:
                cache[task_id] = variants(target, goal)
            case = {"task_id": task_id, "model": label, "category": category, "query": query,
                    "strict_success": status[label], "reward_type": detail["reward_type"],
                    "reward": row["final_reward"], "steps": len(steps),
                    "model_tool_calls": sum(len(m.get("tool_calls") or []) for m in row["messages"] if m.get("role") == "assistant"),
                    "target_asin_offline_only": str(target["asin"]),
                    "target_visible_turns": visible, "target_opened_turns": opened,
                    "required_options_offline_only": goal["required_options_by_key"],
                    "budget": goal["price_upper"], "purchase": {k: purchase.get(k) for k in ("asin", "options", "price")},
                    "failed_dimensions": failed, "hard_gates": detail.get("hard_gates"),
                    "target_variant_audit_oracle_only": cache[task_id],
                    "same_purchased_product_audit_oracle_only": cf,
                    "blocked_tool_calls": row.get("blocked_tool_calls", []),
                    "progress": terminal.get("progress"),
                    "purchase_guard_events": [s["purchase_guard"] for s in states if s.get("purchase_guard")],
                    "actions": [{"turn": i + 1, "tool": s["tool_name"], "parameters": s["parameters"],
                                 "product": state.get("product"), "selected_options": state.get("selected_options"),
                                 "selected_price": state.get("selected_price"), "available_options": state.get("available_options")}
                                for i, (s, state) in enumerate(zip(steps, states))]}
            reviews.append(case)
            pair.update({f"{label}_type": detail["reward_type"], f"{label}_reward": row["final_reward"],
                         f"{label}_steps": len(steps), f"{label}_blocked": len(row.get("blocked_tool_calls", [])),
                         f"{label}_target_visible": bool(visible), f"{label}_target_opened": bool(opened)})
        sequences = [[(s["tool_name"], s["parameters"]) for s in models[label][task_id]["steps"]]
                     for label in ("sft", "grpo")]
        pair["identical_executed_actions"] = sequences[0] == sequences[1]
        pairs.append(pair)
    aggregate = {}
    for label in models:
        failed = [c for c in reviews if c["model"] == label and not c["strict_success"]]
        common = [c for c in failed if c["category"] == "both_fail"]
        aggregate[label] = {"failures": len(failed), "failure_types": dict(Counter(c["reward_type"] for c in failed)),
                            "target_seen_in_failures": sum(bool(c["target_visible_turns"]) for c in failed),
                            "target_opened_in_failures": sum(bool(c["target_opened_turns"]) for c in failed),
                            "common_failure_count": len(common),
                            "common_failure_target_seen": sum(bool(c["target_visible_turns"]) for c in common),
                            "common_failure_target_opened": sum(bool(c["target_opened_turns"]) for c in common),
                            "common_partial_dimensions": dict(Counter(k for c in common if c["reward_type"] == "partial_alternative_purchase" for k in c["failed_dimensions"])),
                            "same_purchase_gold_possible_ids": [c["task_id"] for c in failed if (c["same_purchased_product_audit_oracle_only"] or {}).get("gold_found")]}
    evidence = {"paired_counts": dict(Counter(p["category"] for p in pairs)),
                "identical_executed_action_sequences": sum(p["identical_executed_actions"] for p in pairs),
                "guard_rejections_by_task": {label: {str(tid): len(row["blocked_tool_calls"])
                                                     for tid, row in model.items() if row["blocked_tool_calls"]}
                                             for label, model in models.items()},
                "purchase_reward_replays": replayed, "blocked_call_replays": blocked_replayed,
                "executed_selections_verified_in_environment_state": selections_verified,
                "summary_recomputation": "exact", "all_input_hashes": "exact", "aggregate": aggregate,
                "annotation_infeasible_dev_ids": sorted({c["task_id"] for c in reviews if not c["target_variant_audit_oracle_only"].get("gold_found")}),
                "limits": ["Offline hidden goals only; oracle option enumeration is not achievable improvement.",
                           "Old GRPO trained on old environment; new deployment evaluation is not a v1.1 retraining ablation.",
                           "One deterministic decoding run per model; no statistical/generalization conclusion.",
                           "Original 1000-task contract and Dev denominator preserved; no Final data read."]}
    OUT.mkdir(exist_ok=True)
    write_json(OUT / "evidence.json", evidence)
    write_json(OUT / "case_reviews.json", reviews)
    with (OUT / "paired_tasks.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
