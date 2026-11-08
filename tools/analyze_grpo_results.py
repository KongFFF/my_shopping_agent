"""Reproduce the CPU-only analysis of the frozen GRPO diagnostic snapshot.

Run from the parent repository. No training, model inference or Final evaluation.
Requires PyYAML plus the pinned environment's pure Python reward modules.
"""

import csv
import gzip
import hashlib
import itertools
import json
import math
import runpy
import statistics
import sys
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "analysis/grpo_baseline_20261022_202816"
UPSTREAM = ROOT / "upstream/shopping-grpo-longhorizon"
OUT = SNAPSHOT / "derived"
sys.dont_write_bytecode = True
sys.path.insert(0, str(UPSTREAM / "environments/ShopSimulator/shop_env"))
from web_agent_site.engine.constraints import explicit_budget_from_instruction
from web_agent_site.engine.reward import evaluate_purchase
from web_agent_site.engine.reward_features import compile_reward_features
from web_agent_site.engine.reward_features import normalize_option_text
from web_agent_site.engine.variant_price import option_axes


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [item for key in sorted(set(a) | set(b))
                for item in differences(a.get(key), b.get(key), f"{prefix}.{key}")]
    return [] if a == b else [{"field": prefix, "before": a, "after": b}]


def action_sequence(trajectory):
    return [(s["tool_name"], s["parameters"]) for s in trajectory["steps"]]


def write_csv(path, records):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_overview(windows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ["#334155", "#0f766e", "#b45309", "#be123c"]
    ax = axes[0, 0]
    values = [2498, 485, 313, 1000]
    bars = ax.barh(["Constant reward", "Invalid group", "Valid but not used", "Used for update"], values, color=colors)
    ax.bar_label(bars, padding=4)
    ax.set_xlim(0, 2900)
    ax.set_title("A. Where 4,296 generated groups went")
    ax.set_xlabel("Task-group occurrences (4 rollouts each)")
    ax = axes[0, 1]
    bars = ax.bar(["Gold throughout", "Never gold", "Other tasks"], [365, 224, 411], color=[colors[0], colors[3], colors[1]])
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, 460)
    ax.set_title("B. Training task behavior across sampled visits")
    ax.set_ylabel("Unique tasks, total = 1,000")
    ax = axes[1, 0]
    selection = rows(SNAPSHOT / "selection/dev50_summary.csv")
    ax.plot([int(r["step"]) for r in selection], [100 * float(r["gold_reward_1_rate"]) for r in selection], "o-", color=colors[1], label="In-training Dev")
    ax.scatter([0, 500], [48, 54], marker="*", s=170, color=colors[2], label="Independent Dev", zorder=4)
    ax.set(title="C. Dev-50: evaluation paths kept separate", xlabel="GRPO optimizer step", ylabel="Gold success (%)", ylim=(40, 64))
    ax.legend(loc="upper left")
    ax = axes[1, 1]
    ax.plot([r["end"] for r in windows], [r["entropy"] for r in windows], "o-", color=colors[1])
    ax.set(title="D. Logged token entropy does not collapse", xlabel="GRPO optimizer step (50-step windows)", ylabel="Mean actor/entropy_loss", ylim=(0, .15))
    fig.suptitle("GRPO baseline diagnosis | frozen snapshot, 2026-10-29", fontsize=15)
    fig.savefig(OUT / "overview.png", dpi=160)
    plt.close(fig)


def main():
    manifest = load(SNAPSHOT / "manifest.json")
    integrity = {}
    for name, record in manifest["files"].items():
        content = (SNAPSHOT / name).read_bytes()
        raw = hashlib.sha256(content).hexdigest()
        normalized = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
        integrity[name] = "exact" if raw == record["sha256"] else "CRLF_only" if normalized == record["sha256"] else "mismatch"
    assert "mismatch" not in integrity.values(), integrity
    summary_module = runpy.run_path(str(UPSTREAM / "src/shopping_grpo/evaluation/summary.py"))
    strict_success = summary_module["_is_strict_success"]
    trajectories, summaries = {}, {}
    for label in ("sft", "grpo"):
        data = [json.loads(line) for line in (SNAPSHOT / f"{label}_dev50/trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(data) == len({r["task_id"] for r in data}) == 50
        trajectories[label] = {r["task_id"]: r for r in data}
        summaries[label] = load(SNAPSHOT / f"{label}_dev50/summary.json")
        recomputed = summary_module["summarize_trajectories"]([r["task_id"] for r in data], data)
        assert all(summaries[label][key] == value for key, value in recomputed.items())
    assert trajectories["sft"].keys() == trajectories["grpo"].keys()
    protocol_diff = differences(summaries["sft"]["protocol"], summaries["grpo"]["protocol"])
    assert {r["field"] for r in protocol_diff} == {".model"}

    with gzip.open(UPSTREAM / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz", "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    by_asin = {str(p["asin"]): p for p in products}
    assert len(by_asin) == len(products) == 23421
    indexed_goals = [(p, i) for p in products for i in p.get("instructions", []) if i.get("attributes")]
    for trajectory in trajectories["grpo"].values():
        assert indexed_goals[trajectory["task_id"]][1]["instruction"] == trajectory["initial_result"]["instruction"].removeprefix("Instruction: ")

    def goal_for(task_id):
        product, instruction = indexed_goals[task_id]
        return product, {"asin": product["asin"], "category": product["category"],
                         "price_upper": explicit_budget_from_instruction(instruction["instruction"]),
                         **compile_reward_features(instruction, product)}

    def audit_goal(task_id, split):
        product, goal = goal_for(task_id)
        selected = {v["source_axis"]: v["value"] for v in goal["required_options_by_key"].values()}
        result = evaluate_purchase(product, goal, selected_options=selected)
        variant_search = {"gold_found": result.reward_type == "gold_purchase", "complete": True, "evaluated": 1}
        if result.reward_type != "gold_purchase":
            axes = option_axes(product)
            choices, source_axes = [], []
            required = goal["required_options_by_key"]
            if axes["collisions"] or not set(required) <= set(axes["axes"]):
                variant_search["reason"] = "ambiguous_or_missing_required_axis"
            else:
                for axis, record in axes["axes"].items():
                    source_axes.append(record["source_axis"])
                    if axis in required:
                        entry = record["values"].get(normalize_option_text(required[axis]["value"]))
                        choices.append([entry["value"]] if entry else [])
                    else:
                        # Include unselected optional axes as well as every catalog value.
                        choices.append([None] + [v["value"] for v in record["values"].values()])
                variant_search["candidate_combinations"] = math.prod(map(len, choices))
                for values in itertools.product(*choices):
                    if variant_search["evaluated"] >= 10001:
                        variant_search["complete"] = False
                        break
                    candidate = {axis: value for axis, value in zip(source_axes, values) if value is not None}
                    trial = evaluate_purchase(product, goal, selected_options=candidate)
                    variant_search["evaluated"] += 1
                    if trial.reward_type == "gold_purchase":
                        variant_search["gold_found"] = True
                        variant_search["successful_options"] = candidate
                        break
        return {"task_id": task_id, "split": split, "asin": goal["asin"],
                "reward_type": result.reward_type, "reward": result.reward,
                "price": result.hard_gates["budget"]["actual"], "budget": goal["price_upper"],
                "preference_score": result.weighted_score,
                "query": indexed_goals[task_id][1]["instruction"], "annotated_options": selected,
                "variant_search": variant_search}

    replay_count = 0
    pairs, case_reviews = [], []
    for task_id in sorted(trajectories["sft"]):
        a, b = (trajectories[k][task_id] for k in ("sft", "grpo"))
        a_success, b_success = strict_success(a), strict_success(b)
        category = "both_success" if a_success and b_success else "sft_only" if a_success else "grpo_only" if b_success else "both_fail"
        seq_a, seq_b = action_sequence(a), action_sequence(b)
        common = 0
        for x, y in zip(seq_a, seq_b):
            if x != y:
                break
            common += 1
        product, goal = goal_for(task_id)
        pairs.append({"task_id": task_id, "category": category,
                      "sft_type": a["terminal_result"]["reward_detail"]["reward_type"],
                      "grpo_type": b["terminal_result"]["reward_detail"]["reward_type"],
                      "sft_steps": len(seq_a), "grpo_steps": len(seq_b),
                      "identical_actions": seq_a == seq_b,
                      "first_action_divergence": common + 1 if seq_a != seq_b else None})
        for label, trajectory in (("sft", a), ("grpo", b)):
            terminal = trajectory["terminal_result"]
            purchase = terminal.get("purchase") or {}
            detail = terminal["reward_detail"]
            if purchase.get("asin"):
                result = evaluate_purchase(by_asin[str(purchase["asin"])], terminal["goal"], selected_options=purchase.get("options", {}))
                assert result.reward_type == detail["reward_type"], (label, task_id)
                assert math.isclose(result.reward, trajectory["final_reward"], abs_tol=1e-6), (label, task_id)
                replay_count += 1
            if strict_success(trajectory):
                continue
            dimensions = detail.get("evidence", {}).get("preference_scoring", {}).get("dimensions", {})
            case_reviews.append({
                "task_id": task_id, "model": label, "query": indexed_goals[task_id][1]["instruction"],
                "reward_type": detail["reward_type"], "target_asin_match": detail.get("target_asin_match"),
                "gold_visible_turns": [i + 1 for i, step in enumerate(trajectory["steps"]) if goal["asin"] in step["observation"]],
                "gold_opened_turns": [i + 1 for i, step in enumerate(trajectory["steps"])
                                      if step["tool_name"] == "open_product" and step["parameters"].get("asin") == goal["asin"]],
                "failed_dimensions": {k: v for k, v in dimensions.items() if v.get("active") and v["score"] < 1},
                "purchase": {k: purchase.get(k) for k in ("asin", "options", "price")},
                "hard_gates": detail.get("hard_gates"),
                "actions": [{"turn": i + 1, "tool": step["tool_name"], "parameters": step["parameters"]}
                            for i, step in enumerate(trajectory["steps"])],
            })

    train_ids = [json.loads(line)["task_id"] for line in (UPSTREAM / "data/grpo/train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train_ids) == len(set(train_ids)) == 1000
    assert not set(train_ids) & trajectories["grpo"].keys()
    gold_audits = [audit_goal(tid, "train") for tid in train_ids] + [audit_goal(tid, "dev") for tid in trajectories["grpo"]]
    groups = rows(SNAPSHOT / "training_diagnosis/groups.csv")
    updates = rows(SNAPSHOT / "training_diagnosis/updates.csv")
    tasks = [{key: int(value) for key, value in row.items()} for row in rows(SNAPSHOT / "training_diagnosis/tasks.csv")]
    assert len(groups) == 4296 and [int(u["step"]) for u in updates] == list(range(1, 501))
    assert sum(int(t["trained_groups"]) for t in tasks) == 1000
    assert sum(g["trained"] == "True" for g in groups) == 1000
    windows = []
    for start in range(1, 501, 50):
        gs = [g for g in groups if start <= int(g["step"]) < start + 50]
        us = [u for u in updates if start <= int(u["step"]) < start + 50]
        windows.append({"start": start, "end": start + 49, "generated_groups": len(gs),
                        "unfiltered_gold_rate": sum(int(g["gold_rollouts"]) for g in gs) / (4 * len(gs)),
                        "trained_over_generated": sum(g["trained"] == "True" for g in gs) / len(gs),
                        "entropy": statistics.fmean(float(u["actor/entropy_loss"]) for u in us),
                        "clip_fraction": statistics.fmean(float(u["actor/pg_clipfrac"]) for u in us),
                        "grad_norm": statistics.fmean(float(u["actor/grad_norm"]) for u in us)})
    counts = Counter(p["category"] for p in pairs)
    n = counts["grpo_only"] + counts["sft_only"]
    mcnemar = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(counts["grpo_only"], counts["sft_only"]) + 1)) / 2**n) if n else 1.0
    old_config, new_config = [yaml.safe_load((SNAPSHOT / f"configs/{name}.yaml").read_text(encoding="utf-8"))
                              for name in ("formal_memfix_v1", "formal_resume350_v1")]
    training_summary = load(SNAPSHOT / "training_diagnosis/summary.json")
    evidence = {"integrity": integrity, "dev_summaries_recomputed": True,
                "local_product_archive_sha256": hashlib.sha256((UPSTREAM / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz").read_bytes()).hexdigest(),
                "local_reward_source_sha256": hashlib.sha256((UPSTREAM / "environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                "protocol_differences": protocol_diff, "paired_counts": dict(counts), "exact_mcnemar_p": mcnemar,
                "purchase_reward_replay_count": replay_count, "purchase_reward_replay_mismatches": 0,
                "configuration_differences": differences(old_config, new_config),
                "training_summary": {k: training_summary[k] for k in ("counts", "rates", "task_coverage", "generation_cost_scope")},
                "always_gold_never_trained_tasks": sum(t["generated_gold_rollouts"] == 4 * t["generated_groups"] for t in tasks),
                "never_gold_but_trained_tasks": sum(t["generated_gold_rollouts"] == 0 and t["trained_groups"] > 0 for t in tasks),
                "train_annotation_audit_counts": dict(Counter(a["reward_type"] for a in gold_audits if a["split"] == "train")),
                "dev_annotation_audit_counts": dict(Counter(a["reward_type"] for a in gold_audits if a["split"] == "dev")),
                "static_gold_infeasible_train_ids": [a["task_id"] for a in gold_audits if a["split"] == "train" and a["variant_search"]["complete"] and not a["variant_search"]["gold_found"]],
                "static_gold_infeasible_dev_ids": [a["task_id"] for a in gold_audits if a["split"] == "dev" and a["variant_search"]["complete"] and not a["variant_search"]["gold_found"]],
                "audit_limitations": ["After checking annotated options, non-gold cases enumerate optional axes, including unselected axes, with a 10000-combination cap. Infeasibility is asserted only for complete searches under the pinned pure reward/catalog contract, not for real shopping semantics.",
                                      "Local pure reward matches all 94 observed purchases; this is not a full live environment replay.",
                                      "Task index was independently checked against every Dev query; no Final tasks or results were read."]}
    OUT.mkdir(exist_ok=True)
    for name, value in (("evidence.json", evidence), ("dev_failure_review.json", case_reviews), ("gold_annotation_audit.json", gold_audits)):
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUT / "dev_paired.csv", pairs)
    write_csv(OUT / "training_windows.csv", windows)
    plot_overview(windows)
    print(json.dumps({k: evidence[k] for k in ("paired_counts", "exact_mcnemar_p", "purchase_reward_replay_count", "train_annotation_audit_counts", "dev_annotation_audit_counts")}, ensure_ascii=False, indent=2))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
