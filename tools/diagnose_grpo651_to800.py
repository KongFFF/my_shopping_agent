"""CPU-only audit of completed651..800; reads no checkpoints or Final tasks."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from analyze_grpo_diagnostics import analyze, describe, group_rollouts
from prepare_bounded_grpo_resume import DEFAULT_PROJECT, RUN, digest, read, write

BOUNDS = ((651, 700), (701, 750), (751, 800))


def metric_summary(rows, key):
    return describe([row[key] for row in rows if key in row])


def numeric(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def diagnose(project):
    project = Path(project).resolve()
    run = project / RUN
    prep = run / "preparation/formal_resume650_to800_v1"
    folder = run / "formal_resume650_to800_v1"
    spec, status = read(prep / "launch_spec.json"), read(prep / "exit_status.json")
    if status.get("exit_code") != 0 or (spec.get("resume_step"), spec.get("total_updates")) != (650, 800):
        raise ValueError("Expected completed650->800 training with exit_code=0")
    path = folder / "training_diagnostics.jsonl"
    before = digest(path)
    summary, groups, updates, _ = analyze([path], step_ranges=[(651, 800)])
    if [row["step"] for row in updates] != list(range(651, 801)):
        raise ValueError("Optimizer steps must be exactly651..800, each once")
    for row in updates:
        if row.get("training/optimizer_updated") != 1:
            raise ValueError(f"step{row['step']} has no confirmed optimizer update")

    # Match generation with actual optimizer commits, never count acceptance alone.
    generated = {}
    for group in groups:
        uid = group["uid"]
        if uid in generated:
            raise ValueError(f"Repeated generation uid: {uid}")
        generated[uid] = group
    event_counts = {start: Counter() for start, _ in BOUNDS}
    utilities = {start: {"generated": [], "trained": []} for start, _ in BOUNDS}
    committed = set()
    trained_per_step = {}
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            step = int(event["global_step"])
            if not 651 <= step <= 800:
                continue
            start = next(start for start, end in BOUNDS if start <= step <= end)
            kind = event["event"]
            event_counts[start][kind] += 1
            if kind not in ("generation_batch", "optimizer_step"):
                continue
            phase = "generated" if kind == "generation_batch" else "trained"
            for rollout in event["rollouts"]:
                utility = (rollout.get("reward") or {}).get("terminal_utility")
                if numeric(utility):
                    utilities[start][phase].append(utility)
            if kind == "optimizer_step":
                by_uid = group_rollouts(event["rollouts"], f"{path}:{number}")
                trained_per_step[step] = len(by_uid)
                for uid, rollouts in by_uid.items():
                    group = generated.get(uid)
                    if group is None or group["step"] != step or not group["kept"] or not group["accepted"]:
                        raise ValueError(f"step{step}: committed group has no matching accepted generation: {uid}")
                    if uid in committed:
                        raise ValueError(f"Repeated optimizer consumption: {uid}")
                    if int(rollouts[0]["task_id"]) != group["task_id"]:
                        raise ValueError(f"Committed task differs: {uid}")
                    committed.add(uid)
    if committed != {g["uid"] for g in groups if g["trained"]}:
        raise ValueError("Generated/committed group reconciliation failed")
    if any(n != 2 for n in trained_per_step.values()):
        raise ValueError("Expected exactly2 committed groups per update in unchanged recipe")

    windows = []
    for start, end in BOUNDS:
        gs = [g for g in groups if start <= g["step"] <= end]
        trained = [g for g in gs if g["trained"]]
        rows = [row for row in updates if start <= row["step"] <= end]
        n, t = len(gs), len(trained)
        kept = sum(g["kept"] for g in gs)
        accepted = sum(g["accepted"] for g in gs)
        counts = {
            "updates": len(rows), "generated_groups": n, "generated_rollouts": 4*n,
            "kept_groups": kept, "accepted_groups": accepted,
            "trained_groups": t, "trained_rollouts": 4*t,
            "accepted_not_trained": accepted-t,
            "constant_reward_groups": sum(g["drop_reason"] == "constant_reward" for g in gs),
            "invalid_groups": sum(g["drop_reason"] == "sampling_invalid" for g in gs),
            "generated_mixed_gold_groups": sum(0 < g["gold_rollouts"] < 4 for g in gs),
            "generated_all_gold_groups": sum(g["gold_rollouts"] == 4 for g in gs),
            "generated_no_gold_groups": sum(g["gold_rollouts"] == 0 for g in gs),
            "trained_mixed_gold_groups": sum(0 < g["gold_rollouts"] < 4 for g in trained),
            "trained_no_gold_groups": sum(g["gold_rollouts"] == 0 for g in trained),
            "skipped_updates": event_counts[start]["skipped_update"],
            "generation_batches": event_counts[start]["generation_batch"],
            "generated_unique_tasks": len({g["task_id"] for g in gs}),
            "trained_unique_tasks": len({g["task_id"] for g in trained}),
        }
        if not n or t != 100 or len(rows) != 50 or accepted < t:
            raise ValueError(f"Window{start}..{end} failed group/update integrity")
        rates = {
            "kept_over_generated": kept/n, "trained_over_generated": t/n,
            "constant_reward_over_generated": counts["constant_reward_groups"]/n,
            "invalid_over_generated": counts["invalid_groups"]/n,
            "generated_mixed_gold_over_generated": counts["generated_mixed_gold_groups"]/n,
            "generated_all_gold_over_generated": counts["generated_all_gold_groups"]/n,
            "generated_no_gold_over_generated": counts["generated_no_gold_groups"]/n,
            "trained_mixed_gold_over_trained": counts["trained_mixed_gold_groups"]/t,
            "trained_no_gold_over_trained": counts["trained_no_gold_groups"]/t,
            "generated_groups_per_update": n/50,
            "generated_rollouts_per_update": 4*n/50,
            "generation_batches_per_update": counts["generation_batches"]/50,
            "generated_gold_rollout_rate": sum(g["gold_rollouts"] for g in gs)/(4*n),
            "trained_gold_rollout_rate": sum(g["gold_rollouts"] for g in trained)/(4*t),
        }
        metric_keys = sorted({key for row in rows for key in row if key not in ("source", "step")})
        windows.append({"start": start, "end": end, "counts": counts, "rates": rates,
            "trained_centered_utility_mean_abs": describe([g["centered_utility_mean_abs"] for g in trained]),
            "trained_utility_std": describe([g["utility_std"] for g in trained]),
            "trained_action_sequences_per_group": describe([g["unique_action_sequences"] for g in trained
                                                            if g["unique_action_sequences"] is not None]),
            "generated_terminal_utility": describe(utilities[start]["generated"]),
            "trained_terminal_utility": describe(utilities[start]["trained"]),
            "metrics": {key: metric_summary(rows, key) for key in metric_keys}})

    validation = {}
    val_hashes = {}
    for val_path in sorted((folder / "validation").glob("*.jsonl")):
        val_hashes[str(val_path)] = digest(val_path)
        for line in val_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            step, score = int(row["step"]), float(row["score"])
            if step not in (700, 750, 800) or not math.isfinite(score):
                raise ValueError("Unexpected/nonfinite training Dev record")
            validation.setdefault(step, []).append(score)
    if set(validation) != {700, 750, 800} or any(len(v) != 50 for v in validation.values()):
        raise ValueError("Expected50 training Dev records at700/750/800")
    dev = {step: {"gold": sum(math.isclose(v, 1, abs_tol=1e-6, rel_tol=0) for v in values),
                  "records": 50, "mean_reward": statistics.fmean(values)} for step, values in validation.items()}
    if digest(path) != before:
        raise ValueError("Diagnostic source changed during analysis")
    return {
        "schema": "shopping-grpo-continuation-diagnosis-v1", "coverage_passed": True,
        "source": {"path": str(path), "sha256": before, "bytes": path.stat().st_size},
        "receipts_sha256": {str(p): digest(p) for p in (prep/"launch_spec.json", prep/"exit_status.json")},
        "validation_sha256": val_hashes, "training_exit_status": status,
        "windows": windows, "training_dev": dev,
        "overall_counts": summary["counts"], "overall_task_coverage": summary["task_coverage"],
        "notes": [
            "No Final tasks/results, model tensors, environment, GPU, torch or network were accessed.",
            "Generated groups include attempts discarded before an update; committed counts use optimizer_step only.",
            "Gold is a training-side proxy, not the independent Final strict-success check.",
            "Nonzero group reward contrast or gradient does not prove held-out improvement or nonconvergence.",
            "Centered terminal utility is a contrast proxy, not measured gradient or exact token-level advantage.",
            "Different windows sample different task mixtures; reward-rate changes are not paired evaluation.",
            "actor/ppo_kl is not assumed to measure divergence from SFT; actor/entropy_loss retains its original name.",
            "Step timing describes logged successful updates; may exclude skipped attempts, validation, checkpoint or startup overhead.",
            "Missing metrics are reported as missing, never zero; missing fields limit diagnosis.",
        ],
    }


def report(result):
    ws = result["windows"]
    lines = ["# 651～800训练诊断", "", "完整性：150次更新连续、每步2组×4条，生成与实际更新组对账通过。", "",
             "| 指标 | 651～700 | 701～750 | 751～800 |", "|---|---:|---:|---:|"]
    def add(label, values, percent=False):
        def fmt(v):
            if v is None:
                return "未记录"
            return f"{v:.1%}" if percent else f"{v:.6g}"
        lines.append("| " + " | ".join([label]+[fmt(v) for v in values]) + " |")
    for label, key in (("真实更新数", "updates"), ("生成组数", "generated_groups"),
                       ("实际更新组数", "trained_groups"), ("接受后未更新组数", "accepted_not_trained"),
                       ("跳过更新事件", "skipped_updates"), ("更新覆盖不同任务数", "trained_unique_tasks")):
        add(label, [w["counts"][key] for w in ws])
    for label, key, pct in (("非同分可保留组占比", "kept_over_generated", True),
            ("实际更新组/生成组", "trained_over_generated", True),
            ("同分丢弃组占比", "constant_reward_over_generated", True),
            ("无效组占比", "invalid_over_generated", True),
            ("生成组：全成功占比", "generated_all_gold_over_generated", True),
            ("生成组：全无gold占比", "generated_no_gold_over_generated", True),
            ("生成组：gold混合组占比", "generated_mixed_gold_over_generated", True),
            ("更新组：gold混合组占比", "trained_mixed_gold_over_trained", True),
            ("更新组：全无gold占比", "trained_no_gold_over_trained", True),
            ("每次更新生成轨迹数（含未消费）", "generated_rollouts_per_update", False),
            ("每次更新生成批次数（含跳过）", "generation_batches_per_update", False),
            ("生成轨迹gold率", "generated_gold_rollout_rate", True),
            ("更新轨迹gold率", "trained_gold_rollout_rate", True)):
        add(label, [w["rates"][key] for w in ws], pct)
    for label, key in (("更新组中心化utility绝对值均值", "trained_centered_utility_mean_abs"),
                       ("更新组utility标准差均值", "trained_utility_std"),
                       ("更新组不同动作序列数均值", "trained_action_sequences_per_group"),
                       ("生成轨迹平均terminal utility", "generated_terminal_utility"),
                       ("更新轨迹平均terminal utility", "trained_terminal_utility")):
        add(label, [w[key].get("mean") for w in ws])
    selected = ["actor/lr", "actor/grad_norm", "actor/entropy_loss", "actor/entropy", "actor/pg_loss",
        "actor/pg_clipfrac", "actor/pg_clipfrac_lower", "actor/ppo_kl", "critic/advantages/min",
        "critic/advantages/max", "timing_s/step", "timing_s/gen", "timing_s/update_actor",
        "perf/time_per_step", "perf/total_num_tokens", "response_length/clip_ratio",
        "trajectory/overlong_rate", "trajectory/repeat_loop_rate", "trajectory/infrastructure_invalid_rate"]
    for key in selected:
        add(key+" 均值", [w["metrics"].get(key, {}).get("mean") for w in ws])
    lines.extend(["", "指标记录覆盖与极值（均值比较须同时看尾部及字段缺失）："])
    for key in ("actor/lr", "actor/grad_norm", "actor/entropy_loss", "actor/pg_clipfrac", "actor/ppo_kl", "timing_s/step"):
        parts = []
        for w in ws:
            m = w["metrics"].get(key, {})
            parts.append(f"{w['start']}～{w['end']}: n={m.get('count', 0)}/50" +
                (f", min={m['min']:.6g}, p50={m['p50']:.6g}, p90={m['p90']:.6g}, max={m['max']:.6g}" if m.get("count") else ""))
        lines.append("- "+key+": "+"；".join(parts))
    lines.extend(["", "训练内Dev（保持原推理链路）："])
    for step, dev in result["training_dev"].items():
        lines.append(f"- step{step}: {dev['gold']}/50，平均奖励{dev['mean_reward']:.6f}")
    lines.extend(["", "完整日志覆盖任务数："+json.dumps(result["overall_task_coverage"], ensure_ascii=False),
                  "训练会话退出记录："+json.dumps(result["training_exit_status"], ensure_ascii=False),
                  "", "解释边界：有奖励差异/非零梯度不等于未收敛；全无gold组仍可能包含可学习的部分奖励差异。",
                  "三个窗口任务组成不同；时间指标未必含跳过尝试、验证和保存，不能直接当作总GPU耗时。",
                  "缺失指标标为未记录；本报告不自动宣称应继续训练或已经达到峰值。",
                  "", "日志SHA256："+result["source"]["sha256"]])
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(DEFAULT_PROJECT))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.project / "outputs" / ("diagnosis651_to800_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if output.exists():
        raise ValueError("Output exists; refusing to overwrite")
    print("Reading completed training logs only; no GPU, tensor loading, training or evaluation.", flush=True)
    result = diagnose(args.project)
    result["analyzer_sha256"] = {str(p): digest(p) for p in (Path(__file__), Path(__file__).with_name("analyze_grpo_diagnostics.py"))}
    text = report(result)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "diagnosis.json", result)
    (output / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print("SAVED:", output.resolve())


if __name__ == "__main__":
    main()
