"""Read existing shopping GRPO diagnostics with Python's standard library only.

No model, environment, torch, network, or Final benchmark is accessed.
Use the two successful formal log segments, excluding smoke and failed runs.
When resuming from an earlier checkpoint, select each file's retained step range.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def describe(values):
    values = sorted(values)
    if not values:
        return {"count": 0}

    def quantile(p):
        index = (len(values) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return {"count": len(values), "mean": statistics.fmean(values),
            "min": values[0], "p10": quantile(.1), "p50": quantile(.5),
            "p90": quantile(.9), "max": values[-1]}


def rate(n, d):
    return n / d if d else None


def actions(rollout):
    return tuple(json.dumps(action, sort_keys=True, ensure_ascii=False)
                 for action in rollout.get("actions", []))


def action_digest(sequence):
    return hashlib.sha256(json.dumps(sequence).encode()).hexdigest()


def gold(rollout):
    # Training-side success proxy; independent strict evaluation has more checks.
    return (rollout.get("reward_type") == "gold_purchase"
            and rollout.get("reward_valid") is True
            and rollout.get("done") is True
            and not rollout.get("infrastructure_invalid")
            and not rollout.get("reward_unverifiable"))


def group_rollouts(rollouts, source):
    grouped = defaultdict(list)
    for rollout in rollouts:
        if "uid" not in rollout or "task_id" not in rollout:
            raise ValueError(f"{source}: rollout missing uid/task_id")
        grouped[str(rollout["uid"])].append(rollout)
    return grouped


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_step_range(value):
    try:
        start, end = map(int, value.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("step range must be START:END, e.g. 1:350") from exc
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError("step range must satisfy 1 <= START <= END")
    return start, end


def analyze(paths, expected_updates=None, expected_group_size=4, step_ranges=None):
    if step_ranges is not None:
        if len(step_ranges) != len(paths):
            raise ValueError("provide one --step-ranges START:END per diagnostics file, in the same order")
        for index, (start, end) in enumerate(step_ranges):
            if start < 1 or end < start:
                raise ValueError("step ranges must satisfy 1 <= START <= END")
            for other_start, other_end in step_ranges[:index]:
                if max(start, other_start) <= min(end, other_end):
                    raise ValueError("selected step ranges overlap; specify the retained training chain explicitly")
    counts, reward_types = Counter(), Counter()
    invalid_reasons, event_counts = Counter(), Counter()
    tasks = defaultdict(Counter)
    sequences_by_task = defaultdict(set)
    generated_gold_by_task = defaultdict(int)
    trained_gold_by_task = defaultdict(int)
    group_rows, update_rows, sources = [], [], []
    seen_updates, trained_keys = set(), set()
    trained_advantage_scales = []
    excluded_generated_rollouts = 0
    for file_index, path in enumerate(paths):
        selected_range = step_ranges[file_index] if step_ranges is not None else None
        selected_events, excluded_events = Counter(), Counter()
        excluded_optimizer_steps = []
        source_excluded_generated_rollouts = 0
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(chunk)
        source_info = {"path": str(path.resolve()), "sha256": digest.hexdigest(),
                       "selected_step_range": list(selected_range) if selected_range else None}
        sources.append(source_info)
        with path.open(encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                source = f"{path}:{line_no}"
                try:
                    event = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{source}: invalid JSON") from exc
                kind = event["event"]
                step = int(event["global_step"])
                if selected_range is not None and not selected_range[0] <= step <= selected_range[1]:
                    excluded_events[kind] += 1
                    if kind == "optimizer_step":
                        excluded_optimizer_steps.append(step)
                    elif kind == "generation_batch":
                        number = len(event.get("rollouts", []))
                        excluded_generated_rollouts += number
                        source_excluded_generated_rollouts += number
                    continue
                selected_events[kind] += 1
                event_counts[kind] += 1
                if kind == "skipped_update":
                    counts["skipped_update_events"] += 1
                    continue
                if kind not in {"generation_batch", "optimizer_step"}:
                    continue
                grouped = group_rollouts(event.get("rollouts", []), source)
                if not grouped:
                    raise ValueError(f"{source}: no rollout diagnostics")
                if kind == "optimizer_step":
                    if step in seen_updates:
                        raise ValueError(f"{source}: duplicate optimizer step {step}")
                    seen_updates.add(step)
                    row = {"step": step, "source": str(path)}
                    for key, value in event.get("metrics", {}).items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            if not math.isfinite(value):
                                raise ValueError(f"{source}: non-finite {key}")
                            row[key] = value
                    update_rows.append(row)
                descriptions = {str(g["uid"]): g for g in event.get("groups", [])}
                if kind == "generation_batch" and set(grouped) != set(descriptions):
                    raise ValueError(f"{source}: group/rollout uid mismatch")
                for uid, rollouts in grouped.items():
                    if len(rollouts) != expected_group_size:
                        raise ValueError(f"{source}: group size {len(rollouts)}")
                    ids = {int(r["task_id"]) for r in rollouts}
                    if len(ids) != 1:
                        raise ValueError(f"{source}: group mixes task IDs")
                    task = next(iter(ids))
                    if kind == "optimizer_step":
                        trained_keys.add((str(path), uid))
                        counts["trained_groups"] += 1
                        counts["trained_rollouts"] += len(rollouts)
                        n_gold = sum(map(gold, rollouts))
                        counts["trained_mixed_gold_groups"] += 0 < n_gold < len(rollouts)
                        counts["trained_no_gold_groups"] += n_gold == 0
                        raw_utilities = [(r.get("reward") or {}).get("terminal_utility") for r in rollouts]
                        if all(isinstance(u, (int, float)) and math.isfinite(u) for u in raw_utilities):
                            group_mean = statistics.fmean(raw_utilities)
                            trained_advantage_scales.append(statistics.fmean(abs(u - group_mean) for u in raw_utilities))
                        tasks[task]["trained_groups"] += 1
                        trained_gold_by_task[task] += n_gold
                        continue
                    detail = descriptions[uid]
                    utilities = [float(v) for v in detail["terminal_utilities"]]
                    if len(utilities) != len(rollouts) or not all(map(math.isfinite, utilities)):
                        raise ValueError(f"{source}: invalid group utilities")
                    mean = statistics.fmean(utilities)
                    seqs = [actions(r) for r in rollouts]
                    has_actions = all("actions" in r for r in rollouts)
                    n_gold = sum(map(gold, rollouts))
                    counts["generated_groups"] += 1
                    counts["generated_rollouts"] += len(rollouts)
                    counts["kept_groups"] += bool(detail["kept"])
                    counts["accepted_before_commit_groups"] += bool(detail["accepted"])
                    reason = detail.get("drop_reason")
                    if reason:
                        counts[f"dropped_{reason}_groups"] += 1
                    for reason in detail.get("sampling_invalid_reasons", []):
                        invalid_reasons[reason] += 1
                    counts["mixed_gold_groups"] += 0 < n_gold < len(rollouts)
                    counts["all_gold_groups"] += n_gold == len(rollouts)
                    counts["no_gold_groups"] += n_gold == 0
                    counts["kept_no_gold_groups"] += bool(detail["kept"]) and n_gold == 0
                    tasks[task]["generated_groups"] += 1
                    tasks[task]["kept_groups"] += bool(detail["kept"])
                    tasks[task]["constant_reward_groups"] += detail.get("drop_reason") == "constant_reward"
                    tasks[task]["invalid_groups"] += detail.get("drop_reason") == "sampling_invalid"
                    generated_gold_by_task[task] += n_gold
                    if has_actions:
                        sequences_by_task[task].update(action_digest(s) for s in seqs)
                    reward_types.update(r.get("reward_type", "unknown") for r in rollouts)
                    first_divergences = []
                    if has_actions:
                        for left, right in itertools.combinations(seqs, 2):
                            common = 0
                            for a, b in zip(left, right):
                                if a != b:
                                    break
                                common += 1
                            if left != right:
                                first_divergences.append(common + 1)
                    group_rows.append({
                        "step": step, "source": str(path), "uid": uid, "task_id": task,
                        "kept": bool(detail["kept"]), "accepted": bool(detail["accepted"]),
                        "drop_reason": detail.get("drop_reason") or "",
                        "gold_rollouts": n_gold, "group_size": len(rollouts),
                        "utility_min": min(utilities), "utility_max": max(utilities),
                        "utility_std": statistics.pstdev(utilities),
                        "centered_utility_mean_abs": statistics.fmean(abs(u - mean) for u in utilities),
                        "unique_action_sequences": len(set(seqs)) if has_actions else None,
                        "mean_first_divergence_action": statistics.fmean(first_divergences) if first_divergences else None,
                        "mean_steps": statistics.fmean(r.get("steps", 0) for r in rollouts),
                    })
        source_info.update({
            "selected_events": dict(selected_events),
            "excluded_events": dict(excluded_events),
            "excluded_optimizer_steps": excluded_optimizer_steps,
            "excluded_generated_rollouts": source_excluded_generated_rollouts,
        })
    update_rows.sort(key=lambda r: r["step"])
    for row in group_rows:
        row["trained"] = (row["source"], row["uid"]) in trained_keys
    if expected_updates is not None and seen_updates != set(range(1, expected_updates + 1)):
        missing = sorted(set(range(1, expected_updates + 1)) - seen_updates)
        extra = sorted(seen_updates - set(range(1, expected_updates + 1)))
        raise ValueError(f"optimizer coverage mismatch: missing={missing}, extra={extra}")
    task_fields = ("generated_groups", "trained_groups", "kept_groups", "constant_reward_groups", "invalid_groups")
    task_rows = [{"task_id": task, **{k: tasks[task][k] for k in task_fields},
                  "generated_gold_rollouts": generated_gold_by_task[task],
                  "trained_gold_rollouts": trained_gold_by_task[task],
                  "unique_generated_action_sequences": len(sequences_by_task[task])}
                 for task in sorted(tasks)]
    generated, kept = counts["generated_groups"], counts["kept_groups"]
    summary = {
        "schema": "shopping-grpo-offline-diagnosis-v2", "sources": sources,
        "events": dict(event_counts), "counts": dict(counts),
        "generation_cost_scope": {
            "retained_chain_generated_rollouts": counts["generated_rollouts"],
            "excluded_generated_rollouts": excluded_generated_rollouts,
            "all_input_generated_rollouts": counts["generated_rollouts"] + excluded_generated_rollouts,
            "retained_trained_over_all_input_generated_rollouts": rate(
                counts["trained_rollouts"], counts["generated_rollouts"] + excluded_generated_rollouts),
        },
        "optimizer_steps": sorted(seen_updates),
        "rates": {
            "kept_over_generated_groups": rate(kept, generated),
            "committed_trained_over_generated_groups": rate(counts["trained_groups"], generated),
            "mixed_gold_over_generated_groups": rate(counts["mixed_gold_groups"], generated),
            "no_gold_over_kept_groups": rate(counts["kept_no_gold_groups"], kept),
            "mixed_gold_over_trained_groups": rate(counts["trained_mixed_gold_groups"], counts["trained_groups"]),
            "no_gold_over_trained_groups": rate(counts["trained_no_gold_groups"], counts["trained_groups"]),
        },
        "task_coverage": {
            "generated_unique": sum(t["generated_groups"] > 0 for t in tasks.values()),
            "trained_unique": sum(t["trained_groups"] > 0 for t in tasks.values()),
            "generated_tasks_never_trained": sum(t["generated_groups"] > 0 and t["trained_groups"] == 0 for t in tasks.values()),
            "generated_tasks_never_gold": sum(t["generated_groups"] > 0 and generated_gold_by_task[k] == 0 for k, t in tasks.items()),
        },
        "generated_reward_type_counts": dict(reward_types),
        "invalid_group_reason_counts": dict(invalid_reasons),
        "kept_group_centered_utility_mean_abs": describe([
            r["centered_utility_mean_abs"] for r in group_rows if r["kept"]]),
        "trained_group_centered_utility_mean_abs": describe(trained_advantage_scales),
        "kept_group_unique_action_sequences": describe([
            r["unique_action_sequences"] for r in group_rows
            if r["kept"] and r["unique_action_sequences"] is not None]),
        "metric_windows": [],
        "notes": [
            "Gold is a training-side proxy, not the full independent strict-success check.",
            "Centered utility is an advantage-scale proxy only when std normalization and length shaping are disabled; it is not gradient magnitude.",
            "accepted groups can later be discarded by skipped updates; trained counts use optimizer_step events only.",
            "Action diversity excludes rejected calls absent from the actions field.",
            "Metric names containing kl may concern rollout vs actor; no SFT reference KL is inferred.",
            "Main counts/rates use selected step ranges. Excluded events are audited per source; generation_cost_scope also counts their sampled rollouts. This is input-log cost, not total project cost.",
        ],
    }
    interesting = ("entropy", "clip", "kl", "grad_norm", "actor/lr", "pg_loss", "log_prob")
    for start in range(1, max(seen_updates, default=0) + 1, 50):
        rows = [r for r in update_rows if start <= r["step"] < start + 50]
        keys = sorted({k for r in rows for k in r if any(word in k.lower() for word in interesting)})
        summary["metric_windows"].append({
            "start": start, "end": start + 49, "updates": len(rows),
            "metrics": {k: describe([r[k] for r in rows if k in r]) for k in keys},
        })
    return summary, group_rows, update_rows, task_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", nargs="+", required=True, type=Path)
    parser.add_argument("--step-ranges", nargs="+", type=parse_step_range,
                        help="one retained START:END range per input, e.g. 1:350 351:500; excludes all event types outside each range")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-updates", type=int)
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()
    summary, groups, updates, tasks = analyze(
        args.diagnostics, args.expected_updates, args.group_size, args.step_ranges)
    # Fail rather than overwrite a previous diagnosis or touch source logs.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(args.output / "groups.csv", groups)
    write_csv(args.output / "updates.csv", updates)
    write_csv(args.output / "tasks.csv", tasks)
    print(json.dumps({k: summary[k] for k in ("counts", "rates", "task_coverage",
                     "kept_group_centered_utility_mean_abs", "generation_cost_scope")}, ensure_ascii=False, indent=2))
    for source in summary["sources"]:
        print(json.dumps({k: source[k] for k in (
            "path", "selected_step_range", "excluded_events", "excluded_optimizer_steps")},
            ensure_ascii=False))
    print(f"Saved diagnosis: {args.output.resolve()}")


if __name__ == "__main__":
    main()
