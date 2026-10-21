"""Verify the pinned upstream dataset hashes and task_id boundaries."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "upstream" / "shopping-grpo-longhorizon"
DATASETS = {
    "sft_train": ("data/sft/train.jsonl", "sft", "train", "rows"),
    "sft_validation": ("data/sft/validation.jsonl", "sft", "validation", "rows"),
    "grpo_train": ("data/grpo/train.jsonl", "grpo", "train", "tasks"),
    "grpo_validation": ("data/grpo/validation.jsonl", "grpo", "validation", "tasks"),
    "final_clean": ("data/evaluation/tasks.jsonl", "evaluation", None, "tasks"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_ids_jsonl(path: Path) -> list[int]:
    task_ids = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            task_id = row.get("task_id")
            if not isinstance(task_id, int) or isinstance(task_id, bool):
                raise ValueError(f"{path}:{line_number}: invalid task_id {task_id!r}")
            task_ids.append(task_id)
    return task_ids


def task_ids_parquet(path: Path) -> list[int]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path, columns=["extra_info"]).to_pylist()
    return [int(row["extra_info"]["task_id"]) for row in rows]


def main() -> None:
    metadata = {
        name: json.loads((ROOT / "data" / name / "metadata.json").read_text(encoding="utf-8"))
        for name in ("sft", "grpo", "evaluation")
    }
    ids = {}
    failures = []
    for label, (relative, kind, split, count_key) in DATASETS.items():
        path = ROOT / relative
        item = metadata[kind][split] if split else metadata[kind]
        expected_count = item[count_key]
        actual_ids = task_ids_jsonl(path)
        ids[label] = set(actual_ids)
        repeated = {task_id: count for task_id, count in Counter(actual_ids).items() if count > 1}
        expected_hash = item.get("sha256")
        actual_hash = sha256(path)
        print(f"{label}: rows={len(actual_ids)} unique_task_ids={len(ids[label])} duplicates={len(repeated)}")
        print(f"  sha256={actual_hash}" + (f" expected={expected_hash} match={actual_hash == expected_hash}" if expected_hash else " (no manifest hash)"))
        if len(actual_ids) != expected_count or repeated or (expected_hash and actual_hash != expected_hash):
            failures.append(label)

    for split in ("train", "validation"):
        path = ROOT / "data" / "grpo" / f"{split}.parquet"
        expected_hash = metadata["grpo"][split]["parquet_sha256"]
        actual_hash = sha256(path)
        print(f"grpo_{split}.parquet: sha256={actual_hash} match={actual_hash == expected_hash}")
        if actual_hash != expected_hash:
            failures.append(f"grpo_{split}_parquet_hash")
        try:
            parquet_ids = task_ids_parquet(path)
        except (ImportError, OSError) as exc:
            print(f"  CONTENT NOT VERIFIED: {type(exc).__name__}: {exc}")
            failures.append(f"grpo_{split}_parquet_read")
        else:
            same = set(parquet_ids) == ids[f"grpo_{split}"]
            repeated = len(parquet_ids) != len(set(parquet_ids))
            print(f"  rows={len(parquet_ids)} jsonl_task_ids_match={same} duplicates={repeated}")
            if len(parquet_ids) != metadata["grpo"][split]["tasks"] or not same or repeated:
                failures.append(f"grpo_{split}_parquet_ids")

    for (left, left_ids), (right, right_ids) in combinations(ids.items(), 2):
        overlap = left_ids & right_ids
        print(f"{left} ∩ {right}: {len(overlap)}" + (f" examples={sorted(overlap)[:10]}" if overlap else ""))
        if overlap:
            failures.append(f"{left}_{right}_overlap")

    if failures:
        raise SystemExit("AUDIT INCOMPLETE/FAILED: " + ", ".join(failures))
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
