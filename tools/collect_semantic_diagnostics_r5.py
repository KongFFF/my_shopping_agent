"""Collect remaining diagnostic responses from an immutable existing run.

No schema/prompt relaxation. Saved model failures are retained and skipped.
The existing extractor runs unchanged, once per new task, with its normal
retry/feedback behavior. All history is preserved in a separate directory.
"""
import argparse
import contextlib
import os
from pathlib import Path
import sys

from extract_semantic_requirements_r5 import (ROOT, TARGET, atomic, digest, dump, endpoint,
    execute, journal_read, loads, snapshot, summarize)


def model_reply(record):
    if record.get("http_status") != 200:
        return False
    try:
        message = loads(record["raw_response"])["choices"][0]["message"]
        return isinstance(message, dict) and any(k in message for k in ("content", "refusal", "tool_calls"))
    except (ValueError, KeyError, IndexError, TypeError):
        return False


def prepare_collection(source, output, attempts=2, timeout=180):
    source, output = source.resolve(), output.resolve()
    if output == source or output.is_relative_to(source) or output.is_relative_to(TARGET.resolve()):
        raise ValueError("Use a separate output directory outside the source/frozen pilot")
    if (source / ".run.lock").exists():
        raise ValueError("Source is running/locked; do not run both collectors concurrently")
    manifest_raw = (source / "run_manifest.json").read_bytes()
    original = loads(manifest_raw.decode("utf-8"))
    driver = ROOT / "tools/extract_semantic_requirements_r5.py"
    driver_sha = digest(driver.read_text(encoding="utf-8").encode("utf-8"))
    if original.get("runner_sha256") != driver_sha:
        raise ValueError("Source extractor differs from current extractor; no silent driver upgrade")
    tasks, requests, files, provenance = snapshot(original["dataset"]["mode"], original["api"].get("extraction_profile", "frozen-v1"))
    if provenance != original["dataset"]:
        raise ValueError("Source data/protocol differs from current snapshot")
    for name, expected in files.items():
        raw = (source / name).read_bytes()
        if raw != expected or digest(raw) != original["dataset"]["files_sha256"][name]:
            raise ValueError(f"Source input snapshot changed: {name}")
    # A prepare-only run intentionally has no journal yet. This also supports
    # collecting a fresh fixed protocol without the extractor's three-task stop.
    seed_path = source / "attempts.jsonl"
    seed = seed_path.read_bytes() if seed_path.exists() else b""
    seed_records = journal_read(source / "attempts.jsonl")
    manifest = {"version": "semantic-diagnostic-collection-v1", "source_run": str(source),
                "source_manifest_sha256": digest(manifest_raw), "source_journal_sha256": digest(seed),
                "seed_attempts": len(seed_records), "api": original["api"], "dataset": provenance,
                "runner_sha256": driver_sha,
                "collector_sha256": digest(Path(__file__).read_text(encoding="utf-8").encode("utf-8")),
                "controls": {"attempts_per_new_task": attempts, "timeout": timeout,
                             "skip_saved_model_failures": True, "stop_after_transport_failures": 3}}
    own = output / "run_manifest.json"
    if own.exists():
        if loads(own.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Collection configuration/source changed; use a new output directory")
        for name, expected in files.items():
            if (output / name).read_bytes() != expected:
                raise ValueError(f"Collection snapshot changed: {name}")
        if (output / "seed_attempts.jsonl").read_bytes() != seed or (output / "source_run_manifest.json").read_bytes() != manifest_raw:
            raise ValueError("Saved source evidence changed")
    else:
        if any(p.name != ".run.lock" for p in output.iterdir()):
            raise ValueError("Use an empty output directory for a new collection")
        for name, raw in files.items():
            (output / name).write_bytes(raw)
        (output / "seed_attempts.jsonl").write_bytes(seed)
        (output / "source_run_manifest.json").write_bytes(manifest_raw)
        atomic(own, dump(manifest) + "\n")
    if (seed_path.read_bytes() if seed_path.exists() else b"") != seed:
        raise ValueError("Source journal changed during preparation; stop source run")
    return tasks, requests, loads(files["output.schema.json"].decode("utf-8")), manifest


def aggregate(output, tasks, schema, manifest):
    records = journal_read(output / "seed_attempts.jsonl")
    for task in tasks:
        path = output / "per_task" / str(task["task_id"]) / "attempts.jsonl"
        own = journal_read(path)
        if any(r["task_id"] != task["task_id"] for r in own):
            raise ValueError("Per-task journal contains an unrelated task ID")
        records.extend(own)
    atomic(output / "attempts.jsonl", "".join(dump(r) + "\n" for r in records))
    valid, report = summarize(output, tasks, schema, records, manifest["api"].get("evidence_offsets", "strict"))
    collected = {r["task_id"] for r in records if model_reply(r)}
    all_ids = {t["task_id"] for t in tasks}
    if not collected <= all_ids:
        raise ValueError("Collected response outside frozen task set")
    failures = []
    for tid in sorted(collected - valid.keys()):
        failures.append({"task_id": tid, "attempt_lines": [i for i, r in enumerate(records, 1) if r["task_id"] == tid and model_reply(r)],
                         "errors": [r.get("error", "See original raw response") for r in records if r["task_id"] == tid and model_reply(r)]})
    atomic(output / "collected_failures.jsonl", "".join(dump(r) + "\n" for r in failures))
    report.update(collected_tasks=len(collected), collection_complete=collected == all_ids,
                  not_collected_task_ids=sorted(all_ids-collected),
                  collected_validation_failed_task_ids=sorted(collected-valid.keys()),
                  seed_api_attempts=manifest["seed_attempts"],
                  new_api_attempts=len(records)-manifest["seed_attempts"],
                  collection_note="Collection complete does not mean validation or semantic approval. Usage includes source history; inspect per_task for new attempts.")
    atomic(output / "report.json", dump(report) + "\n")
    return records, report


class TaskLog:
    """Keep live attempt lines while hiding intermediate per-task summaries."""
    def __init__(self, target):
        self.target, self.buffer = target, ""

    def write(self, value):
        self.buffer += value
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.startswith("task="):
                self.target.write(line + "\n")
                self.target.flush()
        return len(value)

    def flush(self):
        self.target.flush()


def collect(output, tasks, requests, schema, manifest, key, limit=None, runner=execute):
    records, report = aggregate(output, tasks, schema, manifest)
    initial_attempts = len(records)
    seen = {r["task_id"] for r in records if model_reply(r)}
    remaining = [(t, r) for t, r in zip(tasks, requests) if t["task_id"] not in seen]
    if limit is not None:
        remaining = remaining[:limit]
    transport_failures, stop_reason = 0, None
    print(f"Collected {len(seen)}/{len(tasks)}; validated {report['validated_tasks']}; not collected {len(tasks)-len(seen)}", flush=True)
    try:
        for task, request in remaining:
            task_dir = output / "per_task" / str(task["task_id"])
            task_dir.mkdir(parents=True, exist_ok=True)
            before = len(journal_read(task_dir / "attempts.jsonl"))
            print(f"Collecting task={task['task_id']}", flush=True)
            with contextlib.redirect_stdout(TaskLog(sys.stdout)):
                runner(task_dir, [task], [request], schema, manifest["api"], key,
                       attempts=manifest["controls"]["attempts_per_new_task"], timeout=manifest["controls"]["timeout"])
            recent = journal_read(task_dir / "attempts.jsonl")[before:]
            _, report = aggregate(output, tasks, schema, manifest)
            fatal = next((r.get("http_status") for r in recent
                          if isinstance(r.get("http_status"), int) and 400 <= r["http_status"] < 500 and r["http_status"] != 429), None)
            if fatal is not None:
                stop_reason = f"HTTP {fatal}; fix authentication/configuration before resuming"
                break
            transport_failures = 0 if any(model_reply(r) for r in recent) else transport_failures + 1
            if transport_failures >= 3:
                stop_reason = "3 consecutive transport/API protocol failures"
                break
    finally:
        records, report = aggregate(output, tasks, schema, manifest)
        report["api_attempts_this_invocation"] = len(records) - initial_attempts
        report["stop_reason"] = stop_reason
        atomic(output / "report.json", dump(report) + "\n")
        print(dump({k: report[k] for k in ("total_tasks", "collected_tasks", "collection_complete", "validated_tasks",
                                         "collected_validation_failed_task_ids", "api_attempts_this_invocation", "stop_reason")}), flush=True)
        print(f"Report: {output / 'report.json'}", flush=True)
    return 0 if report["collection_complete"] and stop_reason is None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--allow-school-http", action="store_true")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int)
    offline = parser.add_mutually_exclusive_group()
    offline.add_argument("--prepare-only", action="store_true")
    offline.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if min(args.attempts, args.timeout) <= 0 or args.limit is not None and args.limit <= 0:
        parser.error("Numeric controls must be positive")
    source, output = args.source_run.resolve(), args.output.resolve()
    # Validate destination BEFORE creating it; never write inside source.
    if output == source or output.is_relative_to(source) or output.is_relative_to(TARGET.resolve()):
        raise ValueError("Use a separate output outside source/frozen pilot")
    original = loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    url = original["api"]["endpoint"]
    if not url.endswith("/chat/completions"):
        raise ValueError("Source endpoint is not Chat Completions")
    if endpoint(url[:-len('/chat/completions')], allow_school_http=args.allow_school_http,
                local_only=args.prepare_only or args.check_only) != url:
        raise ValueError("Source endpoint changed")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".run.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise ValueError("Collection is locked; check for an existing process") from None
    try:
        with handle:
            handle.write(str(os.getpid()))
        tasks, requests, schema, manifest = prepare_collection(source, output, args.attempts, args.timeout)
        if args.prepare_only or args.check_only:
            _, report = aggregate(output, tasks, schema, manifest)
            print(dump({k: report[k] for k in ("total_tasks", "collected_tasks", "validated_tasks", "not_collected_task_ids", "new_api_attempts")}))
            print("Local preparation/check only; no API requests.")
            return 0 if args.prepare_only or report["collection_complete"] else 2
        key = os.getenv(args.api_key_env, "").strip()
        if not key:
            raise ValueError(f"Missing environment variable {args.api_key_env}; no API requests made")
        return collect(output, tasks, requests, schema, manifest, key, args.limit)
    finally:
        lock.unlink()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; rerun the same collection command to resume.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
