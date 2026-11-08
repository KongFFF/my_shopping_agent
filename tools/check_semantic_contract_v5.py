"""Validate derived semantic-v2 contracts against original 50/1000-task queries."""
import argparse
import json
from pathlib import Path
from build_semantic_pilot50 import TARGET, source_tasks, verify
from extract_semantic_requirements_r5 import loads, v2_backend
from semantic_contract_v2 import DIMENSIONS


def check(path, mode="50", allow_partial=False):
    verify()
    tasks = ([loads(x) for x in (TARGET / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
             if mode == "50" else source_tasks()[0])
    if mode == "regression50":
        from build_semantic_holdout import verify as verify_holdout
        tasks = verify_holdout()
    originals = {t["task_id"]: t["query"] for t in tasks}
    seen, errors, unresolved, review_findings = set(), [], [], []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            contract = loads(line)
            tid = contract["task_id"]
            if tid not in originals or tid in seen:
                raise ValueError("Unexpected or duplicate task ID")
            backend = v2_backend({"properties": {"schema_version": {"const": contract.get("schema_version")}}})
            if backend is None:
                raise ValueError("Unsupported semantic contract version")
            backend.validate(contract, source_query=originals[tid], extraction=True)
            seen.add(tid)
            findings = backend.split_flags(contract, backend.lint(contract))[1] if hasattr(backend, "split_flags") else []
            if findings:
                review_findings.append({"task_id": tid, "findings": findings})
            if findings or contract["pending_mentions"] or any(r["importance"] == "uncertain" for d in DIMENSIONS for r in contract[d]):
                unresolved.append(tid)
        except Exception as exc:
            errors.append({"line": number, "error": getattr(exc, "message", str(exc))[:1800]})
    missing = sorted(set(originals) - seen)
    return {"passed": not errors and (allow_partial or not missing), "validated_tasks": len(seen),
            "missing_task_ids": missing, "unresolved_task_ids": unresolved, "errors": errors,
            "semantic_review_findings": review_findings,
            "scope": "Program/source/lint checks, not semantic accuracy or human approval; unresolved tasks cannot be scored."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path)
    parser.add_argument("--mode", choices=["50", "1000", "regression50"], default="50")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = check(args.responses, args.mode, args.allow_partial)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
