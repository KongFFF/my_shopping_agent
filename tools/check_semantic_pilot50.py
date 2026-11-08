"""Check frozen pilot extraction responses, without an LLM or semantic approval."""
import argparse
import json
from pathlib import Path
from jsonschema import Draft202012Validator
from build_semantic_pilot50 import TARGET, verify
from semantic_contract_v1 import DIMENSIONS, validate


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_response(contract, tasks, output_schema):
    Draft202012Validator(output_schema).validate(contract)
    tid = contract["task_id"]
    if tid not in tasks:
        raise ValueError(f"Task is not in frozen pilot: {tid}")
    validate(contract, source_query=tasks[tid])
    for dimension in DIMENSIONS:
        if any(r["review_status"] != "pending" for r in contract[dimension]):
            raise ValueError("An extraction model cannot approve its own output")
    return tid


def check(path, allow_partial=False):
    verify()
    tasks = {row["task_id"]: row["query"] for line in (TARGET / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
             for row in [json.loads(line)]}
    output_schema = json.loads((TARGET / "output.schema.json").read_text(encoding="utf-8"))
    seen, errors, pending = set(), [], 0
    for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            contract = json.loads(line, object_pairs_hook=reject_duplicate_keys)
            tid = validate_response(contract, tasks, output_schema)
            if tid in seen:
                raise ValueError(f"Duplicate response task: {tid}")
            seen.add(tid)
            pending += len(contract["pending_mentions"])
        except Exception as exc:
            errors.append({"line": index, "error": str(exc)[:800]})
    missing = sorted(set(tasks) - seen)
    return {"passed": not errors and (allow_partial or not missing), "validated_tasks": len(seen),
            "missing_task_ids": missing, "errors": errors, "pending_mentions": pending,
            "scope": "Structure/source checks only; not semantic accuracy, completeness or approval."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = check(args.responses, args.allow_partial)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
