"""Offline migration probe of saved r2 answers, NOT a new LLM evaluation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from semantic_contract_v2_r3 import VERSION, DIMENSIONS, ATTRIBUTES, PACKAGE, normalize, chinese_number, price_basis_from_phrase
from semantic_relations_v2_r3 import canonicalize


def replay(source, output):
    source, output = source.resolve(), output.resolve()
    if output == source.parent or output.is_relative_to(source.parent):
        raise ValueError("Use a separate output directory; never overwrite the reviewed inputs")
    rows = [json.loads(l) for l in source.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [r["task_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task in reviewed source")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty probe output directory")
    output.mkdir(parents=True, exist_ok=True)
    results, valid = [], []
    for r in rows:
        raw = copy.deepcopy(r["raw_selected"])
        if raw["schema_version"] != "shopping-semantic-requirements-v2-draft":
            raise ValueError("This explicit migration probe accepts r2 answers only")
        raw["schema_version"] = VERSION
        task = {"task_id": r["task_id"], "query": r["query"]}
        item = {"task_id": r["task_id"], "source_attempt_line": r["selected_attempt_line"],
                "r2_program_valid": r["program_valid"], "r3_program_valid": False,
                "input_version_adapter": "r2 schema_version -> r3 for offline probe only",
                "approved_for_reward": False}
        try:
            _, preview = canonicalize(raw, task, DIMENSIONS, ATTRIBUTES, chinese_number, price_basis_from_phrase)
            item["canonicalization_preview"] = preview
            contract, audit = normalize(raw, task)
        except Exception as exc:
            item["error"] = getattr(exc, "message", str(exc))
        else:
            item.update(r3_program_valid=True, audit=audit, pending_mentions=len(contract["pending_mentions"]))
            valid.append(contract)
        results.append(item)
    write = lambda name, values: (output / name).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")
    write("probe_results.jsonl", results)
    write("replayed_candidates.jsonl", valid)
    report = {"scope": "Offline adapter/canonicalizer/guard replay of saved r2 answers; NOT new r3 LLM quality or semantic approval",
              "input": str(source), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "r3_files_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                  Path(__file__).parent / "semantic_contract_v2_r3.py", Path(__file__).parent / "semantic_coverage_v2_r3.py",
                  Path(__file__).parent / "semantic_relations_v2_r3.py", Path(__file__), PACKAGE / "system_prompt.txt", PACKAGE / "policy.json")},
              "total_tasks": len(rows), "r2_program_valid": sum(r["program_valid"] for r in rows),
              "r3_program_valid": len(valid), "diagnostic_complete": True, "all_candidates_passed": len(valid) == len(rows),
              "recovered_r2_failures": [r["task_id"] for r in results if not r["r2_program_valid"] and r["r3_program_valid"]],
              "newly_blocked_r2_passes": [r["task_id"] for r in results if r["r2_program_valid"] and not r["r3_program_valid"]],
              "canonicalized_task_ids": [r["task_id"] for r in results if r.get("canonicalization_preview")],
              "remaining_failures": [{"task_id": r["task_id"], "error": r["error"]} for r in results if not r["r3_program_valid"]],
              "api_calls": 0, "approved_for_reward": 0}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.input, args.output), ensure_ascii=False, indent=2))
