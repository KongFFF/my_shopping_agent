"""Revalidate saved r3 API answers offline after a normalizer hotfix.

Original requests, replies, acceptance flags, reports and frozen snapshots stay
unchanged. Derived results explicitly bind both source and current code hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
from extract_semantic_requirements import journal_read, contract_from_response
from semantic_contract_v2_r3 import schema


def revalidate(source, output, profile="semantic-v2-r3"):
    if profile not in ("semantic-v2-r3", "semantic-v2-r4"):
        raise ValueError("Unsupported target profile")
    if profile == "semantic-v2-r4":
        from semantic_contract_v2_r4 import schema as target_schema, VERSION
    else:
        from semantic_contract_v2_r3 import schema as target_schema, VERSION
    source, output = source.resolve(), output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Use an independent derived directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Derived output must be empty")
    read = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    tasks = read(source / "tasks.jsonl")
    originals = {t["task_id"]: t for t in tasks}
    records = journal_read(source / "attempts.jsonl")
    if any(r["task_id"] not in originals for r in records):
        raise ValueError("Unexpected task in source journal")
    results, valid, chosen = [], {}, {}
    for line, record in enumerate(records, 1):
        tid = record["task_id"]
        item = {"task_id": tid, "source_attempt_line": line, "original_accepted": record.get("accepted",False),
                "original_error": record.get("error"), "current_valid": False}
        if record.get("http_status") == 200:
            audit = []
            try:
                response = record["raw_response"]
                if profile == "semantic-v2-r4":
                    envelope = json.loads(response)
                    candidate = json.loads(envelope["choices"][0]["message"]["content"])
                    if candidate["schema_version"] != "shopping-semantic-requirements-v2-r3-draft":
                        raise ValueError("Explicit r4 probe accepts saved r3 answers only")
                    candidate["schema_version"] = VERSION
                    envelope["choices"][0]["message"]["content"] = json.dumps(candidate, ensure_ascii=False)
                    response = json.dumps(envelope, ensure_ascii=False)
                    item["version_adapter"] = "r3 -> r4 for offline probe only; original response untouched"
                c = contract_from_response(response, originals[tid], target_schema(), changes=audit)
            except Exception as exc:
                item.update(error=getattr(exc,"message",str(exc)), error_type=type(exc).__name__)
                if record.get("accepted") and profile == "semantic-v2-r3":
                    raise ValueError(f"Originally accepted task no longer validates: {tid}") from exc
            else:
                item.update(current_valid=True, audit=audit)
                if tid not in valid:
                    valid[tid] = c
                    chosen[tid] = line
        results.append(item)
    output.mkdir(parents=True, exist_ok=True)
    write = lambda f, xs: (output/f).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in xs),encoding="utf-8")
    write("attempt_revalidation.jsonl",results)
    write("responses.normalized.jsonl",[valid[t["task_id"]] for t in tasks if t["task_id"] in valid])
    cards = []
    for t in tasks:
        tid = t["task_id"]
        own = [x for x in results if x["task_id"]==tid]
        index = chosen.get(tid,own[-1]["source_attempt_line"])
        raw = json.loads(json.loads(records[index-1]["raw_response"])["choices"][0]["message"]["content"])
        cards.append({**t,"program_valid":tid in valid,"selected_attempt_line":index,"raw_selected":raw,
                      "normalized":valid.get(tid),"attempts":len(own),"errors":[x.get("error") for x in own if not x["current_valid"]]})
    write("review_inputs.jsonl",cards)
    report={"scope":"Offline revalidation of saved r3 answers; r4 mode is an explicit version-adapted probe, not a new prompt evaluation; semantic review still required",
            "profile":profile,
            "source":str(source),"total_tasks":len(tasks),"attempts_revalidated":len(records),"original_program_valid":len({r["task_id"] for r in records if r.get("accepted")}),
            "validated_tasks":len(valid),"failed_task_ids":[t["task_id"] for t in tasks if t["task_id"] not in valid],
            "first_attempt_valid":sum(next(x for x in results if x["task_id"]==t["task_id"])["current_valid"] for t in tasks),
            "unresolved_tasks":sum(bool(c["pending_mentions"]) or any(r["importance"]=="uncertain" for d in ("category","budget","functions","specifications","brand","model") for r in c[d]) for c in valid.values()),
            "source_sha256":{f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in ("report.json","run_manifest.json","attempts.jsonl","tasks.jsonl")},
            "code_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name("semantic_contract_v2_r3.py"),Path(__file__).with_name("semantic_relations_v2_r3.py"),Path(__file__).with_name("semantic_contract_v2_r4.py"),Path(__file__).with_name("semantic_policy_v4.py"))},
            "api_calls":0,"approved_for_reward":0}
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--profile",choices=["semantic-v2-r3","semantic-v2-r4"],default="semantic-v2-r3")
    args=parser.parse_args()
    print(json.dumps(revalidate(args.source_run,args.output,args.profile),ensure_ascii=False,indent=2))
