"""Seal a fresh TRAIN-only query sample after protocol fixation, without inference."""
import argparse
import hashlib
import json
from pathlib import Path
from build_semantic_pilot50 import ROOT, TARGET, source_tasks

PACKAGE=ROOT/"experiments/semantic_holdout50_v1"
SEED="semantic-generalization-holdout50-20261103-v1"


def sha(raw):return hashlib.sha256(raw).hexdigest()


def protocol_signature():
    from extract_semantic_requirements import snapshot
    files=snapshot("50","semantic-v2-r4")[3]["files_sha256"]
    return {k:v for k,v in files.items() if k not in ("tasks.jsonl","requests.jsonl")}


def select(tasks, excluded, excluded_queries, count=50):
    eligible=[t for t in tasks if t["task_id"] not in excluded and sha(t["query"].encode()) not in excluded_queries]
    result=[];seen=set()
    for t in sorted(eligible,key=lambda t:sha(f"{SEED}\0{t['task_id']}".encode())):
        h=sha(t["query"].encode())
        if h not in seen:result.append(t);seen.add(h)
        if len(result)==count:break
    if len(result)!=count:raise ValueError("Insufficient unseen distinct training queries")
    return sorted(result,key=lambda t:t["task_id"])


def freeze():
    if (PACKAGE/"lock.json").exists():
        tasks=verify();return {"tasks":len(tasks),"status":"existing sealed sample verified","api_calls":0}
    tasks,sources=source_tasks()
    excluded={json.loads(l)["task_id"] for l in (TARGET/"tasks.jsonl").read_text(encoding="utf-8").splitlines()}
    logs={}
    for path in sorted((ROOT/"outputs").glob("semantic*/attempts.jsonl")):
        raw=path.read_bytes();logs[str(path.relative_to(ROOT))]=sha(raw)
        for line in raw.decode("utf-8").splitlines():
            if line.strip():excluded.add(json.loads(line)["task_id"])
    queries={sha(t["query"].encode()) for t in tasks if t["task_id"] in excluded}
    chosen=select(tasks,excluded,queries)
    raw="".join(json.dumps(t,ensure_ascii=False,separators=(",",":"))+"\n" for t in chosen).encode()
    lock={"version":"semantic-holdout50-v1","seed":SEED,"sample_scope":"Original training queries only; not Dev-50 or Final Clean","exposure_definition":"Not in the original semantic pilot or recorded API task IDs at freeze; not used for reference editing. Historical static dataset inspection is not claimed absent.","tasks":len(chosen),"tasks_sha256":sha(raw),"source_hashes":sources,"excluded_ids":sorted(excluded),"excluded_query_hashes":sorted(queries),"exposure_logs_sha256":logs,"profile":"semantic-v2-r4","protocol_files_sha256":protocol_signature(),"runner_sha256":sha((ROOT/"tools/extract_semantic_requirements.py").read_text(encoding="utf-8").encode()),"api_calls":0,"answers_generated":False}
    PACKAGE.mkdir(parents=True,exist_ok=True)
    (PACKAGE/"tasks.jsonl").write_bytes(raw)
    (PACKAGE/"lock.json").write_text(json.dumps(lock,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    verify()
    return {"tasks":50,"excluded_recorded_ids":len(excluded),"status":"sealed, no answers or labels generated","api_calls":0}


def verify(profile="semantic-v2-r4"):
    lock=json.loads((PACKAGE/"lock.json").read_text(encoding="utf-8"))
    if profile!=lock["profile"]:raise ValueError("Holdout protocol/profile mismatch")
    raw=(PACKAGE/"tasks.jsonl").read_bytes()
    if sha(raw)!=lock["tasks_sha256"]:raise ValueError("Sealed query file changed")
    tasks=[json.loads(l) for l in raw.decode().splitlines()]
    originals,sources=source_tasks();by={t["task_id"]:t["query"] for t in originals}
    if sources!=lock["source_hashes"] or len(tasks)!=50 or len({t["task_id"] for t in tasks})!=50:raise ValueError("Holdout source/size mismatch")
    seen=set()
    for t in tasks:
        h=sha(t["query"].encode())
        if by.get(t["task_id"])!=t["query"] or t["task_id"] in lock["excluded_ids"] or h in lock["excluded_query_hashes"] or h in seen:raise ValueError("Holdout overlap or source mismatch")
        seen.add(h)
    if protocol_signature()!=lock["protocol_files_sha256"]:raise ValueError("Frozen protocol changed; do not silently tune on the sealed holdout")
    if sha((ROOT/"tools/extract_semantic_requirements.py").read_text(encoding="utf-8").encode())!=lock["runner_sha256"]:raise ValueError("Frozen runner changed")
    return tasks


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--verify",action="store_true");args=parser.parse_args()
    print(json.dumps({"tasks":len(verify()),"status":"sealed sample verified","api_calls":0} if args.verify else freeze(),ensure_ascii=False))
