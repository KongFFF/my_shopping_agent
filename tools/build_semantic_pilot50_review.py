"""Package the assistant's 50 per-task semantic review notes with source evidence.

No LLM calls, automatic semantic decisions or modifications to extraction outputs.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from build_semantic_pilot50 import TARGET, verify

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/semantic_school_50_v2_offsets"
OUT = ROOT / "analysis/semantic_pilot50_review_20261102"
LABELS = {
    "pass_draft": "未发现明显错误的草稿",
    "needs_resolution": "表达或政策待定",
    "needs_correction": "需要语义修正",
    "invalid_output": "程序失败且需修正",
}
KINDS = {
    "strength_error": "必需项与偏好错误", "strength_ambiguity": "强弱语用待定",
    "misclassification": "语义类型错误", "omission": "需求未独立保留",
    "measurement_scope": "测量或计数对象不完整", "component_scope": "部件及属性绑定不完整",
    "bundle_error": "套餐促销关系错误或不完整", "invented_quantity": "补造数量",
    "budget_policy_error": "预算政策误用", "numeric_relation": "数值比较关系问题",
    "price_basis": "计价分母缺失", "supported_pending": "已有类型但误判不支持",
    "schema_gap": "类型或核验标准缺口", "policy_gap": "解释政策待定",
    "dedup_review": "去重与计分重叠待审", "logical_scope": "逻辑或否定作用域",
    "format_failure": "格式结构失败",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def lines_bytes(rows):
    return ("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)).encode("utf-8")


def main():
    verify()
    notes = read_json(OUT / "review_notes.json")
    tasks = [json.loads(s) for s in (SOURCE / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    frozen = [json.loads(s) for s in (TARGET / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert tasks == frozen, "Review tasks must be the exact frozen 50 queries"
    assert {str(t["task_id"]) for t in tasks} == set(notes) and len(notes) == 50
    source_manifest = read_json(SOURCE / "run_manifest.json")
    for name, expected in source_manifest["dataset"]["files_sha256"].items():
        assert sha((SOURCE / name).read_bytes()) == expected, f"Changed input snapshot: {name}"
    outputs = {}
    for number, line in enumerate((SOURCE / "responses.normalized.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        c = json.loads(line)
        assert c["task_id"] not in outputs
        outputs[c["task_id"]] = (number, c)
    attempts = {}
    for number, line in enumerate((SOURCE / "attempts.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        r = json.loads(line)
        attempts.setdefault(r["task_id"], []).append((number, r))
    snapshots, reviews = [], []
    for task in tasks:
        tid = task["task_id"]
        note = notes[str(tid)]
        assert note["verdict"] in LABELS
        assert bool(note["findings"]) == (note["verdict"] != "pass_draft")
        for finding in note["findings"]:
            assert finding["code"] in KINDS
            assert finding["quote"] in task["query"], (tid, finding["quote"])
        for _, record in attempts[tid]:
            if record.get("http_status") == 200:
                response = json.loads(record["raw_response"])
                contract = json.loads(response["choices"][0]["message"]["content"])
                assert contract["query"] == task["query"] and contract["task_id"] == tid
        if tid in outputs:
            number, contract = outputs[tid]
            assert contract["query"] == task["query"]
            for dimension in ("category", "budget", "functions", "specifications", "brand", "model"):
                assert all(r["review_status"] == "pending" for r in contract[dimension])
            snapshot = {**task, "program_status": "validated_with_offset_policy",
                        "selected_contract": contract, "selected_source": {
                            "file": "outputs/semantic_school_50_v2_offsets/responses.normalized.jsonl", "line": number}}
            assert note["verdict"] != "invalid_output"
        else:
            failed = []
            for number, record in attempts[tid]:
                response = json.loads(record["raw_response"])
                failed.append({"source_line": number, "program_error": record.get("error"),
                               "raw_response_sha256": sha(record["raw_response"].encode("utf-8")),
                               "contract": json.loads(response["choices"][0]["message"]["content"])})
            snapshot = {**task, "program_status": "failed", "failed_attempts": failed,
                        "selected_source": {"file": "outputs/semantic_school_50_v2_offsets/attempts.jsonl"}}
            assert note["verdict"] == "invalid_output" and len(failed) == 2
        snapshots.append(snapshot)
        reviews.append({**task, "program_status": snapshot["program_status"], **note,
                        "reviewer": "Codex assistant per-task semantic review",
                        "approval_status": "not_human_approved", "ready_for_reward": False})
    counts = Counter(r["verdict"] for r in reviews)
    code_tasks = {code: [r["task_id"] for r in reviews if any(f["code"] == code for f in r["findings"])]
                  for code in KINDS}
    soft_budgets = [r["task_id"] for r in reviews if r["task_id"] in outputs
                    and any(x["importance"] == "preference" for x in outputs[r["task_id"]][1]["budget"])]
    extraction_report = read_json(SOURCE / "report.json")
    source_names = ["run_manifest.json", "tasks.jsonl", "responses.normalized.jsonl", "attempts.jsonl",
                    "normalizations.jsonl", "report.json", "system_prompt.txt", "output.schema.json"]
    source_hashes = {f"outputs/semantic_school_50_v2_offsets/{name}": sha((SOURCE / name).read_bytes()) for name in source_names}
    summary = {
        "review_version": "semantic-pilot50-per-task-review-v1", "review_date": "2026-11-02",
        "reviewer": "Codex assistant; no external LLM calls or human approval",
        "scope": "All original frozen 50 queries, 46 derived contracts, both attempts of each of 4 failed tasks",
        "total_reviewed": len(reviews), "verdict_counts": dict(counts),
        "program_validated": len(outputs), "program_failed": len(tasks) - len(outputs),
        "pending_tasks": sum(bool(c[1]["pending_mentions"]) for c in outputs.values()),
        "pending_mentions": sum(len(c[1]["pending_mentions"]) for c in outputs.values()),
        "budget_preference_task_ids": soft_budgets,
        "finding_task_ids": code_tasks, "source_sha256": source_hashes,
        "note": "Verdicts are diagnostic judgments with a conservative reward-readiness standard, not measured model accuracy; issue categories overlap. No review_status was changed.",
    }
    md = ["# 50题离线需求提取逐题语义审阅", "", "审阅日期：2026-11-02。审阅者：Codex助手逐题初审；不是另一个外部LLM的自动打分，也不是用户人工金标。",
          "原始Query、模型响应、坐标修正结果、冻结schema和实际奖励均未修改；本目录是独立审阅产物，未批准任何合同。", "",
          "## 范围与判定标准", "", "固定原训练集50题全部审阅：46份通过坐标政策的派生合同，另4题读取各自两次失败响应。只依据原Query及已确定的解释政策，不借用gold ASIN或商品标注补用户需求。",
          "预算‘左右’按既定上限×1.1处理，‘出头’按既定分位规则；这些词不自动把必需项变偏好，也不把预算容差套给承重、长度、数量等其他属性。未定义的‘九百多’等表达保留待审。",
          "主观要求、语用强弱和未定义容差作为待定事项；精确文字已保留但尚不足以让程序可靠核验的对象/层级，明确记为表达缺口，不全算成模型捏造。需求之间的AND/OR、部件、计价分母和买赠层级必须保持。", "",
          "## 总体结论", "", "暂不扩大1000题。先集中修正明确错误、确认表达/政策缺口，并完成这50题的复核。保持六个顶层维度，不因本审阅直接变更schema或重训。", "",
          "| 审阅结论 | 题数 | 含义 |", "|---|---:|---|"]
    explanations = {"pass_draft": "没有发现明显语义遗漏/歪曲，仍不是可直接用来训练的人工批准合同",
                    "needs_resolution": "主要信息保留，但表达、核验标准、去重或政策尚待确定",
                    "needs_correction": "至少发现一处需修正的分类、强弱、覆盖或关系表达",
                    "invalid_output": "程序校验失败，原始响应中还存在需处理的语义或表达问题"}
    for key in LABELS:
        md.append(f"| {LABELS[key]} | {counts[key]} | {explanations[key]} |")
    md += ["", f"程序通过{len(outputs)}/50，不能当作语义准确率；{summary['pending_tasks']}份通过结果含{summary['pending_mentions']}个pending mention。按本次审阅标准的分类也不能直接解释成模型准确率，更不能用覆盖性小样本外推1000题错误率。", "",
           f"现有通过结果中有{len(soft_budgets)}题把预算标为preference（ID：{'、'.join(map(str,soft_budgets))}）。本轮原文没有足以支持这些预算全部降为可选项的明确预算偏好修饰，逐题建议见下文。", "",
           "## 修正优先级与需用户敲定的边界", "",
           "1. 先修确定性错误：预算不因‘左右’降为偏好；味道/图案不能归颜色；产地不能归商家；禁止补造赠品数量、错用数值方向。",
           "2. 保持六个顶层维度，讨论维度内的表达：计数/测量对象、部件作用域、套餐与到手数、单价分母；现字段能保留的先保留忠实文本，不为凑数字丢失对象。",
           "3. 再决定真正的缺口：包装、产地、款式图案、味型、品相和工艺是否增加受控subtype；优先使用已有compatibility、target_user、delivery等字段，不把所有待审都当schema缺失。",
           "4. 用户尚需敲定：非预算‘约/左右’的容差；‘九百多’的预算规则；‘可以赠送’和提及店铺的强弱；‘质量好/百搭/最新款’的证据标准及时间基准。审阅中只列建议，没有替用户确立新规则。",
           "5. 实施后用同50题回归并做程序核验和语义复核，再决定是否换模型或扩大1000题。本次没有调用API，也没有将review_status改为approved。", "",
           "## 问题分布（允许交叉，不相加）", "", "| 问题类型 | 涉及题数 | ID |", "|---|---:|---|"]
    for code, ids in code_tasks.items():
        if ids:
            md.append(f"| {KINDS[code]} | {len(ids)} | {'、'.join(map(str,ids))} |")
    md += ["", "## 逐题索引", "", "| ID | 程序结果 | 审阅结论 |", "|---|---|---|"]
    for r in reviews:
        md.append(f"| {r['task_id']} | {'通过（含坐标政策）' if r['program_status'] != 'failed' else '失败'} | {LABELS[r['verdict']]} |")
    md += ["", "## 逐题审阅记录", "", "完整原始提取字段见同目录input_snapshot.jsonl；下文逐题保留原文、覆盖核对、发现与修改建议。", ""]
    for r in reviews:
        md += [f"### {r['task_id']} · {LABELS[r['verdict']]}", "", f"原文：{r['query']}", "", f"覆盖核对：{r['coverage']}", ""]
        for f in r["findings"]:
            md += [f"- **{KINDS[f['code']]}**（原文：{f['quote']}）：{f['problem']} 建议：{f['suggestion']}", ""]
        if not r["findings"]:
            md += ["未发现明显语义错误，保留为待人工确认草稿。", ""]
        if r.get("notes"):
            md += [f"核验注意：{r['notes']}", ""]
    files = {"input_snapshot.jsonl": lines_bytes(snapshots), "reviewed_tasks.jsonl": lines_bytes(reviews),
             "summary.json": json_bytes(summary), "逐题语义审阅.md": ("\n".join(md) + "\n").encode("utf-8")}
    for name, raw in files.items():
        (OUT / name).write_bytes(raw)
    manifest = {"version": summary["review_version"], "count": 50, "source_sha256": source_hashes,
                "files_sha256": {name: sha(raw) for name, raw in files.items()},
                "review_notes_sha256": sha((OUT / "review_notes.json").read_bytes()),
                "builder_normalized_sha256": sha(Path(__file__).read_text(encoding="utf-8").encode("utf-8"))}
    (OUT / "review_manifest.json").write_bytes(json_bytes(manifest))
    for name, expected in source_hashes.items():
        assert sha((ROOT / name).read_bytes()) == expected, "Review must not modify inputs"
    print(json.dumps({"reviewed": 50, "counts": dict(counts), "budget_preference_tasks": len(soft_budgets),
                      "folder": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
