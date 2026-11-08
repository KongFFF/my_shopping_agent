"""Render the completed v2 audit and verification into reviewable Chinese reports."""

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "repairs/task_contract_v2"
V1 = ROOT / "repairs/task_contract_v1"
LABELS = {
    "approved_direct_isolation_missing_metadata": "按决定直接隔离缺证题",
    "multiple_effective_price_axes": "多价格轴缺完整SKU组合价",
    "wrong_purchase": "新契约下目标不满足价格门控",
    "component_constraints_need_evidence": "工费/定金等分项约束缺核验依据",
    "component_price_evidence_missing": "单价/费用组成缺证据",
    "budget_contract_unverifiable": "预算契约尚不能核验",
    "deposit_transaction_unsupported": "定金交易语义未支持",
    "ready_to_use_vs_unprocessed_unverified": "成品可直接使用与不加工尚有冲突",
    "red_query_vs_purple_annotation": "大红色需求与葡萄紫目标冲突",
    "high_speed_bur_package_inclusion_unverified": "套餐是否包含高速车针未明确",
    "root_measurement_definition_unverified": "根块尺寸与种球周长定义待核对",
    "capacity_and_price_roles_unverified": "电池容量与两轴计价来源待核对",
    "selected_axis_prices_125_vs_145_unverified": "所选轴价125/145来源未核验",
    "selected_axis_prices_25_vs_36_unverified": "所选轴价25/36来源未核验",
    "selected_axis_prices_256_vs_150_unverified": "所选轴价256/150来源未核验",
    "selected_axis_prices_399_vs_1439_unverified": "所选轴价399/1439来源未核验",
    "selected_axis_prices_29_vs_79_unverified": "所选轴价29/79来源未核验",
}


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def reasons(row):
    return "；".join(LABELS.get(key, key) for key in row["reasons"]) or "静态gold通过，未留已知隔离原因"


def interval(row):
    details = row["budget_parse"]
    if details["status"] == "review_required":
        return "待审：" + details["reason"]
    lower, upper = details["lower"], details["upper"]
    if lower is None and upper is None:
        return "未声明价格约束"
    values = []
    if lower is not None:
        values.append((">=" if details["lower_inclusive"] else ">") + f"{lower:g}")
    if upper is not None:
        values.append(("<=" if details["upper_inclusive"] else "<") + f"{upper:g}")
    return " 且 ".join(values)


def main():
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifact_sha256"].items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == expected, name
    audits = [json.loads(line) for line in (PACKAGE / "all_train_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    review = {row["task_id"]: row for row in (json.loads(line) for line in (V1 / "first_pass_review.jsonl").read_text(encoding="utf-8").splitlines())}
    verification = json.loads((PACKAGE / "verification.json").read_text(encoding="utf-8"))
    assert verification["train_tasks"] == manifest["retained_count"]
    frozen = (PACKAGE / "freeze.lock.json").exists()
    bundle_label = "outputs/task_contract_v2_frozen" if frozen else "outputs/task_contract_v2"
    recovered = [row for row in audits if row["task_id"] in manifest["recovered_from_v1_task_ids"]]
    newly_held = [row for row in audits if row["task_id"] in manifest["newly_quarantined_from_v1_task_ids"]]
    remaining = [row for row in audits if row["status"] == "quarantined"]
    groups = {}
    for tid, original in review.items():
        row = next(item for item in audits if item["task_id"] == tid)
        groups.setdefault(original["category_label"], Counter())[row["status"]] += 1
    changes = ["# v1到v2的逐题变化", "", "按用户最终选择实施。规则只读query；以下是数据清单变化，不是模型性能提升。",
               "", f"v1保留893题，v2保留{manifest['retained_count']}题：从107题中恢复{len(recovered)}题，同时新增隔离{len(newly_held)}题。",
               "", "| 原初判主要类别 | 恢复 | 继续隔离 |", "| --- | ---: | ---: |"]
    for label, counts in groups.items():
        changes.append(f"| {label} | {counts['retained']} | {counts['quarantined']} |")
    changes.extend(["", "## 从原107题恢复", "", "| task_id | 原类别 | 新价格约束 | 标注字符恢复 |", "| --- | --- | --- | --- |"])
    for row in recovered:
        changes.append(f"| {row['task_id']} | {review[row['task_id']]['category_label']} | {cell(interval(row))} | {'有' if row['annotation_changes'] else '无'} |")
    changes.extend(["", "## 全1000题复核新增隔离", ""])
    for row in newly_held:
        changes.extend([f"### task_id={row['task_id']}", "", row["query"], "", f"原因：{reasons(row)}。", ""])
    changes.extend(["## 规则已修好但仍不能纳入的代表", "",
                    "- 8181：价格下限已实现，345元满足≥340，但套餐是否包含高速车针仍缺明确证据。",
                    "- 19581：末尾分号已恢复，29元候选可匹配；另一个规格轴79元的价格角色仍未知。",
                    "- 22407：一千五正确解析为1500，仍需确认颜色轴399与尺寸轴1439的计价关系。",
                    "- 8784：按最终出头规则，400出头上限409；419元不满足，不能为恢复该题放宽。",
                    "", "## 原90题结果", "", f"v2共保留其中{manifest['original_90_recovered_count']}题，隔离{90 - manifest['original_90_recovered_count']}题；v1为28题恢复。两个版本差额同时包含新恢复和新增隔离，不能只相减后称作新增修复数。", ""])
    (PACKAGE / "changes.md").write_text("\n".join(changes), encoding="utf-8")
    cards = [f"# v2剩余{len(remaining)}题隔离证据", "", "这是一份实施后的审计记录，隔离任务未删除。预算通过仍可能因为已知价格/规格缺证而隔离。", ""]
    for row in remaining:
        gate = row["new_reward_evidence"]["hard_gates"]["budget"]
        cards.extend([f"## task_id={row['task_id']}", "", f"- 来源：{'原107题' if row['task_id'] in review else '全1000题复核新增隔离'}",
                      f"- 原文：{row['query']}", f"- ASIN：`{row['asin']}`", f"- 隔离依据：{reasons(row)}",
                      f"- 新价格约束：{interval(row)}", f"- 现行价格解析值：{gate.get('actual')}（不是缺失完整SKU价的替代证明）",
                      f"- 预算门控：{gate['status']}；终局类别：{row['new_reward_type']}",
                      f"- 可解析目标规格：`{json.dumps(row['selected_gold_options'], ensure_ascii=False)}`",
                      f"- 标注修改：`{json.dumps(row['annotation_changes'], ensure_ascii=False)}`", ""])
        if row["task_id"] in review:
            cards.extend(["初判说明：" + review[row["task_id"]]["interpretation_and_evidence"], "后续证据要求：" + review[row["task_id"]]["proposed_action"], ""])
    (PACKAGE / "remaining_review.md").write_text("\n".join(cards), encoding="utf-8")
    validation = json.loads((PACKAGE / "validation_audit.json").read_text(encoding="utf-8"))
    not_gold = [f"{row['task_id']}（{row['new_reward_type']}）" for row in validation if row["new_reward_type"] != "gold_purchase"]
    report = ["# 任务与评分契约第二版实施结果", "", "2026-10-29。已按用户选择生成独立v2修复包、完成全1000题复核，并从物化目录独立导入运行时代码验证。原始上游与v1包保持不变，没有启动模型、HTTP服务或GRPO。",
              "", "| 项目 | 结果 |", "| --- | --- |", f"| 可训练候选 | {manifest['retained_count']}题，v1为893题 |",
              f"| 从原107题恢复 | {len(recovered)}题 |", f"| 原107题继续隔离 | {107 - len(recovered)}题 |",
              f"| 全量复核新增隔离 | {len(newly_held)}题 |", f"| 最终隔离 | {len(remaining)}题 |",
              f"| 原90题静态恢复 | {manifest['original_90_recovered_count']}题，v1为28题 |",
              f"| 实际独立运行时gold回放 | {verification['actual_runtime_gold_replay']} |", "", "## 已实施政策", "",
              "1. 价格下限、严格高于、上限和明确数值区间，均接入实际购买评分及候选可接受性判断；中文省略金额与明确货币语境省略元已支持。",
              "2. 出头按最终澄清：两位加4、三位加9、四位加99、五位加999。20→24、100→109、420→429、1000→1099，只设上限；没有采用较早的100→119。",
              "3. 2B的多元/口语范围政策生效，不重复叠加近似容差。普通左右/上下维持10%，明确硬上限不放宽。",
              "4. 5题字符/标点恢复候选均已处理，其中4题纳入、19581因轴价缺证继续隔离。通用规则还恢复14950的损坏字符，但该题仍因组合定价缺证隔离。",
              "5. 35道主要缺证题全部直接隔离；次要缺证、定金流程、需求冲突及真实预算冲突继续隔离。未改query、价格或GRPO采样/更新策略。",
              "", "## 新增隔离与残留问题", "",
              "3828缺少工费单价依据，14888缺少手工费依据，15386涉及订金和后续补款。原先仅核对总价不足以证明全部价格条件满足，因此全量复核后隔离。",
              "", "8181、22300、22407的预算解析已经修好，但还存在套餐组成、尺寸定义或轴价来源的证据问题，暂不纳入。35题是初判的主要类别数量，不代表所有缺证只存在于这35题。",
              "", "8784的目标419元不满足用户最终定义的400出头≤409；保留隔离，未根据目标价格反调阈值。",
              "", "## 验证与限制", "",
              "- v2的21项行为回归测试通过，覆盖低价候选不满足下限、开闭边界、出头上限、非金额数值、单位证明、唯一符号恢复及反例。",
              f"- 从`{bundle_label}`的实际三份修复代码独立复算{manifest['retained_count']}题，全部得到reward_valid=true的gold_purchase；新进程的下限/出头反例检查通过。",
              "- parquet保持原始行与schema，Dev parquet字节不变；训练与Dev/Final ID无交叉。",
              "- 原始商品标题、query、属性、候选、价格、顺序等逐项对照未变；仅批准的训练私有instruction_options变化。独立索引与商品SHA绑定通过。",
              "- v1的全部11份哈希工件保持一致；原始上游三份运行时代码保持原样。",
              "", f"原Dev50名单没有筛选。新契约下静态目标gold为{manifest['validation_static_gold_count']}/50，非gold为{'、'.join(not_gold)}。这是标注可行性检查，不是SFT或GRPO模型成功率；当前仍有未支持的交易/证据或真实预算问题。不能直接将不同评分口径下的模型分数比较。",
              "", "## 文件与下一步", "",
              "- [修复包规则及CPU重建命令](repairs/task_contract_v2/README.md)",
              "- [38题恢复与新增隔离详情](repairs/task_contract_v2/changes.md)",
              "- [剩余隔离题逐题证据](repairs/task_contract_v2/remaining_review.md)",
              "- [全1000题机器审计](repairs/task_contract_v2/all_train_audit.jsonl)",
              "- [版本与任务划分](repairs/task_contract_v2/manifest.json)",
              "- [物化运行时验证结果](repairs/task_contract_v2/verification.json)",
              "", f"大文件位于被Git忽略的`{bundle_label}/`。当前完成本地CPU物化和奖励回放，未部署AutoDL，也未启动环境服务或训练。同步后需在云端重建、校验和做接口冒烟，再固定新训练配置；不能只换parquet而沿用旧环境。", ""]
    if frozen:
        report[2:2] = ["冻结编号：`task-contract-v2-928-20261029`。928/72划分、原Dev50及v2评分依赖已由[冻结锁](repairs/task_contract_v2/freeze.lock.json)绑定；后续采样/算法实验共用此版本。", ""]
    (ROOT / "任务评分契约第二版修复.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"recovered": len(recovered), "newly_held": len(newly_held), "remaining_cards": len(remaining),
                      "validation_static_gold": manifest["validation_static_gold_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
