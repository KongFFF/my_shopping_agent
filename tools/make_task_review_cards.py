"""Render task evidence and optional bound first-pass judgments; apply nothing."""

import gzip
import hashlib
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "repairs/task_contract_v1"


def normalized(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value))).casefold().replace("/", "|")


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def main():
    pending = [json.loads(line) for line in (PACKAGE / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()]
    reviews = {}
    if (PACKAGE / "first_pass_review.jsonl").exists():
        summary = json.loads((PACKAGE / "first_pass_summary.json").read_text(encoding="utf-8"))
        source_hash = hashlib.sha256((PACKAGE / "quarantine.jsonl").read_bytes()).hexdigest()
        if source_hash != summary["source_quarantine_sha256"]:
            raise ValueError("First-pass judgments refer to a different quarantine snapshot")
        if hashlib.sha256((PACKAGE / "manifest.json").read_bytes()).hexdigest() != summary["source_manifest_sha256"]:
            raise ValueError("First-pass judgments refer to a different v1 manifest")
        records_hash = hashlib.sha256((PACKAGE / "first_pass_review.jsonl").read_bytes()).hexdigest()
        if records_hash != summary["review_records_sha256"]:
            raise ValueError("First-pass records changed; rebuild and verify the review first")
        records = [json.loads(line) for line in (PACKAGE / "first_pass_review.jsonl").read_text(encoding="utf-8").splitlines()]
        reviews = {r["task_id"]: r for r in records}
        if len(reviews) != len(records) or set(reviews) != {r["task_id"] for r in pending}:
            raise ValueError("First-pass judgments must cover every quarantined task exactly once")
    with gzip.open(ROOT / "upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz", "rb") as handle:
        source = handle.read()
    if reviews and hashlib.sha256(source).hexdigest() != summary["source_catalog_uncompressed_sha256"]:
        raise ValueError("First-pass judgments refer to a different catalog snapshot")
    products = {str(p["asin"]): p for p in json.loads(source)}
    lines = ["# 待审任务证据卡", "", f"共{len(pending)}题。此文件用于审阅，不会修改训练清单或奖励。", "",
             "先看用户原文，再看目标与商品证据。机器的reward_type只说明现有程序结果，不是最终裁决。",
             "", "已写入助手初判的题数：" + str(len(reviews)) + "。初判不代表已修复或已纳入训练。汇总见 [first_pass_review.md](first_pass_review.md)。",
             "", "注意：不同规格轴的价格不是完整SKU组合价格，不可相加或任取其中一个。", ""]
    for item in pending:
        tid = item["task_id"]
        product = products[item["asin"]]
        selected = item["selected_gold_options"]
        source_instructions = [i for i in product["instructions"] if i["instruction"] == item["query"]]
        if len(source_instructions) != 1:
            raise ValueError(f"No unique source instruction for {tid}")
        original_options = source_instructions[0].get("instruction_options", [])
        lines.extend([f"## task_id={tid}", "", f"- 来源：{'原90题' if item['was_original_90'] else '全量复核新增'}",
                      f"- 原文：{item['query']}", f"- 标注目标：{product['title']}（ASIN {item['asin']}）",
                      f"- 原始目标规格：`{json.dumps(original_options, ensure_ascii=False)}`",
                      f"- v1审计可解析规格：`{json.dumps(selected, ensure_ascii=False)}`（空字典可能表示目标字符串未匹配，不是用户没要求）",
                      f"- 当前结算：{item['new_reward_type']}；候选上限={item['new_budget']}；价格={item['new_reward_evidence']['hard_gates']['budget']['actual']}",
                      f"- 程序标记：{', '.join(item['reasons'])}", "", "| 规格轴 | 标注对应的候选及轴上价格 | 其他候选示例 | 该轴候选数 |", "| --- | --- | --- | --- |"])
        for axis, entries in (product.get("customization_options") or {}).items():
            expected = selected.get(axis)
            matched = [e for e in entries if expected is not None and normalized(e.get("value")) == normalized(expected)]
            others = [e for e in entries if e not in matched][:3]
            fmt = lambda es: "；".join(f"{e['value']}：{e.get('price')}" for e in es) or "未匹配/未标注"
            lines.append(f"| {cell(axis)} | {cell(fmt(matched))} | {cell(fmt(others))} | {len(entries)} |")
        failed = item["new_reward_evidence"]["evidence"]["preference_scoring"]["dimensions"]
        for name, detail in failed.items():
            if detail.get("active") and detail["score"] < 1:
                required = [r["required"] for r in detail["results"] if r["status"] != "pass"]
                lines.append(f"\n- 未通过维度 `{name}`：`{json.dumps(required, ensure_ascii=False)}`")
        if tid in reviews:
            review = reviews[tid]
            if review["query_quote"] not in item["query"] or review["asin"] != item["asin"]:
                raise ValueError(f"First-pass evidence mismatch: {tid}")
            lines.extend(["", "初判记录：", "", f"- 主要结论：**{review['category_label']}**",
                          f"- 用户原文依据：“{review['query_quote']}”",
                          f"- 语义判读与商品证据：{review['interpretation_and_evidence']}",
                          f"- 后续动作/缺失证据：{review['proposed_action']}",
                          f"- 问题定位置信度：{'中' if review['confidence_in_issue_identification'] == 'medium' else '高'}（不表示已修复，也不担保其他语义均正确）"])
            for candidate in review["unapplied_option_candidates"]:
                lines.append(f"- 未应用候选：`{candidate['axis']}={candidate['value']}`；该轴价格{candidate['price']}，SKU标识`{candidate['sku_id']}`。仅是核对线索，不代表完整SKU价格或已批准替换。")
            lines.extend(["- 状态：助手初判完成；需规则/证据复核与全量重审后才能重新纳入。当前训练清单、奖励规则及标注均未因本次初判修改。", ""])
        else:
            lines.extend(["", "审阅记录：", "", "- 结论：待审", "- 用户原文依据：", "- 商品/规格/价格依据：",
                          "- 通用修复规则，或还缺什么证据：", "- 复核状态：", ""])
    (PACKAGE / "review_cards.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Rendered {len(pending)} review cards.")


if __name__ == "__main__":
    main()
