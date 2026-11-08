"""Offline screening of original train-1000/Dev-50 reward semantics, no edits."""
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "upstream/shopping-grpo-longhorizon"
OUT = ROOT / "analysis/purchase_guard_v1_1_dev50_20261031/derived"
sys.path.insert(0, str(UP / "environments/ShopSimulator/shop_env"))
from web_agent_site.engine.reward_features import compile_reward_features, normalize_option_text
from web_agent_site.engine.reward import evaluate_purchase

# Literal screens, not a semantic classifier. Do not interpret all hits as defects.
SCREENS = {
    "bundle_words": r"套餐|套装|组合装|组合套|组合包装",
    "promotion_words": r"买[0-9一二三四五六七八九十两]+.{0,8}(?:送|赠)[0-9一二三四五六七八九十两]+",
    "gift_accessory_words": r"赠送|附赠|加赠|附带",
    "quantity_words": r"[0-9一二三四五六七八九十两]+\s*(?:瓶|包|盒|袋|支|只|个|件|套|对|片|把|卷|组|箱|双)",
}
MARKERS = {normalize_option_text(x) for x in ("规格", "0lb", "默认", "默认选项", "默认规格", "其他", "其它")}


def main():
    archive = UP / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    tasks = [(p, i) for p in products for i in p.get("instructions", []) if i.get("attributes")]
    records, counts = {}, {}
    for split, name, size in (("train", "train.jsonl", 1000), ("dev", "validation.jsonl", 50)):
        ids = [json.loads(line)["task_id"] for line in (UP / "data/grpo" / name).read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(ids) == len(set(ids)) == size
        rows = []
        for tid in ids:
            product, instruction = tasks[tid]
            query = instruction["instruction"]
            features = compile_reward_features(instruction, product)
            required = features["required_options_by_key"]
            singles = []
            for axis, requirement in required.items():
                entries = product.get("customization_options", {}).get(requirement["source_axis"], [])
                values = list(dict.fromkeys(str(e["value"]) for e in entries if e.get("value")))
                if len(values) == 1:
                    singles.append({"axis": requirement["source_axis"], "value": values[0],
                                    "marker_candidate": normalize_option_text(values[0]) in MARKERS})
            row = {"task_id": tid, "query": query,
                   **{key: re.findall(pattern, query) for key, pattern in SCREENS.items()},
                   "required_bundle_axis": "bundle" in required,
                   "single_value_required_axes": singles,
                   "multiple_requirements_with_single_axis": len(required) >= 2 and bool(singles),
                   "numeric_model_tokens": [t for t in features["expected_model"] if t.isdigit()],
                   "all_model_tokens": features["expected_model"]}
            rows.append(row)
        records[split] = rows
        counts[split] = {"tasks": size, **{key: sum(bool(r[key]) for r in rows) for key in SCREENS},
                         "bundle_promotion_accessory_union": sum(any(r[k] for k in ("bundle_words", "promotion_words", "gift_accessory_words")) for r in rows),
                         "required_bundle_axis": sum(r["required_bundle_axis"] for r in rows),
                         "single_value_required_axis_tasks": sum(bool(r["single_value_required_axes"]) for r in rows),
                         "multiple_requirements_with_single_axis": sum(r["multiple_requirements_with_single_axis"] for r in rows),
                         "literal_marker_candidate_tasks": sum(any(a["marker_candidate"] for a in r["single_value_required_axes"]) for r in rows),
                         "numeric_model_token_tasks": sum(bool(r["numeric_model_tokens"]) for r in rows),
                         "numeric_model_tokens": dict(Counter(t for r in rows for t in r["numeric_model_tokens"]))}
    product, instruction = tasks[8666]
    features = compile_reward_features(instruction, product)
    goal = {"asin": str(product["asin"]), "category": product["category"], "price_upper": 100, **features}
    variants = []
    for package in ("隐形单边拉链压脚1个", "单边拉链压脚买10送1"):
        for marker in (False, True):
            options = {"T168": package}
            if marker:
                options["颜色分类"] = "规格"
            result = evaluate_purchase(product, goal, selected_options=options)
            variants.append({"options": options, "type": result.reward_type, "reward": result.reward,
                             "score": result.weighted_score})
    evidence = {"definitions": SCREENS, "marker_candidates": sorted(MARKERS), "counts": counts,
                "task8666_counterfactuals": variants,
                "limitations": ["Literal screening counts, not prevalence of semantic bundle tasks or reward defects.",
                                "Single-value axes can encode real user requirements; not automatically placeholders.",
                                "Pure numeric model tokens can be genuine models; candidates require contextual review.",
                                "Only original train1000 and Dev50; no Final data or scoring/data modifications."]}
    OUT.mkdir(exist_ok=True)
    for name, value in (("reward_semantics_screen.json", evidence), ("reward_semantics_candidates.json", records)):
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    for split, rows in records.items():
        print("MARKER CANDIDATES", split)
        for row in rows:
            if any(a["marker_candidate"] for a in row["single_value_required_axes"]):
                print(row["task_id"], row["single_value_required_axes"], row["query"])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
