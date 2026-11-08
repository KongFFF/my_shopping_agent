# Semantic contract v2 r7 — offline revision

本版用于离线规范化和检查，不自动升级原LLM API提取profile，不改变actor用户输入，也不接入GRPO训练奖励。原r5/r6回答和规则保持可复现。

六个顶层维度保持 `category / budget / functions / specifications / brand / model`。新增内容均位于既有统一需求条目中：

- `operator: excludes`：排除命名组件或值；需要各自的负向原文证据。禁止项不带件数。`不需要`的强解释仍须审阅，不自动推成禁止。
- `specifications / manufacturer`：生产企业角色，不等同品牌、商家或地理产地。
- `quantity / purchase_mass` 和 `quantity / purchase_volume`：总需求量的质量/体积，不是购买件数。
- `quantity / per_variant_count`：每种/每款的数量，要求明确的每种/每款来源及数量、单位一致，不推算总购买数。

组件存在与组件物理尺寸使用独立条目，未给件数不补1。组件长度默认只表一般尺寸，不从单位擅自命名长/宽/高轴。单位不明、非预算近似、未定偏好等保留pending；检查器不批准标签。

固定导出物：`llm_output.schema.json`、`requirements.schema.json`、`policy.json`、合成示例及审计、`validation_report.json`。这些schema仅定义r7格式，不表示源LLM已使用r7提取。

正式1000题重放报告：[修复与重放报告](../../analysis/semantic_r7_replay_20261106/修复与重放报告.md)。`analysis/semantic_r7_replay_20261106/confirmed/`是最终结果，`v1/`为中间诊断。

从仓库根目录复现（选择新的、尚未存在或为空的输出目录；全程CPU离线）：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\replay_semantic_v7.py `
  --original-cards .\analysis\semantic_r6_replay_20261106\confirmed\review_inputs.jsonl `
  --reviewed-drafts .\analysis\semantic_review_four_steps_20261106\candidate_drafts1000.jsonl `
  --decisions .\analysis\semantic_review_four_steps_20261106\user_decisions.json `
  --output .\analysis\semantic_r7_replay_20261106\rerun_unique
```

重放器固定每题的来源回答行，不重选历史回答，不访问网络，不覆写输入。原回答与已有修订稿分两路输出；只有修订稿一路应用显式的用户决定。所有结果都保持整题和奖励未批准。

```powershell
.\.venv-llm\Scripts\python.exe -m unittest discover -s tools -p 'test_semantic_v[567].py'
```
