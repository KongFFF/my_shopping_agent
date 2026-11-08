# r7审阅校验版本

这是第三步数据修订使用的独立校验版本，schema字段结构与冻结r7保持一致，版本标识为 `shopping-semantic-requirements-v2-r7-review-draft`。它不修改原LLM提取profile、actor输入或GRPO训练奖励。

本轮用户确认：“加个／配个／有个”，对象明确时按1个。为识别该来源证据，本版本在r7基础上引入 `semantic_review_source_proofs.py`：

- 仅核验已经标注的数量，不默认创建数量或修改强弱。
- 只对同一明确对象支持1个，不能把“有个朋友”借给朋友所需的商品。
- 不屏蔽其他schema、证据、逻辑和数量错误。
- 三件套各1及“至少A和B”的作用域仍待用户决定。
- 不将任何整题或奖励标记为approved。

原r7及其报告保持冻结。当前修订数据、原文与修改前后、校验结果见 [逐题修订报告](../../analysis/semantic_clear_error_revision_20261106/修订报告.md)。

校验与复现（仓库根目录，CPU离线）：

```powershell
.\.venv-llm\Scripts\python.exe .\analysis\semantic_clear_error_revision_20261106\author_revision.py
.\.venv-llm\Scripts\python.exe .\analysis\semantic_clear_error_revision_20261106\verify_revision.py
.\.venv-llm\Scripts\python.exe -m unittest discover -s tools -p 'test_semantic_review_source_proofs.py'
```

校验报告分别列出“仅改数据、使用冻结r7”和“再应用新确认数量政策”的结果，避免混算收益。`candidate_drafts1000.jsonl`保存输入草稿；`confirmed_policy_review.jsonl`保存规范化后的主结果。所有输入源文件都保留在原目录。
