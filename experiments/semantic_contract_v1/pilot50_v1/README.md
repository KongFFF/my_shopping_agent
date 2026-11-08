# 固定50题离线提取试点 v1

固定日期：2026-11-01。仅准备输入，没有调用LLM或修改训练数据。

50题均来自原始1000题。使用固定SHA-256排序种子`semantic-pilot50-20261101-v1`，优先覆盖稀有表达后依次抽取：买赠2、出头预算2、交易条件6、套餐组成4、数量2、其他预算表达6、复合规格8、品牌/型号6、偏好6、一般抽样8题。特征来自Query，标签可重叠，主分层互斥。一般抽样不限定专题，不代表简单任务。

这是覆盖性诊断样本，不能用错误率外推全部1000题。抽样没有使用gold商品、目标规格、旧奖励或模型成绩。留出集仅取ID用于排重。

## 传给LLM的内容

- `tasks.jsonl`：50条原始task_id/query，query逐字保留。
- `system_prompt.txt`：固定完整system prompt，已经嵌入合成示例和完整schema。
- `user_prompt.template.txt`：固定user模板。
- `requests.jsonl`：已经展开的50份消息，可逐条原样发送；不含抽样标签、商品或答案。
- `output.schema.json`：本试点固定的输出JSON Schema，使用$defs减少重复；与现有六维schema等价。
- `request_settings.json`：模型尚未选择；建议temperature=0，每题独立请求。不同模型/版本不得混作同一轮实验。

每次请求只提取一题，输出一个合同JSON；六维字段、原Query、证据字符位置与pending审阅状态都必须保留。system prompt已固定基本预算政策，复杂/不支持语义保留待审，不强行补齐。

`selection_audit.jsonl`和`selection_spec.json`只用于审阅与复现，不传给LLM。`freeze.lock.json`固定全部文件的精确字节哈希。不得同版本静默更换任务、prompt或schema。

## CPU检查

```bash
python tools/build_semantic_pilot50.py --check
python tools/check_semantic_pilot50.py responses.jsonl
```

responses.jsonl每行必须是原始输出合同JSON，不包装供应商响应；检查器要求50题齐全、无重复、无额外题、原Query一致、所有审阅状态pending。模型原始完整响应、模型名、版本、实际生成参数和校验错误也应另存；不能只保存修复后的答案。若先检查部分结果，使用`--allow-partial`。

结构校验通过不代表语义正确或完整。人工仍需检查强弱、漏项、冗余、单位、组合关系和型号误提取。索引错误须记录，不能悄悄改后当作原始LLM输出。

原始1000题、Actor输入、Reward v3、v1.1运行时及928题冻结包未改变。下一阶段再选定LLM与执行方式；本轮无GPU或API费用。
