# 商品事实与程序比较 v2：独立实验

本实验不修改`semantic_*.py`、训练包、原1000题或真实训练入口。沿用既定六维奖励公式与未知项口径。

## 实际做了什么

1. 取上一轮保存的新顺序v5的47种需求/SKU组合，按商品快照和所选规格去重为46次事实提取请求。模型不接收用户query、结构化需求、正确答案、旧奖励或审阅判断。相同商品/SKU事实可跨需求复用。
2. 冻结新的事实提取提示词和schema。模型只提供`claim`、`label`、`scalar`、`members`、`component`五类候选，附原文字段及连续引用。`actual`不是自动可信的值。
3. 程序从完整原文子句重新解析成员、数值和单位，比较提案是否一致。拒绝虚构成员、借未选规格、丢失部件归属、颜色子串、仿制材质、裁掉否定、将价格加法当规格清单等。无法证明的提案留拒绝记录，不成为人工已审事实。
4. 用经过来源检查的实际值，与原冻结需求比较。成员数、数量、单位换算和近似数值由程序比较；没有设计新的奖励表。S/小码等标准尺码表示仅在本实验比较器修正。
5. 设置纯程序对照：同一套证据语法也自动扫描所选规格。必须分别报告程序本身与LLM候选的增量，不能把程序修复的收益全归给LLM。

## 验收与边界

- 上一轮244项判断作为已知回归，不能再称盲测。本轮还在v2输出前，对其余16种保存购买建立72项来源审阅参考；它们仍属于同一批历史商品，不能证明未见分布泛化。
- 同时统计与参考一致率、误放行、真实满足项识别率及奖励排序。把所有内容改成unknown不会通过验收。
- 新协议仍是有限证明能力：未指定轴的单一长度、中文口语数量、复杂定性同义和隐含角色等可能被拒绝或保持未知。拒绝候选不意味着商品实际不满足。
- 属性原文出现不证明现实功效；本轮仍只核验冻结公开资料的声明。
- 批次未知覆盖及训练采样多样性仍是独立问题。候选事实不自动写回239条已审事实缓存，不自动放行GRPO。
- `checker`里的完整公开商品对象仅供程序做未选规格泄漏否决，不发送给模型。模型只见`public`字段；其中只有共同文字和实际所选规格。

脚本：`tools/run_purchase_fact_experiment.py`；证明编译器：`tools/purchase_fact_proofs.py`。提示词见本目录`system_prompt.txt`，schema与全部哈希随实验输出冻结。

```powershell
Set-Location D:\shopping_agent
$env:PYTHONIOENCODING = 'utf-8'
.\.venv-llm\Scripts\python.exe .\tools\run_purchase_fact_experiment.py prepare
# 真实API提取；仅使用已被Git忽略的本地配置，不在命令里填写密钥。
.\.venv-llm\Scripts\python.exe .\tools\run_purchase_fact_experiment.py run `
  --env-file .\.env.purchase-judge.local
# 用保存的回答重编译、重评分，不调用API。
.\.venv-llm\Scripts\python.exe .\tools\run_purchase_fact_experiment.py report
```

接口沿用默认HTTPS和学校HTTP白名单校验；HTTP只能是已授权学校地址。API错误正文和鉴权头不写日志。每次调用记录实际耗时、用量及通过/拒绝事实；格式错误最多重试一次。程序检查通过仍不能等同全语义正确。
