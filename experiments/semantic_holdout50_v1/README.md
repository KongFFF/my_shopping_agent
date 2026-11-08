# 新50题封存验证样本

这是从原训练1000题中确定性随机挑选的50题，排除原语义试点、已记录API任务及重复query。没有使用商品gold、奖励、提取效果或商品类别来挑题；不是原项目Dev-50或Final Clean。

封存表示样本、排除清单及提取协议已固定；不宣称历史上所有静态数据检查都从未接触过这些任务。题文未在本轮人工审阅，未生成答案、参考标签或API请求。

本轮仅封存，不自动调用API。先核对原50题参考草稿和通用规范，再用这一批验证泛化。不要先看新题后逐题改规则，再把成绩称作独立验证。

## 本地核验

```powershell
.\.venv-llm\Scripts\python.exe .\tools\build_semantic_holdout.py --verify
```

## 后续运行入口

PowerShell从`D:\shopping_agent`执行，沿用现有校园网关密钥变量。先恢复上一轮非密钥配置，准备新运行（不调用API）：

```powershell
$fresh50Config = Get-Content .\outputs\semantic_school_contract_v2_r3_50_v1_collect\run_manifest.json -Raw -Encoding UTF8 | ConvertFrom-Json
$env:LLM_BASE_URL = $fresh50Config.api.endpoint -replace '/chat/completions$', ''
$env:LLM_MODEL = $fresh50Config.api.model
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode holdout50 --profile semantic-v2-r4 --thinking omit --output .\outputs\semantic_school_holdout50_v1 --prepare-only
```

准备成功后，下面这一条才会调用API；本轮未执行：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\collect_semantic_diagnostics.py --source-run .\outputs\semantic_school_holdout50_v1 --output .\outputs\semantic_school_holdout50_v1_collect --allow-school-http
```

中断后相同采集命令续跑。结果在采集目录report.json。collection_complete只表示收齐回答，仍需语义审阅和可核验覆盖检查。输出校验支持：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\check_semantic_contract_v2.py .\outputs\semantic_school_holdout50_v1_collect\responses.normalized.jsonl --mode holdout50 --allow-partial
```

全部可本地CPU运行，无需GPU。实际请求须能访问学校网关；不得把开发参考标注当成这批新题的模型回答复用。
