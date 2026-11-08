# semantic-v2-r3：完整50题初审后的修订

## 实际运行后的hotfix1（2026-11-03）

本轮50题已收齐。原报告5题通过被预处理器类型异常污染：pending_mentions.evidence的schema要求对象，模型却普遍返回数组，预处理器在schema校验前按对象访问。已修复类型检查，并对单元素数组做有审计的无损解包；多元素数组仍拒绝，不猜测合并作用域。新增故障形状和离线恢复测试，r3专项共31项通过。

不必重跑API。96次原始回答已在`analysis/semantic_r3_actual50_hotfix_20261103`离线重验，恢复28题程序通过，22题仍失败；全部50题初审见该目录`分析报告.md`。当前尚不满足扩大1000题的门槛。

原prompt/schema/政策及protocol.lock.json均保留，处理器修复单独记录于normalizer.hotfix1.lock.json。下方命令是原r3运行的历史步骤；该运行已经完成，旧目录保留只读。当前使用派生结果继续分析，不向旧目录追加或重新发请求。新提取实验需要另定版本和新目录。

独立于r2的协议修订。固定原训练集1000题、既有50题ID和原始query；六个顶层语义维度及已确认奖励区间不变。没有改上游环境、SFT/GRPO或928题冻结包。本包尚未接入训练或奖励运行时。

## 修改范围

- `semantic_contract_v2_r3.py`：独立版本号、schema、归一化和校验；`packaging`新增受控属性`pack_count`，表达每份销售包装中的计数，与购买数、净含量和附件数分开。
- `semantic_relations_v2_r3.py`：封闭映射的字段路由，字面来源支持的确定性纠正，以及对象、比较关系、原子性和功能覆盖检查。每次纠正记录原值、新值、来源路径和规则，不批准合同。
- `semantic_coverage_v2_r3.py`：沿用数量/强弱覆盖检查，增加“把”的购买计数。
- `system_prompt.txt`：重写规则顺序，明确多段原文证据、字面主体、计价角色、包装层级及待审原因；由提取器追加r3 schema和合成示例。不把50题答案写入提示词。
- 共享提取器新增`--profile semantic-v2-r3`，保留旧profile；检查入口支持按schema_version选择r2/r3。r3运行还保存关系检查器源码快照，跨运行复用也检查该文件。
- 诊断收集入口支持没有`attempts.jsonl`的本地准备目录。因此可先准备固定协议，再一次收集全部50题，不因连续三道模型校验失败卡在前排。网络/鉴权停止规则保留。

明确纠正包括：普通售价被误作无分母单价、包邮运费0、书本计数别名、单支包装计数、重量包装归净含量、防水等功能误作成色、发货来源误作生产地、达到的明确能力下限，以及明示赠品误作常规组件。未知子类型不自动进入兜底；不猜主体、SKU、单位、型号作用域或AND/OR关系。

新增拦截覆盖类别中混入规格、功能/关系遗漏、包装与订单数量混淆、材质与颜色合项、客观与主观成分合项、型号对象不明、课程级别误作版本等。仅把错误改成pending不能绕过角色与覆盖检查。

## 用户选择与发布状态

已确认：

1. “2个和10个装”保留原文待审，不自动改成二选一。
2. “可以是石膏”保留允许含义并待审，不自动加偏好分。

3. 2026-11-03已确认：裸“包含4种音色”按恰好4种；明确至少/达到按下限；“20多”保留严格超过20及范围待审。比较词只约束所属计数，不跨分句传播。三个政策选项已全部确认，`unresolved_decisions=[]`。合同条目的review_status仍为pending，协议确定不代表输出已获批准。

## 离线回放

`replay_semantic_v2_r3.py`读取上一轮`review_inputs.jsonl`中选定的50份r2原始回答，显式适配版本字段，再运行r3纠正和检查。源文件不变，迁移适配和纠正均记录；所有合同仍pending。

```powershell
.\.venv-llm\Scripts\python.exe .\tools\replay_semantic_v2_r3.py --input .\analysis\semantic_contract_v2_full50_review_20261102\review_inputs.jsonl --output .\outputs\semantic_r3_probe_new
```

输出目录须为空。`report.json`区分旧失败被恢复、旧通过被新检查拦截，`probe_results.jsonl`保留具体错误与纠正记录。输出不是新prompt调用结果，不可通过`--reuse-from`假装新实验，也不能把通过数当作模型准确率。

## 后续实际验证流程

政策已确认。在全新目录使用r3准备协议，再用诊断入口收集同50题。沿用校园网关及当前真实模型ID、`thinking=omit`和密钥环境变量。HTTP实际调用仍显式加`--allow-school-http`；prepare-only不联网、不要求密钥。50/1000两种任务选择均保留，现阶段只实际验证50题。

在D:\shopping_agent的PowerShell中，先从上一轮记录恢复非密钥配置，再准备新目录（均不调用API）：

```powershell
$r3PreviousConfig = Get-Content .\outputs\semantic_school_contract_v2_50_v2_collect\run_manifest.json -Raw -Encoding UTF8 | ConvertFrom-Json
$env:LLM_BASE_URL = $r3PreviousConfig.api.endpoint -replace '/chat/completions$', ''
$env:LLM_MODEL = $r3PreviousConfig.api.model
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 50 --profile semantic-v2-r3 --thinking omit --output .\outputs\semantic_school_contract_v2_r3_50_v1 --prepare-only
```

待本地准备成功后，沿用当前窗口已有的LLM_API_KEY运行（这条会调用API）：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\collect_semantic_diagnostics.py --source-run .\outputs\semantic_school_contract_v2_r3_50_v1 --output .\outputs\semantic_school_contract_v2_r3_50_v1_collect --allow-school-http
```

中断后使用同一条采集命令续跑。报告在采集目录report.json；collection_complete表示回答收齐，complete仅表示程序校验全过，仍需语义审阅。本轮不使用旧答案复用为新版LLM结果。以上均可在本地CPU环境运行，无需GPU；实际调用须能访问学校网关。

旧运行的提取器源码哈希已经固定；共享入口增加r3后，旧命令会拒绝把当前入口继续写进旧目录。旧的完整采集结果仍可直接读取、审阅与离线回放，不应删除旧日志或修改manifest来绕过版本检查。

## 验证边界

正反例应同时覆盖：合法拆分/多段证据能通过；非法原文引用、错误物理单位、虚构分母、漏选功能、猜测对象及未批准合同继续被拒绝。另回归旧提取器的学校HTTP例外、日志/重试和密钥保护，以及诊断收集器的断点续跑。

新prompt的实际质量必须靠新模型回答衡量。离线回放只验证程序怎样处理旧回答；原文确有歧义的题仍需人工裁决，已通过的合同也还需商品事实核验。
