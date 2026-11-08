# 独立语义需求契约 v2（离线原型，等待LLM回归验证）

2026-11-02，根据固定50题逐题审阅实施。该版本与旧的`guided-v2`不同：后者仍属于旧契约v1，只改善提示词。本包是新的需求表达和程序核验协议。六个顶层语义维度保持不变；原训练集1000题及Actor用户输入保持原文。

## 已确认的政策

用户已逐项确认：

| 项目 | 规则 |
|---|---|
| 非预算的约/左右 | 保留原文和属性对象，容差待审，不统一加减比例 |
| 整百加多 | 一百多至九百多按该百位末尾作上限，例如九百多≤999，不设最低消费；一千多、五十多不外推 |
| 弱语气赠品/店铺 | “可以赠送”“听说某店不错”等保留待审强弱，不自动判必需/偏好 |
| 20元一瓶等裸单价 | 单价上限≤20元/瓶，保留计价分母 |

既有预算政策沿用：硬上限不放宽；左右等按×1.1上限；出头两位+4，三位及以上+10^(位数−2)−1；下限和严格不等式不变。近似词本身不把必需预算降为偏好。

2026-11-02 r2补充用户确认：“希望”本身不决定强弱；没有其他明确强弱标志时保留待审，不自动降为偏好或升为必需。有明确“最好/必须”等标志时再按对应语境处理，保留原文和程序审计记录。

主观质量、风格、相对最新等仍保留待审，尚无用户批准的客观判定标准。单题AND/OR、计数层级等无法确定时同样待审，不替用户补答案。

## 新表达

每条需求增加固定`subject.kind/name`、`attribute`和`price_basis`。subtype仍使用受控枚举，模型不能自由发明字段。

- subject区分主商品、部件、赠品、适配对象、适用人和订单；部件、赠品、适配对象必须在该条证据中出现。
- attribute区分购买数量、档位数、预设数、功率、电压、储能、承重、适穿体重等。
- `quantity/purchase_count`是购买数，不能放档位/音色/棱面计数；特征计数用`attribute_count`和`count`单位。
- 新增包装、产地、图案、造型、味型、品相、工艺、定制、安装服务、成分等受控子类型。
- 单价保留`price_basis.amount/unit`；20元/瓶和20元/米不能互相替代，不从未知数量推算总价。
- 同类买赠保留付费、赠送、到手数量及件/袋/套层级；只有同类且两数明确时，程序才计算到手数。异类赠品按赠品组件表达。
- 组件数量不明可为null；不默认各1，也不把未知赠品数补成0。

这改善表达和部分确定性校验，不等于已经实现商品/SKU事实核验。

## 模型与程序分工

`llm_output.schema.json`是LLM输出格式：没有id、start、end；证据只给逐字原文text，预算只给完整原文子句，不让模型计算金额。

`requirements.schema.json`是派生合同格式：程序为唯一原文证据填坐标，按来源顺序分配ID，依据确认政策计算预算和单价基准。若预算摘录漏掉前后比较词，程序会回到原句子句，避免把“不低于340”误变成上限340。不能可靠解析的预算进入待审，不猜测。

非预算近似值的模型计算结果会退回原文，标uncertain并记录待审容差；原始importance、数值和修改原因保存在audit中。弱语气、主观标准和相对时间同样保留待审；该步骤不会把未决内容当作成功。

校验会拒绝明显的预算强弱错误、味道/图案塞颜色、产地塞商家、计数对象错误、物理属性单位错配、严格边界变闭边界、明确买赠数字不符、补造组件数量、完全重复事实等。规则是保守检查，不能保证发现所有语义错误，也不证明没有漏项；同义重复仍需人工审阅。

原始响应仍在`attempts.jsonl`，派生结果在`responses.normalized.jsonl`，每项程序操作在`normalizations.jsonl`及调用记录的audit字段中保留。v2每题本来就需要生成ID和坐标，因此`normalized_tasks`通常等于通过数；不要把它或raw_validated_tasks当作质量准确率。

## 本地运行

当前为r2检查规则，新增显式数量覆盖和混合强弱检查，详见同目录`数量与强弱检查r2.md`。以下命令使用新的`semantic_school_contract_v2_50_v2`目录；旧版目录不能直接续写或混用。

沿用已配置的校园网关、真实模型ID和密钥环境变量。不需要GPU，不在此处填写/提交密钥。以下为PowerShell命令，从D:\shopping_agent执行。

先本地准备，不调用API：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 50 --output .\outputs\semantic_school_contract_v2_50_v2 --thinking omit --profile semantic-v2 --prepare-only
```

先真实试跑5题（这条会调用API）：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 50 --output .\outputs\semantic_school_contract_v2_50_v2 --thinking omit --profile semantic-v2 --allow-school-http --limit 5
```

原有`--evidence-offsets`用于v1坐标修正，v2无需添加：它本来就由程序生成坐标。不要使用旧50题目录，不要从旧契约v1/guided-v2导入结果。任务ID和原query相同，但输出结构、证据协议和对象表达不同，不能自动视为新版结果。

试跑后校验：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\check_semantic_contract_v2.py .\outputs\semantic_school_contract_v2_50_v2\responses.normalized.jsonl --mode 50 --allow-partial
```

看报告的validated_tasks、unresolved_tasks、pending_mentions，并逐题审阅派生合同及audit。程序通过但uncertain/pending仍不得进入奖励。5题审阅后再去掉limit完成同50题，复核后才扩展1000题。

1000题模式已支持，后续同模型、同参数、同v2协议复用已通过的新版50题：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 1000 --output .\outputs\semantic_school_contract_v2_1000_v2 --thinking omit --profile semantic-v2 --allow-school-http --reuse-from .\outputs\semantic_school_contract_v2_50_v2
```

## 本轮验证范围

`pilot50_regression_cases.json`记录固定50条原文、来自逐题审阅的关注点和48条独立指定的预算预期。测试核对预算计算和源数据边界，另用正负例检查对象、单位、促销、重复、待审门禁及奖励区间。

这不是新版LLM在50题上的实际准确率：本轮没有调用LLM，没有生成新的50份机器提取结果。实际提取质量仍要通过上面的API小样本验证。

`semantic_reward_v2.py`只是聚合已人工批准合同和已核验pass/fail/unknown事实的离线原型，沿用已确认奖励区间和六维等权。它不查询商品、不判断任意SKU是否满足需求，也没有接入真实训练或改动Reward v3。存在pending、uncertain或未批准条目时拒绝评分。

旧v1包、50题原始响应、928题修复冻结包及上游训练运行时均未改动。每轮输出目录保存新协议、策略和normalizer源码快照；同一目录拒绝混用版本，跨目录复用也会检查这些内容。
