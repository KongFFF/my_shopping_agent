# 通用语义规范与50题开发参考稿

2026-11-03。执行用户确认的1A/2A/3A/4A/5A，同时限制范围：只处理跨题复用的字段身份、原子事实、数量/主体/比较关系及未决标准保留。不为特殊商品名或单题含义增加提取规则。

## 本轮边界

- 运行时只增加两个通用表达：`specifications/weight/mass`、`specifications/dimensions/thickness`。课程级别沿用`content/content`，不继续加字段。整体重量与包装净含量分开，厚度与材质分开。
- `field_guide.json`是字段定义和审阅规范，不是“遇到某个词就套某标签”的规则库。主观词不补造数值阈值，单位缺失、特殊词和多义关系继续待审。
- `analysis/semantic_reference_20261103/reference_edits.json`是对原50题的标注补丁，记录每项before/after和理由。题号仅索引参考数据；自动提取器不读取这些补丁，也没有新增题号分支。
- 原始用户query、原始API回答、原训练1000题和评测划分均未改变。参考稿不替换actor输入，不作为新模型输出复用，不自动批准进入奖励。

## 已交付

`pilot50/reference_candidates.jsonl`：50道候选参考合同，全部条目的review_status仍为pending。36题有数据补丁，其余沿用或仅由既定预算政策归一化。

`pilot50/reference_audit.jsonl`：原回答行号、通用归一化记录及标注补丁。

`参考集审阅.md`：逐题修改范围与待审原文，便于人工核对。

`pilot50/unresolved_mentions.jsonl`：保留的未决要求。38题含未决项，不追求本轮清零，不为此追加特殊语义规则。原50题按复杂表述分层挑选，不能把38/50外推为1000题的未决比例。

候选参考通过的是结构、来源、对象绑定、受控属性和基础关系检查。它是本助手起草的开发参考，不是独立专家真值，也不代表所有商品事实已核验。没有批准任何合同用于训练。

## 下一步验收

新随机50题已经封存在`experiments/semantic_holdout50_v1`，排除了原试点及已记录API任务，并排除重复query；未生成答案或标签，未人工审阅其题文。仅来自原训练集，与原项目Dev-50及Final Clean分开。

采用当前固定r4通用字段协议，新增`--mode holdout50`可加载该封存样本。若协议、输入或运行器改变，会拒绝静默沿用封存实验。

在新50题上分别报告：

1. 结构/原文来源通过情况。
2. 语义需要实质修改的题数。
3. 正确保留的真实未决项及审阅负担。
4. 能进入新奖励核验的覆盖率。

正确保留真实歧义不算提取理解错误；没有待审也不代表自动满足了所有要求。是否扩大1000题，依据新的验证结果和剩余审阅成本决定，不依据手工修订的开发50题得分。

本轮没有新的API调用，不启动训练。代码验证包括参考补丁禁止修改query、原文证据一致、全部合同不能自我批准、通用物理单位约束及封存集无重叠等正反例。

## 复现参考稿

只读原答和数据补丁，输出目录必须为空：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\build_semantic_reference.py --source .\analysis\semantic_r4_offline_20261103\review_inputs.jsonl --edits .\analysis\semantic_reference_20261103\reference_edits.json --output .\outputs\semantic_reference_check_new
```

此前r4源码、prompt和生成元数据的快照保存在`runtime_before_general`，原始r3运行目录保持不变。
