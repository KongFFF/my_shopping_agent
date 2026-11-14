# step650独立评测：复用step500冻结协议

用户于2026-11-14明确同意评测预定续训终点650的Final Clean。训练内Dev50为500=29/50、550=28/50、600=28/50、650=29/50；平均奖励分别0.508667、0.504958、0.472389、0.508238。按Dev选模仍为500，本次650是续训诊断终点，不据Final选择新checkpoint或修改prompt／奖励。

工具 `tools/step650_evaluation.py` 分两步，不启动训练、模型服务、评测或关机。

## 1. 无卡准备

```bash
cd /root/autodl-tmp/work/my_shopping_agent
upstream/shopping-grpo-longhorizon/.venv/bin/python \
  tools/step650_evaluation.py prepare
```

读取原 `preparation/posttrain_step500_v1/evaluation_spec.json` 和 `run_eval.py`。校验原500的Final为137/200，冻结Final Clean清单SHA256为d99112a2…93662f5；三套smoke3／Dev50／Final200任务数、顺序和清单哈希不变。将原评测代码绑定到隔离副本，逐份要求与500源哈希相同，veRL导出器保持原安装包绑定；不使用当前语义实验代码。CLI仅替换Python入口、同哈希脚本位置、模型名和输出／摘要路径。旧runner仅修复直接关机exec为shell调用，其余评测与验收流程不变。

新目录为 `preparation/posttrain_step650_v1/`、`export/step650_v1/` 和 `evaluation/step650_independent_v1/`。已有任一目录则拒绝覆盖，准备阶段仅创建元数据目录。

## 2. 有卡实例导出（需要原高内存配额）

```bash
cd /root/autodl-tmp/work/my_shopping_agent
upstream/shopping-grpo-longhorizon/.venv/bin/python -u \
  tools/step650_evaluation.py export --launch
```

主要使用CPU，但无卡实例2 GiB内存不足，沿用至少110 GiB CPU内存配额的有卡实例。磁盘按actor模型文件2倍＋5 GiB预留实测；不复制／删除原训练checkpoint。先用veRL FSDP merger导出底座和LoRA，再合并adapter生成完整BF16推理模型，沿用step500的真实成功流程。

检查r16／alpha32／world_size1、adapter有限值与非零LoRA B、合并权重无残留LoRA参数，记录源actor和合并权重哈希。已有完整验收可校验复用，未验收的非空中间目录拒绝覆盖。此阶段不启动推理或评测。日志位于新POST的 `fsdp_export.log`、`adapter_merge.log`，完成标志为 `export_receipt.json`。

`--launch` 独占创建 `export_driver.log`，启动独立后台进程，锁文件防止重复导出；`export_exit_status.json`记录导出退出状态。进程被外部强杀时可能无退出记录，不能仅凭PID判断已完成。没有自动关机，便于随后继续启动模型服务；若暂不继续应手动关机。可用 `tail -n 40 -F "$POST/export_driver.log"` 查看总日志，具体导出／合并进度在上述两个阶段日志中。

## 3. 后续顺序

导出成功后才启动8000模型服务，核对模型名 `shopping-agent-grpo-step650`、模型路径与上下文24576。沿用500的独立链路：smoke3验收→独立Dev50验收→Final200 Clean。Dev50的高低不改变预定650 Final评测决定；只用执行／身份／完整性验收阻断基础设施问题，不根据成功率挑选任务或重复采样。

模型服务和评测命令待导出确认后提供，不复制旧500会话的passed标志。Final完成后同时报告SFT134/200、500的137/200和650结果，做650↔500以及650↔SFT任务配对。当前没有650 Final结果。

## 本地验证范围

六项CPU测试覆盖：协议／任务／非身份CLI字段不变；准备不导出不覆盖；任务清单变化阻断；隔离源码变化阻断；未知runner关机调用阻断；模拟后台启动独占日志、防重复且不在父进程加载张量。尚未在本地加载真实权重或启动任何服务，实际导出、模型身份和在线评测需云端验证。

## 2026-11-14：云端执行与回传结果

以下来自用户回传的云端输出，尚未将本轮完整原轨迹下载到本地。step650导出与LoRA合并验收通过，导出退出码0；模型服务身份及环境检查通过。smoke3为2/3，1条reward_unverifiable，退出码0且passed=true。独立Dev50为26/50、平均奖励0.4190366、平均步数9.56，非上下文预算执行异常为空，退出码0且passed=true。旧step500独立Dev为27/50，因此不将训练内29/50与独立结果混用。

Final Clean退出码0，passed=true，200/200条完整，无缺题，耗时483.36秒。严格成功／购买成功141/200=70.5%，平均奖励0.6122954852，平均步数9.7；gold141、partial35、循环13、错误购买9、unknown2。上下文超限2题，仍保留在200题分母；done198、reward_valid198。Guard总拒绝25（click6、search19）。不将unknown或上下文失败剔除。

| 同一原协议Final指标 | SFT step300 | GRPO step500 | GRPO step650 |
|---|---:|---:|---:|
| 严格成功 | 134/200 | 137/200 | 141/200 |
| 严格成功率 | 67.0% | 68.5% | 70.5% |
| 平均奖励（四位小数） | 0.5569 | 0.5741 | 0.6123 |
| 平均工具步数 | 9.270 | 9.935 | 9.700 |
| 循环／错误购买 | 16／12 | 15／12 | 13／9 |

step650相对500净增4题（+2.0个百分点），相对SFT净增7题（+3.5个百分点）。这修正了先前仅凭Dev持平而倾向认为续训无收益的判断：旧Dev没有反映这批Final上的正向增量。但原配方未改算法；不能称为新算法改进，也尚未证明跨训练seed的稳定泛化收益。650是预先固定的诊断终点，Dev选模仍是500，不据本次Final改称Dev选出的最优。

下一步只读对比三份Final摘要成功ID，统计新增成功、退化与共同失败，计算配对exact McNemar和按任务配对bootstrap区间；配对结果未返回前，不声称统计显著。分析用于解释本次试验，不根据Final具体题目硬编码prompt、奖励或后续采样规则，也不据+2点立即自动扩训。

用户回传的Final轨迹SHA256：`936869b22540805f1eb470789d5c4228ebfde9ee2f97f14020ede5eb5d21211a`；摘要SHA256：`ab879b3724cf8d9be275cdc3e3340fef739520ecad776ab76b83ba623cba079b`。两者是云端passed记录中的指纹，本地尚未重新计算。本次Final自动关机是否成功尚未由用户回填，不能仅凭评测退出码推断。
