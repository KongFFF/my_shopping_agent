# 历史A500条件匹配核验

2026-11-17。结论：**历史A可作为B从同一SFT起训到500的历史对照，但不是跨seed、逐位相同的同期对照。已完成本地证据核验；云端实际权重与原baseline_plan的最终比对仍是开跑硬门槛。** 不将协议冻结等同于训练已可执行。

> 2026-11-18补证：[云端预检诊断](20261118身份预检诊断.md)确认原baseline_plan与实际SFT权重、数据及31项已完成文件检查匹配；唯一失败是预检把共享uv.lock定位到了隔离目录。修正后仍待实物哈希复核，不宣称预检全项通过。

## 已完成的本地核验

| 条件 | 结果与B的匹配要求 |
|---|---|
| 历史身份 | 原A运行grpo_baseline_20261022_202816，代码基线a583dba4c149be09622ce58ad6f7a160db77bb41 |
| SFT起点 | sft_20261021_165243/merged，即选定step300；B从此模型新挂GRPO LoRA，不加载A的Actor/优化器 |
| SFT指纹 | 两份后续独立运行身份收据的8个文件指纹一致，model.safetensors为c79c26f6d39cc42b4d4766b276d54dfd00d241b6b1bec159976e270a2930f16a；未在本地读取权重 |
| 数据 | 原train1000、Dev50、Final200任务ID互斥；原4个JSONL/Parquet文件哈希均匹配（文本允许核验LF换行），与在线窗口训练ID一致 |
| 环境 | ShopSimulator v2.1、Reward v3、Observation v2、Tool v2；原环境资产/源文件哈希固定，已包含当时的runtime_changes.patch，不能只检出裸基线提交 |
| 有效更新 | CSV重算唯一连续1—500，每次optimizer_updated=1；旧351—358回滚不计入有效链 |
| 学习率 | 原1—15步线性warmup，第15步达到1e-6，之后恒定；500条LR全核对。不能从第1步直接用1e-6 |
| 恢复条件 | 原有效链为formal_memfix_v1的1—350，加formal_resume350_v1的351—500；两份配置只有7项预期运行/恢复字段不同 |
| Actor | LoRA rank16/alpha32、all-linear；AdamW、lr1e-6、weight_decay0.01、betas0.9/0.999、clip_grad1；ppo_epochs1，token-mean，无KL/熵项 |
| 采样 | 每次两题、每题4条；温度0.7/top_p0.9；模型上下文24576，但有效输入预算16384；每步最多512生成token、工具35步，无历史压缩 |
| 动态筛组 | 原seq_reward筛选，max_num_gen_batches3，max_consecutive_skipped_updates10，容差1e-8。B保持它们，不顺带回收组或改数据调度 |
| 随机设置 | data/Actor loader/FSDP为42，PYTHONHASHSEED42；原未显式指定新vLLM seed，不追加2026系列seed；异步生成仍不保证逐位相同 |
| 评测 | 原A500独立Dev27/50，SFT独立Dev24/50。两份协议除服务名称外一致；训练内A500的29/50属于另一harness，不作B独立Dev对照 |

基线证据包11个文件全部通过原SHA256或仅CRLF归一化比对。未使用旧4192轨迹；只为核对SFT身份读取旧运行的模型元数据，不读取其任务选择、采样结果或结论。Final只核对ID边界，未新读Final成绩或运行Final。

完整核验、配置差异及数据/模型指纹见[historical_a_audit.json](historical_a_audit.json)。复算：`python -X utf8 tools/freeze_group_weight_b500.py`。

## 不能装作已经匹配的条件

1. 最初A的preparation/baseline_plan.json与真实SFT权重在云端；本地只有后续两份一致身份收据。必须在云端核对“原A计划指纹=冻结收据指纹=当前权重”，不以路径相同代替。
2. 原0—350的完整源码/运行状态没有全部独立归档到本地。原配置、后段源码绑定、有效更新日志与历史收据支持配方连续性，但不能证明每个包、GPU内核或随机状态完全一致。
3. 原A有350步重启及被回滚的8步。B预设一次350步完整保存/恢复，匹配恢复结构；不故意重做失败分支，也不声称DataLoader/RNG逐位一致。B若意外恢复，完整记录而不隐藏。
4. 历史A只有一个训练seed，B也为单seed；不能把任务配对检验当作跨seed方法可靠性证据。Final已多次观察，不称未见测试。

因此，主结论的适当表述是“该固定配置下、从同一SFT起点、相同500次有效更新的单seed历史对照结果”，而非已经严格隔离所有运行因素的因果证明。

## 成本口径

历史A有效链生成17184条，回滚额外208条，共17392条训练生成。有效更新500次、学习轨迹4000条；全部跳过、无效和回滚生成必须计费/计成本。历史有效更新timing_s/step相加约10.46小时，只是保留更新记录的时间下界，不是整个A的GPU占用时长，不能拿它精确宣称B更便宜。

500步约为800步的62.5%更新量，已是有明显成本的阶段训练，不再等同于40步工程探测。B协议设置24000条全部训练生成及20小时GPU占用上限；预算截断时报告实际步数，不伪称完成B500。

## 云端无卡预检

已提供只读工具[check_b500_history.py](../../tools/check_b500_history.py)：不载入张量、不启动服务或训练，核对原baseline_plan、当前SFT、原数据、隔离基线代码和共享依赖源文件哈希，并输出receipt。执行方式与Git回传见[协议说明](../../experiments/group_weight_b500_v1/README.md)。

若实际资产与绑定不符，先恢复可验证的基线runtime或补证；不把旧实验补丁、奖励修复或新依赖悄悄带入B。预检通过也只表示身份条件满足，B损失实现及其验收仍需后续完成。
