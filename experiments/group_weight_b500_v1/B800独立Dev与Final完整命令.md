# 固定B800原协议独立Dev50与Final200完整命令

2026-11-22。用户报告从750恢复的B800训练已跑完，并明确要求执行独立Dev和Final。新入口在AutoDL准备阶段先验证训练完成证据，再固定800导出；本地尚未收到最后50步原始材料，不能提前声称已完成本地训练核验。

固定顺序为导出→服务身份/槽位核验→smoke3→原独立Dev50→Final200 Clean。Dev或smoke成功率不作为后续分数门槛；身份、协议、缺题/重复或非上下文执行异常会停止。每题执行一次，不按分数重跑、改参数或换750；不自动继续训练或关机。

## 1. 拉取并准备，可无卡执行

```bash
if (
  cd /root/autodl-tmp/work/my_shopping_agent &&
  mkdir -p outputs &&
  git -c http.version=HTTP/1.1 fetch origin codex/b800-evaluation &&
  git show FETCH_HEAD:tools/b800_eval_commands.sh > outputs/b800_eval_commands.sh &&
  bash outputs/b800_eval_commands.sh prepare
); then
  printf '准备完成；确认B800_EVAL_PREPARED=true后再执行第2步，尚未导出或评测。\n'
else
  printf '准备失败；保留输出及EVAL_CPU_LOG，不要执行run。\n'
fi
```

入口读取outputs/b800_finish750_active.txt，不读取指向失败运行的旧B800指针。检查以下材料：

- training_complete中completed_updates=800、新更新50，无stopped/budget_truncated；最后800更新验收收据、唯一连续1—800链、恢复750前缀、最后50步实际LR1e-6及有限非零Actor更新。
- 651—800对应150次组权重调用的返回链；物理生成账本与tokens逐项求和及完成收据一致。首次失败和37步回滚的成本保留。
- 原核心方法/基准协议身份、尾段skip50补充协议哈希、完整800恢复点及latest指针、全文件CRC、r16/alpha32/world_size1。记录800 Actor SHA256，导出后再次核对。
- 评测依据原A800的posttrain_step800_v1规格与原摘要（独立Dev29/50、Final143/200），保持harness源码哈希、三套任务顺序及原CLI/默认值。只替换解释器/已核验隔离脚本路径、B800模型名和输出位置。

准备重跑新评测模块CPU测试；本地118项完整CPU验收通过不替代AutoDL验收。需要顺序读取约11 GiB检查点做CRC和模型哈希，可能耗时几分钟，不加载模型张量或GPU。

磁盘要求模型文件2倍＋5 GiB，当前约26 GiB额外空闲，以实际打印为准。只检查，不删除旧检查点或导出。

新目录outputs/runs/b800_evaluation_时间_后缀、新指针outputs/b800_eval_active.txt，旧训练及评测不覆盖。准备成功后用status/run，不得删除指针重复prepare。

## 2. 开卡后后台执行全流程

```bash
if bash /root/autodl-tmp/work/my_shopping_agent/outputs/b800_eval_commands.sh run; then
  printf '后台请求已提交，请立即执行status确认实际启动。\n'
else
  printf '启动失败；保留现场，不要反复run。\n'
fi
```

沿用原实例要求：CPU内存配额至少110 GiB，GPU空闲至少90 GiB；5700/8000端口须空闲。固定800走既有FSDP导出、LoRA合并BF16及vLLM服务流程。服务名shopping-agent-b800-group-weight-v1，核对模型根目录、ID和上下文24576。

smoke3用于执行核验，不并入Dev/Final。按原summary重算严格gold、协议及完整轨迹；ContextBudgetError、invalid_action_limit等原协议失败保留50/200分母，不补采样。Final仍使用原Clean清单SHA256 d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5。

预算由用户自行记录，不套旧训练时间/生成上限，不自动关机；原训练生成账本不被本次独立评测改写。

## 3. 查看状态及日志

```bash
bash /root/autodl-tmp/work/my_shopping_agent/outputs/b800_eval_commands.sh status
```

```bash
bash /root/autodl-tmp/work/my_shopping_agent/outputs/b800_eval_commands.sh follow
```

Ctrl+C只退出日志查看。总日志supervisor.log；阶段明细在preparation/posttrain_step800_v1下的fsdp_export.log、adapter_merge.log、vllm.log，以及evaluation各suite/driver.log。

Dev完成会写dev50_complete.json，Final完成写final200_complete.json。整个流程完成必须同时有evaluation_complete.json的passed=true及pipeline_exit_status.json的exit_code=0，监督器输出B800_DEV_AND_FINAL_COMPLETE。只导出成功或只完成Dev不算结束。

## 4. 正常结束或失败停止后Git回传

```bash
if bash /root/autodl-tmp/work/my_shopping_agent/outputs/b800_eval_commands.sh return; then
  printf '回传完成，请发送RETURN_BRANCH和RETURN_COMMIT。\n'
else
  printf '回传失败；保留现场，若仍在运行则等待结束。\n'
fi
```

回传三套原轨迹/摘要、导出和服务指纹、评测完成收据、A800摘要及仍保留的原轨迹；同时附带training_evidence中的B800完成收据、配置、完整有效链、权重/物理计数日志、750恢复补充协议、原生验证及中断快照。不含模型权重，无需为了取得训练材料另跑GPU任务。

主要同更新数历史对照为A800（29/50、143/200），B650（25/50、143/200）作为本方法新增训练前的参照。必须保留750恢复、尾段停止阈值50及额外生成/回滚差异，不称严格同计算成本或完全相同训练条件；Final已多次观察，不称完全未见测试。
