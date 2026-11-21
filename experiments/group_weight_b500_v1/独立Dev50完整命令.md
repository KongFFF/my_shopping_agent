# 固定B500独立Dev50完整流程

2026-11-21。先无卡准备，再有卡后台一次性导出→服务身份核验→原smoke3→原独立Dev50。只改模型名、同哈希隔离脚本入口及输出位置，原CLI其余字段/默认值、任务顺序、环境/奖励/投影/guard均绑定原A500。无Final、重训、端点挑选、自动续训或自动关机。

## 1. 无卡准备

```bash
if (
  cd /root/autodl-tmp/work/my_shopping_agent &&
  mkdir -p outputs &&
  git -c http.version=HTTP/1.1 fetch origin codex/b500-independent-dev50 &&
  git show FETCH_HEAD:tools/b500_dev50_commands.sh > outputs/b500_dev50_commands.sh &&
  bash outputs/b500_dev50_commands.sh prepare
); then
  printf '准备完成，须看到DEV50_PREPARED=true；尚未导出或评测。\n'
else
  printf '准备失败，请保留输出与DEV50_CPU_LOG；不要执行run。\n'
fi
```

准备读取outputs/b500_resume150_active.txt锁定已完成训练的global_step_500，核对完成标记、500检查点全文件CRC、r16/alpha32/world_size1，以及隔离runtime与原A评测源哈希。会在原Python环境执行本流程CPU测试；不加载模型张量、启动GPU或服务。CRC需读取约11 GiB文件，可等待几分钟。

原A评测规格来源必须仍在共享资产preparation/posttrain_step500_v1/evaluation_spec.json；原Dev50数据、隔离runtime和.venv也需保留。任务顺序与归档A50条轨迹相同，原A独立27/50及原冻结参数再次匹配。准备不触发原规格里的Final命令。

新运行目录为outputs/runs/b500_dev50_时间_后缀，指针outputs/b500_dev50_active.txt；原训练和检查点不修改。磁盘预检按模型文件2倍＋5 GiB，当前约25.5 GiB额外空闲。元数据与完整参数写在新目录preparation/posttrain_step500_v1，指纹由dev50_code_manifest.json单独管理。

## 2. 开卡后启动后台流水线

```bash
if bash /root/autodl-tmp/work/my_shopping_agent/outputs/b500_dev50_commands.sh run; then
  printf '后台请求已提交，请立即执行status；PID不代表已完成。\n'
else
  printf '启动请求失败，请保留现场，不要反复run。\n'
fi
```

导出复用原step500/650已成功的FSDP merger＋PEFT合并逻辑：CPU gloo初始化、导出底座和LoRA、验证adapter有限值及非零B、合并BF16，记录源Actor和推理权重哈希。无需无卡小内存实例加载模型；有卡实例仍需CPU配额≥110 GiB，模型启动要求GPU空闲≥90 GiB。中间目录有残留会停止，完整结果存在时也不会重新启动整条流水线。

5700/8000端口在流水线开始时必须空闲，已有服务则停止，不杀未知进程。服务模型名shopping-agent-b500-group-weight-v1，模型路径绑定新merged目录，服务上下文24576，环境检查8个槽位reset/release。运行期间冻结源文件漂移即停止。

smoke3与Dev50直接执行原CLI，输出目录全新，逐题一次，不追加尝试。按原生summarize_trajectories重新核对完整摘要及严格gold成功；非ContextBudgetError执行错误、释放错误、缺题/重复题、协议不符均阻断。上下文失败和reward_unverifiable保留在分母中，不补采样。smoke成功题数不设门槛，失败不靠重跑追分。

## 3. 查看状态或持续日志

```bash
bash /root/autodl-tmp/work/my_shopping_agent/outputs/b500_dev50_commands.sh status
```

```bash
bash /root/autodl-tmp/work/my_shopping_agent/outputs/b500_dev50_commands.sh follow
```

Ctrl+C只退出follow。导出细节日志为POST下fsdp_export.log/adapter_merge.log，模型为vllm.log，各套评测在evaluation/smoke3和evaluation/dev50的driver.log。服务启动或导出通过不是Dev完成；必须看到dev50_complete.json中passed=true，并且pipeline_exit_status.json中exit_code=0。成功后报告A500=27、B500、B-only/A-only、净增及冻结筛查门槛是否达到；不自动改变任何路线。

退出时只清理带本次唯一标记的服务/评测子进程，不自动关机。预算由用户自行记录，无旧20小时截止。异常现场保留，禁止重复run；先回传诊断。

## 4. 正常完成或失败后Git回传

```bash
if bash /root/autodl-tmp/work/my_shopping_agent/outputs/b500_dev50_commands.sh return; then
  printf '回传完成，请发送RETURN_BRANCH和RETURN_COMMIT。\n'
else
  printf '回传失败，请保留日志；仍运行时需等结束后再回传。\n'
fi
```

包括命令规格、导出指纹、服务身份、smoke3/Dev50全轨迹/摘要/日志及通过收据，并补带先前未回传的训练validation/*.jsonl；不包含模型权重或Final轨迹。后续本地按A-only/B-only配对分析，报告单seed历史对照和150额外恢复的限制。
