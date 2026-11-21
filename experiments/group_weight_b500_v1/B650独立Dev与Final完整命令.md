# 固定B650原协议独立Dev50与Final200完整流程

2026-11-21。用户明确授权固定B650导出、原独立Dev50和Final200 Clean。顺序为导出→服务身份/槽位核验→smoke3→Dev50→Final200。Dev成功率不作为本次已授权Final的开关；只有身份、协议、完整性或非上下文执行错误阻断，不根据分数重复采样、改参数或换600端点，不自动续800。

## 1. 无卡准备

```bash
if (
  cd /root/autodl-tmp/work/my_shopping_agent &&
  mkdir -p outputs &&
  git -c http.version=HTTP/1.1 fetch origin codex/b650-evaluation &&
  git show FETCH_HEAD:tools/b650_eval_commands.sh > outputs/b650_eval_commands.sh &&
  bash outputs/b650_eval_commands.sh prepare
); then
  printf '准备完成，须看到B650_EVAL_PREPARED=true；尚未导出或评测。\n'
else
  printf '准备失败，请保留输出及EVAL_CPU_LOG，不要执行run。\n'
fi
```

从outputs/b650_active.txt定位已验收运行，核对650完成标记、新150步、核心方法/配置身份，检查650完整checkpoint、CRC与r16/alpha32/world_size1。读取约11 GiB模型做CRC/SHA256，不加载张量，可需几分钟。无卡原Python执行新流程CPU测试。

参考原A650的preparation/posttrain_step650_v1/evaluation_spec.json，要求checkpoint/model_tag明确650、Dev26/50和Final141/200原摘要；保持三套任务顺序、原CLI参数/默认值及哈希相同的原harness。只替换Python/隔离脚本路径、B650模型名及新输出位置。Final Clean清单SHA256仍为d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5。

新输出outputs/runs/b650_evaluation_时间_后缀，指针outputs/b650_eval_active.txt，不覆盖B500或训练目录。按原导出器预留模型文件2倍＋5 GiB，当前约26 GiB额外空闲；不足在无卡阶段停止，不删除检查点或旧导出。

## 2. 开卡后启动后台全流程

```bash
if bash /root/autodl-tmp/work/my_shopping_agent/outputs/b650_eval_commands.sh run; then
  printf '后台请求已提交，请立即执行status核查。\n'
else
  printf '启动失败，请保留现场，不要反复run。\n'
fi
```

需要原同规格实例：CPU配额≥110 GiB，GPU空闲≥90 GiB；5700/8000端口应空闲，已有服务则停止，不终止未知进程。复用此前真实成功的FSDP导出＋LoRA合并BF16逻辑；记录源650 Actor SHA256和merged SHA256，与准备记录核对，再启动服务核对实际model id/模型根目录/上下文24576。服务名shopping-agent-b650-group-weight-v1。

原harness顺序执行三套评测，每题一次。用原summarize_trajectories重算严格gold、完整摘要和协议，缺题/重复/非ContextBudgetError错误/释放错误都停止；上下文失败、无法核验保留在原分母，不补采样。smoke或Dev分数低本身不阻止后续已授权评测。

## 3. 查看状态及持续日志

```bash
bash /root/autodl-tmp/work/my_shopping_agent/outputs/b650_eval_commands.sh status
```

```bash
bash /root/autodl-tmp/work/my_shopping_agent/outputs/b650_eval_commands.sh follow
```

Ctrl+C只退出查看。根日志supervisor.log；阶段细节在preparation/posttrain_step650_v1的fsdp_export.log、adapter_merge.log、vllm.log，以及evaluation各suite/driver.log。

Dev完成会先写dev50_complete.json；Final完成写final200_complete.json。整个任务必须同时有evaluation_complete.json的passed=true和pipeline_exit_status.json的exit_code=0。导出通过或仅Dev完成不算全部结束。

正常或异常退出后只清理带本次唯一标记的服务和评测子进程，不自动关机，预算仍由用户自行记录。失败不自动重跑；先保留现场回传。

## 4. 正常结束或失败后Git回传

```bash
if bash /root/autodl-tmp/work/my_shopping_agent/outputs/b650_eval_commands.sh return; then
  printf '回传完成，请发送RETURN_BRANCH和RETURN_COMMIT。\n'
else
  printf '回传失败，请保留现场；仍运行时请等待结束。\n'
fi
```

回传三套原轨迹/摘要/日志、导出指纹、服务身份、完成收据及A650 Dev/Final摘要；若历史A650原轨迹仍在，也一并带回以便深层配对。不上传模型权重。

结果主要比较同650次更新的A650（独立Dev26/50、Final141/200），同时以B500（独立Dev28/50、Final141/200）判断本方法新增150步的变化。训练内650的score=1计数30/50不能替代本独立Dev结果。固定650是在已观察B500 Dev/补充Final后另行授权的延长端点，不称未见测试、自动选出的最佳模型或严格同计算成本实验。
