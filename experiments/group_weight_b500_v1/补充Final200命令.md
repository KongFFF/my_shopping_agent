# 固定B500补充Final200（2026-11-21）

用户在知晓Dev28/50、未达到30/50及净增3题门槛后，明确授权固定B500进行一次原协议Final200 Clean。此为补充评测，不修改此前Dev结论，不称未见测试，不因Final结果回选checkpoint、重跑或自动续800。

复用已验收的B500 merged模型及导出收据，准备和启动前分别核对实际权重SHA256，不重新导出、训练、smoke或Dev。需保留原Dev运行目录中的export/merged及元数据，原A posttrain_step500_v1/evaluation_spec.json、Final200 Clean任务文件和A500 Final摘要（137/200）。

## 1. 无卡准备

```bash
if (
  cd /root/autodl-tmp/work/my_shopping_agent &&
  mkdir -p outputs &&
  git -c http.version=HTTP/1.1 fetch origin codex/b500-final200 &&
  git show FETCH_HEAD:tools/b500_final200_commands.sh > outputs/b500_final200_commands.sh &&
  bash outputs/b500_final200_commands.sh prepare
); then
  printf '准备完成，须看到FINAL200_PREPARED=true；尚未启动服务或评测。\n'
else
  printf '准备失败，请保留输出及FINAL_CPU_LOG，不要执行run。\n'
fi
```

校验原Dev成功退出、B500权重、原A规格哈希、200题顺序/唯一性及清单SHA256 d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5。读取数GiB推理权重计算哈希可需几分钟，不加载模型张量。新输出为outputs/runs/b500_final200_时间_后缀，新指针outputs/b500_final200_active.txt，不覆盖Dev。

## 2. 开卡后台启动

```bash
if bash /root/autodl-tmp/work/my_shopping_agent/outputs/b500_final200_commands.sh run; then
  printf '后台请求已提交，请执行status确认实际进展。\n'
else
  printf '启动失败，保留现场并执行status，不要反复run。\n'
fi
```

5700/8000端口应空闲；已有服务会停止，不杀未知进程。沿用原模型服务参数（上下文24576、空闲显存≥90 GiB）及环境8槽位检查。只执行原Final命令，参数/任务/投影/guard/严格gold定义不变。上下文失败和无法核验保留在200题分母，不追加尝试。

## 3. 状态与日志

```bash
bash /root/autodl-tmp/work/my_shopping_agent/outputs/b500_final200_commands.sh status
```

```bash
bash /root/autodl-tmp/work/my_shopping_agent/outputs/b500_final200_commands.sh follow
```

Ctrl+C只退出follow。完成判据为final200_complete.json的passed=true且pipeline_exit_status.json的exit_code=0。流水线成功或失败后只清理带本次标记的自有服务，不自动关机；预算仍由用户记录。

## 4. Git回传

```bash
if bash /root/autodl-tmp/work/my_shopping_agent/outputs/b500_final200_commands.sh return; then
  printf '回传完成，请发送RETURN_BRANCH和RETURN_COMMIT。\n'
else
  printf '回传失败，请保留现场；仍运行时请等待结束后回传。\n'
fi
```

包括原A Final摘要、B500导出收据、Final原始轨迹/摘要/日志、模型服务身份及完成收据，不含模型权重。失败不自动重跑，先回传诊断。后续与A500按同200题配对分析，保留补充评测和单seed历史对照限制。
