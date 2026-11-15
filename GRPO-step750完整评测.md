# step750完整独立评测

750由650→800训练内Dev预先选定：700为28/50、750为30/50、800为28/50；同分依次比较平均奖励和更早step。此处尚无750 Final结果。复用step500冻结的smoke3、Dev50、Final200 Clean任务、顺序、推理参数、评测代码及完整性验收。独立Dev仍为相同50题，不视为新增验证集，其得分不会改变750的选择或决定是否运行Final。

工具文件保留原名`tools/step650_evaluation.py`，本轮所有命令必须带`--step 750`。prepare仅准备元数据；pipeline依次完成FSDP导出、LoRA合并、环境与vLLM服务身份检查、smoke3、Dev50和Final200 Clean。任何阶段执行或完整性验收失败即停止；不按成功率设置通过阈值，不覆盖已有结果或自动重试。

## 同步与准备

在AutoDL终端执行。可在无卡模式先做prepare，有任一750准备、导出或评测目录时拒绝覆盖。

```bash
(
set -e
cd /root/autodl-tmp/work/my_shopping_agent
git fetch origin codex/bounded-grpo-resume650
git restore --source=origin/codex/bounded-grpo-resume650 --worktree -- \
  tools/step650_evaluation.py tools/test_step650_evaluation.py GRPO-step750完整评测.md
upstream/shopping-grpo-longhorizon/.venv/bin/python \
  tools/step650_evaluation.py prepare --step 750
)
```

prepare检查650→800正常退出、训练Dev选出750、750的LoRA/FSDP元数据、隔离训练及评测源码哈希、冻结任务清单及原500 Final参考值。生成`$RUN/preparation/posttrain_step750_v1/`，不加载张量、不启动服务。

## 后台运行

切换回原有卡实例（CPU内存120 GiB、GPU约95 GiB），保留至少110 GiB CPU内存配额。导出磁盘门槛为actor模型文件的2倍加5 GiB，约25.45 GiB；本次回传空闲41.71 GiB足够。模型服务启动前检查GPU空闲至少90 GiB。不要同时运行其他训练或模型服务。

```bash
cd /root/autodl-tmp/work/my_shopping_agent
upstream/shopping-grpo-longhorizon/.venv/bin/python -u \
  tools/step650_evaluation.py pipeline --step 750 --launch --shutdown-after-exit
```

启动器返回`pipeline launched`后后台独立运行，终端断开不终止任务。总流程成功或报错退出后均等待60秒再通过`/bin/bash -c /usr/bin/shutdown`请求AutoDL关机。该方式修复了此前直接exec无shebang关机脚本的问题，但关机请求记录不能独立证明平台实际已停机。取消本轮自动关机可创建`$POST/pipeline_auto_shutdown.cancel`；它不会取消评测。

## 查看进度

每个新终端都需重新加载session，避免未定义的`EVAL_ROOT`变成根目录路径。

```bash
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step750_v1/session.env
tail -n 60 -F "$POST/pipeline_driver.log"
```

Ctrl+C只退出tail。导出细节在`$POST/export_driver.log`、`fsdp_export.log`和`adapter_merge.log`，服务日志为`$POST/vllm.log`。各套评测的`driver.log`和`eval.log`在`$EVAL_ROOT/smoke3/`、`dev50/`、`final200_clean/`。阶段未开始时文件尚不存在是正常的。

```bash
# 可选：取消评测结束后的自动关机
touch "$POST/pipeline_auto_shutdown.cancel"
```

## 完成检查

关机后可无卡开机读取，不需要重新导出、启动模型或重跑评测。

```bash
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step750_v1/session.env
python - <<'PY'
import json, os
from pathlib import Path
post, root = Path(os.environ['POST']), Path(os.environ['EVAL_ROOT'])
for name in ('pipeline_exit_status.json', 'pipeline_result.json', 'pipeline_shutdown_request.json', 'pipeline_shutdown_error.json'):
    path = post / name
    print(f'【{name}】')
    print(path.read_text() if path.exists() else '不存在')
for name in ('exit_status.json', 'passed.json', 'summary.json'):
    path = root / 'final200_clean' / name
    print(f'【Final {name}】')
    print(path.read_text() if path.exists() else '不存在')
PY
```

完整完成应同时具备总流程exit_code=0、Final exit_code=0、passed=true、expected_tasks=completed_tasks=200且missing_tasks为空。上下文预算失败仍计入200题分母，保持原协议。若报错，保留输出并查看总日志和相应阶段日志；独占日志与锁文件阻止重复启动覆盖，不应删除标记后盲目重试。

Final完成后与SFT134/200、500的137/200、650的141/200做同题配对比较。750本轮是Dev选定模型；历次Final已用于诊断，后续报告应披露这些重复观察，不据Final再挑700或800。

## 本地验证范围

CPU测试使用临时小文件与模拟子进程，覆盖原协议与任务保持不变、准备不覆盖、源码或任务漂移阻断、750的Dev排序、后台启动参数及防重复、smoke→Dev→Final顺序、Dev执行失败阻断Final、通过shell请求关机。没有在本地加载真实权重、调用关机或运行在线评测，真实导出与服务身份需由云端流程核验。
