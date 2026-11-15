# step800终点诊断评测

用户在查看750的Final142/200后，明确授权补测原定续训终点800。800用于诊断650→800的终点表现，不替换训练内Dev选出的750。报告应披露已查看750及更早Final的事实，不将本次比较称为未曾观察测试集的最终选模。

沿用step500冻结的smoke3、Dev50、Final200 Clean任务、顺序、代码与推理参数。800仍走导出→环境及模型身份检查→smoke→Dev→Final的完整流程；不根据Dev分数决定是否运行Final。准备目录`preparation/posttrain_step800_v1`，导出目录`export/step800_v1`，评测目录`evaluation/step800_independent_v1`，模型名`shopping-agent-grpo-step800`。不复制750的passed标记，不覆盖750结果。

## 同步与准备

可无卡执行prepare。工具仍使用原文件名，所有命令明确带`--step 800`。

```bash
(
set -e
cd /root/autodl-tmp/work/my_shopping_agent
git fetch origin codex/bounded-grpo-resume650
git restore --source=origin/codex/bounded-grpo-resume650 --worktree -- \
  tools/step650_evaluation.py tools/test_step650_evaluation.py GRPO-step800终点评测.md
upstream/shopping-grpo-longhorizon/.venv/bin/python \
  tools/step650_evaluation.py prepare --step 800
)
```

prepare验证650→800训练正常退出、训练Dev仍选择750、800 checkpoint元数据、隔离源码及原评测任务哈希。已有800目录拒绝覆盖。不会启动导出、GPU服务、评测或关机。

## 有卡后台执行

使用原120 GiB CPU内存、约95 GiB GPU的实例。导出前检查CPU内存配额至少110 GiB，磁盘空闲至少actor模型文件的2倍加5 GiB，约25.45 GiB。上次750导出已占用空间，实际空闲需重新检查，不能沿用此前41.71 GiB。模型启动前要求GPU空闲至少90 GiB。建议在上一轮实际关机后重新启动的实例执行，避免750服务占用8000端口；若端口模型身份不符，流程会停止，不会杀死未知服务。

```bash
cd /root/autodl-tmp/work/my_shopping_agent
upstream/shopping-grpo-longhorizon/.venv/bin/python -u \
  tools/step650_evaluation.py pipeline --step 800 --launch --shutdown-after-exit
```

`pipeline launched`后独立后台执行，可断开本地终端。任一阶段执行或完整性检查失败则停止；成功或报错退出后均等待60秒，通过shell请求AutoDL关机。关机请求记录不能独立证明平台实际停机。

## 查看日志与取消自动关机

每个新终端先source本轮session，避免变量为空或沿用750路径。

```bash
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step800_v1/session.env
tail -n 60 -F "$POST/pipeline_driver.log"
```

Ctrl+C只退出日志查看。导出明细在`$POST/export_driver.log`、`fsdp_export.log`、`adapter_merge.log`，vLLM日志为`$POST/vllm.log`。进入Final后可另开终端重新source，再执行：

```bash
tail -n 40 -F "$EVAL_ROOT/final200_clean/eval.log"
```

阶段尚未开始时相关日志不存在正常。可选取消本轮结束后的自动关机，不取消评测：

```bash
touch "$POST/pipeline_auto_shutdown.cancel"
```

## 完成检查

结束后可无卡开机读取，禁止因仅需读取结果而再次启动pipeline。

```bash
source /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/preparation/posttrain_step800_v1/session.env
python - <<'PY'
import os
from pathlib import Path
post = Path(os.environ['POST'])
final = Path(os.environ['EVAL_ROOT']) / 'final200_clean'
for title, path in [
    ('总流程退出状态', post / 'pipeline_exit_status.json'),
    ('各阶段结果', post / 'pipeline_result.json'),
    ('Final退出状态', final / 'exit_status.json'),
    ('Final验收', final / 'passed.json'),
    ('Final完整摘要', final / 'summary.json'),
    ('关机请求', post / 'pipeline_shutdown_request.json'),
    ('关机异常（无此文件通常正常）', post / 'pipeline_shutdown_error.json'),
]:
    print(f'\n【{title}】')
    print(path.read_text() if path.exists() else '不存在')
PY
```

完整完成须总流程与Final exit_code均为0、passed=true、expected_tasks=completed_tasks=200、missing_tasks为空。上下文失败继续计入200题分母。若失败，保留日志及中间结果，先定位，不删除日志或标记盲目重跑。

## 验证范围

13项本地CPU测试通过，新增检查800终点与750 Dev选模身份分别记录、协议及任务保持不变、750输出保留、800后台子进程参数和独立输出路径。既有测试覆盖阶段顺序、错误阻断和shell关机调用。均为小文件或模拟子进程测试，未在本地加载真实checkpoint、执行在线评测或调用实际关机。云端是否成功以用户运行输出为准。
