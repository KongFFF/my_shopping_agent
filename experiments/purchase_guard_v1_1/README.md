# 购买保护 v1.1：规格动作字符串保真

2026-10-30，在 v1 的原 Dev-50 复评中发现并复现了一个原有接口错误。训练继续使用原始 1000 题，原商品库和评分代码保持不变。

## 发现与更正

用户提供的 v1 结果为 26/50，旧 SFT 为 24/50；新增成功 4128、14141，无退化。这两题没有触发购买保护，不能将新增成功直接归因于该保护。

2718、16252 原先均因漏选规格导致 `reward_unverifiable`；v1 中变成有效 `wrong_purchase`。但查看模型消息发现，两题在提示后都尝试了 `select_option`，这些调用被动作守卫拒绝，没有进入已执行的 `steps`。因此，不能根据 `steps` 连续出现 `buy_now` 就判断模型没有尝试纠错。

2718 的真实值是 `bykski cpu冷头  黑色款`（冷头后两个空格）。`available_options` 保留原值，但 footer 使用展示文本的空格归一化，把两个空格压成一个。模型复制真实值时，守卫对照被改写的 footer，误报 `click_not_in_previous_observation`。即使模型改为复制单空格 footer，环境的真实 HTML 按钮仍使用双空格，选择也可能不执行。

16252 的模型输出是 `3包不锈钢料箱-全自动高速+无料停机（质保一年`，真实值为 `3包不锈钢料箱- 全自动高速+无料停机（质保一年`。模型漏了连字符后的一个空格；其缺少右括号本来就存在于原始选项，不能把右括号当作模型漏字。本次不对错误参数做模糊匹配或自动猜测。

## 修复

只修复 Agent 共用 observation renderer 的 footer：可执行动作字符串原样通过 JSON 输出，不再套用正文展示的空格压缩。规格正文、footer、守卫和环境按钮由此使用同一个原始字符串。正文中的普通标题和描述仍保留原来的展示处理。

购买拒绝、最多三次不完整购买、终局奖励、数据集、商品库、工具 schema 和 GRPO 优化算法均沿用 v1。v1.1 的环境服务源码与 v1 一致，运行差异仅在 Agent 的 renderer。原 v1 代码覆盖包和既有输出目录保持原样。

构建器新增显式开关，未传开关时仍生成原 v1：

```bash
python tools/build_purchase_guard_v1.py \
  --materialize outputs/purchase_guard_v1_1 \
  --preserve-action-values
python tools/verify_purchase_guard_v1.py outputs/purchase_guard_v1_1
```

本地验证：v1.1 原有 20 项测试通过；新增 7 项接口回归通过，包括旧版错误复现、双空格值的真实 HTML 选择、GRPO 工具路径、长 observation 投影，以及严格拒绝模型漏空格的参数。原 v1 的 20 项测试、两版本完整性检查和 928 题冻结锁检查也通过。

## AutoDL 下一步

先拉取并构建，均可无卡执行：

```bash
cd /root/autodl-tmp/work/my_shopping_agent
git pull --ff-only origin main
source outputs/runs/purchase_guard_sft300_dev50_v1/session.env
"$PY" tools/build_purchase_guard_v1.py \
  --materialize outputs/purchase_guard_v1_1 --preserve-action-values
"$PY" tools/verify_purchase_guard_v1.py outputs/purchase_guard_v1_1
```

不要覆盖旧的运行目录。以下无卡准备从已完成的 v1 运行复制固定协议和原 Dev 文件，生成独立 v1.1 运行目录；不复制旧评测结果。

```bash
"$PY" - <<'PY'
import json, os, shutil
from pathlib import Path

root = Path(os.environ['ROOT'])
old = root / 'outputs/runs/purchase_guard_sft300_dev50_v1'
new = root / 'outputs/runs/purchase_guard_sft300_dev50_v1_1'
bundle = root / 'outputs/purchase_guard_v1_1'
assert not new.exists(), f'已有目录，请检查已有运行：{new}'
manifest = json.loads((bundle / 'manifest.json').read_text())
old_manifest = json.loads((root / 'outputs/purchase_guard_v1/manifest.json').read_text())
assert manifest['preserve_action_values'] is True
assert manifest['dataset_unchanged_sha256'] == old_manifest['dataset_unchanged_sha256']
for name, digest in manifest['changed_files_sha256'].items():
    if name.startswith('environment/'):
        assert old_manifest['changed_files_sha256'][name] == digest, name

spec = json.loads((old / 'evaluation_spec.json').read_text())
spec.update(experiment='purchase-guard-v1.1', bundle=str(bundle), bundle_manifest=manifest)
new.mkdir()
for name in ('run_eval.py', 'dev50.jsonl', 'smoke3.jsonl'):
    shutil.copyfile(old / name, new / name)
(new / 'evaluation_spec.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n')
(new / 'session.env').write_text(
    'source /root/autodl-tmp/work/my_shopping_agent/outputs/runs/purchase_guard_sft300_dev50_v1/session.env\n'
    'export BUNDLE="$ROOT/outputs/purchase_guard_v1_1"\n'
    'export RUN="$ROOT/outputs/runs/purchase_guard_sft300_dev50_v1_1"\n'
    'export PYTHONPATH="$BUNDLE/agent"\n'
)
print('准备完成：', new)
PY
```

有卡模式评测前恢复新 session：

```bash
source /root/autodl-tmp/work/my_shopping_agent/outputs/runs/purchase_guard_sft300_dev50_v1_1/session.env
"$PY" -c 'import shopping_grpo.environment.observation as m; print(m.__file__)'
```

打印路径必须在 `outputs/purchase_guard_v1_1/agent/` 下。

若 v1 的 5711 环境和 8001 SFT step300 服务仍运行，可复用：环境源码已在准备阶段检查与 v1 相同，模型权重和推理设置也未变。若已经关机，需要按前次启动步骤重启服务，并确认 8001 的模型仍是选定的 SFT step300；新环境启动时使用本 session 的 `BUNDLE`。不得把仍运行的其他模型当作本轮模型。

继续使用前次的身份检查和 smoke3 验收，先执行：

```bash
mkdir -p "$RUN/smoke3"
nohup "$PY" -u "$RUN/run_eval.py" smoke3 \
  > "$RUN/smoke3/driver.log" 2>&1 < /dev/null &
echo $! > "$RUN/smoke3/driver.pid"
tail -n 40 -F "$RUN/smoke3/driver.log"
```

只有 smoke3 退出码为 0、三题完整且无执行/释放异常时，再使用前次相同命令运行 Dev-50（此时 `RUN` 已指向 v1.1）。模型服务需要 GPU；CPU 汇总检查不需要 GPU。两题诊断可以用于回归检查，但不能根据其结果改模型或筛选 Dev 任务。

本次未在本地启动模型、训练或正式评测。云端 v1.1 是否改善纠错行为和 Dev 成绩，需要以上述新运行结果为准。
