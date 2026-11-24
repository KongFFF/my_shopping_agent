# 固定C500原协议独立Dev50＋Final200完整命令

2026-11-24。用户明确授权两套评测。固定C500，不按中间分数换checkpoint；自动依次导出一次→smoke3→Dev50→Final200。Dev分数不阻断Final，执行/身份错误停止，不重跑失败题。无自动续训或关机。

## 1. 无卡准备

仅首次执行。轻量/稀疏克隆只取所需代码和冻结包，避免拉取大量历史结果。读取并保存原生CPU收据、首次磁盘预检失败场景，验证既有C源码、1—500更新/权重链、500恢复点CRC及LoRA元数据。仅跑新评测入口CPU测试，不重跑原C训练CPU验收、不训练、不加载模型。

```bash
cd /root/autodl-tmp/work/my_shopping_agent
bash <<'BASH'
set -euo pipefail
ROOT=/root/autodl-tmp/work/my_shopping_agent
cd "$ROOT"
mkdir -p "$ROOT/outputs"
LOG="$ROOT/outputs/c500_eval_prepare_$(date -u +%Y%m%dT%H%M%SZ)_$RANDOM.log"
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; printf "PREPARE_EXIT_CODE=%s\nLOG=%s\n" "$rc" "$LOG"; if [[ "$rc" != 0 ]]; then echo "准备失败，不要执行有卡run，保留本日志。"; fi' EXIT
[[ ! -e "$ROOT/outputs/c500_eval_active.txt" ]] || { echo '已有评测准备，请使用status/return，勿重复prepare。'; exit 1; }
CODE=$(mktemp -d "$ROOT/outputs/c500-eval-code-XXXXXX")
printf 'CODE=%s\n' "$CODE"
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch --filter=blob:none --sparse --branch codex/c500-evaluation "$(git remote get-url origin)" "$CODE"
git -C "$CODE" sparse-checkout set tools experiments/group_weight_c500_v1 experiments/group_weight_b500_v1
export OMP_NUM_THREADS=1
bash "$CODE/tools/c500_eval_commands.sh" prepare
BASH
```

必须出现C500_EVAL_PREPARED=true且PREPARE_EXIT_CODE=0才继续。checkpoint CRC需顺序读取约10GiB，请等待。原生CPU收据或旧失败现场缺失/不匹配时停止，保留LOG返回诊断，不用重训补证。

## 2. 有卡启动

切回原高内存有卡实例（导出需至少110GiB CPU配额、服务需GPU空闲至少90GiB；磁盘需约26GiB额外导出空间），执行：

```bash
cd /root/autodl-tmp/work/my_shopping_agent
bash <<'BASH'
set -euo pipefail
ROOT=/root/autodl-tmp/work/my_shopping_agent
PY="$ROOT/upstream/shopping-grpo-longhorizon/.venv/bin/python"
IFS= read -r OUT < "$ROOT/outputs/c500_eval_active.txt"
CODE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"])' "$OUT/posttrain.json")
bash "$CODE/tools/c500_eval_commands.sh" run
BASH
```

后台执行，输出SUPERVISOR_PID和OUT。PID仅表示已启动监督器；实际导出/服务/评测进展看第3段。完成Dev后自动继续Final，无需再启动一遍，也无需再次导出。

## 3. 查状态（可重复）

```bash
cd /root/autodl-tmp/work/my_shopping_agent
bash <<'BASH'
set -euo pipefail
ROOT=/root/autodl-tmp/work/my_shopping_agent
PY="$ROOT/upstream/shopping-grpo-longhorizon/.venv/bin/python"
IFS= read -r OUT < "$ROOT/outputs/c500_eval_active.txt"
CODE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"])' "$OUT/posttrain.json")
bash "$CODE/tools/c500_eval_commands.sh" status
BASH
```

成功标准：evaluation_complete.json的passed=true，固定step500，pipeline_exit_status.json的exit_code=0，Dev50及Final200任务完整且原协议核验通过。出现失败时不要重跑run，先执行第4段回传。

## 4. 完成或失败后Git回传

确认监督器已退出后执行；仍运行时监督锁会拒绝回传。仅读轻量证据，不上传权重；含准备阶段自动补回的原始CPU收据/旧失败现场、训练快照与核验结果、固定checkpoint来源哈希、导出/服务身份、smoke/Dev/Final轨迹及原A500可用对照。

```bash
cd /root/autodl-tmp/work/my_shopping_agent
bash <<'BASH'
set -euo pipefail
ROOT=/root/autodl-tmp/work/my_shopping_agent
PY="$ROOT/upstream/shopping-grpo-longhorizon/.venv/bin/python"
IFS= read -r OUT < "$ROOT/outputs/c500_eval_active.txt"
CODE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"])' "$OUT/posttrain.json")
bash "$CODE/tools/c500_eval_commands.sh" return
BASH
```

发回RETURN_BRANCH及RETURN_COMMIT。本地22项CPU测试与Bash语法通过；真实云端准备、导出和评测尚待执行，测试中的模拟分数不是C成绩。比较目标为A500 Dev27/50、Final137/200，B500 Dev28/50、Final141/200；Final已被多次观察，不称未见测试。
