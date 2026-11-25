# 固定C800独立Dev50和Final200完整命令

2026-11-25。用户明确授权。固定已验收step800，一次导出→smoke3→原独立Dev50→原Final200。Dev分数不阻断Final；执行/身份错误停止。无模型换点、重训、自动续800或自动关机。

## 1. 无卡准备

稀疏克隆只取工具和冻结包，不下载大历史归档。复核完整1—800、γ4实际权重及保留的C650前缀；原C650记录用发布过的固定canonical哈希和原weights SHA核对。校验800实物CRC和LoRA16/32/world_size1；会读约10GiB文件并计算源哈希，请等待。只运行新评测CPU检查，不加载模型或启动服务。

```bash
cd /root/autodl-tmp/work/my_shopping_agent
bash <<'BASH'
set -euo pipefail
ROOT=/root/autodl-tmp/work/my_shopping_agent
cd "$ROOT"
mkdir -p "$ROOT/outputs"
LOG="$ROOT/outputs/c800_eval_prepare_$(date -u +%Y%m%dT%H%M%SZ)_$RANDOM.log"
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; printf "PREPARE_EXIT_CODE=%s\nLOG=%s\n" "$rc" "$LOG"; if [[ "$rc" != 0 ]]; then echo "准备失败，保留日志，不要启动有卡run。"; fi' EXIT
[[ ! -e "$ROOT/outputs/c800_eval_active.txt" ]] || { echo '已准备过C800评测，请使用status/return。'; exit 1; }
CODE=$(mktemp -d "$ROOT/outputs/c800-eval-code-XXXXXX")
printf 'CODE=%s\n' "$CODE"
git -c http.version=HTTP/1.1 clone --depth 1 --single-branch --filter=blob:none --sparse --branch codex/c800-evaluation "$(git remote get-url origin)" "$CODE"
git -C "$CODE" sparse-checkout set tools experiments/group_weight_c500_v1 experiments/group_weight_c650_v1 experiments/group_weight_c800_v1 experiments/group_weight_b500_v1
export OMP_NUM_THREADS=1
bash "$CODE/tools/c800_eval_commands.sh" prepare
BASH
```

必须出现C800_EVAL_PREPARED=true和PREPARE_EXIT_CODE=0才继续；失败保留LOG，不执行第2段，不重复prepare覆盖目录。

## 2. 原有卡实例启动

原高内存环境：导出至少110GiB CPU配额，服务GPU空闲至少90GiB，磁盘额外导出空间约26GiB（程序按实物大小检查）。

```bash
cd /root/autodl-tmp/work/my_shopping_agent
bash <<'BASH'
set -euo pipefail
ROOT=/root/autodl-tmp/work/my_shopping_agent
PY="$ROOT/upstream/shopping-grpo-longhorizon/.venv/bin/python"
IFS= read -r OUT < "$ROOT/outputs/c800_eval_active.txt"
CODE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"])' "$OUT/posttrain.json")
bash "$CODE/tools/c800_eval_commands.sh" run
BASH
```

后台自动导出并依次跑三套流程，无需手动再启动Final或重复导出；不要重复run。PID本身不等于导出或评测成功。

## 3. 查询状态（可重复）

```bash
cd /root/autodl-tmp/work/my_shopping_agent
bash <<'BASH'
set -euo pipefail
ROOT=/root/autodl-tmp/work/my_shopping_agent
PY="$ROOT/upstream/shopping-grpo-longhorizon/.venv/bin/python"
IFS= read -r OUT < "$ROOT/outputs/c800_eval_active.txt"
CODE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"])' "$OUT/posttrain.json")
bash "$CODE/tools/c800_eval_commands.sh" status
BASH
```

完成标准为evaluation_complete.json的passed=true、fixed_step=800以及pipeline_exit_status.json的exit_code=0，且Dev/Final完成收据存在。实际GPU执行/恢复点实物核验仍待用户运行；CPU模拟不代表C800独立成绩。

## 4. 完成或报错停止后Git回传

当前入口不自动关机。确认评测监督器已退出后，可手动关机并无卡启动回传；仍运行时回传锁会拒绝。上传日志/配置/收据/原协议轨迹及训练快照，权重不上传。

```bash
cd /root/autodl-tmp/work/my_shopping_agent
bash <<'BASH'
set -euo pipefail
ROOT=/root/autodl-tmp/work/my_shopping_agent
PY="$ROOT/upstream/shopping-grpo-longhorizon/.venv/bin/python"
IFS= read -r OUT < "$ROOT/outputs/c800_eval_active.txt"
CODE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"])' "$OUT/posttrain.json")
bash "$CODE/tools/c800_eval_commands.sh" return
BASH
```

发回RETURN_BRANCH和RETURN_COMMIT。对照固定同800步：A800 Dev29/50、Final143/200，B800 Dev28/50、Final142/200；原C500为25/50、138/200。Final已多次观察，不称未见测试，不由Dev分数换成750。

C800保持连续跳过阈值10；B800尾段调为50的历史差异保留披露。C650独立Dev24/50、Final136/200的既有结论保持。
