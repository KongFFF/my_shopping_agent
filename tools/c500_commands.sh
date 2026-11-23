#!/usr/bin/env bash
# Run as a child shell; strict mode never alters the interactive parent shell.
set -euo pipefail
ROOT=/root/autodl-tmp/work/my_shopping_agent
ACTIVE="$ROOT/outputs/c500_active_run.txt"
PHASE="${1:?Use prepare, run, status or return}"
PROJECT="$ROOT/upstream/shopping-grpo-longhorizon"
PY="$PROJECT/.venv/bin/python"
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ "$PHASE" == prepare ]]; then
  [[ ! -e "$ACTIVE" ]] || { echo 'C500 pointer exists; use status/return. Preserve prior run.' >&2; exit 1; }
  mkdir -p "$CODE/outputs" "$ROOT/outputs/runs"
  CUDA_VISIBLE_DEVICES='' "$PY" "$CODE/tools/run_c500_cpu_acceptance.py" --asset-project "$PROJECT" --output "$CODE/outputs/c500_native_cpu.json"
  RUN="$ROOT/outputs/runs/c500_gamma4_$(date -u +%Y%m%dT%H%M%SZ)_$RANDOM"
  CUDA_VISIBLE_DEVICES='' "$PY" "$CODE/tools/c500_run.py" prepare --asset-project "$PROJECT" --run-dir "$RUN"
  # Exclusive creation: never replace an existing active pointer.
  (set -o noclobber; printf '%s\n' "$RUN" > "$ACTIVE")
  printf 'C500_PREPARED=true\nRUN=%s\nNo training started.\n' "$RUN"
  exit 0
fi
[[ -s "$ACTIVE" ]] || { echo 'No prepared C500 pointer.' >&2; exit 1; }
IFS= read -r RUN < "$ACTIVE"
[[ "$RUN" == "$ROOT/outputs/runs/c500_gamma4_"* && "$RUN" != *'/../'* ]] || { echo 'Unexpected C500 run path.' >&2; exit 1; }
CODE=$("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["experiment"]=="C500" and r["gamma"]==4; print(r["repo"])' "$RUN/run.json")
case "$PHASE" in
  run)
    "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert not (r["training_started"] or r.get("launch_attempted")), "Already attempted: preserve evidence and use status/return"' "$RUN/run.json"
    [[ ! -e "$RUN/supervisor.log" ]] || { echo 'Log exists; refusing overwrite.' >&2; exit 1; }
    nohup "$PY" -u "$CODE/tools/c500_run.py" run --run-dir "$RUN" > "$RUN/supervisor.log" 2>&1 < /dev/null &
    PID=$!; printf '%s\n' "$PID" > "$RUN/supervisor.pid"
    printf 'SUPERVISOR_PID=%s\nLOG=%s/supervisor.log\nA PID alone does not prove startup; inspect status.\n' "$PID" "$RUN"
    ;;
  status)
    "$PY" "$CODE/tools/c500_run.py" status --run-dir "$RUN"
    [[ ! -f "$RUN/supervisor.log" ]] || tail -n 20 "$RUN/supervisor.log"
    ;;
  return) "$PY" "$CODE/tools/c500_run.py" return --run-dir "$RUN" ;;
  *) echo 'Unknown phase' >&2; exit 2 ;;
esac
