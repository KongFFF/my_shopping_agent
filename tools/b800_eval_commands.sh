#!/usr/bin/env bash
set -euo pipefail
set -C
ROOT=/root/autodl-tmp/work/my_shopping_agent
PY="$ROOT/upstream/shopping-grpo-longhorizon/.venv/bin/python"
ACTIVE="$ROOT/outputs/b800_eval_active.txt"
PHASE="${1:?Use prepare, run, status, follow, or return}"
cd "$ROOT"
if [[ "$PHASE" == prepare ]]; then
  [[ ! -e "$ACTIVE" ]] || { printf 'STOP: B800 evaluation preparation exists; use status.\n' >&2; exit 1; }
  CODE=$(mktemp -d "$ROOT/outputs/b800-eval-code-XXXXXX")
  git -c http.version=HTTP/1.1 clone --depth 1 --single-branch --branch codex/b800-evaluation "$(git remote get-url origin)" "$CODE"
  IFS= read -r TRAINING < "$ROOT/outputs/b800_finish750_active.txt"
  OUT="$ROOT/outputs/runs/b800_evaluation_$(date -u +%Y%m%dT%H%M%SZ)_${CODE##*-}"
  "$PY" -u "$CODE/tools/b800_evaluation.py" prepare --training "$TRAINING" --out "$OUT"
  printf '%s\n' "$OUT" > "$ACTIVE"
else
  IFS= read -r OUT < "$ACTIVE"
  [[ "$OUT" == "$ROOT/outputs/runs/b800_evaluation_"* ]] || { printf 'STOP: unexpected evaluation path.\n' >&2; exit 1; }
  CODE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"])' "$OUT/posttrain.json")
  case "$PHASE" in
    run)
      [[ ! -e "$OUT/supervisor.log" ]] || { printf 'STOP: already attempted; use status/return.\n' >&2; exit 1; }
      nohup "$PY" -u "$CODE/tools/b800_evaluation.py" run --out "$OUT" > "$OUT/supervisor.log" 2>&1 < /dev/null &
      PID=$!; printf '%s\n' "$PID" > "$OUT/supervisor.pid"
      printf 'SUPERVISOR_PID=%s\nOUT=%s\nUse status; PID does not prove successful evaluation.\n' "$PID" "$OUT"
      ;;
    status)
      "$PY" "$CODE/tools/b800_evaluation.py" status --out "$OUT"
      if [[ -f "$OUT/supervisor.log" ]]; then tail -n 25 "$OUT/supervisor.log"; fi
      ;;
    follow) tail -n 35 -F "$OUT/supervisor.log" ;;
    return) "$PY" "$CODE/tools/b800_evaluation.py" return --out "$OUT" ;;
    *) printf 'Unknown phase: %s\n' "$PHASE" >&2; exit 2 ;;
  esac
fi
