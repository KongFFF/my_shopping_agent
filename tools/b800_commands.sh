#!/usr/bin/env bash
set -euo pipefail
set -C
ROOT=/root/autodl-tmp/work/my_shopping_agent
PY="$ROOT/upstream/shopping-grpo-longhorizon/.venv/bin/python"
ACTIVE="$ROOT/outputs/b800_active.txt"
PHASE="${1:?Use prepare, run, status, follow, or return}"
cd "$ROOT"
if [[ "$PHASE" == prepare ]]; then
  [[ ! -e "$ACTIVE" ]] || { printf 'STOP: B800 already prepared; use status.\n' >&2; exit 1; }
  CODE=$(mktemp -d "$ROOT/outputs/b800-code-XXXXXX")
  git -c http.version=HTTP/1.1 clone --depth 1 --single-branch --branch codex/b800-training "$(git remote get-url origin)" "$CODE"
  IFS= read -r PARENT < "$ROOT/outputs/b650_active.txt"
  OUT="$ROOT/outputs/runs/b800_$(date -u +%Y%m%dT%H%M%SZ)_${CODE##*-}"
  "$PY" -u "$CODE/tools/b800_training.py" --parent "$PARENT" --out "$OUT"
  printf '%s\n' "$OUT" > "$ACTIVE"
else
  IFS= read -r RUN < "$ACTIVE"
  [[ "$RUN" == "$ROOT/outputs/runs/b800_"* ]] || { printf 'STOP: unexpected B800 path.\n' >&2; exit 1; }
  CODE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"])' "$RUN/run.json")
  case "$PHASE" in
    run)
      [[ ! -e "$RUN/supervisor.log" ]] || { printf 'STOP: already attempted; use status/return.\n' >&2; exit 1; }
      nohup "$PY" -u "$CODE/tools/b500_run.py" run --run-dir "$RUN" > "$RUN/supervisor.log" 2>&1 < /dev/null &
      PID=$!; printf '%s\n' "$PID" > "$RUN/supervisor.pid"
      printf 'SUPERVISOR_PID=%s\nRUN=%s\nUse status; PID alone is not verified training.\n' "$PID" "$RUN"
      ;;
    status)
      "$PY" "$CODE/tools/b500_run.py" status --run-dir "$RUN"
      for LOG in supervisor.log train_to_800.log; do if [[ -f "$RUN/$LOG" ]]; then printf '\nLOG=%s\n' "$RUN/$LOG"; tail -n 14 "$RUN/$LOG"; fi; done
      ;;
    follow) tail -n 35 -F "$RUN/train_to_800.log" ;;
    return) "$PY" "$CODE/tools/b500_run.py" return --run-dir "$RUN" ;;
    *) printf 'Unknown phase: %s\n' "$PHASE" >&2; exit 2 ;;
  esac
fi
