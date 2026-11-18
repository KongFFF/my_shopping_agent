#!/usr/bin/env bash
# Execute with bash, never source. All strict-mode failures end only this child shell.
set -euo pipefail
ROOT=/root/autodl-tmp/work/my_shopping_agent
PROJECT="$ROOT/upstream/shopping-grpo-longhorizon"
RUN="$ROOT/outputs/runs/b500_group_weight_v1"
PY="$PROJECT/.venv/bin/python"
PHASE="${1:?Use prepare, run, status, or return}"
cd "$ROOT"
if [[ "$PHASE" == prepare ]]; then
  if [[ -e "$RUN" ]]; then
    printf 'STOP: run directory exists; preserve it and use status/return: %s\n' "$RUN" >&2
    exit 1
  fi
  mkdir -p "$ROOT/outputs"
  CODE=$(mktemp -d "$ROOT/outputs/b500-live-code-XXXXXX")
  git -c http.version=HTTP/1.1 clone --single-branch --branch codex/b500-live-run-v1 "$(git remote get-url origin)" "$CODE"
  "$PY" "$CODE/tools/b500_run.py" prepare --run-dir "$RUN"
else
  CODE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"])' "$RUN/run.json")
  case "$PHASE" in
    run)
      "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert not (r["training_started"] or r.get("launch_attempted")), "Preserve attempted run; use status/return"' "$RUN/run.json"
      if [[ -e "$RUN/supervisor.log" ]]; then
        printf 'STOP: supervisor.log exists; do not overwrite it. Use status/return.\n' >&2
        exit 1
      fi
      nohup "$PY" -u "$CODE/tools/b500_run.py" run --run-dir "$RUN" > "$RUN/supervisor.log" 2>&1 < /dev/null &
      PID=$!
      printf '%s\n' "$PID" > "$RUN/supervisor.pid"
      printf 'SUPERVISOR_PID=%s\nLOG=%s/supervisor.log\n' "$PID" "$RUN"
      printf 'Use status to inspect actual startup; a PID alone does not prove training started.\n'
      ;;
    status)
      "$PY" "$CODE/tools/b500_run.py" status --run-dir "$RUN"
      if [[ -f "$RUN/supervisor.pid" ]] && kill -0 "$(cat "$RUN/supervisor.pid")" 2>/dev/null; then
        printf 'SUPERVISOR_PROCESS=alive (confirm progress in logs)\n'
      else
        printf 'SUPERVISOR_PROCESS=not_alive\n'
      fi
      if [[ -f "$RUN/supervisor.log" ]]; then tail -n 18 "$RUN/supervisor.log"; fi
      ;;
    return)
      "$PY" "$CODE/tools/b500_run.py" return --run-dir "$RUN"
      ;;
    *) printf 'Unknown phase: %s\n' "$PHASE" >&2; exit 2 ;;
  esac
fi
