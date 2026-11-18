#!/usr/bin/env bash
# Execute with bash, never source. All strict-mode failures end only this child shell.
set -euo pipefail
ROOT=/root/autodl-tmp/work/my_shopping_agent
PROJECT="$ROOT/upstream/shopping-grpo-longhorizon"
RUN="$ROOT/outputs/runs/b500_group_weight_v1"
ACTIVE="$ROOT/outputs/b500_active_run.txt"
if [[ -f "$ACTIVE" ]]; then
  IFS= read -r RUN < "$ACTIVE"
  [[ "$RUN" == "$ROOT/outputs/runs/"* ]] || { printf 'STOP: unexpected active-run path.\n' >&2; exit 1; }
fi
PY="$PROJECT/.venv/bin/python"
PHASE="${1:?Use recover, run, status, or return}"
cd "$ROOT"
if [[ "$PHASE" == recover ]]; then
  [[ ! -e "$ACTIVE" ]] || { printf 'STOP: recovery already prepared; use status/return.\n' >&2; exit 1; }
  mkdir -p "$ROOT/outputs"
  CODE=$(mktemp -d "$ROOT/outputs/b500-guard-code-XXXXXX")
  git -c http.version=HTTP/1.1 clone --depth 1 --single-branch --branch codex/b500-bypass-guard-fix "$(git remote get-url origin)" "$CODE"
  "$PY" "$CODE/tools/recover_b500_guard.py" --old-run "$RUN" --active-pointer "$ACTIVE"
elif [[ "$PHASE" == prepare ]]; then
  printf 'STOP: this repair helper is for recover, run, status, return; do not repeat prepare.\n' >&2
  exit 1
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
