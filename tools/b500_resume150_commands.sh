#!/usr/bin/env bash
# Always execute with bash, never source into the interactive shell.
set -euo pipefail
ROOT=/root/autodl-tmp/work/my_shopping_agent
PY="$ROOT/upstream/shopping-grpo-longhorizon/.venv/bin/python"
POINTER="$ROOT/outputs/b500_resume150_active.txt"
OLD="$ROOT/outputs/runs/b500_group_weight_v1_recovery_20261118T075525Z_8e06cb"
PHASE="${1:?Use prepare, run, status, follow, or return}"
cd "$ROOT"
if [[ "$PHASE" == prepare ]]; then
  [[ ! -e "$POINTER" ]] || { printf 'STOP: already prepared; use status.\n' >&2; exit 1; }
  CODE=$(mktemp -d "$ROOT/outputs/b500-resume150-code-XXXXXX")
  git -c http.version=HTTP/1.1 clone --depth 1 --single-branch --branch codex/b500-resume150 "$(git remote get-url origin)" "$CODE"
  "$PY" -u "$CODE/tools/prepare_b500_resume150.py" --old-run "$OLD" --active-pointer "$POINTER"
else
  IFS= read -r RUN < "$POINTER"
  [[ "$RUN" == "$ROOT/outputs/runs/b500_resume150_"* ]] || { printf 'STOP: unexpected run path.\n' >&2; exit 1; }
  CODE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"])' "$RUN/run.json")
  case "$PHASE" in
    run)
      "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["resume_step"]==150 and not (r["training_started"] or r.get("launch_attempted")), "Attempted run: preserve it and use status/return"' "$RUN/run.json"
      [[ ! -e "$RUN/supervisor.log" ]] || { printf 'STOP: supervisor.log already exists.\n' >&2; exit 1; }
      nohup "$PY" -u "$CODE/tools/b500_run.py" run --run-dir "$RUN" > "$RUN/supervisor.log" 2>&1 < /dev/null &
      PID=$!; printf '%s\n' "$PID" > "$RUN/supervisor.pid"
      printf 'SUPERVISOR_PID=%s\nRUN=%s\nUse status to confirm actual progress.\n' "$PID" "$RUN"
      ;;
    status)
      "$PY" "$CODE/tools/b500_run.py" status --run-dir "$RUN"
      for LOG in supervisor.log train_to_350.log train_to_500.log; do
        if [[ -f "$RUN/$LOG" ]]; then printf '\nLOG=%s\n' "$RUN/$LOG"; tail -n 12 "$RUN/$LOG"; fi
      done
      ;;
    follow)
      tail -n 30 -F "$RUN/train_to_350.log" "$RUN/train_to_500.log"
      ;;
    return)
      "$PY" "$CODE/tools/b500_run.py" return --run-dir "$RUN"
      ;;
    *) printf 'Unknown phase: %s\n' "$PHASE" >&2; exit 2 ;;
  esac
fi
