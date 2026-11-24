#!/usr/bin/env bash
set -euo pipefail
set -C
ROOT=/root/autodl-tmp/work/my_shopping_agent
PY="$ROOT/upstream/shopping-grpo-longhorizon/.venv/bin/python"
ACTIVE="$ROOT/outputs/c800_active.txt"
PHASE="${1:?Use prepare, run, status, follow, or return}"
cd "$ROOT"
if [[ "$PHASE" == prepare ]]; then
  [[ ! -e "$ACTIVE" ]] || { printf 'STOP: C800 already prepared; use status.\n' >&2; exit 1; }
  CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  IFS= read -r PARENT < "$ROOT/outputs/c650_active.txt"
  OUT="$ROOT/outputs/runs/c800_$(date -u +%Y%m%dT%H%M%SZ)_${CODE##*-}"
  "$PY" -u "$CODE/tools/c800_training.py" --parent "$PARENT" --out "$OUT"
  printf '%s\n' "$OUT" > "$ACTIVE"
else
  IFS= read -r RUN < "$ACTIVE"
  [[ "$RUN" == "$ROOT/outputs/runs/c800_"* ]] || { printf 'STOP: unexpected C800 path.\n' >&2; exit 1; }
  CODE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"])' "$RUN/run.json")
  case "$PHASE" in
    run)
      [[ ! -e "$RUN/supervisor.log" ]] || { printf 'STOP: already attempted; use status/return.\n' >&2; exit 1; }
      nohup "$PY" -u "$CODE/tools/c800_run.py" run --run-dir "$RUN" > "$RUN/supervisor.log" 2>&1 < /dev/null &
      PID=$!; printf '%s\n' "$PID" > "$RUN/supervisor.pid"
      printf 'SUPERVISOR_PID=%s\nRUN=%s\nUse status; PID alone is not verified training.\n' "$PID" "$RUN"
      ;;
    status)
      "$PY" "$CODE/tools/c800_run.py" status --run-dir "$RUN"
      for LOG in supervisor.log train_to_800.log; do if [[ -f "$RUN/$LOG" ]]; then printf '\nLOG=%s\n' "$RUN/$LOG"; tail -n 14 "$RUN/$LOG"; fi; done
      ;;
    auto-shutdown)
      LOG="$RUN/auto_shutdown_$(date -u +%Y%m%dT%H%M%SZ)_$RANDOM.log"
      nohup "$PY" -u "$CODE/tools/watch_c800_shutdown.py" --run-dir "$RUN" --grace-seconds 60 > "$LOG" 2>&1 < /dev/null &
      GUARD=$!
      sleep 2
      if ! kill -0 "$GUARD" 2>/dev/null; then cat "$LOG"; echo 'Guard not armed'; exit 1; fi
      "$PY" - "$RUN" "$GUARD" <<'PY'
import json,sys
from pathlib import Path
rows=[json.loads(x) for x in (Path(sys.argv[1])/'auto_shutdown_events.jsonl').read_text().splitlines()]
assert any(x['event']=='armed' and x.get('guard_pid')==int(sys.argv[2])for x in rows),'No armed receipt for this guard'
print('AUTO_SHUTDOWN_ARMED=true')
PY
      printf 'GUARD_PID=%s\nLOG=%s\n' "$GUARD" "$LOG"
      ;;
    cancel-shutdown) touch "$RUN/auto_shutdown.cancel"; echo 'Cancellation requested before shutdown call.' ;;
    follow) tail -n 35 -F "$RUN/train_to_800.log" ;;
    return) "$PY" "$CODE/tools/c800_run.py" return --run-dir "$RUN" ;;
    *) printf 'Unknown phase: %s\n' "$PHASE" >&2; exit 2 ;;
  esac
fi
