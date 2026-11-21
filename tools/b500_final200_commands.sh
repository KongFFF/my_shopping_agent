#!/usr/bin/env bash
set -euo pipefail
set -C
ROOT=/root/autodl-tmp/work/my_shopping_agent
PY="$ROOT/upstream/shopping-grpo-longhorizon/.venv/bin/python"
ACTIVE="$ROOT/outputs/b500_final200_active.txt"
PHASE="${1:?Use prepare, run, status, follow, or return}"
cd "$ROOT"
if [[ "$PHASE" == prepare ]]; then
  [[ ! -e "$ACTIVE" ]] || { printf 'STOP: Final preparation already exists; use status.\n' >&2; exit 1; }
  CODE=$(mktemp -d "$ROOT/outputs/b500-final-code-XXXXXX")
  git -c http.version=HTTP/1.1 clone --depth 1 --single-branch --branch codex/b500-final200 "$(git remote get-url origin)" "$CODE"
  IFS= read -r DEV < "$ROOT/outputs/b500_dev50_active.txt"
  OUT="$ROOT/outputs/runs/b500_final200_$(date -u +%Y%m%dT%H%M%SZ)_${CODE##*-}"
  "$PY" -u "$CODE/tools/b500_final200.py" prepare --dev "$DEV" --out "$OUT"
  printf '%s\n' "$OUT" > "$ACTIVE"
else
  IFS= read -r OUT < "$ACTIVE"
  [[ "$OUT" == "$ROOT/outputs/runs/b500_final200_"* ]] || { printf 'STOP: unexpected path.\n' >&2; exit 1; }
  CODE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"])' "$OUT/posttrain.json")
  case "$PHASE" in
    run)
      [[ ! -e "$OUT/supervisor.log" ]] || { printf 'STOP: already attempted; use status/return.\n' >&2; exit 1; }
      nohup "$PY" -u "$CODE/tools/b500_final200.py" run --out "$OUT" > "$OUT/supervisor.log" 2>&1 < /dev/null &
      PID=$!; printf '%s\n' "$PID" > "$OUT/supervisor.pid"
      printf 'SUPERVISOR_PID=%s\nOUT=%s\nUse status to verify actual completion.\n' "$PID" "$OUT"
      ;;
    status)
      "$PY" "$CODE/tools/b500_final200.py" status --out "$OUT"
      if [[ -f "$OUT/supervisor.log" ]]; then tail -n 25 "$OUT/supervisor.log"; fi
      ;;
    follow) tail -n 35 -F "$OUT/supervisor.log" ;;
    return) "$PY" "$CODE/tools/b500_final200.py" return --out "$OUT" ;;
    *) printf 'Unknown phase: %s\n' "$PHASE" >&2; exit 2 ;;
  esac
fi
