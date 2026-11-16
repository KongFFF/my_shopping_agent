#!/usr/bin/env bash
# Invoke with bash, never source: strict options belong to a child shell only.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'Please run: bash tools/decision_probe_commands.sh <prepare|sample|return>'
  return 2
fi
set -uo pipefail
PHASE="${1:-}"
case "$PHASE" in prepare|sample|return) ;; *) printf '%s\n' 'Usage: bash tools/decision_probe_commands.sh <prepare|sample|return>'; exit 2 ;; esac
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd -- "$ROOT" || exit 1
STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p outputs/decision_probe_v1/command_logs || exit 1
LOG="$ROOT/outputs/decision_probe_v1/command_logs/${STAMP}_${PHASE}.log"
printf 'PHASE=%s\nLOG=%s\n' "$PHASE" "$LOG"
(
  set -Eeuo pipefail
  trap 'status=$?; printf "FAILED: line %s, exit %s\n" "$LINENO" "$status" >&2; exit "$status"' ERR
  PY=upstream/shopping-grpo-longhorizon/.venv/bin/python
  BUNDLE=outputs/decision_probe_v1/runtime
  MODEL=upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/export/step800_v1/merged
  RUN=outputs/decision_probe_v1/run800_v1
  export PYTHONUNBUFFERED=1
  case "$PHASE" in
    prepare)
      "$PY" -m unittest discover -s tools -p test_decision_probe.py -v
      "$PY" tools/prepare_decision_probe.py --bundle "$BUNDLE"
      if [[ -e "$RUN/run.json" ]]; then
        # Recheck existing preparation without deleting or resetting any run.
        "$PY" - "$RUN" "$BUNDLE" "$MODEL" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'tools')
from run_decision_probe import checked_run, file_hash
run, bundle, model = map(lambda p: Path(p).resolve(), sys.argv[1:])
cfg = checked_run(run)
if (run/'budget.json').exists():
    raise ValueError('Sampling was already started. Preserve this run; use return, not prepare/restart.')
if Path(cfg['bundle']).resolve() != bundle or Path(cfg['model_dir']).resolve() != model:
    raise ValueError('Existing run identifies another bundle or model')
for name, expected in cfg['model_hashes'].items():
    if file_hash(model/name) != expected:
        raise ValueError('Prepared model file changed: '+name)
print('PASS: existing CPU preparation reverified; no run files reset')
PY
      else
        "$PY" tools/run_decision_probe.py init --bundle "$BUNDLE" --model-dir "$MODEL" --run-dir "$RUN"
      fi
      git rev-parse --verify 'refs/remotes/origin/codex/decision-probe800-v1^{commit}' > "$RUN/code_ref.txt"
      CPU_DIR="outputs/decision_probe_v1/cpu_check_cloud_${STAMP}"
      "$PY" tools/run_decision_probe.py cpu-check --bundle "$BUNDLE" --run-dir "$CPU_DIR"
      cp -- "$CPU_DIR/cpu_check.json" "$RUN/cpu_check.json"
      ;;
    sample)
      "$PY" tools/run_decision_probe.py pipeline --run-dir "$RUN"
      ;;
    return)
      "$PY" tools/export_decision_probe.py --run-dir "$RUN" --code-ref "$(cat "$RUN/code_ref.txt")" --git-upload
      ;;
  esac
) 2>&1 | tee "$LOG"
STATUSES=("${PIPESTATUS[@]}")
STATUS="${STATUSES[0]}"
if [[ "${STATUSES[1]}" -ne 0 ]]; then
  printf 'Logging failed: tee exit %s\n' "${STATUSES[1]}" >&2
  [[ "$STATUS" -ne 0 ]] || STATUS="${STATUSES[1]}"
fi
printf '\nPHASE=%s EXIT_CODE=%s\nLOG=%s\n' "$PHASE" "$STATUS" "$LOG"
if [[ "$STATUS" -ne 0 ]]; then
  printf '%s\n' 'Phase stopped. Preserve the log and run directory; do not start GPU sampling after a failed prepare.'
fi
exit "$STATUS"
