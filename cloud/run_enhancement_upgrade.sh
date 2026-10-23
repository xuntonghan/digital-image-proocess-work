#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/sied30}"
PROJECT="$ROOT/project"
PYTHON="${PYTHON:-/root/autodl-tmp/cv-homework/.venv/bin/python}"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"
STATUS="$LOGS/enhancement_upgrade_status.txt"
status() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$STATUS"; }
export PYTHON
cd "$PROJECT"
exec 9>"$LOGS/enhancement_upgrade_queue.lock"
flock -n 9 || { echo "Enhancement upgrade queue already exists" >&2; exit 1; }
touch "$ROOT/HOLD_FINALIZE"

status waiting_for_phase4_native_validation
for _attempt in $(seq 1 1440); do
  [[ -s "$ROOT/runs/canon_raw_guided_naf_phase4_val_native/summary.json" ]] && break
  sleep 30
done
if [[ ! -s "$ROOT/runs/canon_raw_guided_naf_phase4_val_native/summary.json" ]]; then
  status phase4_timeout_review_original_training_log
  exit 1
fi
status phase4_native_validation_complete
status phase5_context_started
bash cloud/run_sied30_phase5_context.sh "$ROOT" > "$LOGS/phase5_context.log" 2>&1
status phase5_context_complete_native_validation_pending_review

# Review all native metrics before the one final frozen-test evaluation.
# Geometry and compression are explicitly paused by the user's latest request.
