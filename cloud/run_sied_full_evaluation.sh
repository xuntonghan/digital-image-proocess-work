#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/root/autodl-tmp/sied30_eval1000}"
PROJECT="$ROOT/project"
PYTHON=/root/autodl-tmp/cv-homework/.venv/bin/python
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/logs/evaluation_wrapper.lock"
flock -n 9 || { echo 'Evaluation wrapper already running' >&2; exit 1; }
status() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$ROOT/logs/evaluation_status.txt"; }
trap 'status "FAILED exit=$? line=$LINENO"' ERR
status WAITING_FOR_VERIFIED_MIGRATION
for attempt in $(seq 1 720); do
  [[ -e "$ROOT/logs/MIGRATION_VERIFIED" ]] && break
  sleep 10
done
[[ -e "$ROOT/logs/MIGRATION_VERIFIED" ]] || { status MIGRATION_TIMEOUT; exit 1; }
cd "$PROJECT"
"$PYTHON" -m pip install --no-deps --no-build-isolation -e .
"$PYTHON" -m pytest -q tests/test_sied1000_summary.py tests/test_infer_official_sied.py \
  tests/test_metrics.py tests/test_train_sied_restorer_native.py
status TESTS_PASSED
"$PYTHON" -m dip_course.cli.audit_dataset --data-root "$ROOT/data/SIED_dataset" \
  --manifest manifests/source/canon_all.txt --output-json "$ROOT/logs/data_audit.json"
status AUDIT_PASSED
exec 8>"$ROOT/logs/gpu_work.lock"
flock 8
"$PYTHON" cloud/evaluate_sied1000.py --root "$ROOT" --cache-base-fp32 --cache-color-pairs --limit-new 3
status PILOT_PASSED_RESUMING_FULL_EVALUATION
"$PYTHON" cloud/evaluate_sied1000.py --root "$ROOT" --cache-base-fp32 --cache-color-pairs
status EVALUATION_COMPLETE_AWAITING_LOCAL_VERIFIED_BACKUP
# No automatic shutdown here: first back up the only new rows and PNGs locally.
