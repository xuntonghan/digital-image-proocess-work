#!/usr/bin/env bash
# One Sony-only experiment. No deletion, paid expansion or automatic shutdown.
set -euo pipefail
ROOT="${1:-/root/autodl-tmp/sony1000_20261025}"
PYTHON="${2:-/root/autodl-tmp/cv-homework/.venv/bin/python}"
export PATH="/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/project/src"
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/logs/orchestrator.lock"
flock -n 9 || { echo 'Sony orchestrator already active' >&2; exit 1; }
trap 'printf "FAILED line=%s\n" "$LINENO" > "$ROOT/logs/status.txt"' ERR
cd "$ROOT/project"
printf 'WAITING_FOR_SONY_DATA\n' > "$ROOT/logs/status.txt"
for _ in $(seq 1 1440); do
  [[ -s "$ROOT/logs/sony_data_verified.json" ]] && break
  sleep 5
done
[[ -s "$ROOT/logs/sony_data_verified.json" ]]
sleep 1
"$PYTHON" -m dip_course.cli.audit_dataset --data-root "$ROOT/data/SIED_dataset" \
  --manifest manifests/source/sony_all.txt --output-json "$ROOT/logs/audit.json"
printf 'RUNNING_3_TRAIN_IMAGE_SMOKE\n' > "$ROOT/logs/status.txt"
"$PYTHON" -u cloud/sony1000_pipeline.py --root "$ROOT" --phase prepare --limit-new 3
printf 'PREPARING_913_TRAIN_VALIDATION_IMAGES\n' > "$ROOT/logs/status.txt"
"$PYTHON" -u cloud/sony1000_pipeline.py --root "$ROOT" --phase prepare
printf 'BOUNDED_SONY_COLOR_TRAINING\n' > "$ROOT/logs/status.txt"
"$PYTHON" -u cloud/sony1000_pipeline.py --root "$ROOT" --phase research
printf 'FROZEN_FINAL_1000_IMAGE_EVALUATION\n' > "$ROOT/logs/status.txt"
"$PYTHON" -u cloud/sony1000_pipeline.py --root "$ROOT" --phase finalize
"$PYTHON" cloud/verify_sony1000.py --directory "$ROOT/submission1000_native" \
  > "$ROOT/logs/independent_statistics.json"
printf 'COMPLETE_PENDING_INDEPENDENT_BACKUP_AND_REPORT\n' > "$ROOT/logs/status.txt"
