#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/root/autodl-tmp/sied30_eval1000}"
PYTHON=/root/autodl-tmp/cv-homework/.venv/bin/python
mkdir -p "$ROOT/research" "$ROOT/logs"
exec 9>"$ROOT/logs/research_wrapper.lock"
flock -n 9 || exit 1
for attempt in $(seq 1 2880); do
  if [[ -s "$ROOT/evaluation1000_native/verification.json" ]] && \
     grep -q 'EVALUATION_COMPLETE_AWAITING_LOCAL_VERIFIED_BACKUP' "$ROOT/logs/evaluation_status.txt"; then
    break
  fi
  if tail -n 1 "$ROOT/logs/evaluation_status.txt" | grep -q 'FAILED\|TIMEOUT'; then
    echo 'BASELINE_FAILED: research will not start' >&2
    exit 1
  fi
  sleep 10
done
test -s "$ROOT/evaluation1000_native/verification.json"
cd "$ROOT/project"
"$PYTHON" cloud/research_sied_color.py --root "$ROOT" --steps 1500
date -Is
echo RESEARCH_COMPLETE_AWAITING_VERIFIED_BACKUP
