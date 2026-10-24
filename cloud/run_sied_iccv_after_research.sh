#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/root/autodl-tmp/sied30_eval1000}"
PYTHON=/root/autodl-tmp/cv-homework/.venv/bin/python
mkdir -p "$ROOT/research" "$ROOT/logs"
exec 9>"$ROOT/logs/iccv_wrapper.lock"
flock -n 9 || exit 1
for attempt in $(seq 1 2880); do
  if [[ -s "$ROOT/research/color_lut_tlc/result.json" ]]; then
    break
  fi
  if ! pgrep -f '[r]un_sied_research_after_eval.sh|[r]esearch_sied_color.py' >/dev/null; then
    echo 'PRECEDING_RESEARCH_NOT_RUNNING_OR_FAILED: not starting a new job' >&2
    exit 1
  fi
  sleep 10
done
test -s "$ROOT/research/color_lut_tlc/result.json"
cd "$ROOT/project"
"$PYTHON" cloud/research_sied_iccv.py --root "$ROOT" --steps 1000
date -Is
echo ICCV_RESEARCH_COMPLETE_AWAITING_VERIFIED_BACKUP
