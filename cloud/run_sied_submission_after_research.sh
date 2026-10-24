#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/root/autodl-tmp/sied30_eval1000}"
PYTHON=/root/autodl-tmp/cv-homework/.venv/bin/python
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/logs/submission_wrapper.lock"
flock -n 9 || exit 1
for attempt in $(seq 1 720); do
  if [[ -s "$ROOT/research/iccv_pairwise/result.json" ]]; then
    break
  fi
  if ! pgrep -f '[r]un_sied_iccv_after_research.sh|[r]esearch_sied_iccv.py' >/dev/null; then
    echo 'PRECEDING_RESEARCH_FAILED: save evidence and request review, do not change the candidate set' >&2
    exit 1
  fi
  sleep 10
done
test -s "$ROOT/research/iccv_pairwise/result.json"
cd "$ROOT/project"
"$PYTHON" cloud/finalize_sied_submission1000.py --root "$ROOT" --limit-new 3
"$PYTHON" cloud/finalize_sied_submission1000.py --root "$ROOT"
date -Is
echo SUBMISSION1000_COMPLETE_AWAITING_VERIFIED_BACKUP
