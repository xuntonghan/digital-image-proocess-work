#!/usr/bin/env bash
# User-authorized, bounded follow-up, followed by one frozen full evaluation.
set -euo pipefail
ROOT="${1:-/root/autodl-tmp/sony1000_20261025}"
PYTHON="${2:-/root/autodl-tmp/cv-homework/.venv/bin/python}"
export PYTHONPATH="$ROOT/project/src"
cd "$ROOT/project"
exec 9>"$ROOT/logs/orchestrator.lock"
flock -n 9 || { echo 'Sony orchestrator already active' >&2; exit 1; }
"$PYTHON" - "$ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
proof=json.loads((root/'logs/final_execution_equivalence.json').read_text())
assert proof['equivalent'] and proof['png_sha256_equal'] and proof['workers']==4
assert proof['short_run_ratio']>=1.15
assert proof['executor_sha256']==hashlib.sha256((root/'project/cloud/sony_deadline.py').read_bytes()).hexdigest()
PY
printf 'DEADLINE_BOUNDED_GTMEAN_RESEARCH\n' > "$ROOT/logs/status.txt"
if ! "$PYTHON" -u cloud/sony_deadline.py --root "$ROOT" --phase research --workers 4; then
  printf 'Follow-up failed or exceeded its bound. Preserve evidence and retain a validated selection.\n' >&2
  "$PYTHON" - "$ROOT" <<'PY'
import json,sys,datetime
from pathlib import Path
root=Path(sys.argv[1]); p=root/'research/gtmean_deadline/failure.json'
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({'failed':True,'time':datetime.datetime.now().astimezone().isoformat(),'action':'No further training. Continue only with the existing validated selection; inspect deadline log.'},indent=2)+'\n')
PY
fi
printf 'PARALLEL_FROZEN_FINAL_1000_IMAGE_EVALUATION\n' > "$ROOT/logs/status.txt"
"$PYTHON" -u cloud/sony_deadline.py --root "$ROOT" --phase finalize --workers 4
"$PYTHON" cloud/verify_sony1000.py --directory "$ROOT/submission1000_native" > "$ROOT/logs/independent_statistics.json"
printf 'COMPLETE_PENDING_INDEPENDENT_BACKUP_AND_REPORT\n' > "$ROOT/logs/status.txt"
