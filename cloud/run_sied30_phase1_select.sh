#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/sied30}"
PROJECT="$ROOT/project"
DATA_ROOT="$ROOT/data/SIED_dataset"
OFFICIAL_REPO="$ROOT/third_party/SIED-official"
OFFICIAL_WEIGHT="$ROOT/checkpoints/official/canon_0.0001.pt"
PYTHON="${PYTHON:-/root/autodl-tmp/cv-homework/.venv/bin/python}"
RUNS="$ROOT/runs"
PHASE1="$RUNS/canon_raw_guided_naf_phase1"
BEST_VAL="$RUNS/canon_raw_guided_naf_phase1_val_native"
LAST_VAL="$RUNS/canon_raw_guided_naf_phase1_last_val_native"

for _attempt in $(seq 1 960); do
  if [[ -s "$PHASE1/best.pt" && -s "$PHASE1/last.pt" && -s "$BEST_VAL/summary.json" ]]; then
    break
  fi
  sleep 30
done

if [[ ! -s "$PHASE1/last.pt" || ! -s "$BEST_VAL/summary.json" ]]; then
  echo "Phase 1 validation did not finish within eight hours" >&2
  exit 1
fi

cd "$PROJECT"
if [[ ! -s "$LAST_VAL/summary.json" ]]; then
  "$PYTHON" -m dip_course.cli.infer_official_sied \
    --official-repo "$OFFICIAL_REPO" \
    --checkpoint "$OFFICIAL_WEIGHT" \
    --refiner-checkpoint "$PHASE1/last.pt" \
    --data-root "$DATA_ROOT" \
    --manifest "$PROJECT/manifests/enhancement/canon_val.txt" \
    --output-dir "$LAST_VAL" \
    --device cuda --precision fp16 \
    --sampling-steps 20 --ensemble-size 3 --seed 3407 \
    --refiner-tile-size 768 --refiner-overlap 96
fi

"$PYTHON" - "$PHASE1" "$BEST_VAL/summary.json" "$LAST_VAL/summary.json" <<'PY'
import json
import shutil
import sys
from pathlib import Path

phase1 = Path(sys.argv[1])
best_summary_path = Path(sys.argv[2])
last_summary_path = Path(sys.argv[3])
best = json.loads(best_summary_path.read_text())
last = json.loads(last_summary_path.read_text())
candidates = [
    {"name": "thumbnail_mean_best", "checkpoint": phase1 / "best.pt", "summary": best, "summary_path": best_summary_path},
    {"name": "thumbnail_last", "checkpoint": phase1 / "last.pt", "summary": last, "summary_path": last_summary_path},
]
selected = max(candidates, key=lambda row: float(row["summary"]["psnr_mean"]))
shutil.copy2(selected["checkpoint"], phase1 / "selected.pt")
shutil.copy2(selected["summary_path"], phase1 / "selected_validation_summary.json")
payload = {
    "selection_metric": "full_native_validation_psnr_mean",
    "selected": selected["name"],
    "candidates": [
        {
            "name": row["name"],
            "checkpoint": str(row["checkpoint"].resolve()),
            "summary": str(row["summary_path"].resolve()),
            "psnr_mean": row["summary"]["psnr_mean"],
            "ssim_mean": row["summary"]["ssim_mean"],
        }
        for row in candidates
    ],
}
(phase1 / "native_selection.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
