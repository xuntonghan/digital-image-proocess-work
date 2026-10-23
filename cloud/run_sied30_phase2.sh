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
PHASE1_INITIAL="$PHASE1/selected.pt"
PHASE1_VAL="$PHASE1/selected_validation_summary.json"
PHASE2="$RUNS/canon_raw_guided_naf_phase2_native"
VAL_NATIVE="$RUNS/canon_raw_guided_naf_phase2_val_native"

for _attempt in $(seq 1 720); do
  if [[ -s "$PHASE1/summary.json" && -s "$PHASE1_INITIAL" && -s "$PHASE1_VAL" ]]; then
    break
  fi
  sleep 30
done

if [[ ! -s "$PHASE1/summary.json" || ! -s "$PHASE1_INITIAL" || ! -s "$PHASE1_VAL" ]]; then
  echo "Phase 1 did not finish within six hours" >&2
  exit 1
fi

cd "$PROJECT"
if [[ ! -s "$PHASE2/summary.json" || ! -s "$PHASE2/best.pt" ]]; then
  "$PYTHON" -m dip_course.cli.train_sied_restorer_native \
    --official-repo "$OFFICIAL_REPO" \
    --checkpoint "$OFFICIAL_WEIGHT" \
    --initial-checkpoint "$PHASE1_INITIAL" \
    --data-root "$DATA_ROOT" \
    --train-manifest "$PROJECT/manifests/enhancement/canon_train.txt" \
    --val-manifest "$PROJECT/manifests/enhancement/canon_val.txt" \
    --output-dir "$PHASE2" \
    --device cuda --precision fp16 --require-cuda \
    --sampling-steps 20 --max-steps 2500 --patch-size 512 \
    --learning-rate 5e-5 --weight-decay 1e-4 \
    --validation-interval 250 --val-limit 48 \
    --tile-size 768 --tile-overlap 96 \
    --ema-decay 0.999 --num-workers 2 --freeze-photometric
fi

if [[ ! -s "$VAL_NATIVE/summary.json" ]]; then
  "$PYTHON" -m dip_course.cli.infer_official_sied \
    --official-repo "$OFFICIAL_REPO" \
    --checkpoint "$OFFICIAL_WEIGHT" \
    --refiner-checkpoint "$PHASE2/best.pt" \
    --data-root "$DATA_ROOT" \
    --manifest "$PROJECT/manifests/enhancement/canon_val.txt" \
    --output-dir "$VAL_NATIVE" \
    --device cuda --precision fp16 \
    --sampling-steps 20 --ensemble-size 3 --seed 3407 \
    --refiner-tile-size 768 --refiner-overlap 96
fi

"$PYTHON" - "$PHASE1/summary.json" "$PHASE2/summary.json" "$VAL_NATIVE/summary.json" <<'PY'
import json
import sys
from pathlib import Path

phase1 = json.loads(Path(sys.argv[1]).read_text())
phase2 = json.loads(Path(sys.argv[2]).read_text())
validation = json.loads(Path(sys.argv[3]).read_text())
print(
    json.dumps(
        {
            "phase": "raw_guided_naf_phase2_native",
            "phase1_thumbnail_validation_best_psnr": phase1[
                "best_refined_psnr_mean"
            ],
            "phase2_subset_validation_best_psnr": phase2[
                "best_refined_native_psnr"
            ],
            "full_native_validation_psnr": validation["psnr_mean"],
            "full_native_validation_ssim": validation["ssim_mean"],
            "full_native_validation_delta_e2000": validation[
                "delta_e2000_mean"
            ],
        },
        indent=2,
    )
)
PY
