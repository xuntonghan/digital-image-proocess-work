#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/sied30}"
PROJECT="$ROOT/project"
DATA_ROOT="$ROOT/data/SIED_dataset"
OFFICIAL_REPO="$ROOT/third_party/SIED-official"
OFFICIAL_WEIGHT="$ROOT/checkpoints/official/canon_0.0001.pt"
PYTHON="${PYTHON:-/root/autodl-tmp/cv-homework/.venv/bin/python}"
RUNS="$ROOT/runs"
PHASE2="$RUNS/canon_raw_guided_naf_phase2_native"
PHASE2_VAL="$RUNS/canon_raw_guided_naf_phase2_val_native"
PHASE3="$RUNS/canon_raw_guided_naf_phase3_psnr"
VAL_NATIVE="$RUNS/canon_raw_guided_naf_phase3_val_native"

for _attempt in $(seq 1 960); do
  if [[ -s "$PHASE2/best.pt" && -s "$PHASE2_VAL/summary.json" ]]; then
    break
  fi
  sleep 30
done

if [[ ! -s "$PHASE2/best.pt" || ! -s "$PHASE2_VAL/summary.json" ]]; then
  echo "Phase 2 did not finish within eight hours" >&2
  exit 1
fi

cd "$PROJECT"
if [[ ! -s "$PHASE3/summary.json" || ! -s "$PHASE3/best.pt" ]]; then
  "$PYTHON" -m dip_course.cli.train_sied_restorer_native \
    --official-repo "$OFFICIAL_REPO" \
    --checkpoint "$OFFICIAL_WEIGHT" \
    --initial-checkpoint "$PHASE2/best.pt" \
    --data-root "$DATA_ROOT" \
    --train-manifest "$PROJECT/manifests/enhancement/canon_train.txt" \
    --val-manifest "$PROJECT/manifests/enhancement/canon_val.txt" \
    --output-dir "$PHASE3" \
    --device cuda --precision fp16 --require-cuda \
    --sampling-steps 20 --max-steps 2000 --patch-size 512 \
    --learning-rate 2e-5 --weight-decay 1e-4 \
    --validation-interval 200 --val-limit 48 \
    --tile-size 768 --tile-overlap 96 \
    --ema-decay 0.999 --num-workers 2 --freeze-photometric \
    --loss-profile psnr
fi

if [[ ! -s "$VAL_NATIVE/summary.json" ]]; then
  "$PYTHON" -m dip_course.cli.infer_official_sied \
    --official-repo "$OFFICIAL_REPO" \
    --checkpoint "$OFFICIAL_WEIGHT" \
    --refiner-checkpoint "$PHASE3/best.pt" \
    --data-root "$DATA_ROOT" \
    --manifest "$PROJECT/manifests/enhancement/canon_val.txt" \
    --output-dir "$VAL_NATIVE" \
    --device cuda --precision fp16 \
    --sampling-steps 20 --ensemble-size 3 --seed 3407 \
    --refiner-tile-size 768 --refiner-overlap 96
fi

"$PYTHON" - "$PHASE2_VAL/summary.json" "$VAL_NATIVE/summary.json" <<'PY'
import json
import sys
from pathlib import Path

phase2 = json.loads(Path(sys.argv[1]).read_text())
phase3 = json.loads(Path(sys.argv[2]).read_text())
print(
    json.dumps(
        {
            "phase": "raw_guided_naf_phase3_psnr",
            "phase2_full_native_validation_psnr": phase2["psnr_mean"],
            "phase3_full_native_validation_psnr": phase3["psnr_mean"],
            "phase3_full_native_validation_ssim": phase3["ssim_mean"],
            "phase3_full_native_validation_delta_e2000": phase3[
                "delta_e2000_mean"
            ],
        },
        indent=2,
    )
)
PY
