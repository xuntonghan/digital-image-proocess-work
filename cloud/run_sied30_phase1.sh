#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/sied30}"
PROJECT="$ROOT/project"
DATA_ROOT="$ROOT/data/SIED_dataset"
OFFICIAL_REPO="$ROOT/third_party/SIED-official"
OFFICIAL_WEIGHT="$ROOT/checkpoints/official/canon_0.0001.pt"
PYTHON="${PYTHON:-/root/autodl-tmp/cv-homework/.venv/bin/python}"
CACHE="$ROOT/cache/official_sied/canon_train_val_768"
RUNS="$ROOT/runs"
LOGS="$ROOT/logs"

mkdir -p "$CACHE/images" "$RUNS" "$LOGS"

count_files() {
  local directory="$1"
  local pattern="$2"
  if [[ ! -d "$directory" ]]; then
    echo 0
    return
  fi
  find "$directory" -maxdepth 1 -type f -name "$pattern" | wc -l
}

for _attempt in $(seq 1 360); do
  RAW_COUNT="$(count_files "$DATA_ROOT/0.0001" '*.raw')"
  GT_COUNT="$(count_files "$DATA_ROOT/gt" '*.jpg')"
  if [[ "$RAW_COUNT" -eq 1000 && "$GT_COUNT" -eq 1000 && -s "$OFFICIAL_WEIGHT" ]]; then
    break
  fi
  sleep 30
done

RAW_COUNT="$(count_files "$DATA_ROOT/0.0001" '*.raw')"
GT_COUNT="$(count_files "$DATA_ROOT/gt" '*.jpg')"
if [[ "$RAW_COUNT" -ne 1000 || "$GT_COUNT" -ne 1000 || ! -s "$OFFICIAL_WEIGHT" ]]; then
  echo "SIED assets did not become ready within three hours" >&2
  exit 1
fi

cd "$PROJECT"
"$PYTHON" -m dip_course.cli.audit_dataset \
  --data-root "$DATA_ROOT" \
  --manifest "$PROJECT/manifests/source/canon_all.txt" \
  --output-json "$RUNS/canon_cloud_audit.json"

CACHE_COUNT="$(find "$CACHE/images" -maxdepth 1 -type f -name '*.png' | wc -l)"
if [[ "$CACHE_COUNT" -ne 924 ]]; then
  "$PYTHON" -m dip_course.cli.infer_official_sied \
    --official-repo "$OFFICIAL_REPO" \
    --checkpoint "$OFFICIAL_WEIGHT" \
    --data-root "$DATA_ROOT" \
    --manifest "$PROJECT/manifests/enhancement/canon_train.txt" \
    --output-dir "$CACHE" \
    --device cuda --precision fp16 \
    --sampling-steps 20 --ensemble-size 1 --seed 3407 \
    --skip-metrics --save-max-side 768

  "$PYTHON" -m dip_course.cli.infer_official_sied \
    --official-repo "$OFFICIAL_REPO" \
    --checkpoint "$OFFICIAL_WEIGHT" \
    --data-root "$DATA_ROOT" \
    --manifest "$PROJECT/manifests/enhancement/canon_val.txt" \
    --output-dir "$CACHE" \
    --device cuda --precision fp16 \
    --sampling-steps 20 --ensemble-size 1 --seed 3407 \
    --skip-metrics --save-max-side 768
fi

CACHE_COUNT="$(find "$CACHE/images" -maxdepth 1 -type f -name '*.png' | wc -l)"
if [[ "$CACHE_COUNT" -ne 924 ]]; then
  echo "Expected 924 cached train/validation predictions, found $CACHE_COUNT" >&2
  exit 1
fi

RESTORER_RUN="$RUNS/canon_raw_guided_naf_phase1"
if [[ ! -s "$RESTORER_RUN/summary.json" || ! -s "$RESTORER_RUN/best.pt" ]]; then
  "$PYTHON" -m dip_course.cli.train_sied_restorer \
    --data-root "$DATA_ROOT" \
    --train-manifest "$PROJECT/manifests/enhancement/canon_train.txt" \
    --val-manifest "$PROJECT/manifests/enhancement/canon_val.txt" \
    --prediction-dir "$CACHE/images" \
    --output-dir "$RESTORER_RUN" \
    --photometric-checkpoint "$PROJECT/reports/assets/enhancement/refiner_v2/best.pt" \
    --device cuda --precision fp16 --require-cuda \
    --width 48 --encoder-blocks 2,2,4 --middle-blocks 6 \
    --decoder-blocks 2,2,2 --max-residual 0.35 \
    --batch-size 1 --num-workers 4 \
    --max-steps 20000 --learning-rate 2e-4 --weight-decay 1e-4 \
    --validation-interval 250 --checkpoint-interval 2000 \
    --gradient-accumulation 1 --ema-decay 0.999
fi

VAL_NATIVE="$RUNS/canon_raw_guided_naf_phase1_val_native"
if [[ ! -s "$VAL_NATIVE/summary.json" ]]; then
  "$PYTHON" -m dip_course.cli.infer_official_sied \
    --official-repo "$OFFICIAL_REPO" \
    --checkpoint "$OFFICIAL_WEIGHT" \
    --refiner-checkpoint "$RESTORER_RUN/best.pt" \
    --data-root "$DATA_ROOT" \
    --manifest "$PROJECT/manifests/enhancement/canon_val.txt" \
    --output-dir "$VAL_NATIVE" \
    --device cuda --precision fp16 \
    --sampling-steps 20 --ensemble-size 3 --seed 3407 \
    --refiner-tile-size 768 --refiner-overlap 96
fi

"$PYTHON" - "$RESTORER_RUN/summary.json" "$VAL_NATIVE/summary.json" <<'PY'
import json
import sys
from pathlib import Path

training = json.loads(Path(sys.argv[1]).read_text())
validation = json.loads(Path(sys.argv[2]).read_text())
print(
    json.dumps(
        {
            "phase": "raw_guided_naf_phase1",
            "thumbnail_validation_best_psnr": training["best_refined_psnr_mean"],
            "native_validation_psnr": validation["psnr_mean"],
            "native_validation_ssim": validation["ssim_mean"],
        },
        indent=2,
    )
)
PY
