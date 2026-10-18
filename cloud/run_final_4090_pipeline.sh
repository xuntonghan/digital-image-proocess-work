#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/cv-homework
PROJECT="$ROOT/project"
DATA="$ROOT/data"
RUNS="$ROOT/runs"
PYTHON=/root/autodl-tmp/venvs/cv-homework/bin/python
export PYTHONPATH="$PROJECT/src"
export CUDA_VISIBLE_DEVICES=0

GEO_PID="${1:-}"
STATUS="$RUNS/final_pipeline_status.txt"
mkdir -p "$RUNS"

phase() {
  printf '%s  %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$STATUS"
}

fail() {
  phase "FAILED at line $1"
}
trap 'fail $LINENO' ERR

if [[ -n "$GEO_PID" ]]; then
  phase "waiting for GeoCorrection training PID $GEO_PID"
  while kill -0 "$GEO_PID" 2>/dev/null; do
    sleep 30
  done
fi

test -s "$RUNS/geocorrection_segmentation_v1/summary.json"
mkdir -p "$RUNS/geocorrection_segmentation_predictions/masks"
MASK_COUNT="$(find "$RUNS/geocorrection_segmentation_predictions/masks" -type f -name '*.png' 2>/dev/null | wc -l)"
if [[ ! -s "$RUNS/geocorrection_segmentation_predictions/summary.json" || "$MASK_COUNT" -ne 200 ]]; then
  phase "predicting page masks"
  "$PYTHON" -m dip_course.cli.infer_page_segmentation \
    --data-root "$DATA/GeoCorrection" \
    --checkpoint "$RUNS/geocorrection_segmentation_v1/best.pt" \
    --output-dir "$RUNS/geocorrection_segmentation_predictions" \
    --split all --device cuda --precision bf16 \
    > "$RUNS/geocorrection_segmentation_infer.log" 2>&1
else
  phase "reusing 200 completed page-mask predictions"
fi

TEST_KEYS="$($PYTHON - <<'PY'
import json
path = "/root/autodl-tmp/cv-homework/runs/geocorrection_segmentation_v1/split.json"
print(",".join(json.load(open(path, encoding="utf-8"))["test"]))
PY
)"

if [[ ! -s "$RUNS/geocorrection_predicted_mask_test/summary.json" ]]; then
  phase "rectifying the held-out GeoCorrection test split with predicted masks"
  "$PYTHON" -m dip_course.cli.run_geocorrection \
    --data-root "$DATA/GeoCorrection" \
    --mask-subdir "$RUNS/geocorrection_segmentation_predictions/masks" \
    --mask-source-label "PageMaskUNet predictions on a held-out test split" \
    --output-dir "$RUNS/geocorrection_predicted_mask_test" \
    --keys "$TEST_KEYS" --max-side 1400 --expected-count 200 \
    > "$RUNS/geocorrection_predicted_mask_test.log" 2>&1
else
  phase "reusing predicted-mask GeoCorrection test results"
fi

if [[ ! -s "$RUNS/geocorrection_oracle_mask_test/summary.json" ]]; then
  phase "rectifying the same test split with oracle masks"
  "$PYTHON" -m dip_course.cli.run_geocorrection \
    --data-root "$DATA/GeoCorrection" \
    --output-dir "$RUNS/geocorrection_oracle_mask_test" \
    --keys "$TEST_KEYS" --max-side 1400 --expected-count 200 \
    > "$RUNS/geocorrection_oracle_mask_test.log" 2>&1
else
  phase "reusing oracle-mask GeoCorrection test results"
fi

if [[ ! -s "$RUNS/canon_cloud_audit.json" ]]; then
  phase "auditing Canon SIED data"
  "$PYTHON" -m dip_course.cli.audit_dataset \
    --data-root "$DATA/SIED_dataset" \
    --manifest "$PROJECT/manifests/source/canon_all.txt" \
    --output-json "$RUNS/canon_cloud_audit.json" \
    > "$RUNS/canon_cloud_audit.log" 2>&1
else
  phase "reusing completed Canon SIED audit"
fi

OFFICIAL_REPO="$ROOT/third_party/SIED-official"
OFFICIAL_WEIGHT="$PROJECT/checkpoints/official/canon_0.0001.pt"
CACHE="$ROOT/cache/official_sied/canon_train_val_768"
mkdir -p "$CACHE/images"

CACHE_COUNT="$(find "$CACHE/images" -type f -name '*.png' | wc -l)"
if [[ "$CACHE_COUNT" -ne 924 ]]; then
  phase "caching 832 blind SIED training predictions"
  "$PYTHON" -m dip_course.cli.infer_official_sied \
    --official-repo "$OFFICIAL_REPO" \
    --checkpoint "$OFFICIAL_WEIGHT" \
    --data-root "$DATA/SIED_dataset" \
    --manifest "$PROJECT/manifests/enhancement/canon_train.txt" \
    --output-dir "$CACHE" --device cuda --precision fp16 \
    --sampling-steps 20 --ensemble-size 1 --seed 3407 \
    --skip-metrics --save-max-side 768 \
    > "$RUNS/canon_sied_cache_train.log" 2>&1

  phase "caching 92 blind SIED validation predictions"
  "$PYTHON" -m dip_course.cli.infer_official_sied \
    --official-repo "$OFFICIAL_REPO" \
    --checkpoint "$OFFICIAL_WEIGHT" \
    --data-root "$DATA/SIED_dataset" \
    --manifest "$PROJECT/manifests/enhancement/canon_val.txt" \
    --output-dir "$CACHE" --device cuda --precision fp16 \
    --sampling-steps 20 --ensemble-size 1 --seed 3407 \
    --skip-metrics --save-max-side 768 \
    > "$RUNS/canon_sied_cache_val.log" 2>&1
  CACHE_COUNT="$(find "$CACHE/images" -type f -name '*.png' | wc -l)"
else
  phase "reusing 924 cached SIED train/validation predictions"
fi
if [[ "$CACHE_COUNT" -ne 924 ]]; then
  phase "unexpected SIED cache count $CACHE_COUNT"
  exit 1
fi

if [[ ! -s "$RUNS/canon_sied_refiner_v2/training_summary.json" || ! -s "$RUNS/canon_sied_refiner_v2/best.pt" ]]; then
  phase "fine-tuning the exposure and color refiner from the previous best checkpoint"
  "$PYTHON" -m dip_course.cli.train_sied_refiner \
  --official-repo "$OFFICIAL_REPO" \
  --checkpoint "$OFFICIAL_WEIGHT" \
  --initial-checkpoint "$PROJECT/checkpoints/canon_sied_refiner_fullscene_bf16_best.pt" \
  --data-root "$DATA/SIED_dataset" \
  --train-manifest "$PROJECT/manifests/enhancement/canon_train.txt" \
  --val-manifest "$PROJECT/manifests/enhancement/canon_val.txt" \
  --prediction-dir "$CACHE/images" \
  --output-dir "$RUNS/canon_sied_refiner_v2" \
  --device cuda --precision bf16 --batch-size 8 --num-workers 4 \
  --max-steps 1500 --learning-rate 2e-5 --weight-decay 1e-4 \
  --validation-interval 25 --checkpoint-interval 250 \
  --gradient-accumulation 1 --require-cuda \
    > "$RUNS/canon_sied_refiner_v2.log" 2>&1
else
  phase "reusing completed SIED refiner fine-tuning"
fi

if [[ ! -s "$RUNS/compression_cheng2020_anchor/summary.json" ]]; then
  phase "benchmarking pretrained learned compression on Kodak"
  "$PYTHON" -m dip_course.cli.benchmark_learned_compression \
  --data-dir "$DATA/kodak" \
  --output-dir "$RUNS/compression_cheng2020_anchor" \
  --architecture cheng2020-anchor --qualities 1 3 6 --metric mse \
  --device cuda --repeats 1 --warmup 0 \
  --baseline-summary "$PROJECT/reports/assets/compression/jpeg_baseline/summary.json" \
    > "$RUNS/compression_cheng2020_anchor.log" 2>&1
else
  phase "reusing completed learned-compression benchmark"
fi

FINAL="$RUNS/enhancement_final_test"
mkdir -p "$FINAL/base_images"
if [[ ! -s "$FINAL/summary.json" ]]; then
  phase "evaluating the final refiner on the frozen 76-image Canon test split"
  "$PYTHON" -m dip_course.cli.infer_official_sied \
  --official-repo "$OFFICIAL_REPO" \
  --checkpoint "$OFFICIAL_WEIGHT" \
  --refiner-checkpoint "$RUNS/canon_sied_refiner_v2/best.pt" \
  --save-base-dir "$FINAL/base_images" \
  --data-root "$DATA/SIED_dataset" \
  --manifest "$PROJECT/manifests/enhancement/canon_test.txt" \
  --output-dir "$FINAL" --device cuda --precision fp16 \
  --sampling-steps 20 --ensemble-size 3 --seed 3407 \
    > "$RUNS/enhancement_final_test.log" 2>&1
else
  phase "reusing completed Canon final-test evaluation"
fi

if [[ ! -s "$FINAL/comparison.png" ]]; then
  phase "building enhancement comparison figure"
  "$PYTHON" -m dip_course.cli.visualize_comparison \
  --data-root "$DATA/SIED_dataset" \
  --manifest "$PROJECT/manifests/enhancement/canon_test.txt" \
  --reference-prediction-dir "$FINAL/base_images" \
  --prediction-dir "$FINAL/images" \
  --output "$FINAL/comparison.png" --count 6 --auto \
    > "$RUNS/enhancement_final_visualize.log" 2>&1
fi

phase "COMPLETE"
