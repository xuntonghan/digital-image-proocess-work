#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/sied30}"
PROJECT="$ROOT/project"
DATA_ROOT="$ROOT/data/SIED_dataset"
OFFICIAL_REPO="$ROOT/third_party/SIED-official"
OFFICIAL_WEIGHT="$ROOT/checkpoints/official/canon_0.0001.pt"
PYTHON="${PYTHON:-/root/autodl-tmp/cv-homework/.venv/bin/python}"
FINAL="$ROOT/final"

mkdir -p "$ROOT/logs"
exec 9>"$ROOT/logs/sied_finalization.lock"
flock -n 9 || { echo "Another SIED finalizer is running" >&2; exit 1; }
if [[ -e "$ROOT/HOLD_FINALIZE" ]]; then
  echo "FINALIZATION_DEFERRED: validation experiments are still in progress"
  exit 0
fi

cd "$PROJECT"
# All decisions use complete native validation. Existing acceptance artifacts
# must match immutable checkpoint/manifest/inference bindings before any writes.
ACTION="$($PYTHON -m dip_course.cli.select_sied_final prepare --root "$ROOT")"
read -r TILE OVERLAP SEED SAMPLING_STEPS ENSEMBLE_SIZE < <(
  "$PYTHON" -m dip_course.cli.select_sied_final parameters --root "$ROOT"
)

if [[ "$ACTION" == "run" ]]; then
  # A started evaluation without a completed summary cannot silently rerun.
  "$PYTHON" -m dip_course.cli.select_sied_final start --root "$ROOT"
  "$PYTHON" -m dip_course.cli.infer_official_sied \
    --official-repo "$OFFICIAL_REPO" \
    --checkpoint "$OFFICIAL_WEIGHT" \
    --refiner-checkpoint "$FINAL/best.pt" \
    --data-root "$DATA_ROOT" \
    --manifest "$PROJECT/manifests/enhancement/canon_test.txt" \
    --output-dir "$FINAL/frozen_test" \
    --save-base-dir "$FINAL/base_images" \
    --device cuda --precision fp16 \
    --sampling-steps "$SAMPLING_STEPS" --ensemble-size "$ENSEMBLE_SIZE" --seed "$SEED" \
    --refiner-tile-size "$TILE" --refiner-overlap "$OVERLAP" --lpips
elif [[ "$ACTION" != "reuse" ]]; then
  echo "Unexpected finalization action: $ACTION" >&2
  exit 1
fi

# Validate and seal results before creating presentation artifacts.
"$PYTHON" -m dip_course.cli.select_sied_final seal --root "$ROOT"
if [[ ! -e "$FINAL/comparison.png" ]]; then
  "$PYTHON" -m dip_course.cli.visualize_comparison \
    --data-root "$DATA_ROOT" \
    --manifest "$PROJECT/manifests/enhancement/canon_test.txt" \
    --reference-prediction-dir "$FINAL/base_images" \
    --prediction-dir "$FINAL/frozen_test/images" \
    --output "$FINAL/comparison.png" \
    --count 6 --auto --panel-width 640
fi
