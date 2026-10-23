#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/sied30}"
PROJECT="$ROOT/project"
PYTHON="${PYTHON:-/root/autodl-tmp/cv-homework/.venv/bin/python}"
RUNS="$ROOT/runs"
PHASE5="$RUNS/canon_raw_guided_naf_phase5_context"
PHASE5_VAL="$RUNS/canon_raw_guided_naf_phase5_val_native"
mkdir -p "$PHASE5"
cd "$PROJECT"

# Compare complete native-resolution validation summaries, never test scores.
INITIAL_CHECKPOINT="$($PYTHON - "$RUNS" "$PHASE5/initial_selection.json" <<'PY'
import json
import sys
from pathlib import Path

root, destination = map(Path, sys.argv[1:])
pairs = [
    ("phase1", "canon_raw_guided_naf_phase1/selected_validation_summary.json", "canon_raw_guided_naf_phase1/selected.pt"),
    ("phase2", "canon_raw_guided_naf_phase2_val_native/summary.json", "canon_raw_guided_naf_phase2_native/best.pt"),
    ("phase3", "canon_raw_guided_naf_phase3_val_native/summary.json", "canon_raw_guided_naf_phase3_psnr/best.pt"),
    ("phase4", "canon_raw_guided_naf_phase4_val_native/summary.json", "canon_raw_guided_naf_phase4_cached_native/best.pt"),
]
candidates = []
for phase, summary, checkpoint in pairs:
    summary_path, checkpoint_path = root / summary, root / checkpoint
    if not summary_path.is_file() or not checkpoint_path.is_file():
        continue
    result = json.loads(summary_path.read_text())
    if result.get("count") != 92 or result.get("psnr_mean") is None:
        continue
    candidates.append({"phase": phase, "checkpoint": str(checkpoint_path), "psnr_mean": result["psnr_mean"], "validation_summary": str(summary_path)})
if not candidates:
    raise RuntimeError("No checkpoint has a complete 92-image native validation summary")
selected = max(candidates, key=lambda item: item["psnr_mean"])
destination.write_text(json.dumps({"selection_metric": "native_validation_psnr_mean", "selected": selected, "candidates": candidates}, indent=2) + "\n")
print(selected["checkpoint"])
PY
)"

if [[ ! -s "$PHASE5/summary.json" || ! -s "$PHASE5_VAL/summary.json" ]]; then
  "$PYTHON" -m dip_course.cli.train_sied_restorer_context \
    --data-root "$ROOT/data/SIED_dataset" \
    --train-manifest "$PROJECT/manifests/enhancement/canon_train.txt" \
    --val-manifest "$PROJECT/manifests/enhancement/canon_val.txt" \
    --prediction-dir "$ROOT/cache/official_sied/canon_train_val_native_ensemble3/images" \
    --cache-dir "$ROOT/cache/official_sied/canon_full_scene_corrected" \
    --initial-checkpoint "$INITIAL_CHECKPOINT" \
    --output-dir "$PHASE5" --validation-output-dir "$PHASE5_VAL" \
    --device cuda --precision fp16 --require-cuda \
    --crop-size 512 --batch-size 1 --num-workers 4 \
    --max-steps "${PHASE5_MAX_STEPS:-10000}" \
    --min-steps 2500 --early-stop-patience 6 \
    --validation-interval 500 --val-limit 12 \
    --tile-size 512 --tile-overlap 64 \
    --learning-rate 2e-5 --weight-decay 1e-4 --ema-decay 0.999 \
    --resume
fi

echo "Phase 5 complete: $PHASE5_VAL/summary.json"
# The outer queue selects phases on validation and performs frozen test once.
