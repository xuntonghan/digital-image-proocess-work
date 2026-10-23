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
PHASE3_VAL="$RUNS/canon_raw_guided_naf_phase3_val_native"
FULL_CACHE="$ROOT/cache/official_sied/canon_train_val_native_ensemble3"
VAL_CACHE="$ROOT/cache/official_sied/canon_val_768_ensemble3"
RAW_MEAN_CACHE="$ROOT/cache/official_sied/canon_raw_means.json"
PHASE4="$RUNS/canon_raw_guided_naf_phase4_cached_native"
VAL_NATIVE="$RUNS/canon_raw_guided_naf_phase4_val_native"

for _attempt in $(seq 1 1200); do
  if [[ -s "$PHASE3/best.pt" && -s "$PHASE3_VAL/summary.json" ]] || \
     [[ -s "$PHASE2/best.pt" && -s "$PHASE2_VAL/summary.json" ]]; then
    break
  fi
  sleep 30
done

if [[ -s "$PHASE3/best.pt" && -s "$PHASE3_VAL/summary.json" ]]; then
  INITIAL_CHECKPOINT="$PHASE3/best.pt"
  INITIAL_VALIDATION="$PHASE3_VAL/summary.json"
  INITIAL_PHASE="phase3"
elif [[ -s "$PHASE2/best.pt" && -s "$PHASE2_VAL/summary.json" ]]; then
  INITIAL_CHECKPOINT="$PHASE2/best.pt"
  INITIAL_VALIDATION="$PHASE2_VAL/summary.json"
  INITIAL_PHASE="phase2"
else
  echo "Neither Phase 2 nor Phase 3 became ready within ten hours" >&2
  exit 1
fi
echo "Phase 4 initialization: $INITIAL_PHASE ($INITIAL_CHECKPOINT)"

cd "$PROJECT"
mkdir -p "$FULL_CACHE/images" "$VAL_CACHE/images"
"$PYTHON" - "$FULL_CACHE/images" <<'PY'
import sys
from pathlib import Path

from PIL import Image

root = Path(sys.argv[1])
invalid = []
for path in sorted(root.glob("*.png")):
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, ValueError):
        invalid.append(path)
for path in invalid:
    path.unlink()
print(f"verified {len(list(root.glob('*.png')))} cached PNGs; removed {len(invalid)} invalid files")
PY
CACHE_COUNT="$(find "$FULL_CACHE/images" -maxdepth 1 -type f -name '*.png' | wc -l)"
if [[ "$CACHE_COUNT" -ne 924 ]]; then
  build_missing_manifest() {
    local source_manifest="$1"
    local missing_manifest="$2"
    local seed_file="$3"
    "$PYTHON" - "$source_manifest" "$FULL_CACHE/images" "$missing_manifest" \
      "$seed_file" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1])
images = Path(sys.argv[2])
output = Path(sys.argv[3])
seed_output = Path(sys.argv[4])
cached = {path.stem.split("_", 1)[-1] for path in images.glob("*.png")}
lines = [line for line in source.read_text().splitlines() if line.strip()]
present = [Path(line.split()[0]).stem in cached for line in lines]
first_missing = next((index for index, value in enumerate(present) if not value), len(lines))
if any(present[first_missing:]):
    raise RuntimeError(
        f"Cached predictions for {source.name} are not a contiguous prefix; "
        "refusing to change per-image diffusion seeds during resume"
    )
missing = lines[first_missing:]
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(missing) + ("\n" if missing else ""))
seed_output.write_text(f"{3407 + first_missing}\n")
print(
    f"{source.name}: {len(missing)} missing cached predictions; "
    f"resume seed {3407 + first_missing}"
)
PY
  }

  MISSING_TRAIN="$FULL_CACHE/missing_train.txt"
  MISSING_VAL="$FULL_CACHE/missing_val.txt"
  TRAIN_SEED_FILE="$FULL_CACHE/missing_train.seed"
  VAL_SEED_FILE="$FULL_CACHE/missing_val.seed"
  build_missing_manifest \
    "$PROJECT/manifests/enhancement/canon_train.txt" "$MISSING_TRAIN" \
    "$TRAIN_SEED_FILE"
  build_missing_manifest \
    "$PROJECT/manifests/enhancement/canon_val.txt" "$MISSING_VAL" \
    "$VAL_SEED_FILE"

  for MISSING_MANIFEST in "$MISSING_TRAIN" "$MISSING_VAL"; do
    if [[ -s "$MISSING_MANIFEST" ]]; then
      if [[ "$MISSING_MANIFEST" == "$MISSING_TRAIN" ]]; then
        RESUME_SEED="$(<"$TRAIN_SEED_FILE")"
      else
        RESUME_SEED="$(<"$VAL_SEED_FILE")"
      fi
      "$PYTHON" -m dip_course.cli.infer_official_sied \
        --official-repo "$OFFICIAL_REPO" \
        --checkpoint "$OFFICIAL_WEIGHT" \
        --data-root "$DATA_ROOT" \
        --manifest "$MISSING_MANIFEST" \
        --output-dir "$FULL_CACHE" \
        --device cuda --precision fp16 \
        --sampling-steps 20 --ensemble-size 3 --seed "$RESUME_SEED" --skip-metrics
    fi
  done
fi

CACHE_COUNT="$(find "$FULL_CACHE/images" -maxdepth 1 -type f -name '*.png' | wc -l)"
if [[ "$CACHE_COUNT" -ne 924 ]]; then
  echo "Expected 924 native cached predictions, found $CACHE_COUNT" >&2
  exit 1
fi

"$PYTHON" - "$PROJECT/manifests/enhancement/canon_val.txt" \
  "$FULL_CACHE/images" "$VAL_CACHE/images" <<'PY'
import sys
from pathlib import Path

from PIL import Image

manifest = Path(sys.argv[1])
source = Path(sys.argv[2])
destination = Path(sys.argv[3])
destination.mkdir(parents=True, exist_ok=True)
for line in manifest.read_text().splitlines():
    if not line.strip():
        continue
    key = Path(line.split()[0]).stem
    matches = list(source.glob(f"*_{key}.png"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one native cache entry for {key}, got {matches}")
    output = destination / matches[0].name
    if output.is_file():
        continue
    with Image.open(matches[0]) as image:
        rgb = image.convert("RGB")
        scale = 768 / max(rgb.size)
        size = (round(rgb.width * scale), round(rgb.height * scale))
        rgb.resize(size, Image.Resampling.LANCZOS).save(output)
PY

if [[ ! -s "$RAW_MEAN_CACHE" ]]; then
  "$PYTHON" - "$DATA_ROOT" "$PROJECT/manifests/enhancement/canon_train.txt" \
    "$RAW_MEAN_CACHE" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

data_root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
output = Path(sys.argv[3])
means = {}
for line in manifest.read_text().splitlines():
    if not line.strip():
        continue
    relative = Path(line.split()[0])
    raw = np.memmap(
        data_root / relative,
        dtype=np.uint16,
        mode="r",
        shape=(2160, 3840),
    )
    total = 0.0
    count = 0
    for top in range(0, 2160, 128):
        stripe = np.asarray(raw[top : top + 128], dtype=np.float32)
        normalized = np.clip((stripe - 512.0) / (16383.0 - 512.0), 0.0, 1.0)
        total += float(normalized.sum(dtype=np.float64))
        count += normalized.size
    means[relative.stem] = total / count
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(means, indent=2) + "\n")
PY
fi

if [[ ! -s "$PHASE4/summary.json" || ! -s "$PHASE4/best.pt" ]]; then
  "$PYTHON" -m dip_course.cli.train_sied_restorer \
    --data-root "$DATA_ROOT" \
    --train-manifest "$PROJECT/manifests/enhancement/canon_train.txt" \
    --val-manifest "$PROJECT/manifests/enhancement/canon_val.txt" \
    --prediction-dir "$FULL_CACHE/images" \
    --val-prediction-dir "$VAL_CACHE/images" \
    --raw-mean-cache "$RAW_MEAN_CACHE" \
    --output-dir "$PHASE4" \
    --initial-checkpoint "$INITIAL_CHECKPOINT" \
    --device cuda --precision fp16 --require-cuda \
    --width 48 --encoder-blocks 2,2,4 --middle-blocks 6 \
    --decoder-blocks 2,2,2 --max-residual 0.35 \
    --crop-size 512 --batch-size 1 --num-workers 4 \
    --max-steps 20000 --learning-rate 2e-5 --weight-decay 1e-4 \
    --validation-interval 500 --checkpoint-interval 5000 \
    --gradient-accumulation 1 --ema-decay 0.999 --loss-profile psnr
fi

if [[ ! -s "$VAL_NATIVE/summary.json" ]]; then
  "$PYTHON" -m dip_course.cli.infer_official_sied \
    --official-repo "$OFFICIAL_REPO" \
    --checkpoint "$OFFICIAL_WEIGHT" \
    --refiner-checkpoint "$PHASE4/best.pt" \
    --data-root "$DATA_ROOT" \
    --manifest "$PROJECT/manifests/enhancement/canon_val.txt" \
    --output-dir "$VAL_NATIVE" \
    --device cuda --precision fp16 \
    --sampling-steps 20 --ensemble-size 3 --seed 3407 \
    --refiner-tile-size 768 --refiner-overlap 96
fi

"$PYTHON" - "$INITIAL_VALIDATION" "$VAL_NATIVE/summary.json" \
  "$INITIAL_PHASE" <<'PY'
import json
import sys
from pathlib import Path

initial = json.loads(Path(sys.argv[1]).read_text())
phase4 = json.loads(Path(sys.argv[2]).read_text())
initial_phase = sys.argv[3]
print(
    json.dumps(
        {
            "phase": "raw_guided_naf_phase4_cached_native",
            "initial_phase": initial_phase,
            "initial_full_native_validation_psnr": initial["psnr_mean"],
            "phase4_full_native_validation_psnr": phase4["psnr_mean"],
            "phase4_full_native_validation_ssim": phase4["ssim_mean"],
            "phase4_full_native_validation_delta_e2000": phase4[
                "delta_e2000_mean"
            ],
        },
        indent=2,
    )
)
PY
