#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/sied30}"
DOWNLOADER="${2:-$ROOT/project/cloud/download_official_sied_assets.py}"
ASSETS="$ROOT/assets"
DATA_ROOT="$ROOT/data/SIED_dataset"
CHECKPOINT_DIR="$ROOT/checkpoints/official"
MANIFEST="$ROOT/project/manifests/source/canon_all.txt"

# Preparation is only for a fresh experiment directory. Existing raw data,
# shared GT or weights may be the user's only copy: never delete/overwrite them.
for EXISTING_TARGET in "$DATA_ROOT/0.0001" "$DATA_ROOT/gt" "$CHECKPOINT_DIR/canon_0.0001.pt"; do
  if [[ -e "$EXISTING_TARGET" || -L "$EXISTING_TARGET" ]]; then
    echo "Refusing to replace existing data/weights: $EXISTING_TARGET" >&2
    echo "Use a fresh ROOT, or audit and reuse the existing files without this preparation step." >&2
    exit 1
  fi
done
mkdir -p "$ASSETS" "$DATA_ROOT" "$CHECKPOINT_DIR" "$ROOT/logs"

python "$DOWNLOADER" \
  --output-dir "$ASSETS" \
  --dataset-assets Canon_0.0001.zip \
  --checkpoint-assets
unzip -tq "$ASSETS/Canon_0.0001.zip"
if [[ ! -s "$MANIFEST" ]]; then
  echo "Missing Canon source manifest: $MANIFEST" >&2
  exit 1
fi
mapfile -t RAW_FILES < <(awk 'NF >= 2 {print $1}' "$MANIFEST")
unzip -q "$ASSETS/Canon_0.0001.zip" "${RAW_FILES[@]}" -d "$DATA_ROOT"
RAW_COUNT="$(find "$DATA_ROOT/0.0001" -maxdepth 1 -type f -name '*.raw' | wc -l)"
if [[ "$RAW_COUNT" -ne 1000 ]]; then
  echo "Expected 1000 manifest-selected Canon RAW files, found $RAW_COUNT" >&2
  exit 1
fi

python "$DOWNLOADER" \
  --output-dir "$ASSETS" \
  --dataset-assets Canon_gt.zip \
  --checkpoint-assets
unzip -tq "$ASSETS/Canon_gt.zip"
mapfile -t GT_FILES < <(awk 'NF >= 2 {print $2}' "$MANIFEST")
unzip -q "$ASSETS/Canon_gt.zip" "${GT_FILES[@]}" -d "$DATA_ROOT"
GT_COUNT="$(find "$DATA_ROOT/gt" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
if [[ "$GT_COUNT" -ne 1000 ]]; then
  echo "Expected 1000 manifest-selected Canon GT images, found $GT_COUNT" >&2
  exit 1
fi

python "$DOWNLOADER" \
  --output-dir "$ASSETS" \
  --dataset-assets \
  --checkpoint-assets canon_0.0001.pt
mv -n "$ASSETS/canon_0.0001.pt" "$CHECKPOINT_DIR/canon_0.0001.pt"

# Downloaded archives stay in assets/; move them to verified external storage
# manually if space is needed. This script never removes original datasets.

python - "$DATA_ROOT" "$CHECKPOINT_DIR/canon_0.0001.pt" <<'PY'
import json
import sys
from pathlib import Path

data_root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
print(
    json.dumps(
        {
            "data_root": str(data_root),
            "raw_count": len(list((data_root / "0.0001").glob("*.raw"))),
            "gt_count": len(list((data_root / "gt").glob("*.jpg"))),
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
        },
        indent=2,
    )
)
PY
