#!/usr/bin/env bash
# Copy only project-one evaluation inputs. Never delete source or destination data.
set -euo pipefail
ROOT="${1:-/root/autodl-tmp/sied30_eval1000}"
SOURCE_HOST=connect.nmb1.seetacloud.com
SOURCE_PORT=18868
SOURCE_ROOT=/root/autodl-tmp/sied30
KEY=/root/.ssh/sied_eval20261024_ed25519
SSH="ssh -i $KEY -p $SOURCE_PORT -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o StrictHostKeyChecking=accept-new"
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/logs/migration.lock"
flock -n 9 || { echo 'Migration already running' >&2; exit 1; }
VENV=/root/autodl-tmp/cv-homework/.venv
mkdir -p "$VENV"
rsync -a --partial -e "$SSH" "root@$SOURCE_HOST:$VENV/" "$VENV/"
for relative in \
  third_party/SIED-official \
  checkpoints/official \
  final \
  runs/canon_raw_guided_naf_phase4_val_native \
  data/SIED_dataset/gt \
  data/SIED_dataset/0.0001; do
  printf '%s copy %s\n' "$(date -Is)" "$relative"
  mkdir -p "$ROOT/$relative"
  # Do not race extraction from the optional compressed RAW transfer.
  if [[ "$relative" == data/SIED_dataset/0.0001 ]]; then
    exec 8>"$ROOT/logs/raw_stream.lock"
    flock 8
  fi
  rsync -a --partial --info=progress2 -e "$SSH" \
    "root@$SOURCE_HOST:$SOURCE_ROOT/$relative/" "$ROOT/$relative/"
  printf '%s checksum %s\n' "$(date -Is)" "$relative"
  CHECK="$ROOT/logs/verify_$(printf '%s' "$relative" | tr / _).txt"
  rsync -rcni --out-format='%i %n' -e "$SSH" \
    "root@$SOURCE_HOST:$SOURCE_ROOT/$relative/" "$ROOT/$relative/" > "$CHECK"
  if [[ -s "$CHECK" ]]; then
    echo "Checksum differences: $CHECK" >&2
    exit 1
  fi
  if [[ "$relative" == data/SIED_dataset/0.0001 ]]; then
    flock -u 8
  fi
done
printf '%s MIGRATION_VERIFIED\n' "$(date -Is)"
touch "$ROOT/logs/MIGRATION_VERIFIED"
