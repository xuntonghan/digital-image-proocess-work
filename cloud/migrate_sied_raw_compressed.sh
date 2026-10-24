#!/usr/bin/env bash
# Canon RAWs compress well; avoid sending 16.6 GB of nearly-black RAW verbatim.
set -euo pipefail
ROOT="${1:-/root/autodl-tmp/sied30_eval1000}"
mkdir -p "$ROOT/data" "$ROOT/logs"
exec 9>"$ROOT/logs/raw_stream.lock"
flock -n 9 || exit 1
ssh -i /root/.ssh/sied_eval20261024_ed25519 -p 18868 \
  -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=8 root@connect.nmb1.seetacloud.com \
  'set -o pipefail; tar -C /root/autodl-tmp/sied30/data -cf - SIED_dataset/0.0001 | /root/miniconda3/bin/zstd -1 -T1 -q' \
  | /root/miniconda3/bin/zstd -d -q | tar -C "$ROOT/data" -xf -
printf '%s RAW_STREAM_COMPLETE\n' "$(date -Is)"
