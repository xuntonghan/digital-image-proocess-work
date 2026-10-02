#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/root/autodl-tmp/cv-homework"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader

echo "== Active compute processes =="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

echo "== Data disk =="
df -h /root/autodl-tmp

echo "== Python =="
python3 --version

echo "== Workspace =="
if [[ -d "${WORK_ROOT}" ]]; then
  du -sh "${WORK_ROOT}"
else
  echo "Workspace has not been created yet: ${WORK_ROOT}"
fi
