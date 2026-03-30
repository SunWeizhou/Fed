#!/bin/bash
set -euo pipefail

HOST="${HOST:-dell@10.24.1.131}"
REMOTE_ROOT="${REMOTE_ROOT:-/c/Users/DELL/Desktop/FedViM}"
REMOTE_DATASET="${REMOTE_ROOT}/Plankton_OOD_Dataset"
REMOTE_LOG_ROOT="${REMOTE_ROOT}/logs_fedvim_5gpu_watch_$(date +%Y%m%d_%H%M%S)"
REMOTE_LOG_FILE="${REMOTE_LOG_ROOT}/resnet101_cuda_1.log"
WINDOWS_PYTHON_BIN="${WINDOWS_PYTHON_BIN:-/c/Users/DELL/anaconda3/envs/dlsr/python.exe}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
TARGET_KB="$(du -sk /home/dell7960/桌面/FedOOD/Plankton_OOD_Dataset | awk '{print $1}')"

remote_dataset_kb() {
  ssh "${HOST}" bash -lc "\"du -sk '${REMOTE_DATASET}' 2>/dev/null | cut -f1\""
}

echo "[watch] waiting for Windows dataset sync on ${HOST}"
echo "[watch] target size: ${TARGET_KB} KB"

while true; do
  current_kb="$(remote_dataset_kb || true)"
  current_kb="${current_kb:-0}"
  echo "[watch] current remote size: ${current_kb} KB"
  if [[ "${current_kb}" -ge "${TARGET_KB}" ]]; then
    break
  fi
  sleep "${CHECK_INTERVAL}"
done

echo "[watch] dataset sync complete, launching resnet101 on cuda:1"
ssh "${HOST}" bash -lc "\"mkdir -p '${REMOTE_LOG_ROOT}' && cd '${REMOTE_ROOT}' && PYTHON_BIN='${WINDOWS_PYTHON_BIN}' nohup bash './run_fedvim_model_pipeline.sh' 'resnet101' 'cuda:1' '${REMOTE_ROOT}' > '${REMOTE_LOG_FILE}' 2>&1 < /dev/null &\""
echo "[watch] launched -> ${REMOTE_LOG_FILE}"
