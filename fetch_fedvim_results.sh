#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
LOCAL_OUTPUT_ROOT="${PROJECT_ROOT}/experiments/experiments_rerun_v1"
WINDOWS_HOST="dell@10.24.1.131"
mkdir -p "${LOCAL_OUTPUT_ROOT}"

resolve_remote_root() {
  local host="$1"
  if [[ "${host}" == "${WINDOWS_HOST}" ]]; then
    printf "%s" "/c/Users/DELL/Desktop/FedViM"
  else
    ssh "${host}" "bash -lc 'if [ -d \"\$HOME/桌面\" ]; then root=\"\$HOME/桌面/FedViM\"; else root=\"\$HOME/Desktop/FedViM\"; fi; printf \"%s\" \"\$root\"'"
  fi
}

remote_has_rsync() {
  local host="$1"
  if [[ "${host}" == "${WINDOWS_HOST}" ]]; then
    return 1
  fi
  ssh "${host}" "bash -lc 'command -v rsync >/dev/null 2>&1'"
}

fetch_model_dir() {
  local host="$1"
  local model="$2"
  local remote_root
  remote_root="$(resolve_remote_root "${host}")"
  mkdir -p "${LOCAL_OUTPUT_ROOT}/${model}"
  if [[ "${host}" == "${WINDOWS_HOST}" ]]; then
    scp -r "${host}:Desktop/FedViM/experiments/experiments_rerun_v1/${model}/"* "${LOCAL_OUTPUT_ROOT}/${model}/"
    echo "[fetch] ${host} -> ${model}"
    return
  fi
  if remote_has_rsync "${host}"; then
    rsync -az "${host}:${remote_root}/experiments/experiments_rerun_v1/${model}/" "${LOCAL_OUTPUT_ROOT}/${model}/"
  else
    ssh "${host}" "bash -lc \"cd '${remote_root}' && tar -cf - 'experiments/experiments_rerun_v1/${model}'\"" | tar -xf - -C "${PROJECT_ROOT}"
  fi
  echo "[fetch] ${host} -> ${model}"
}

fetch_model_dir "dell@10.24.1.131" "resnet101"
fetch_model_dir "dell7960@10.4.47.203" "resnet50"
fetch_model_dir "dell7960@10.4.47.203" "mobilenetv3_large"

echo "[fetch] remote results copied to ${LOCAL_OUTPUT_ROOT}"
