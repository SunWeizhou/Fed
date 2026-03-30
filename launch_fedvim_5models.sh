#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOCAL_LOG_ROOT="${PROJECT_ROOT}/logs_fedvim_5gpu_${TIMESTAMP}"
WINDOWS_HOST="dell@10.24.1.131"
WINDOWS_PYTHON_BIN="${WINDOWS_PYTHON_BIN:-/c/Users/DELL/anaconda3/envs/dlsr/python.exe}"
mkdir -p "${LOCAL_LOG_ROOT}"

resolve_remote_root() {
  local host="$1"
  if [[ "${host}" == "${WINDOWS_HOST}" ]]; then
    ssh "${host}" "cmd /c mkdir %USERPROFILE%\\Desktop\\FedViM >nul 2>nul || exit 0" >/dev/null
    printf "%s" "/c/Users/DELL/Desktop/FedViM"
  else
    ssh "${host}" "bash -lc 'if [ -d \"\$HOME/桌面\" ]; then root=\"\$HOME/桌面/FedViM\"; else root=\"\$HOME/Desktop/FedViM\"; fi; mkdir -p \"\$root\"; printf \"%s\" \"\$root\"'"
  fi
}

launch_local() {
  local model="$1"
  local device="$2"
  local log_file="${LOCAL_LOG_ROOT}/${model}_$(echo "${device}" | tr ':' '_').log"
  nohup bash "${PROJECT_ROOT}/run_fedvim_model_pipeline.sh" "${model}" "${device}" "${PROJECT_ROOT}" > "${log_file}" 2>&1 < /dev/null &
  echo "[launch][local] ${model} on ${device} -> pid=$! log=${log_file}"
}

launch_remote() {
  local host="$1"
  local model="$2"
  local device="$3"
  local remote_root
  remote_root="$(resolve_remote_root "${host}")"
  local remote_log_root="${remote_root}/logs_fedvim_5gpu_${TIMESTAMP}"
  local remote_log_file="${remote_log_root}/${model}_$(echo "${device}" | tr ':' '_').log"

  if [[ "${host}" == "${WINDOWS_HOST}" ]]; then
    ssh "${host}" bash -lc "\"mkdir -p '${remote_log_root}' && cd '${remote_root}' && PYTHON_BIN='${WINDOWS_PYTHON_BIN}' nohup bash './run_fedvim_model_pipeline.sh' '${model}' '${device}' '${remote_root}' > '${remote_log_file}' 2>&1 < /dev/null &\""
  else
    ssh "${host}" "bash -lc \"mkdir -p '${remote_log_root}' && cd '${remote_root}' && nohup bash './run_fedvim_model_pipeline.sh' '${model}' '${device}' '${remote_root}' > '${remote_log_file}' 2>&1 < /dev/null &\""
  fi
  echo "[launch][remote] ${host} ${model} on ${device} -> log=${remote_log_file}"
}

echo "[launch] starting five-model FedViM training/evaluation jobs"
echo "[launch] local logs: ${LOCAL_LOG_ROOT}"

launch_local "densenet169" "cuda:0"
launch_local "efficientnet_v2_s" "cuda:1"

launch_remote "dell7960@10.4.47.203" "resnet50" "cuda:0"
launch_remote "dell7960@10.4.47.203" "mobilenetv3_large" "cuda:1"
launch_remote "dell@10.24.1.131" "resnet101" "cuda:1"

echo "[launch] all jobs submitted"
