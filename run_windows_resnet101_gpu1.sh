#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${1:-$(cd "$(dirname "$0")" && pwd)}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs_fedvim_5gpu_20260330_193900}"
LOG_FILE="${LOG_ROOT}/resnet101_cuda_1.log"

export PYTHON_BIN="${PYTHON_BIN:-/c/Users/DELL/anaconda3/envs/dlsr/python.exe}"
export DATA_ROOT="${DATA_ROOT:-/c/Users/DELL/Desktop/FedOOD/Plankton_OOD_Dataset}"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

exec bash ./run_fedvim_model_pipeline.sh resnet101 cuda:1 "${PROJECT_ROOT}" >> "${LOG_FILE}" 2>&1
