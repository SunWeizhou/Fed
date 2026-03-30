#!/bin/bash
set -euo pipefail

MODEL_TYPE="${1:?Usage: run_fedvim_model_pipeline.sh <model_type> <device> [project_root]}"
DEVICE="${2:?Usage: run_fedvim_model_pipeline.sh <model_type> <device> [project_root]}"
PROJECT_ROOT="${3:-$(pwd)}"

cd "${PROJECT_ROOT}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
else
  echo "[FedViM pipeline] Python interpreter not found" >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/Plankton_OOD_Dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/experiments/experiments_rerun_v1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
N_CLIENTS="${N_CLIENTS:-5}"
ALPHA="${ALPHA:-0.1}"
IMAGE_SIZE="${IMAGE_SIZE:-320}"
COMM_ROUNDS="${COMM_ROUNDS:-50}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-4}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-10}"
OMP_THREADS="${OMP_THREADS:-8}"

export PYTHONUNBUFFERED=1
export KMP_USE_SHM=0
export OMP_NUM_THREADS="${OMP_THREADS}"
export MKL_NUM_THREADS="${OMP_THREADS}"

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON_BIN}" train_federated.py \
  --model_type "${MODEL_TYPE}" \
  --use_fedvim \
  --data_root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --output_dir "${OUTPUT_ROOT}" \
  --n_clients "${N_CLIENTS}" \
  --alpha "${ALPHA}" \
  --image_size "${IMAGE_SIZE}" \
  --communication_rounds "${COMM_ROUNDS}" \
  --local_epochs "${LOCAL_EPOCHS}" \
  --batch_size "${TRAIN_BATCH_SIZE}" \
  --save_frequency "${SAVE_FREQUENCY}" \
  --seed "${SEED}" \
  --num_workers "${NUM_WORKERS}"

EXPERIMENT_DIR="$(ls -td "${OUTPUT_ROOT}/${MODEL_TYPE}"/experiment_* | head -1)"
CHECKPOINT_PATH="${EXPERIMENT_DIR}/best_model.pth"

"${PYTHON_BIN}" evaluate_fedvim.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --data_root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

"${PYTHON_BIN}" evaluate_act_fedvim.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --data_root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

"${PYTHON_BIN}" evaluate_baselines.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --data_root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --methods msp energy

echo "[FedViM pipeline] Completed ${MODEL_TYPE} on ${DEVICE}"
