#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/桌面/FedOOD"
LOG="$REPO/experiments/queue_final_linux_gpu0.log"

mkdir -p "$REPO/experiments"
exec > >(tee -a "$LOG") 2>&1

cd "$REPO"

timestamp() {
  date '+%F %T'
}

run_step() {
  local step="$1"
  local label="$2"
  shift 2
  echo "[$(timestamp)] [$step] START $label"
  "$@"
  echo "[$(timestamp)] [$step] END $label"
}

echo "[$(timestamp)] queue start: linux gpu0"

run_step "1/2" "FedAvg resnet101" \
  python3 train_federated.py \
  --model_type resnet101 \
  --data_root ./Plankton_OOD_Dataset \
  --communication_rounds 50 \
  --local_epochs 4 \
  --batch_size 32 \
  --image_size 320 \
  --n_clients 5 \
  --alpha 0.1 \
  --seed 42 \
  --device cuda:0 \
  --num_workers 8 \
  --output_dir ./experiments/fedavg_manifest_final_v1

run_step "2/2" "FedAvg densenet169" \
  python3 train_federated.py \
  --model_type densenet169 \
  --data_root ./Plankton_OOD_Dataset \
  --communication_rounds 50 \
  --local_epochs 4 \
  --batch_size 32 \
  --image_size 320 \
  --n_clients 5 \
  --alpha 0.1 \
  --seed 42 \
  --device cuda:0 \
  --num_workers 8 \
  --output_dir ./experiments/fedavg_manifest_final_v1

echo "[$(timestamp)] queue finished: linux gpu0"
