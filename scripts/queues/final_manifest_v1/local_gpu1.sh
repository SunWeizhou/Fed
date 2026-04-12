#!/usr/bin/env bash
set -euo pipefail

REPO="/home/dell7960/桌面/FedOOD"
LOG="$REPO/experiments/queue_final_local_gpu1.log"

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

echo "[$(timestamp)] queue start: local gpu1"

run_step "1/3" "FedLN mobilenetv3_large" \
  python3 train_fedln.py \
  --model_type mobilenetv3_large \
  --data_root ./Plankton_OOD_Dataset \
  --communication_rounds 50 \
  --local_epochs 4 \
  --batch_size 32 \
  --image_size 320 \
  --n_clients 5 \
  --alpha 0.1 \
  --seed 42 \
  --device cuda:1 \
  --num_workers 8 \
  --output_dir ./experiments/fedln_manifest_final_v1

run_step "2/3" "FedLN resnet50" \
  python3 train_fedln.py \
  --model_type resnet50 \
  --data_root ./Plankton_OOD_Dataset \
  --communication_rounds 50 \
  --local_epochs 4 \
  --batch_size 32 \
  --image_size 320 \
  --n_clients 5 \
  --alpha 0.1 \
  --seed 42 \
  --device cuda:1 \
  --num_workers 8 \
  --output_dir ./experiments/fedln_manifest_final_v1

run_step "3/3" "FedAvg efficientnet_v2_s" \
  python3 train_federated.py \
  --model_type efficientnet_v2_s \
  --data_root ./Plankton_OOD_Dataset \
  --communication_rounds 50 \
  --local_epochs 4 \
  --batch_size 32 \
  --image_size 320 \
  --n_clients 5 \
  --alpha 0.1 \
  --seed 42 \
  --device cuda:1 \
  --num_workers 8 \
  --output_dir ./experiments/fedavg_manifest_final_v1

echo "[$(timestamp)] queue finished: local gpu1"
