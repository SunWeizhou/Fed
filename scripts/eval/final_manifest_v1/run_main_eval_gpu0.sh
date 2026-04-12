#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

checkpoints=(
  "experiments/fedavg_manifest_final_v1/mobilenetv3_large/experiment_20260411_225014/best_model.pth"
  "experiments/fedavg_manifest_final_v1/resnet50/experiment_20260412_034104/best_model.pth"
  "experiments/fedavg_manifest_final_v1/efficientnet_v2_s/experiment_20260412_024911/best_model.pth"
  "experiments/fedavg_manifest_final_v1/resnet101/experiment_20260411_225014/best_model.pth"
  "experiments/fedavg_manifest_final_v1/densenet169/experiment_20260412_035859/best_model.pth"
)

for ckpt in "${checkpoints[@]}"; do
  python3 evaluate_fedvim.py --checkpoint "$ckpt" --data_root ./Plankton_OOD_Dataset --device cuda:0 --batch_size 64 --num_workers 8
  python3 advanced_fedvim.py --checkpoint "$ckpt" --data_root ./Plankton_OOD_Dataset --device cuda:0 --batch_size 64 --num_workers 8
  python3 evaluate_pooled_vim.py --checkpoint "$ckpt" --data_root ./Plankton_OOD_Dataset --device cuda:0 --batch_size 64 --num_workers 8
  python3 evaluate_pooled_act_vim.py --checkpoint "$ckpt" --data_root ./Plankton_OOD_Dataset --device cuda:0 --batch_size 64 --num_workers 8
  python3 evaluate_baselines.py --checkpoint "$ckpt" --data_root ./Plankton_OOD_Dataset --device cuda:0 --batch_size 64 --num_workers 8 --methods msp energy
done
