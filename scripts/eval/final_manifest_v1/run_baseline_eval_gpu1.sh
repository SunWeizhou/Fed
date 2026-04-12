#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

fedln_checkpoints=(
  "experiments/fedln_manifest_final_v1/mobilenetv3_large/experiment_20260411_225013/best_model.pth"
  "experiments/fedln_manifest_final_v1/resnet50/experiment_20260412_004716/best_model.pth"
  "experiments/fedln_manifest_final_v1/resnet101/experiment_20260411_225013/best_model.pth"
  "experiments/fedln_manifest_final_v1/densenet169/experiment_20260412_011359/best_model.pth"
  "experiments/fedln_manifest_final_v1/efficientnet_v2_s/experiment_20260412_041553/best_model.pth"
)

foster_checkpoints=(
  "experiments/foster_manifest_final_v1/mobilenetv3_large/experiment_20260412_014037/best_model.pth"
  "experiments/foster_manifest_final_v1/resnet50/experiment_20260412_022431/best_model.pth"
  "experiments/foster_manifest_final_v1/resnet101/experiment_20260411_225015/best_model.pth"
  "experiments/foster_manifest_final_v1/densenet169/experiment_20260411_225015/best_model.pth"
  "experiments/foster_manifest_final_v1/efficientnet_v2_s/experiment_20260412_065956/best_model.pth"
)

for ckpt in "${fedln_checkpoints[@]}"; do
  python3 evaluate_fedln.py --checkpoint "$ckpt" --data_root ./Plankton_OOD_Dataset --device cuda:1 --batch_size 64 --num_workers 8 --evaluation_score msp
done

for ckpt in "${foster_checkpoints[@]}"; do
  python3 evaluate_foster.py --checkpoint "$ckpt" --data_root ./Plankton_OOD_Dataset --device cuda:1 --batch_size 64 --num_workers 8 --evaluation_score msp
done
