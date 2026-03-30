#!/usr/bin/env bash
set -euo pipefail

cd /home/dell7960/桌面/FedOOD

MODELS=(
  "/home/dell7960/桌面/OOD_model/plankton54_resnet50_base_e100_lr0.1_default/plankton54_resnet50_base_e100_lr0.1_default__s2__best_epoch88_acc0.9553.ckpt"
  "/home/dell7960/桌面/OOD_model/plankton54_resnet101_base_e100_lr0.1_default/plankton54_resnet101_base_e100_lr0.1_default__s0__best_epoch95_acc0.9563.ckpt"
  "/home/dell7960/桌面/OOD_model/plankton54_resnet152_base_e100_lr0.1_default/plankton54_resnet152_base_e100_lr0.1_default__s0__best_epoch88_acc0.9606.ckpt"
  "/home/dell7960/桌面/OOD_model/plankton54_densenet169_base_e100_lr0.1_default/plankton54_densenet169_base_e100_lr0.1_default__s0__best_epoch98_acc0.9642.ckpt"
  "/home/dell7960/桌面/OOD_model/plankton54_vit-b-16_base_e100_lr0.1_default/plankton54_vit-b-16_base_e100_lr0.1_default__s2__best_epoch98_acc0.9023.ckpt"
)

for ckpt in "${MODELS[@]}"; do
  base=$(basename "$ckpt")
  model_dir=${base%%__s*}
  out_dir="/home/dell7960/桌面/OOD_model/${model_dir}"
  mkdir -p "$out_dir"

  if [ ! -f "$out_dir/${base%.ckpt}__scr_ood.json" ]; then
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 KMP_USE_SHM=0 \
      python3 paper_tools/evaluate_scr_ood_desktop_models.py \
      --method scr_ood \
      --checkpoint "$ckpt" \
      --data_root ./Plankton_OOD_Dataset \
      --output_dir "$out_dir" \
      --device cuda:0 \
      --batch_size 32 \
      --num_workers 4 \
      --target_variance_ratio 0.95
  fi

  if [ ! -f "$out_dir/${base%.ckpt}__vim.json" ]; then
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 KMP_USE_SHM=0 \
      python3 paper_tools/evaluate_scr_ood_desktop_models.py \
      --method vim \
      --checkpoint "$ckpt" \
      --data_root ./Plankton_OOD_Dataset \
      --output_dir "$out_dir" \
      --device cuda:0 \
      --batch_size 32 \
      --num_workers 4
  fi
done
