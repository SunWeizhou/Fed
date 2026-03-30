#!/bin/bash
set -euo pipefail

################################################################################
# ACT-FedViM 9 个主模型双卡批量训练脚本
# 设计目标：
# 1. 两张 3090 并行，各自串行跑一条模型队列
# 2. 优先把实验跑完，完整 OOD 评估降频到每 10 轮一次
# 3. 其余 batch size / accumulation / warmup 交给 train_federated.py 自动配置
################################################################################

DATA_ROOT="${DATA_ROOT:-./Plankton_OOD_Dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-./experiments/experiments_rerun_v1}"
N_CLIENTS="${N_CLIENTS:-5}"
ALPHA="${ALPHA:-0.1}"
IMAGE_SIZE="${IMAGE_SIZE:-320}"
COMM_ROUNDS="${COMM_ROUNDS:-50}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-4}"
SEED="${SEED:-42}"
EVAL_FREQUENCY="${EVAL_FREQUENCY:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OMP_THREADS="${OMP_THREADS:-8}"

LOG_ROOT="${LOG_ROOT:-./logs_2x3090_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_ROOT"
mkdir -p "$OUTPUT_DIR"

GPU0_MODELS=(
  "convnext_base"
  "vit_b_16"
  "resnet50"
  "mobilenetv3_large"
  "efficientnet_v2_s"
)

GPU1_MODELS=(
  "deit_base"
  "vit_b_32"
  "resnet101"
  "densenet169"
)

cleanup() {
  jobs -pr | xargs -r kill
}

trap cleanup INT TERM

run_model_queue() {
  local gpu_id="$1"
  shift
  local models=("$@")

  for model in "${models[@]}"; do
    local log_file="$LOG_ROOT/${model}_gpu${gpu_id}.log"
    local model_image_size="${IMAGE_SIZE}"

    case "${model}" in
      vit_b_16|vit_b_32|deit_base|swin_t)
        model_image_size=224
        ;;
    esac

    echo "[$(date '+%F %T')] [GPU ${gpu_id}] 启动 ${model}"
    echo "[$(date '+%F %T')] [GPU ${gpu_id}] 日志: ${log_file}"
    echo "[$(date '+%F %T')] [GPU ${gpu_id}] 图像尺寸: ${model_image_size}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    PYTHONUNBUFFERED=1 \
    KMP_USE_SHM=0 \
    OMP_NUM_THREADS="${OMP_THREADS}" \
    MKL_NUM_THREADS="${OMP_THREADS}" \
    python3 train_federated.py \
      --model_type "${model}" \
      --use_fedvim \
      --data_root "${DATA_ROOT}" \
      --output_dir "${OUTPUT_DIR}" \
      --n_clients "${N_CLIENTS}" \
      --alpha "${ALPHA}" \
      --image_size "${model_image_size}" \
      --communication_rounds "${COMM_ROUNDS}" \
      --local_epochs "${LOCAL_EPOCHS}" \
      --seed "${SEED}" \
      --eval_frequency "${EVAL_FREQUENCY}" \
      --num_workers "${NUM_WORKERS}" \
      > "${log_file}" 2>&1

    echo "[$(date '+%F %T')] [GPU ${gpu_id}] 完成 ${model}"
  done
}

echo "======================================"
echo "ACT-FedViM 双卡主实验启动"
echo "======================================"
echo "开始时间: $(date)"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志目录: ${LOG_ROOT}"
echo "数据目录: ${DATA_ROOT}"
echo "客户端数: ${N_CLIENTS}"
echo "Dirichlet alpha: ${ALPHA}"
echo "图像尺寸: ${IMAGE_SIZE}"
echo "通信轮次: ${COMM_ROUNDS}"
echo "本地轮次: ${LOCAL_EPOCHS}"
echo "随机种子: ${SEED}"
echo "完整 OOD 评估频率: 每 ${EVAL_FREQUENCY} 轮"
echo "DataLoader workers: ${NUM_WORKERS}"
echo ""
echo "GPU 0 队列: ${GPU0_MODELS[*]}"
echo "GPU 1 队列: ${GPU1_MODELS[*]}"
echo "======================================"

run_model_queue 0 "${GPU0_MODELS[@]}" &
PID0=$!

run_model_queue 1 "${GPU1_MODELS[@]}" &
PID1=$!

wait "${PID0}"
wait "${PID1}"

echo ""
echo "======================================"
echo "9 个主模型训练完成"
echo "结束时间: $(date)"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志目录: ${LOG_ROOT}"
echo "======================================"
