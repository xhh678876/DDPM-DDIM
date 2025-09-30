#!/usr/bin/env bash
# ================================================================
# DDPM Lab1 单卡（GPU 2）串行训练脚本 —— 只跑 x0 预测器
# 任务： (linear, quad, cosine) × x0
# ================================================================
set -uo pipefail

ENV_NAME="${ENV_NAME:-ddpm}"
REPO_DIR="${REPO_DIR:-/data/XHH/DDPM/Lab1-DDPM/image_diffusion_todo}"
LOG_DIR="${REPO_DIR}/results/logs"
mkdir -p "$LOG_DIR"

# 固定 GPU 2
export CUDA_VISIBLE_DEVICES=2

timestamp() { date +"%m-%d-%H%M%S"; }

activate_env() {
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
  fi
}

train_once() {
  local mode="$1"
  local predictor="x0"
  local tag="${mode}_${predictor}"
  local log="${LOG_DIR}/train_${tag}_$(timestamp).log"

  echo "=============================================================="
  echo ">>> [TRAIN] mode=${mode}, predictor=${predictor}, GPU=${CUDA_VISIBLE_DEVICES}"
  echo "    日志：${log}"
  echo "=============================================================="

  activate_env
  cd "${REPO_DIR}"
  python train.py --mode "${mode}" --predictor "${predictor}" \
    2>&1 | tee "${log}"
  echo ">>> [OK] 完成 ${tag}"
}

# 只跑三个 mode
modes=(linear quad cosine)

for m in "${modes[@]}"; do
  train_once "${m}"
done

echo "=============================================================="
echo ">>> 所有 x0 任务完成，日志目录：${LOG_DIR}"
echo "=============================================================="
