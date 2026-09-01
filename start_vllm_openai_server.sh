#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="/home/admin01/桌面/Desktop/durian-training"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

BASE_MODEL="/home/admin01/durian_qwen3_14b_awq"
LORA_MODEL="/home/admin01/下载/ToDesk/outputs/durian_qwen3_14b_qlora_v2_industry_final"

BIND_HOST="127.0.0.1"
BIND_PORT="8010"
MAX_MODEL_LEN="${DURIAN_MAX_MODEL_LEN:-6144}"
GPU_MEMORY_UTILIZATION="${DURIAN_GPU_MEMORY_UTILIZATION:-0.90}"

cd "${PROJECT_DIR}"

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"

source "/home/admin01/miniconda3/etc/profile.d/conda.sh"
conda activate durian

echo "[vLLM] Project dir: ${PROJECT_DIR}"
echo "[vLLM] Base model: ${BASE_MODEL}"
echo "[vLLM] LoRA model: ${LORA_MODEL}"
echo "[vLLM] Listen: ${BIND_HOST}:${BIND_PORT}"
echo "[vLLM] Log: ${LOG_DIR}/vllm_openai.log"

pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true

nohup python -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL}" \
  --served-model-name "durian-base" \
  --host "${BIND_HOST}" \
  --port "${BIND_PORT}" \
  --dtype float16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs 4 \
  --trust-remote-code \
  --enforce-eager \
  --disable-custom-all-reduce \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules "durian-lora=${LORA_MODEL}" \
  > "${LOG_DIR}/vllm_openai.log" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_DIR}/vllm_openai.pid"

echo "[vLLM] Started with PID=${PID}"
echo "[vLLM] Wait 20-60 seconds, then test:"
echo "curl http://127.0.0.1:8010/v1/models"
