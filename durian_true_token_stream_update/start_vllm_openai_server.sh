#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/admin01/桌面/Desktop/durian-training"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate durian
fi

# 只杀独立 vLLM OpenAI Server，不杀 FastAPI 业务后端。
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

BASE_MODEL="${DURIAN_BASE_MODEL:-/home/admin01/durian_qwen3_14b_awq}"
LORA_MODEL="${DURIAN_LORA_MODEL_PATH:-/home/admin01/下载/ToDesk/outputs/durian_qwen3_14b_qlora_v2_industry_final}"
MAX_MODEL_LEN="${DURIAN_MAX_MODEL_LEN:-6144}"
GPU_UTIL="${DURIAN_GPU_MEMORY_UTILIZATION:-0.90}"
PORT="${DURIAN_VLLM_OPENAI_PORT:-8010}"

nohup python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name durian-base \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype float16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-num-seqs 1 \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules durian-lora="$LORA_MODEL" \
  > "$LOG_DIR/vllm_openai.log" 2>&1 &

echo "[vLLM] OpenAI server starting on 127.0.0.1:$PORT"
echo "[vLLM] log: $LOG_DIR/vllm_openai.log"
echo "[vLLM] test after ready: curl http://127.0.0.1:$PORT/v1/models"
