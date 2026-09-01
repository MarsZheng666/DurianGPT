#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/admin01/桌面/Desktop/durian-training"
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cp "$PKG_DIR/durian__inference_api_true_stream.py" "$ROOT/durian__inference_api.py"
cp "$PKG_DIR/App_true_stream.vue" "$ROOT/frontend-vue/src/App.vue"
cp "$PKG_DIR/style_true_stream.css" "$ROOT/frontend-vue/src/style.css"
cp "$PKG_DIR/start_vllm_openai_server.sh" "$ROOT/start_vllm_openai_server.sh"
chmod +x "$ROOT/start_vllm_openai_server.sh"

ENV_FILE="$ROOT/.env"
touch "$ENV_FILE"
add_env() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" "$ENV_FILE"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
  else
    echo "${key}=${value}" >> "$ENV_FILE"
  fi
}

add_env "DURIAN_ENGINE_MODE" "openai_proxy"
add_env "DURIAN_OPENAI_BASE_URL" "http://127.0.0.1:8010/v1"
add_env "DURIAN_OPENAI_MODEL" "durian-lora"
add_env "DURIAN_PROXY_DISABLE_THINKING" "1"

cd "$ROOT/frontend-vue"
npm run build

echo "[update] true streaming files applied."
echo "[update] start vLLM first: bash $ROOT/start_vllm_openai_server.sh"
echo "[update] then start DurianGPT backend/frontend with your launcher."
