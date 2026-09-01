#!/bin/zsh
# 本地启动榴莲GPT推理服务（openai_proxy 模式，通过 tunnel.sh 连远端 vLLM）
# 前置：先运行 ./tunnel.sh，确认服务器上 vLLM(8010) 在跑
cd "$(dirname "$0")"
set -a
source .env
set +a
exec .venv/bin/python -m uvicorn durian__inference_api:app --host 127.0.0.1 --port 8001 --workers 1
