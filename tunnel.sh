#!/bin/zsh
# SSH 隧道（自动重连版）：把服务器上的 vLLM(8010) 和 RAGFlow(19380) 转发到本地同名端口
# 用法: ./tunnel.sh    （Ctrl+C 退出；断线后 3 秒自动重连）
# 验证: curl http://127.0.0.1:8010/v1/models

 Forward() {
  echo "[$(date '+%H:%M:%S')] 隧道连接中..."
  ssh -N \
    -L 8010:127.0.0.1:8010 \
    -L 19380:127.0.0.1:19380 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o ConnectTimeout=10 \
    durian-server
}

while true; do
  Forward
  code=$?
  # Ctrl+C (130) 时直接退出，不重连
  if [ $code -eq 130 ] || [ $code -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] 隧道已退出"
    break
  fi
  echo "[$(date '+%H:%M:%S')] 隧道断开(exit=$code)，3 秒后重连..."
  sleep 3
done
