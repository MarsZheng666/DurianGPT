#!/bin/zsh
# SSH 隧道：把服务器上的 vLLM(8010) 和 RAGFlow(19380) 转发到本地同名端口
# 用法: ./tunnel.sh   （Ctrl+C 断开）
echo "[tunnel] 127.0.0.1:8010 -> durian-server:8010 (vLLM OpenAI API)"
echo "[tunnel] 127.0.0.1:19380 -> durian-server:19380 (RAGFlow)"
ssh -N \
  -L 8010:127.0.0.1:8010 \
  -L 19380:127.0.0.1:19380 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  durian-server
