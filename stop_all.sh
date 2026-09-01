#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT/.run"

stop_pid_file() {
  local name="$1"
  local pid_file="$2"

  if [[ ! -f "$pid_file" ]]; then
    echo "[DurianGPT] $name is not running: no pid file."
    return
  fi

  local pid
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "[DurianGPT] Stopping $name, pid $pid ..."
    kill "$pid" >/dev/null 2>&1 || true
    sleep 2
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "[DurianGPT] $name still running, forcing stop..."
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  else
    echo "[DurianGPT] $name is not running."
  fi

  rm -f "$pid_file"
}

stop_pid_file "frontend" "$RUN_DIR/frontend.pid"
stop_pid_file "backend" "$RUN_DIR/backend.pid"

echo "[DurianGPT] Stopped."
