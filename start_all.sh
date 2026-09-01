#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-durian}"

if [[ "${DURIAN_NO_PAUSE:-0}" != "1" ]]; then
  trap 'status=$?; if [[ $status -ne 0 ]]; then echo; echo "[$APP_NAME] Failed with exit code $status"; echo "Check logs or the error above."; read -r -p "Press Enter to close..."; fi' EXIT
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT/frontend-vue"
LOG_DIR="$ROOT/logs"
RUN_DIR="$ROOT/.run"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

export DURIAN_API_KEY="${DURIAN_API_KEY:-${API_KEY:-change-me}}"
export VITE_DURIAN_API_KEY="${VITE_DURIAN_API_KEY:-$DURIAN_API_KEY}"
export QWEN_VL_MAX_API_KEY="${QWEN_VL_MAX_API_KEY:-}"
export DURIAN_APP_USERS="${DURIAN_APP_USERS:-admin,admin2}"

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8001}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${FRONTEND_PORT:-5000}"
DURIAN_CONDA_ENV="${DURIAN_CONDA_ENV:-durian}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

load_conda() {
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_base/etc/profile.d/conda.sh"
      return 0
    fi
  fi

  local candidate
  for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$candidate" ]]; then
      # shellcheck disable=SC1090
      source "$candidate"
      return 0
    fi
  done

  return 1
}

activate_durian_env() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$DURIAN_CONDA_ENV" ]]; then
    return 0
  fi

  if load_conda; then
    set +u
    conda activate "$DURIAN_CONDA_ENV"
    local rc=$?
    set -u
    return "$rc"
  fi

  if command -v activate >/dev/null 2>&1; then
    # Some servers expose the old-style activate command.
    # shellcheck disable=SC1091
    set +u
    source activate "$DURIAN_CONDA_ENV"
    local rc=$?
    set -u
    return "$rc"
  fi

  echo "[ERROR] Could not load conda. Please run this once in your shell:"
  echo "  conda init bash"
  echo "Then reopen the terminal and run:"
  echo "  conda activate $DURIAN_CONDA_ENV"
  echo "  bash start_all.sh"
  return 1
}

is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1
}

echo "[$APP_NAME] Project: $ROOT"

echo "[$APP_NAME] Activating conda environment: $DURIAN_CONDA_ENV"
activate_durian_env
echo "[$APP_NAME] Python: $(command -v python)"
python -V

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  if ! python -m pip show fastapi >/dev/null 2>&1 || ! python -m pip show pypdf >/dev/null 2>&1; then
    echo "[$APP_NAME] Installing backend dependencies..."
    python -m pip install -r "$ROOT/requirements.txt"
  fi
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "[ERROR] npm not found. Please install Node.js first."
  exit 1
fi

if [[ "${SKIP_INSTALL:-0}" != "1" && ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "[$APP_NAME] Installing frontend dependencies..."
  (cd "$FRONTEND_DIR" && npm install)
fi

if is_running "$RUN_DIR/backend.pid"; then
  echo "[$APP_NAME] Backend already running, pid $(cat "$RUN_DIR/backend.pid")"
else
  echo "[$APP_NAME] Starting backend on $BACKEND_HOST:$BACKEND_PORT ..."
  (
    cd "$ROOT"
    nohup python durian__inference_api.py --host "$BACKEND_HOST" --port "$BACKEND_PORT" \
      > "$LOG_DIR/backend.log" 2>&1 &
    echo $! > "$RUN_DIR/backend.pid"
  )
fi

if is_running "$RUN_DIR/frontend.pid"; then
  echo "[$APP_NAME] Frontend already running, pid $(cat "$RUN_DIR/frontend.pid")"
else
  echo "[$APP_NAME] Starting frontend on $FRONTEND_HOST:$FRONTEND_PORT ..."
  (
    cd "$FRONTEND_DIR"
    nohup npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" \
      > "$LOG_DIR/frontend.log" 2>&1 &
    echo $! > "$RUN_DIR/frontend.pid"
  )
fi

echo
echo "[$APP_NAME] Started."
echo "Frontend: http://SERVER_IP:$FRONTEND_PORT"
echo "Backend:  http://SERVER_IP:$BACKEND_PORT/docs"
echo
echo "Logs:"
echo "  tail -f \"$LOG_DIR/backend.log\""
echo "  tail -f \"$LOG_DIR/frontend.log\""
echo
echo "Stop:"
echo "  bash \"$ROOT/stop_all.sh\""

echo "=== warming up LlamaIndex RAG ==="
(
  sleep 25
  timeout 45s curl -s -N -X POST http://127.0.0.1:80/gpt/api/chat/stream2 \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [
        {
          "role": "user",
          "content": "榴莲品种"
        }
      ],
      "response_language": "zh",
      "use_rag": true,
      "stream": true
    }' > /tmp/durian_rag_warmup.log || true
  echo "RAG warmup done"
) &

