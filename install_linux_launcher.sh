#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_NAME="DurianGPT.desktop"
LOCAL_LAUNCHER="$ROOT/$LAUNCHER_NAME"

DESKTOP_DIR="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
if [[ ! -d "$DESKTOP_DIR" && -d "$HOME/桌面" ]]; then
  DESKTOP_DIR="$HOME/桌面"
fi

cat > "$LOCAL_LAUNCHER" <<EOF
[Desktop Entry]
Type=Application
Name=DurianGPT
Comment=Start DurianGPT backend and frontend
Terminal=true
WorkingDirectory=$ROOT
Exec=bash -lc 'cd "$ROOT" && bash start_all.sh; echo; echo "Press Enter to close"; read'
Icon=utilities-terminal
Categories=Development;
EOF

chmod +x "$ROOT/start_all.sh" "$ROOT/stop_all.sh" "$LOCAL_LAUNCHER"

if [[ -d "$DESKTOP_DIR" ]]; then
  cp "$LOCAL_LAUNCHER" "$DESKTOP_DIR/$LAUNCHER_NAME"
  chmod +x "$DESKTOP_DIR/$LAUNCHER_NAME"
  if command -v gio >/dev/null 2>&1; then
    gio set "$DESKTOP_DIR/$LAUNCHER_NAME" metadata::trusted true >/dev/null 2>&1 || true
  fi
  echo "[DurianGPT] Launcher installed:"
  echo "  $DESKTOP_DIR/$LAUNCHER_NAME"
else
  echo "[DurianGPT] Desktop directory not found. Launcher created here:"
  echo "  $LOCAL_LAUNCHER"
fi

echo
echo "If your desktop shows a warning, right click the icon and choose:"
echo "  Allow Launching / Trust and Launch"
