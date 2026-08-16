#!/bin/zsh
set -e
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$APP_DIR/runtime/gsplat_core/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Splat Studio is not installed yet."
  echo "Run: Install Splat Studio.command"
  read -k 1 "?Press any key to close..."
  exit 1
fi
cd "$APP_DIR"
exec "$PYTHON" -m streamlit run SplatStudio.py
