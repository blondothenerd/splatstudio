#!/bin/zsh
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$APP_DIR/runtime/gsplat_core/bin/python"
HF_HOME_DIR="$APP_DIR/models/huggingface"
[[ -x "$PYTHON" ]] || { echo "Run Install Splat Studio.command first."; exit 1; }
mkdir -p "$HF_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"
export TRANSFORMERS_CACHE="$HF_HOME_DIR"
export PYTORCH_ENABLE_MPS_FALLBACK=1
"$PYTHON" -m pip install -r "$APP_DIR/requirements-ai.txt"
"$PYTHON" "$APP_DIR/scripts/prefetch_ai_models.py"
echo
echo "AI Vision ready. Restart Splat Studio."
read -k 1 "?Press any key to close..."
