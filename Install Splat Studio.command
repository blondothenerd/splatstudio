#!/bin/zsh
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$APP_DIR/runtime"
MINIFORGE_DIR="$RUNTIME_DIR/miniforge3"
ENV_DIR="$RUNTIME_DIR/gsplat_core"
BACKEND_DIR="$APP_DIR/backend/gsplat-metal"
BACKEND_URL="https://github.com/a1091150/gsplat-mlx.git"
BACKEND_COMMIT="3c947a6d4d5a1df4a17096b03c0ac4d39032311a"
SPZ_URL="https://github.com/nianticlabs/spz.git"
SPZ_COMMIT="ef094fd1a96ca6ff414d72d7904ee4f4f6d97be9"

banner() { printf "\n=== %s ===\n" "$1"; }
fail() { printf "\nERROR: %s\n" "$1"; printf "\nPress Return to close..."; read -r _; exit 1; }

banner "Splat Studio installer"
echo "Install folder: $APP_DIR"

[[ "$(uname -s)" == "Darwin" ]] || fail "Splat Studio's native MLX/Metal backend currently requires macOS."
[[ "$(uname -m)" == "arm64" ]] || fail "Splat Studio currently requires an Apple Silicon Mac (M-series)."

# Full Xcode is required to compile the Metal extension.
if [[ ! -d "/Applications/Xcode.app" ]]; then
  fail "Full Xcode is required. Install Xcode from the Mac App Store, open it once, then run this installer again."
fi

if [[ "$(xcode-select -p 2>/dev/null || true)" != "/Applications/Xcode.app/Contents/Developer" ]]; then
  banner "Selecting full Xcode"
  sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
  sudo xcodebuild -runFirstLaunch
fi

if ! xcrun --find metal >/dev/null 2>&1; then
  banner "Installing Apple Metal Toolchain"
  xcodebuild -downloadComponent metalToolchain
fi

if ! command -v brew >/dev/null 2>&1; then
  echo
  echo "Homebrew is required for COLMAP, FFmpeg and Node.js."
  echo "Install it from https://brew.sh/ and run this installer again."
  fail "Homebrew was not found."
fi

banner "Installing system tools"
brew install ffmpeg colmap cmake node

mkdir -p "$RUNTIME_DIR" "$APP_DIR/backend" "$APP_DIR/models/huggingface" "$APP_DIR/projects"

if [[ ! -x "$MINIFORGE_DIR/bin/conda" ]]; then
  banner "Installing private Miniforge runtime"
  CACHE_DIR="$APP_DIR/.install-cache"
  mkdir -p "$CACHE_DIR"
  INSTALLER="$CACHE_DIR/Miniforge3-MacOSX-arm64.sh"
  curl -fL --retry 3 -o "$INSTALLER" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
  bash "$INSTALLER" -b -p "$MINIFORGE_DIR"
fi

CONDA="$MINIFORGE_DIR/bin/conda"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  banner "Creating Python 3.11 environment"
  "$CONDA" create -y -p "$ENV_DIR" python=3.11
fi
PYTHON="$ENV_DIR/bin/python"

banner "Installing Python dependencies"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install -r "$APP_DIR/requirements-core.txt"

if [[ ! -d "$BACKEND_DIR/.git" ]]; then
  banner "Downloading native MLX/Metal backend"
  rm -rf "$BACKEND_DIR"
  git clone --recurse-submodules "$BACKEND_URL" "$BACKEND_DIR"
fi

banner "Pinning native backend"
git -C "$BACKEND_DIR" fetch --tags origin
git -C "$BACKEND_DIR" checkout --detach "$BACKEND_COMMIT"
git -C "$BACKEND_DIR" submodule sync --recursive
git -C "$BACKEND_DIR" submodule update --init --recursive --force

if [[ ! -d "$BACKEND_DIR/submodules/spz" ]]; then
  banner "Repairing SPZ submodule"
  mkdir -p "$BACKEND_DIR/submodules"
  git clone "$SPZ_URL" "$BACKEND_DIR/submodules/spz"
fi
git -C "$BACKEND_DIR/submodules/spz" checkout --detach "$SPZ_COMMIT"

banner "Building native C++/Metal backend"
cd "$BACKEND_DIR"
"$PYTHON" -m pip install . --no-build-isolation
"$PYTHON" -m pip install ./submodules/spz

banner "Validating backend"
"$PYTHON" -c 'import mlx, gsplat_core, streamlit, pycolmap; print("Core runtime OK")'
if [[ -f "$BACKEND_DIR/scripts/test/training_dense_3dgs_loop_smoke.py" ]]; then
  "$PYTHON" "$BACKEND_DIR/scripts/test/training_dense_3dgs_loop_smoke.py"
fi

printf "\nInstall optional local AI vision dependencies/models now? [Y/n] "
read -r INSTALL_AI
INSTALL_AI=${INSTALL_AI:-Y}
if [[ "$INSTALL_AI" == [Yy]* ]]; then
  banner "Installing AI vision support"
  "$PYTHON" -m pip install -r "$APP_DIR/requirements-ai.txt"
  HF_HOME="$APP_DIR/models/huggingface" \
  TRANSFORMERS_CACHE="$APP_DIR/models/huggingface" \
  SPLAT_AI_DEPTH_MODEL="${SPLAT_AI_DEPTH_MODEL:-depth-anything/Depth-Anything-V2-Small-hf}" \
  SPLAT_AI_SEGMENT_MODEL="${SPLAT_AI_SEGMENT_MODEL:-nvidia/segformer-b5-finetuned-ade-640-640}" \
  "$PYTHON" "$APP_DIR/scripts/prefetch_ai_models.py"
fi

chmod +x "$APP_DIR/Launch Splat Studio.command" "$APP_DIR/Install Splat Studio.command" "$APP_DIR/Setup AI Vision.command"
rm -rf "$APP_DIR/.install-cache"

banner "Installation complete"
echo "Everything managed by Splat Studio is stored under:"
echo "  $APP_DIR"
echo
echo "Double-click: Launch Splat Studio.command"
printf "\nPress Return to close..."
read -r _
