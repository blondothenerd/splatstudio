#!/bin/zsh

# Splat Studio - best-effort public installer
# This installer intentionally keeps going after recoverable failures, retries
# network/build steps, and reports anything still missing at the end.

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$APP_DIR/runtime"
MINIFORGE_DIR="$RUNTIME_DIR/miniforge3"
ENV_DIR="$RUNTIME_DIR/gsplat_core"
BACKEND_DIR="$APP_DIR/backend/gsplat-metal"
MODELS_DIR="$APP_DIR/models/huggingface"
THIRD_PARTY_DIR="$APP_DIR/.splat_studio/third_party"
SUPERSPLAT_DIR="$THIRD_PARTY_DIR/supersplat"
SUPERSPLAT_VIEWER_DIR="$THIRD_PARTY_DIR/supersplat-viewer"
SPLAT_TRANSFORM_DIR="$THIRD_PARTY_DIR/splat-transform"
REPORT_DIR="$APP_DIR/.splat_studio"
REPORT_FILE="$REPORT_DIR/install_report.txt"

BACKEND_URL="https://github.com/a1091150/gsplat-mlx.git"
BACKEND_COMMIT="3c947a6d4d5a1df4a17096b03c0ac4d39032311a"
SPZ_URL="https://github.com/nianticlabs/spz.git"
SPZ_COMMIT="ef094fd1a96ca6ff414d72d7904ee4f4f6d97be9"

SUPERSPLAT_URL="https://github.com/playcanvas/supersplat.git"
SUPERSPLAT_REF="v2.32.3"
SUPERSPLAT_VIEWER_URL="https://github.com/playcanvas/supersplat-viewer.git"
SUPERSPLAT_VIEWER_REF="v1.29.1"

typeset -a FAILURES
typeset -a WARNINGS
typeset -a MISSING
typeset -a OPTIONAL_MISSING
FAILURES=()
WARNINGS=()
MISSING=()
OPTIONAL_MISSING=()

BREW=""
PYTHON=""
XCODE_APP=""
DEVELOPER_DIR=""
INSTALL_AI="N"

banner() {
  printf "\n============================================================\n"
  printf "%s\n" "$1"
  printf "============================================================\n"
}

note() {
  printf "  %s\n" "$1"
}

add_failure() {
  FAILURES+=("$1")
  printf "  WARNING: %s\n" "$1"
}

add_warning() {
  WARNINGS+=("$1")
  printf "  NOTE: %s\n" "$1"
}

add_missing() {
  MISSING+=("$1")
}

add_optional_missing() {
  OPTIONAL_MISSING+=("$1")
}

retry_cmd() {
  local label="$1"
  local attempts="$2"
  shift 2

  banner "$label"
  local attempt=1
  local rc=0

  while (( attempt <= attempts )); do
    "$@"
    rc=$?
    if (( rc == 0 )); then
      note "OK"
      return 0
    fi

    note "Attempt $attempt/$attempts failed with exit code $rc."
    if (( attempt < attempts )); then
      note "Retrying in 3 seconds..."
      sleep 3
    fi
    (( attempt++ ))
  done

  add_failure "$label failed after $attempts attempt(s), exit code $rc."
  return $rc
}

find_xcode() {
  local candidate=""

  for candidate in \
    "/Applications/Xcode.app" \
    "$HOME/Applications/Xcode.app"
  do
    if [[ -d "$candidate/Contents/Developer" ]]; then
      XCODE_APP="$candidate"
      DEVELOPER_DIR="$candidate/Contents/Developer"
      return 0
    fi
  done

  if command -v mdfind >/dev/null 2>&1; then
    candidate="$(mdfind "kMDItemCFBundleIdentifier == 'com.apple.dt.Xcode'" 2>/dev/null | head -n 1)"
    if [[ -n "$candidate" && -d "$candidate/Contents/Developer" ]]; then
      XCODE_APP="$candidate"
      DEVELOPER_DIR="$candidate/Contents/Developer"
      return 0
    fi
  fi

  return 1
}

find_brew() {
  local candidate=""

  if command -v brew >/dev/null 2>&1; then
    BREW="$(command -v brew)"
    return 0
  fi

  for candidate in "/opt/homebrew/bin/brew" "/usr/local/bin/brew"; do
    if [[ -x "$candidate" ]]; then
      BREW="$candidate"
      return 0
    fi
  done

  return 1
}

configure_brew_path() {
  [[ -n "$BREW" && -x "$BREW" ]] || return 1

  eval "$("$BREW" shellenv 2>/dev/null)"

  local prefix
  prefix="$("$BREW" --prefix 2>/dev/null)"
  if [[ -n "$prefix" ]]; then
    local profile="$HOME/.zprofile"
    local marker="$prefix/bin/brew shellenv"
    touch "$profile" 2>/dev/null || true

    if ! grep -Fq "$marker" "$profile" 2>/dev/null; then
      {
        printf "\n# Added by Splat Studio installer\n"
        printf 'eval "$(%s/bin/brew shellenv)"\n' "$prefix"
      } >> "$profile" 2>/dev/null || add_warning "Could not update ~/.zprofile with Homebrew PATH. The installer will still use Homebrew directly."
    fi
  fi

  return 0
}

install_homebrew() {
  local cache="$APP_DIR/.install-cache"
  local script="$cache/homebrew-install.sh"
  mkdir -p "$cache"

  banner "Installing Homebrew"
  note "Homebrew was not found. Splat Studio will try the official installer automatically."

  if ! curl -fsSL --retry 3 --retry-delay 2 \
    "https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh" \
    -o "$script"
  then
    add_failure "Could not download the official Homebrew installer."
    return 1
  fi

  chmod +x "$script" 2>/dev/null || true

  # Try unattended first. If sudo/user interaction is required, retry using the
  # normal official installer so the user can enter their password.
  NONINTERACTIVE=1 /bin/bash "$script"
  local rc=$?

  if (( rc != 0 )); then
    note "Automatic Homebrew setup needs user interaction; retrying normally."
    /bin/bash "$script"
    rc=$?
  fi

  if (( rc != 0 )); then
    add_failure "Homebrew installation did not complete."
    return $rc
  fi

  find_brew || return 1
  configure_brew_path
  return 0
}

install_brew_package() {
  local package="$1"
  [[ -n "$BREW" && -x "$BREW" ]] || return 1

  if "$BREW" list --versions "$package" >/dev/null 2>&1; then
    note "$package is already installed."
    return 0
  fi

  retry_cmd "Installing $package" 2 "$BREW" install "$package"
}

prepare_miniforge() {
  if [[ -x "$MINIFORGE_DIR/bin/conda" ]]; then
    note "Private Miniforge runtime already exists."
    return 0
  fi

  local cache="$APP_DIR/.install-cache"
  local package="$cache/Miniforge3-MacOSX-arm64.sh"
  mkdir -p "$cache"

  retry_cmd "Downloading private Miniforge runtime" 3 \
    curl -fL --retry 3 --retry-delay 2 \
    -o "$package" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh" || return 1

  retry_cmd "Installing private Miniforge runtime" 2 \
    /bin/bash "$package" -b -p "$MINIFORGE_DIR"
}

create_python_env() {
  local conda="$MINIFORGE_DIR/bin/conda"

  if [[ ! -x "$conda" ]]; then
    add_failure "Miniforge/conda is unavailable, so the Python runtime could not be created."
    return 1
  fi

  if [[ -x "$ENV_DIR/bin/python" ]]; then
    PYTHON="$ENV_DIR/bin/python"
    note "Python environment already exists."
    return 0
  fi

  retry_cmd "Creating Splat Studio Python 3.11 environment" 2 \
    "$conda" create -y -p "$ENV_DIR" python=3.11 || return 1

  PYTHON="$ENV_DIR/bin/python"
  return 0
}

prepare_backend() {
  if ! command -v git >/dev/null 2>&1; then
    add_failure "Git is unavailable, so the native backend could not be downloaded."
    return 1
  fi

  if [[ -d "$BACKEND_DIR" && ! -d "$BACKEND_DIR/.git" ]]; then
    local backup="$BACKEND_DIR.incomplete-$(date +%Y%m%d-%H%M%S)"
    note "Existing backend folder is not a Git repository; moving it aside:"
    note "$backup"
    mv "$BACKEND_DIR" "$backup" 2>/dev/null || rm -rf "$BACKEND_DIR"
  fi

  if [[ ! -d "$BACKEND_DIR/.git" ]]; then
    mkdir -p "$(dirname "$BACKEND_DIR")"
    retry_cmd "Downloading native MLX/Metal backend" 3 \
      git clone --recurse-submodules "$BACKEND_URL" "$BACKEND_DIR" || return 1
  fi

  retry_cmd "Fetching native backend" 2 \
    git -C "$BACKEND_DIR" fetch --tags origin || true

  retry_cmd "Pinning native backend" 2 \
    git -C "$BACKEND_DIR" checkout --detach "$BACKEND_COMMIT" || return 1

  retry_cmd "Synchronising backend submodules" 2 \
    git -C "$BACKEND_DIR" submodule sync --recursive || true

  retry_cmd "Updating backend submodules" 2 \
    git -C "$BACKEND_DIR" submodule update --init --recursive --force || true

  local spz="$BACKEND_DIR/submodules/spz"
  if [[ ! -d "$spz/.git" && ! -f "$spz/.git" ]]; then
    rm -rf "$spz"
    mkdir -p "$(dirname "$spz")"
    retry_cmd "Repairing SPZ submodule" 3 \
      git clone "$SPZ_URL" "$spz" || return 1
  fi

  retry_cmd "Pinning SPZ support" 2 \
    git -C "$spz" checkout --detach "$SPZ_COMMIT" || true

  return 0
}

build_backend() {
  if [[ ! -x "$PYTHON" ]]; then
    add_failure "Python runtime is unavailable, so the native backend could not be built."
    return 1
  fi

  if [[ ! -d "$BACKEND_DIR" ]]; then
    add_failure "Native backend source is unavailable, so it could not be built."
    return 1
  fi

  if ! xcrun --find metal >/dev/null 2>&1; then
    add_failure "Apple Metal compiler is unavailable, so the native backend build was skipped."
    return 1
  fi

  (
    cd "$BACKEND_DIR" || exit 1
    export CMAKE_BUILD_PARALLEL_LEVEL="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
    "$PYTHON" -m pip install . --no-build-isolation
  )
}

clone_or_update_ref() {
  local url="$1"
  local ref="$2"
  local directory="$3"

  if [[ -d "$directory" && ! -d "$directory/.git" ]]; then
    rm -rf "$directory"
  fi

  if [[ ! -d "$directory/.git" ]]; then
    git clone "$url" "$directory" || return 1
  fi

  git -C "$directory" fetch --tags origin || return 1
  git -C "$directory" checkout --detach "$ref" || return 1
  return 0
}

build_npm_project() {
  local directory="$1"
  (
    cd "$directory" || exit 1
    npm ci &&
    npm run build
  )
}

install_supersplat_tools() {
  if ! command -v git >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    add_failure "Git/npm are unavailable, so SuperSplat tools could not be installed."
    return 1
  fi

  mkdir -p "$THIRD_PARTY_DIR"

  retry_cmd "Downloading SuperSplat Editor" 3 \
    clone_or_update_ref "$SUPERSPLAT_URL" "$SUPERSPLAT_REF" "$SUPERSPLAT_DIR" || return 1

  retry_cmd "Building SuperSplat Editor" 2 \
    build_npm_project "$SUPERSPLAT_DIR" || return 1

  retry_cmd "Downloading SuperSplat Viewer" 3 \
    clone_or_update_ref "$SUPERSPLAT_VIEWER_URL" "$SUPERSPLAT_VIEWER_REF" "$SUPERSPLAT_VIEWER_DIR" || return 1

  retry_cmd "Building SuperSplat Viewer" 2 \
    build_npm_project "$SUPERSPLAT_VIEWER_DIR" || return 1

  retry_cmd "Installing SplatTransform" 2 \
    npm install --prefix "$SPLAT_TRANSFORM_DIR" @playcanvas/splat-transform || return 1

  return 0
}

install_ai_support() {
  [[ -x "$PYTHON" ]] || return 1

  if [[ ! -f "$APP_DIR/requirements-ai.txt" ]]; then
    add_failure "requirements-ai.txt is missing."
    return 1
  fi

  retry_cmd "Installing optional AI Python dependencies" 2 \
    "$PYTHON" -m pip install -r "$APP_DIR/requirements-ai.txt" || return 1

  if [[ ! -f "$APP_DIR/scripts/prefetch_ai_models.py" ]]; then
    add_failure "scripts/prefetch_ai_models.py is missing."
    return 1
  fi

  mkdir -p "$MODELS_DIR"

  banner "Downloading optional local AI models"
  env \
    HF_HOME="$MODELS_DIR" \
    TRANSFORMERS_CACHE="$MODELS_DIR" \
    SPLAT_AI_DEPTH_MODEL="${SPLAT_AI_DEPTH_MODEL:-depth-anything/Depth-Anything-V2-Small-hf}" \
    SPLAT_AI_SEGMENT_MODEL="${SPLAT_AI_SEGMENT_MODEL:-nvidia/segformer-b5-finetuned-ade-640-640}" \
    "$PYTHON" "$APP_DIR/scripts/prefetch_ai_models.py"

  local rc=$?
  if (( rc != 0 )); then
    add_failure "Optional AI model download failed. Core Splat Studio can still work without AI."
    return $rc
  fi

  note "AI support ready."
  return 0
}

check_command() {
  local display="$1"
  local command_name="$2"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    add_missing "$display"
    return 1
  fi
  return 0
}

check_optional_path() {
  local display="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    add_optional_missing "$display"
    return 1
  fi
  return 0
}

write_report() {
  mkdir -p "$REPORT_DIR"

  {
    echo "Splat Studio installation report"
    echo "Generated: $(date)"
    echo "Folder: $APP_DIR"
    echo

    echo "Final required missing items: ${#MISSING[@]}"
    for item in "${MISSING[@]}"; do
      echo "  - $item"
    done
    echo

    echo "Optional/recommended missing items: ${#OPTIONAL_MISSING[@]}"
    for item in "${OPTIONAL_MISSING[@]}"; do
      echo "  - $item"
    done
    echo

    echo "Intermediate failed/retried steps: ${#FAILURES[@]}"
    for item in "${FAILURES[@]}"; do
      echo "  - $item"
    done
    echo

    echo "Warnings/notes: ${#WARNINGS[@]}"
    for item in "${WARNINGS[@]}"; do
      echo "  - $item"
    done
  } > "$REPORT_FILE" 2>/dev/null || true
}

banner "Splat Studio automatic installer"
echo "Install folder:"
echo "  $APP_DIR"
echo
echo "The installer is safe to rerun. Existing working components are reused"
echo "where possible, failed network/build steps are retried, and final checks"
echo "are shown at the end."

# ---------------------------------------------------------------------------
# Hard platform checks
# ---------------------------------------------------------------------------

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo
  echo "Splat Studio's current native MLX/Metal backend requires macOS."
  printf "\nPress Return to close..."
  IFS= read -r _
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo
  echo "Splat Studio currently requires an Apple Silicon Mac (M-series)."
  printf "\nPress Return to close..."
  IFS= read -r _
  exit 1
fi

MACOS_VERSION="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
MACOS_MAJOR="${MACOS_VERSION%%.*}"
if [[ "$MACOS_MAJOR" =~ ^[0-9]+$ ]] && (( MACOS_MAJOR < 14 )); then
  add_warning "macOS $MACOS_VERSION detected. Current Homebrew officially supports macOS 14 or newer; older versions may still work but are not the recommended target."
fi

# ---------------------------------------------------------------------------
# Make sure the public repository itself is complete
# ---------------------------------------------------------------------------

banner "Checking Splat Studio files"

REQUIRED_REPO_FILES=(
  "$APP_DIR/SplatStudio.py"
  "$APP_DIR/requirements-core.txt"
  "$APP_DIR/requirements-ai.txt"
  "$APP_DIR/Launch Splat Studio.command"
  "$APP_DIR/scripts/doctor.py"
  "$APP_DIR/scripts/prefetch_ai_models.py"
)

for file in "${REQUIRED_REPO_FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    add_missing "Repository file: ${file#$APP_DIR/}"
  fi
done

mkdir -p \
  "$RUNTIME_DIR" \
  "$APP_DIR/backend" \
  "$MODELS_DIR" \
  "$APP_DIR/projects" \
  "$THIRD_PARTY_DIR" \
  "$REPORT_DIR"

# ---------------------------------------------------------------------------
# Xcode + Metal
# ---------------------------------------------------------------------------

banner "Checking Xcode and Metal"

if find_xcode; then
  note "Found Xcode: $XCODE_APP"

  retry_cmd "Selecting full Xcode" 2 \
    sudo xcode-select --switch "$DEVELOPER_DIR" || true

  retry_cmd "Completing Xcode first-launch setup" 2 \
    sudo xcodebuild -runFirstLaunch -checkForNewerComponents || true

  if ! xcrun --find metal >/dev/null 2>&1; then
    retry_cmd "Installing Apple Metal Toolchain" 2 \
      xcodebuild -downloadComponent metalToolchain || true
  fi
else
  add_missing "Full Xcode"
  add_warning "Full Xcode was not found. Opening Apple's Xcode page. Finish installing/opening Xcode, then rerun this installer."
  open "macappstore://itunes.apple.com/app/id497799835" >/dev/null 2>&1 || \
    open "https://apps.apple.com/app/xcode/id497799835" >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------------------
# Homebrew + system tools
# ---------------------------------------------------------------------------

if ! find_brew; then
  install_homebrew || true
else
  note "Homebrew found: $BREW"
  configure_brew_path || true
fi

if find_brew; then
  configure_brew_path || true

  for package in git ffmpeg colmap cmake node; do
    install_brew_package "$package" || true
  done

  # Refresh PATH after installs.
  configure_brew_path || true
else
  add_missing "Homebrew"
  add_warning "Homebrew is unavailable. FFmpeg, COLMAP, CMake, Node.js and Git may also be missing."
fi

# ---------------------------------------------------------------------------
# Private Python runtime
# ---------------------------------------------------------------------------

prepare_miniforge || true
create_python_env || true

if [[ -x "$ENV_DIR/bin/python" ]]; then
  PYTHON="$ENV_DIR/bin/python"

  retry_cmd "Updating Python packaging tools" 2 \
    "$PYTHON" -m pip install --upgrade pip setuptools wheel || true

  if [[ -f "$APP_DIR/requirements-core.txt" ]]; then
    retry_cmd "Installing Splat Studio Python dependencies" 2 \
      "$PYTHON" -m pip install -r "$APP_DIR/requirements-core.txt" || true
  fi
else
  add_missing "Splat Studio Python 3.11 runtime"
fi

# ---------------------------------------------------------------------------
# Native MLX / Metal training backend
# ---------------------------------------------------------------------------

prepare_backend || true

if [[ -x "$PYTHON" && -d "$BACKEND_DIR" ]]; then
  retry_cmd "Building native C++/Metal Gaussian backend" 2 \
    build_backend || true

  if [[ -d "$BACKEND_DIR/submodules/spz" ]]; then
    retry_cmd "Installing SPZ Python support" 2 \
      "$PYTHON" -m pip install "$BACKEND_DIR/submodules/spz" || true
  fi
fi

# ---------------------------------------------------------------------------
# SuperSplat tools used by Review/Edit/View
# ---------------------------------------------------------------------------

install_supersplat_tools || true

# ---------------------------------------------------------------------------
# Optional AI
# ---------------------------------------------------------------------------

echo
echo "Optional AI vision can add local depth assistance and semantic masking."
echo "The current optional NVIDIA SegFormer model has separate non-commercial"
echo "research/evaluation license terms. Core Splat Studio does not require it."
printf "Install optional AI support/models now? [y/N] "
IFS= read -r INSTALL_AI_REPLY
INSTALL_AI_REPLY="${INSTALL_AI_REPLY:-N}"

if [[ "$INSTALL_AI_REPLY" == [Yy]* ]]; then
  INSTALL_AI="Y"
  install_ai_support || true
else
  add_warning "Optional AI support was skipped by the user."
fi

# ---------------------------------------------------------------------------
# Permissions / cleanup
# ---------------------------------------------------------------------------

for command_file in \
  "$APP_DIR/Install Splat Studio.command" \
  "$APP_DIR/Launch Splat Studio.command" \
  "$APP_DIR/Setup AI Vision.command" \
  "$APP_DIR/Create Splat Studio App.command"
do
  [[ -f "$command_file" ]] && chmod +x "$command_file" 2>/dev/null || true
done

# ---------------------------------------------------------------------------
# Final verification - this is the authoritative result
# ---------------------------------------------------------------------------

banner "Final installation checks"

# Reset required final check list; repository-file misses from above remain.
# Avoid duplicate items where possible by only adding final missing items below
# if they have not already been recorded.

if ! find_xcode; then
  [[ " ${MISSING[*]} " == *" Full Xcode "* ]] || add_missing "Full Xcode"
else
  if ! xcrun --find metal >/dev/null 2>&1; then
    add_missing "Apple Metal Toolchain"
  fi
fi

if ! find_brew; then
  [[ " ${MISSING[*]} " == *" Homebrew "* ]] || add_missing "Homebrew"
else
  configure_brew_path || true
fi

check_command "Git" git || true
check_command "FFmpeg" ffmpeg || true
check_command "COLMAP" colmap || true
check_command "CMake" cmake || true
check_command "Node.js" node || true
check_command "npm" npm || true

if command -v node >/dev/null 2>&1; then
  node -e '
    const p = process.versions.node.split(".").map(Number);
    const ok = p[0] > 20 || (p[0] === 20 && p[1] >= 19);
    process.exit(ok ? 0 : 1);
  ' >/dev/null 2>&1 || add_missing "Node.js 20.19 or newer"
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  [[ " ${MISSING[*]} " == *" Splat Studio Python 3.11 runtime "* ]] || add_missing "Splat Studio Python 3.11 runtime"
else
  PYTHON="$ENV_DIR/bin/python"

  "$PYTHON" -c 'import streamlit, mlx, pycolmap, cv2, plyfile, PIL, scipy, tyro' >/dev/null 2>&1 || \
    add_missing "Core Python dependencies"

  "$PYTHON" -c 'import gsplat_core' >/dev/null 2>&1 || \
    add_missing "Native gsplat_core Metal backend"
fi

if [[ -x "$PYTHON" && -f "$BACKEND_DIR/scripts/test/training_dense_3dgs_loop_smoke.py" && -x "$(command -v xcrun 2>/dev/null)" ]]; then
  if xcrun --find metal >/dev/null 2>&1; then
    banner "Native training smoke test"
    "$PYTHON" "$BACKEND_DIR/scripts/test/training_dense_3dgs_loop_smoke.py"
    rc=$?
    if (( rc != 0 )); then
      add_missing "Native Metal training smoke test"
    else
      note "Native training smoke test passed."
    fi
  fi
fi

check_optional_path "SuperSplat Editor build" "$SUPERSPLAT_DIR/dist/index.html" || true
check_optional_path "SuperSplat Viewer build" "$SUPERSPLAT_VIEWER_DIR/public/index.html" || true
check_optional_path "SplatTransform" "$SPLAT_TRANSFORM_DIR/node_modules/.bin/splat-transform" || true

if [[ "$INSTALL_AI" == "Y" && -x "$PYTHON" ]]; then
  "$PYTHON" -c 'import torch, transformers, safetensors, huggingface_hub' >/dev/null 2>&1 || \
    add_optional_missing "AI Python dependencies/models"
fi

if [[ -x "$PYTHON" && -f "$APP_DIR/scripts/doctor.py" ]]; then
  banner "Splat Studio doctor"
  "$PYTHON" "$APP_DIR/scripts/doctor.py" || true
fi

write_report

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

banner "Installation summary"

if (( ${#MISSING[@]} == 0 )); then
  echo "Core Splat Studio installation: READY"
else
  echo "Core Splat Studio installation: NEEDS ATTENTION"
  echo
  echo "Still missing:"
  for item in "${MISSING[@]}"; do
    echo "  - $item"
  done
fi

if (( ${#OPTIONAL_MISSING[@]} > 0 )); then
  echo
  echo "Optional/recommended components not ready:"
  for item in "${OPTIONAL_MISSING[@]}"; do
    echo "  - $item"
  done
fi

if (( ${#FAILURES[@]} > 0 )); then
  echo
  echo "Steps that reported an error during installation:"
  for item in "${FAILURES[@]}"; do
    echo "  - $item"
  done

  if (( ${#MISSING[@]} == 0 )); then
    echo
    echo "Final required checks passed despite the errors above."
  fi
fi

echo
echo "Full report:"
echo "  $REPORT_FILE"
echo

if (( ${#MISSING[@]} == 0 )); then
  echo "Splat Studio is ready."
  echo "Double-click:"
  echo "  Launch Splat Studio.command"
  echo
  echo "Optional macOS app launcher:"
  echo "  Create Splat Studio App.command"

  printf "\nLaunch Splat Studio now? [Y/n] "
  IFS= read -r LAUNCH_REPLY
  LAUNCH_REPLY="${LAUNCH_REPLY:-Y}"
  if [[ "$LAUNCH_REPLY" == [Yy]* ]]; then
    open "$APP_DIR/Launch Splat Studio.command" >/dev/null 2>&1 || true
  fi
else
  echo "Fix the missing item(s) above and simply run this installer again."
  echo "It is designed to be safely rerun."
fi

if (( ${#OPTIONAL_MISSING[@]} > 0 )); then
  echo
  echo "Optional tools can also be repaired later from Splat Studio Settings."
fi

printf "\nPress Return to close..."
IFS= read -r _
