#!/bin/zsh

# Splat Studio macOS app creator
# Creates a local .app launcher that:
#   - uses SplatStudio_icon.png to build a proper macOS .icns icon
#   - starts SplatStudio.py directly with the private Splat Studio runtime
#   - restores Homebrew/Xcode paths that Finder-launched apps do not inherit
#   - waits for Streamlit to become genuinely healthy before opening the browser
#   - offers the installer if Splat Studio is not installed yet
#   - writes a useful launch log if startup fails

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_APP="$APP_DIR/Splat Studio.app"
PNG_ICON="$APP_DIR/SplatStudio_icon.png"
ICNS_FALLBACK="$APP_DIR/SplatStudio.icns"
PYTHON="$APP_DIR/runtime/gsplat_core/bin/python"
STUDIO_PY="$APP_DIR/SplatStudio.py"
INSTALLER="$APP_DIR/Install Splat Studio.command"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/splatstudio-app.XXXXXX")"
APPLESCRIPT="$TMP_DIR/SplatStudioLauncher.applescript"
ICONSET="$TMP_DIR/SplatStudio.iconset"
GENERATED_ICNS="$TMP_DIR/SplatStudio.icns"

cleanup() {
  rm -rf "$TMP_DIR" 2>/dev/null || true
}
trap cleanup EXIT

pause_close() {
  printf "\nPress Return to close..."
  IFS= read -r _
}

escape_applescript_string() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

echo
echo "============================================================"
echo "Create Splat Studio.app"
echo "============================================================"
echo
echo "Splat Studio folder:"
echo "  $APP_DIR"
echo

# ---------------------------------------------------------------------------
# Required public files
# ---------------------------------------------------------------------------

if [[ ! -f "$STUDIO_PY" ]]; then
  echo "ERROR: SplatStudio.py was not found:"
  echo "  $STUDIO_PY"
  pause_close
  exit 1
fi

if [[ ! -f "$INSTALLER" ]]; then
  echo "ERROR: Install Splat Studio.command was not found:"
  echo "  $INSTALLER"
  pause_close
  exit 1
fi

# The app can still be created before installation. On first launch it will
# offer to open the installer if the private runtime is not ready.
if [[ ! -x "$PYTHON" ]]; then
  echo "NOTE: The private Python runtime is not installed yet."
  echo "      The generated app will offer to run the installer when opened."
  echo
fi

# ---------------------------------------------------------------------------
# Compile the AppleScript app shell.
#
# The shell script that actually launches Streamlit will be stored INSIDE the
# generated app bundle. It reads install-root.txt so the .app itself can be
# moved to /Applications without losing its launcher resources.
# ---------------------------------------------------------------------------

cat > "$APPLESCRIPT" <<'APPLESCRIPT_EOF'
on run
    set resourcesPath to POSIX path of (path to resource "install-root.txt")
    set launcherResource to POSIX path of (path to resource "launch_splatstudio.sh")
    set installRoot to do shell script "/bin/cat " & quoted form of resourcesPath
    set installerPath to installRoot & "/Install Splat Studio.command"
    set logPath to installRoot & "/.splat_studio/app-launch.log"

    try
        do shell script "/bin/zsh " & quoted form of launcherResource

    on error errorMessage number errorNumber
        if errorNumber is 3 then
            set resultDialog to display dialog ¬
                "Splat Studio is not installed yet." & return & return & ¬
                "The automatic installer needs to run before the app can start." ¬
                buttons {"Cancel", "Run Installer"} ¬
                default button "Run Installer" ¬
                with icon caution

            if button returned of resultDialog is "Run Installer" then
                do shell script "/usr/bin/open " & quoted form of installerPath
            end if
            return
        end if

        set resultDialog to display dialog ¬
            "Splat Studio could not start." & return & return & ¬
            "The browser was NOT opened because the Streamlit server did not become ready." & return & return & ¬
            "Launch log:" & return & logPath & return & return & ¬
            errorMessage ¬
            buttons {"OK", "Open Log"} ¬
            default button "Open Log" ¬
            with icon stop

        if button returned of resultDialog is "Open Log" then
            try
                do shell script "/usr/bin/open -a TextEdit " & quoted form of logPath
            end try
        end if
    end try
end run
APPLESCRIPT_EOF

rm -rf "$OUTPUT_APP"

if ! /usr/bin/osacompile -o "$OUTPUT_APP" "$APPLESCRIPT"; then
  echo
  echo "ERROR: macOS could not create the application bundle."
  pause_close
  exit 1
fi

RESOURCES="$OUTPUT_APP/Contents/Resources"
PLIST="$OUTPUT_APP/Contents/Info.plist"
mkdir -p "$RESOURCES"

# ---------------------------------------------------------------------------
# Store the local installation location inside the generated LOCAL app.
# This path is never added to the GitHub source repository.
# ---------------------------------------------------------------------------

printf '%s' "$APP_DIR" > "$RESOURCES/install-root.txt"

# ---------------------------------------------------------------------------
# Embedded launch script
# ---------------------------------------------------------------------------

cat > "$RESOURCES/launch_splatstudio.sh" <<'LAUNCHER_EOF'
#!/bin/zsh

RESOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_FILE="$RESOURCE_DIR/install-root.txt"

if [[ ! -f "$ROOT_FILE" ]]; then
  echo "Splat Studio launcher is missing install-root.txt." >&2
  exit 4
fi

APP_DIR="$(cat "$ROOT_FILE")"
PYTHON="$APP_DIR/runtime/gsplat_core/bin/python"
STUDIO_PY="$APP_DIR/SplatStudio.py"
INSTALLER="$APP_DIR/Install Splat Studio.command"
STATE_DIR="$APP_DIR/.splat_studio"
LOG_FILE="$STATE_DIR/app-launch.log"
PID_FILE="$STATE_DIR/app-streamlit.pid"

URL="http://127.0.0.1:8501"
HEALTH_URL="$URL/_stcore/health"

mkdir -p "$STATE_DIR"

# Finder-launched apps do not inherit the user's normal interactive shell PATH.
# Add the standard Apple Silicon / Intel Homebrew locations explicitly.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

if [[ -x /opt/homebrew/bin/brew ]]; then
  eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null)" || true
elif [[ -x /usr/local/bin/brew ]]; then
  eval "$(/usr/local/bin/brew shellenv 2>/dev/null)" || true
fi

# Make the full Xcode toolchain visible to subprocesses such as Metal/CMake.
if [[ -d "/Applications/Xcode.app/Contents/Developer" ]]; then
  export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
fi

{
  echo
  echo "============================================================"
  echo "Splat Studio app launch"
  echo "============================================================"
  echo "Time: $(date)"
  echo "Install root: $APP_DIR"
  echo "Python: $PYTHON"
  echo "Application: $STUDIO_PY"
  echo "PATH: $PATH"
  echo "DEVELOPER_DIR: ${DEVELOPER_DIR:-<not set>}"
} >> "$LOG_FILE"

if [[ ! -f "$STUDIO_PY" ]]; then
  echo "ERROR: SplatStudio.py does not exist." >> "$LOG_FILE"
  echo "SplatStudio.py is missing from the Splat Studio folder." >&2
  exit 4
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Private Splat Studio Python runtime is not installed." >> "$LOG_FILE"
  exit 3
fi

# If Splat Studio is already running, just bring up the existing UI.
if /usr/bin/curl -fsS --max-time 1 "$HEALTH_URL" 2>/dev/null | /usr/bin/grep -qi "ok"; then
  echo "Existing Streamlit server is healthy; opening it." >> "$LOG_FILE"
  /usr/bin/open "$URL"
  exit 0
fi

# If a previous launcher PID still exists but the health endpoint is dead,
# terminate only that PID. We never kill an arbitrary process occupying 8501.
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null)"
  if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && /bin/kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Previous Splat Studio launcher PID $OLD_PID is unhealthy; terminating it." >> "$LOG_FILE"
    /bin/kill "$OLD_PID" 2>/dev/null || true
    sleep 1
  fi
  rm -f "$PID_FILE"
fi

# Do a quick import check before launching so failures are useful rather than a
# blank localhost page.
if ! "$PYTHON" -c 'import streamlit; import sys; print("streamlit", streamlit.__version__, "python", sys.version.split()[0])' >> "$LOG_FILE" 2>&1; then
  echo "ERROR: Streamlit cannot be imported from the Splat Studio runtime." >> "$LOG_FILE"
  echo "Splat Studio's Python runtime is incomplete. Run Install Splat Studio.command again." >&2
  exit 3
fi

cd "$APP_DIR" || {
  echo "ERROR: Could not enter Splat Studio directory." >> "$LOG_FILE"
  exit 4
}

echo "Starting SplatStudio.py..." >> "$LOG_FILE"

# Launch Streamlit directly instead of indirectly opening the .command file.
# nohup detaches it from the short-lived AppleScript app process.
nohup "$PYTHON" -m streamlit run "$STUDIO_PY" \
  --server.address=127.0.0.1 \
  --server.port=8501 \
  --server.headless=true \
  >> "$LOG_FILE" 2>&1 < /dev/null &

STREAMLIT_PID=$!
printf '%s\n' "$STREAMLIT_PID" > "$PID_FILE"
echo "Started Streamlit PID $STREAMLIT_PID." >> "$LOG_FILE"

# Wait up to ~45 seconds. Do NOT open the browser unless Streamlit actually
# reports itself healthy.
for attempt in {1..90}; do
  if ! /bin/kill -0 "$STREAMLIT_PID" 2>/dev/null; then
    echo "ERROR: Streamlit process exited before becoming healthy." >> "$LOG_FILE"
    rm -f "$PID_FILE"
    echo "SplatStudio.py exited during startup. See $LOG_FILE" >&2
    exit 5
  fi

  if /usr/bin/curl -fsS --max-time 1 "$HEALTH_URL" 2>/dev/null | /usr/bin/grep -qi "ok"; then
    echo "Streamlit health check passed." >> "$LOG_FILE"
    /usr/bin/open "$URL"
    exit 0
  fi

  sleep 0.5
done

echo "ERROR: Timed out waiting for Streamlit health endpoint." >> "$LOG_FILE"
echo "SplatStudio.py did not become ready within 45 seconds. See $LOG_FILE" >&2
exit 6
LAUNCHER_EOF

chmod +x "$RESOURCES/launch_splatstudio.sh"

# ---------------------------------------------------------------------------
# Build a proper macOS ICNS icon directly from SplatStudio_icon.png.
# Use the existing SplatStudio.icns only as a fallback.
# ---------------------------------------------------------------------------

ICON_READY="N"

if [[ -f "$PNG_ICON" ]]; then
  echo "Building app icon from:"
  echo "  $PNG_ICON"

  mkdir -p "$ICONSET"

  make_icon() {
    local pixels="$1"
    local filename="$2"
    /usr/bin/sips -s format png -z "$pixels" "$pixels" "$PNG_ICON" \
      --out "$ICONSET/$filename" >/dev/null 2>&1
  }

  make_icon 16   "icon_16x16.png"       || true
  make_icon 32   "icon_16x16@2x.png"    || true
  make_icon 32   "icon_32x32.png"       || true
  make_icon 64   "icon_32x32@2x.png"    || true
  make_icon 128  "icon_128x128.png"     || true
  make_icon 256  "icon_128x128@2x.png"  || true
  make_icon 256  "icon_256x256.png"     || true
  make_icon 512  "icon_256x256@2x.png"  || true
  make_icon 512  "icon_512x512.png"     || true
  make_icon 1024 "icon_512x512@2x.png"  || true

  if /usr/bin/iconutil -c icns "$ICONSET" -o "$GENERATED_ICNS" >/dev/null 2>&1; then
    cp "$GENERATED_ICNS" "$RESOURCES/SplatStudio.icns"
    ICON_READY="Y"
  else
    echo "WARNING: Could not build ICNS from the PNG."
  fi
fi

if [[ "$ICON_READY" != "Y" && -f "$ICNS_FALLBACK" ]]; then
  echo "Using existing SplatStudio.icns fallback."
  cp "$ICNS_FALLBACK" "$RESOURCES/SplatStudio.icns"
  ICON_READY="Y"
fi

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------

/usr/bin/plutil -replace CFBundleName -string "Splat Studio" "$PLIST" 2>/dev/null || true
/usr/bin/plutil -replace CFBundleDisplayName -string "Splat Studio" "$PLIST" 2>/dev/null || true
/usr/bin/plutil -replace CFBundleIdentifier -string "com.blondothenerd.splatstudio.launcher" "$PLIST" 2>/dev/null || true
/usr/bin/plutil -replace CFBundleShortVersionString -string "1.0" "$PLIST" 2>/dev/null || true
/usr/bin/plutil -replace CFBundleVersion -string "1" "$PLIST" 2>/dev/null || true
/usr/bin/plutil -replace LSMinimumSystemVersion -string "14.0" "$PLIST" 2>/dev/null || true

if [[ "$ICON_READY" == "Y" ]]; then
  /usr/bin/plutil -replace CFBundleIconFile -string "SplatStudio.icns" "$PLIST" 2>/dev/null || true
fi

# Remove inherited quarantine from the locally generated bundle where possible.
# This does not bypass Gatekeeper for downloaded public releases; this .app is
# generated locally by the user.
xattr -cr "$OUTPUT_APP" 2>/dev/null || true

# Ad-hoc sign the locally generated launcher after all resources are modified.
if command -v codesign >/dev/null 2>&1; then
  /usr/bin/codesign --force --deep --sign - "$OUTPUT_APP" >/dev/null 2>&1 || true
fi

touch "$OUTPUT_APP"

echo
echo "============================================================"
echo "Created successfully"
echo "============================================================"
echo
echo "App:"
echo "  $OUTPUT_APP"
echo
if [[ "$ICON_READY" == "Y" ]]; then
  echo "Icon:"
  echo "  SplatStudio_icon.png -> SplatStudio.icns  ✓"
else
  echo "Icon:"
  echo "  No usable PNG/ICNS icon was found.       !"
fi
echo
echo "The app will:"
echo "  ✓ start SplatStudio.py directly"
echo "  ✓ restore Homebrew and Xcode paths"
echo "  ✓ wait until Streamlit is actually healthy"
echo "  ✓ open the browser only after startup succeeds"
echo "  ✓ offer the installer if the runtime is missing"
echo "  ✓ write startup diagnostics to:"
echo "    $APP_DIR/.splat_studio/app-launch.log"
echo
echo "You can drag Splat Studio.app into /Applications."
echo
echo "The generated app points back to this installation:"
echo "  $APP_DIR"
echo
echo "If that folder is moved or renamed later, run this creator again."
echo

printf "Reveal Splat Studio.app in Finder now? [Y/n] "
IFS= read -r REVEAL_REPLY
REVEAL_REPLY="${REVEAL_REPLY:-Y}"

if [[ "$REVEAL_REPLY" == [Yy]* ]]; then
  open -R "$OUTPUT_APP" >/dev/null 2>&1 || true
fi

pause_close
