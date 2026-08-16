#!/bin/zsh

# Creates a small local macOS .app launcher for an installed Splat Studio.
# The generated app stores the current local Splat Studio folder path so the
# repository itself remains fully portable and contains no user-specific path.

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCHER="$APP_DIR/Launch Splat Studio.command"
ICON="$APP_DIR/SplatStudio.icns"
OUTPUT_APP="$APP_DIR/Splat Studio.app"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/splatstudio-app.XXXXXX")"
SCRIPT_FILE="$TMP_DIR/launcher.applescript"

cleanup() {
  rm -rf "$TMP_DIR" 2>/dev/null || true
}
trap cleanup EXIT

echo
echo "=== Create Splat Studio.app ==="
echo

if [[ ! -f "$LAUNCHER" ]]; then
  echo "Launch Splat Studio.command was not found:"
  echo "  $LAUNCHER"
  echo
  echo "Download/restore the complete Splat Studio repository first."
  printf "\nPress Return to close..."
  IFS= read -r _
  exit 1
fi

if [[ ! -x "$APP_DIR/runtime/gsplat_core/bin/python" ]]; then
  echo "Splat Studio does not appear to be installed yet."
  echo "Run Install Splat Studio.command first."
  printf "\nPress Return to close..."
  IFS= read -r _
  exit 1
fi

# Escape the absolute launcher path for an AppleScript string.
ESCAPED_LAUNCHER="${LAUNCHER//\\/\\\\}"
ESCAPED_LAUNCHER="${ESCAPED_LAUNCHER//\"/\\\"}"

cat > "$SCRIPT_FILE" <<EOF
on run
    set launcherPath to "$ESCAPED_LAUNCHER"

    try
        do shell script "/usr/bin/nohup /bin/zsh " & quoted form of launcherPath & " > /tmp/splatstudio-launch.log 2>&1 < /dev/null &"
    on error errorMessage
        display dialog "Splat Studio could not be started:" & return & return & errorMessage buttons {"OK"} default button "OK" with icon stop
        return
    end try

    repeat 30 times
        delay 0.5
        try
            do shell script "/usr/bin/curl -fsS http://127.0.0.1:8501/_stcore/health > /dev/null"
            exit repeat
        end try
    end repeat

    open location "http://127.0.0.1:8501"
end run
EOF

rm -rf "$OUTPUT_APP"

if ! /usr/bin/osacompile -o "$OUTPUT_APP" "$SCRIPT_FILE"; then
  echo
  echo "Could not create the .app bundle."
  printf "\nPress Return to close..."
  IFS= read -r _
  exit 1
fi

PLIST="$OUTPUT_APP/Contents/Info.plist"

/usr/bin/plutil -replace CFBundleName -string "Splat Studio" "$PLIST" 2>/dev/null || true
/usr/bin/plutil -replace CFBundleDisplayName -string "Splat Studio" "$PLIST" 2>/dev/null || true
/usr/bin/plutil -replace CFBundleIdentifier -string "com.blondothenerd.splatstudio.launcher" "$PLIST" 2>/dev/null || true
/usr/bin/plutil -replace CFBundleShortVersionString -string "1.0" "$PLIST" 2>/dev/null || true
/usr/bin/plutil -replace CFBundleVersion -string "1" "$PLIST" 2>/dev/null || true

if [[ -f "$ICON" ]]; then
  cp "$ICON" "$OUTPUT_APP/Contents/Resources/SplatStudio.icns"
  /usr/bin/plutil -replace CFBundleIconFile -string "SplatStudio.icns" "$PLIST" 2>/dev/null || true
fi

# Ad-hoc signing is enough for a locally generated launcher and avoids leaving
# the bundle completely unsigned. This is not App Store/notarization signing.
if command -v codesign >/dev/null 2>&1; then
  /usr/bin/codesign --force --deep --sign - "$OUTPUT_APP" >/dev/null 2>&1 || true
fi

touch "$OUTPUT_APP"

echo
echo "Created:"
echo "  $OUTPUT_APP"
echo
echo "You can drag Splat Studio.app into /Applications if you want."
echo
echo "IMPORTANT:"
echo "The generated app points to this Splat Studio installation:"
echo "  $APP_DIR"
echo "If you move or rename the Splat Studio folder later, run this creator again."
echo

printf "Reveal the app in Finder now? [Y/n] "
IFS= read -r REVEAL_REPLY
REVEAL_REPLY="${REVEAL_REPLY:-Y}"

if [[ "$REVEAL_REPLY" == [Yy]* ]]; then
  open -R "$OUTPUT_APP" >/dev/null 2>&1 || true
fi

printf "\nPress Return to close..."
IFS= read -r _
