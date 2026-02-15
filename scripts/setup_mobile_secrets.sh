#!/usr/bin/env bash
# Version: 0.9.2
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ANDROID_LOCAL="$ROOT_DIR/android/local.properties"
IOS_CONFIG="$ROOT_DIR/ios/Config.xcconfig"

mkdir -p "$ROOT_DIR/android" "$ROOT_DIR/ios"

if [[ ! -f "$ANDROID_LOCAL" ]]; then
  cat > "$ANDROID_LOCAL" <<'PROPS'
# Local secrets (do not commit)
MAPBOX_ACCESS_TOKEN=YOUR_MAPBOX_TOKEN_HERE
PROPS
  echo "[secrets] created $ANDROID_LOCAL"
else
  echo "[secrets] exists: $ANDROID_LOCAL"
fi

if [[ ! -f "$IOS_CONFIG" ]]; then
  cat > "$IOS_CONFIG" <<'XCC'
// Local secrets (do not commit)
MAPBOX_ACCESS_TOKEN = YOUR_MAPBOX_TOKEN_HERE
XCC
  echo "[secrets] created $IOS_CONFIG"
else
  echo "[secrets] exists: $IOS_CONFIG"
fi

echo "[secrets] done. keep these files local and never commit them."
