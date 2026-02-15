#!/usr/bin/env bash
# Version: 0.9.0
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

check_cmd() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "[setup] missing dependency: $name" >&2
    return 1
  fi
  echo "[setup] found: $name"
}

check_cmd python3
check_cmd node
check_cmd pod

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio pytest pytest-bdd opencv-python numpy

if [[ -f "ios/Podfile" ]]; then
  (cd ios && pod install)
else
  echo "[setup] ios/Podfile not found, skipping pod install"
fi

if [[ -f "android/gradlew" ]]; then
  (cd android && chmod +x ./gradlew && ./gradlew build)
else
  echo "[setup] android/gradlew not found, skipping Android build"
fi

echo "[setup] completed"
