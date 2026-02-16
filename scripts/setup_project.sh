#!/usr/bin/env bash
# Version: 0.9.4
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
# Author: Dr. Babak Sorkhpour with support from ChatGPT
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing dependency: $1"; exit 1; }
}

require_cmd node
require_cmd python3

if ! command -v pod >/dev/null 2>&1; then
  echo "warning: CocoaPods not found; iOS dependency install skipped"
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/dev.txt

if [[ -d ios ]] && command -v pod >/dev/null 2>&1; then
  (cd ios && pod install || true)
fi

if [[ -f android/gradlew ]]; then
  (cd android && ./gradlew build || true)
fi

echo "setup complete (pinned requirements)"
