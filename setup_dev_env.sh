#!/usr/bin/env bash
# Version: 0.9.12
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

set -euo pipefail

echo "[Ampel] Setting up unified Samsung + Raspberry Pi development environment"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y \
    git curl unzip zip jq openjdk-17-jdk python3 python3-venv python3-pip docker.io
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r proto/python/requirements.txt || true
pip install pytest pytest-bdd pre-commit

pre-commit install || true

if ! command -v docker >/dev/null 2>&1; then
  echo "[WARN] Docker not found after setup; install manually for gadget image builds."
else
  sudo systemctl enable docker || true
  sudo systemctl start docker || true
fi

ANDROID_SDK_ROOT_DEFAULT="${HOME}/Android/Sdk"
export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$ANDROID_SDK_ROOT_DEFAULT}"
mkdir -p "${ANDROID_SDK_ROOT}/cmdline-tools"

if [ ! -d "${ANDROID_SDK_ROOT}/cmdline-tools/latest" ]; then
  echo "[INFO] Android cmdline-tools not installed automatically in this script."
  echo "[INFO] Download commandline-tools and place under ${ANDROID_SDK_ROOT}/cmdline-tools/latest"
fi

echo "[DONE] Setup complete. Activate Python env with: source .venv/bin/activate"
