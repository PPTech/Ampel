#!/usr/bin/env bash
# Version: 0.9.4
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
# Author: Dr. Babak Sorkhpour with support from ChatGPT
set -euo pipefail

v="${1:?usage: $0 <version>}"
echo "$v" > VERSION
sed -i "s/SEMVER = \"[0-9]\+\.[0-9]\+\.[0-9]\+\"/SEMVER = \"$v\"/" src/ampel_app/cli.py
sed -i "s/Version: [0-9]\+\.[0-9]\+\.[0-9]\+/Version: $v/g" CHANGELOG.md MEMORY.md README.md || true
echo "bumped version to $v"
