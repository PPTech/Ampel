#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="$ROOT_DIR/traffic_ai_runner.sh"

supports_arg() {
  local arg="$1"
  "$PYTHON_BIN" "$ROOT_DIR/traffic_ai_assist.py" --help 2>&1 | grep -q -- "$arg"
}

run_repo_instructions() {
  echo "Running repository instructions on main..."
  mkdir -p "$ROOT_DIR/var/data" "$ROOT_DIR/var/log" "$ROOT_DIR/features"

  if [[ -x "$RUNNER" ]]; then
    "$RUNNER" compile
  else
    "$PYTHON_BIN" -m py_compile "$ROOT_DIR/traffic_ai_assist.py"
    echo "Compile check passed: $ROOT_DIR/traffic_ai_assist.py"
  fi

  if supports_arg "--self-test"; then
    "$PYTHON_BIN" "$ROOT_DIR/traffic_ai_assist.py" --self-test
  fi

  if supports_arg "--export-gherkin"; then
    "$PYTHON_BIN" "$ROOT_DIR/traffic_ai_assist.py" --export-gherkin "$ROOT_DIR/features/traffic_ai_agent.feature"
  fi

  if supports_arg "--dataset-manifest"; then
    "$PYTHON_BIN" "$ROOT_DIR/traffic_ai_assist.py" --dataset-manifest > "$ROOT_DIR/var/data/dataset_manifest.json"
  fi

  if supports_arg "--sync-datasets"; then
    "$PYTHON_BIN" "$ROOT_DIR/traffic_ai_assist.py" --sync-datasets --db "$ROOT_DIR/var/data/traffic_ai.sqlite3"
  fi

  if supports_arg "--fetch-dataset-metadata"; then
    "$PYTHON_BIN" "$ROOT_DIR/traffic_ai_assist.py" --fetch-dataset-metadata "$ROOT_DIR/dataset_meta.json"
  fi

  if supports_arg "--serve"; then
    if command -v systemctl >/dev/null 2>&1; then
      systemctl --user restart ampel.service || true
      systemctl --user is-active ampel.service || true
    fi
  else
    echo "No --serve mode in current main; skipped service restart."
  fi
}

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Error: commit or stash tracked local changes before sync."
  exit 1
fi

current_ref="$(git rev-parse --abbrev-ref HEAD)"

cleanup() {
  if [[ "$current_ref" != "main" ]]; then
    git switch "$current_ref" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

git fetch origin --prune

latest_remote_branch="$(
  git for-each-ref --sort=-committerdate --format='%(refname:short)' refs/remotes/origin \
    | awk '$0 != "origin/main" && $0 != "origin/HEAD" {print; exit}'
)"

if [[ -z "${latest_remote_branch:-}" ]]; then
  echo "No remote branch found to merge."
  exit 0
fi

echo "Latest remote branch: $latest_remote_branch"

git switch main >/dev/null
git pull --ff-only origin main

if git merge-base --is-ancestor "$latest_remote_branch" main; then
  echo "main already contains $latest_remote_branch"
else
  git merge --no-ff "$latest_remote_branch" -m "Merge $latest_remote_branch into main"
fi

git push origin main
git fetch origin --prune

run_repo_instructions

echo "Sync complete: main is updated and pushed."
