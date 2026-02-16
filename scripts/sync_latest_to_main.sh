#!/usr/bin/env bash
# Version: 0.9.0
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
set -euo pipefail

REMOTE="${1:-origin}"
TARGET_BRANCH="${2:-main}"
WORK_BRANCH="$(git branch --show-current)"

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "Remote '$REMOTE' is not configured." >&2
  exit 2
fi

echo "[sync] fetching latest from $REMOTE ..."
git fetch "$REMOTE" --prune

if git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
  git checkout "$TARGET_BRANCH"
else
  git checkout -b "$TARGET_BRANCH"
fi

if git show-ref --verify --quiet "refs/remotes/$REMOTE/$TARGET_BRANCH"; then
  echo "[sync] attempting fast-forward from $REMOTE/$TARGET_BRANCH ..."
  git merge --ff-only "$REMOTE/$TARGET_BRANCH" || echo "[warn] ff merge skipped"
fi

if [[ "$WORK_BRANCH" != "$TARGET_BRANCH" ]]; then
  echo "[sync] merging $WORK_BRANCH -> $TARGET_BRANCH ..."
  git merge --no-ff "$WORK_BRANCH" -m "merge: sync $WORK_BRANCH into $TARGET_BRANCH"
fi

echo "[sync] running compile/safety health checks ..."
python3 scripts/dataset_manager.py --compile-manifest
python3 scripts/dataset_manager.py --sync
python3 scripts/dataset_manager.py --build-training-plan docs/TRAINING_PLAN.json
python3 traffic_ai_assist.py --sync-datasets --db traffic_ai.sqlite3
python3 traffic_ai_assist.py --ab-test --db traffic_ai.sqlite3 >/tmp/ab_test_sync.json
python3 traffic_ai_assist.py --security-check >/tmp/security_sync.json

echo "[sync] AB result"
cat /tmp/ab_test_sync.json

echo "[sync] Security result"
cat /tmp/security_sync.json

echo "[sync] done. push with: git push $REMOTE $TARGET_BRANCH"
