#!/usr/bin/env bash
# Version: 1.0.0
# License: MIT
# Code generated with support from CODEX and CODEX CLI.
# Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
set -euo pipefail

REMOTE="${1:-origin}"
TARGET_BRANCH="${2:-main}"
SERVICE_NAME="${3:-ampel.service}"
HEALTH_URL="${4:-http://127.0.0.1:8080/health}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_PATHS=("traffic_ai.sqlite3" "var/data/traffic_ai.sqlite3" "var/log/ampel.log")
STASH_NAME="runtime-sync-$(date +%Y%m%d-%H%M%S)"

cd "$REPO_ROOT"

log() {
  printf '[sync] %s\n' "$*"
}

warn() {
  printf '[warn] %s\n' "$*" >&2
}

die() {
  printf '[error] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

is_runtime_path() {
  local path="$1"
  for runtime_path in "${RUNTIME_PATHS[@]}"; do
    if [[ "$path" == "$runtime_path" ]]; then
      return 0
    fi
  done
  return 1
}

stash_created=0
service_managed=0
service_stopped=0
service_restarted=0

cleanup() {
  local exit_code=$?

  if (( stash_created == 1 )); then
    if (( exit_code == 0 )); then
      log "restoring runtime stash ($STASH_NAME)"
      if ! git stash pop --quiet; then
        warn "could not auto-restore runtime stash. Apply manually with: git stash pop"
      fi
    else
      warn "runtime stash kept for safety ($STASH_NAME)"
    fi
  fi

  if (( service_managed == 1 )) && (( service_stopped == 1 )) && (( service_restarted == 0 )); then
    warn "service was stopped before failure; attempting emergency restart"
    systemctl --user start "$SERVICE_NAME" >/dev/null 2>&1 || warn "failed to restart $SERVICE_NAME"
  fi

  exit "$exit_code"
}
trap cleanup EXIT

require_cmd git
require_cmd curl

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  die "not inside a git repository"
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  die "remote '$REMOTE' is not configured"
fi

if command -v systemctl >/dev/null 2>&1 && systemctl --user show "$SERVICE_NAME" --property=Id --value >/dev/null 2>&1; then
  service_managed=1
  log "stopping service: $SERVICE_NAME"
  systemctl --user stop "$SERVICE_NAME"
  service_stopped=1
else
  warn "user service '$SERVICE_NAME' not found; skip stop/start"
fi

mapfile -t dirty_files < <(
  {
    git diff --name-only
    git diff --cached --name-only
    git ls-files --others --exclude-standard
  } | sed '/^$/d' | sort -u
)

if (( ${#dirty_files[@]} > 0 )); then
  non_runtime=()
  runtime_dirty=()
  for file in "${dirty_files[@]}"; do
    if ! is_runtime_path "$file"; then
      non_runtime+=("$file")
    else
      runtime_dirty+=("$file")
    fi
  done

  if (( ${#non_runtime[@]} > 0 )); then
    printf '[error] repo has non-runtime changes; clean them first:\n' >&2
    printf '  - %s\n' "${non_runtime[@]}" >&2
    exit 2
  fi

  if (( ${#runtime_dirty[@]} > 0 )); then
    log "stashing runtime drift"
    git stash push -u -m "$STASH_NAME" -- "${runtime_dirty[@]}" >/dev/null
    stash_created=1
  fi
fi

log "fetching from $REMOTE"
git fetch "$REMOTE" --prune

if git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
  git switch "$TARGET_BRANCH" >/dev/null 2>&1 || git checkout "$TARGET_BRANCH" >/dev/null
else
  git checkout -b "$TARGET_BRANCH" >/dev/null
fi

if git show-ref --verify --quiet "refs/remotes/$REMOTE/$TARGET_BRANCH"; then
  log "fast-forwarding from $REMOTE/$TARGET_BRANCH"
  git merge --ff-only "$REMOTE/$TARGET_BRANCH"
fi

latest_remote_branch=""
while read -r ref; do
  case "$ref" in
    "$REMOTE/HEAD"|"$REMOTE/$TARGET_BRANCH") continue ;;
  esac
  latest_remote_branch="$ref"
  break
done < <(git for-each-ref --sort=-committerdate --format='%(refname:short)' "refs/remotes/$REMOTE")

if [[ -z "$latest_remote_branch" ]]; then
  log "no remote branch found to merge"
else
  log "latest remote branch: $latest_remote_branch"
  if git merge-base --is-ancestor "$latest_remote_branch" HEAD; then
    log "$latest_remote_branch already merged into $TARGET_BRANCH"
  else
    log "merging $latest_remote_branch into $TARGET_BRANCH"
    if ! git merge --no-ff "$latest_remote_branch" -m "Merge $latest_remote_branch into $TARGET_BRANCH"; then
      warn "merge conflict detected; applying resolution policy"
      mapfile -t conflict_files < <(git diff --name-only --diff-filter=U)
      if (( ${#conflict_files[@]} == 0 )); then
        die "merge failed but no conflicted files were listed"
      fi
      for file in "${conflict_files[@]}"; do
        if [[ "$file" == "MEMORY.md" ]]; then
          git checkout --ours -- "$file"
        else
          git checkout --theirs -- "$file"
        fi
      done
      git add -A
      git commit -m "Merge $latest_remote_branch into $TARGET_BRANCH"
    fi
  fi
fi

log "pushing $TARGET_BRANCH to $REMOTE"
git push "$REMOTE" "$TARGET_BRANCH"

mkdir -p var/log var/data
touch var/log/ampel.log

if (( service_managed == 1 )); then
  log "starting service: $SERVICE_NAME"
  systemctl --user start "$SERVICE_NAME"
  service_restarted=1

  log "checking health: $HEALTH_URL"
  health_ok=0
  for attempt in $(seq 1 20); do
    if curl -fsS "$HEALTH_URL" >/tmp/ampel-health.json 2>/dev/null; then
      health_ok=1
      break
    fi
    sleep 1
  done
  if (( health_ok == 0 )); then
    die "service did not become healthy at $HEALTH_URL"
  fi
  cat /tmp/ampel-health.json
fi

log "sync complete"
