#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-origin}"
MAIN_BRANCH="${2:-main}"
SERVICE_NAME="${3:-ampel.service}"
HEALTH_URL="${4:-http://127.0.0.1:8080/health}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

log() {
  printf '[update] %s\n' "$*"
}

service_exists=0
if command -v systemctl >/dev/null 2>&1 && systemctl --user show "$SERVICE_NAME" --property=Id --value >/dev/null 2>&1; then
  service_exists=1
  log "stopping service: $SERVICE_NAME"
  systemctl --user stop "$SERVICE_NAME"
else
  log "service not found, continuing without service control: $SERVICE_NAME"
fi

log "fetching latest refs"
git fetch "$REMOTE" --prune

log "switching to $MAIN_BRANCH"
git checkout "$MAIN_BRANCH"

log "fast-forward pull from $REMOTE/$MAIN_BRANCH"
git pull --ff-only "$REMOTE" "$MAIN_BRANCH"

latest_remote_branch="$(
  git for-each-ref --sort=-committerdate --format='%(refname:short)' "refs/remotes/$REMOTE" \
  | awk -v remote="$REMOTE" -v main="$MAIN_BRANCH" '$0 != remote"/HEAD" && $0 != remote"/"main {print; exit}'
)"

if [[ -n "$latest_remote_branch" ]]; then
  log "latest branch: $latest_remote_branch"
  if git merge-base --is-ancestor "$latest_remote_branch" HEAD; then
    log "already merged: $latest_remote_branch"
  else
    log "merging $latest_remote_branch"
    if ! git merge --no-ff "$latest_remote_branch" -m "Merge $latest_remote_branch into $MAIN_BRANCH"; then
      log "conflicts detected; applying policy (MEMORY.md ours, others theirs)"
      while IFS= read -r file; do
        [[ -z "$file" ]] && continue
        if [[ "$file" == "MEMORY.md" ]]; then
          git checkout --ours -- "$file"
        else
          git checkout --theirs -- "$file"
        fi
      done < <(git diff --name-only --diff-filter=U)
      git add -A
      git commit -m "Merge $latest_remote_branch into $MAIN_BRANCH"
    fi
  fi
else
  log "no remote feature branch found"
fi

log "pushing $MAIN_BRANCH"
git push "$REMOTE" "$MAIN_BRANCH"

mkdir -p var/log
: > var/log/ampel.log

if [[ "$service_exists" -eq 1 ]]; then
  log "starting service: $SERVICE_NAME"
  systemctl --user start "$SERVICE_NAME"

  log "health check: $HEALTH_URL"
  for attempt in $(seq 1 20); do
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
      log "health check passed"
      exit 0
    fi
    sleep 1
  done
  printf '[error] health check failed: %s\n' "$HEALTH_URL" >&2
  exit 1
fi

log "update complete"
