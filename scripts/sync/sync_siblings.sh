#!/usr/bin/env bash
# scripts/sync/sync_siblings.sh
# =============================
# Clone the 6 sibling spinouts into src/<name>/ for the editable-siblings workflow.
#
# Resolution order for each sibling:
#   1. If src/<name>/ already exists → skip (development workspace; develop in place).
#   2. If $LEROBOT_SPINOUTS_BASE/<name> is reachable (local bare repo or
#      file:// path) → clone from there. Falls back to GitHub on failure.
#   3. Else clone from https://github.com/kvgork/<name>.git.
#
# To pull updates after initial clone, use `pixi run sync-update`.
#
# This script is invoked by the `pixi run sync` task and by
# `scripts/setup.sh` (canonical first-time bootstrap).
set -eu

mkdir -p src

LOCAL_BASE="${LEROBOT_SPINOUTS_BASE:-file://${HOME}/workspaces/spinouts}"
GITHUB_BASE="${LEROBOT_SIBLINGS_GITHUB_BASE:-https://github.com/kvgork}"

SIBLINGS=(
  lerobot-isaac-configs
  lerobot-isaac-dashboard
  lerobot-isaac-autoresearch
  lerobot-isaac-env
  lerobot-isaac-adapters
  lerobot-isaac-synthetic
  lerobot-isaac-deploy
)

_try_clone() {
  local url="$1"
  local dest="$2"
  # ls-remote is cheap and works for both file:// and https://.
  git ls-remote --exit-code -h "$url" >/dev/null 2>&1 || return 1
  git clone "$url" "$dest"
}

for name in "${SIBLINGS[@]}"; do
  if [ -d "src/$name" ]; then
    echo "[sync] $name: exists in src/ (development), skipping"
    continue
  fi

  echo "[sync] $name: not in src/, cloning..."

  # Tier 1: local bare repo (fast, offline-friendly)
  if _try_clone "${LOCAL_BASE}/${name}" "src/$name"; then
    echo "[sync] $name: cloned from ${LOCAL_BASE}/${name}"
    continue
  fi

  # Tier 2: GitHub HTTPS
  if _try_clone "${GITHUB_BASE}/${name}.git" "src/$name"; then
    echo "[sync] $name: cloned from ${GITHUB_BASE}/${name}.git"
    continue
  fi

  echo "[sync] ERROR: cannot clone $name from any source" >&2
  echo "[sync]        tried: ${LOCAL_BASE}/${name}" >&2
  echo "[sync]        tried: ${GITHUB_BASE}/${name}.git" >&2
  echo "[sync]        check network + LEROBOT_SPINOUTS_BASE env var" >&2
  exit 1
done

echo "[sync] done. All 6 siblings present in src/."
echo "[sync] next: pixi install     # uses editable installs from src/"
