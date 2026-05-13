#!/usr/bin/env bash
# scripts/sync/sync_siblings.sh
# =============================
# Clone the 6 sibling spinouts into src/<name>/ for the editable-siblings workflow.
#
# Idempotent: skips any directory that already exists. To pull updates after the
# initial clone, use `pixi run sync-update` (or `scripts/sync/sync_update.sh`).
#
# Set LEROBOT_SPINOUTS_BASE to override the bare-repo base URL (default:
# file:///home/koen/workspaces/spinouts).
#
# This script is invoked by the `pixi run sync` task; do not invoke directly
# unless you know what you're doing (run it from the workspace root).
set -e
mkdir -p src
BASE="${LEROBOT_SPINOUTS_BASE:-file:///home/koen/workspaces/spinouts}"
for name in lerobot-isaac-configs lerobot-isaac-dashboard lerobot-isaac-autoresearch \
            lerobot-isaac-env lerobot-isaac-adapters lerobot-isaac-synthetic; do
  if [ -d "src/$name" ]; then
    echo "[sync] $name: exists, skipping (use sync-update to pull)"
  else
    echo "[sync] $name: cloning from $BASE/$name"
    git clone "$BASE/$name" "src/$name"
  fi
done
echo "[sync] done. next: pixi install -e editable"
