#!/usr/bin/env bash
# scripts/sync/sync_update.sh
# ===========================
# Pull updates from origin for each local sibling clone in src/lerobot-isaac-*.
# Only fast-forward pulls are performed — diverged clones are reported but not
# rewritten. Resolve divergence manually inside the affected src/<pkg>/ checkout.
#
# This script is invoked by the `pixi run sync-update` task.
set -e
shopt -s nullglob
for d in src/lerobot-isaac-*; do
  [ -d "$d" ] || continue
  echo "[sync-update] $d"
  git -C "$d" fetch && git -C "$d" pull --ff-only
done
