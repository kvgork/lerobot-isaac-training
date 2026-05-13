#!/usr/bin/env bash
# scripts/sync/sync_recorder.sh
# =============================
# Opt-in: clone robot_data_recorder into src/robot-data-recorder/ for local
# development. Recorder is intentionally NOT a workspace dep — install it
# manually after this script with:
#   pixi run -e default pip install -e src/robot-data-recorder
#
# NOTE: the on-disk bare repo currently lives at
# $LEROBOT_SPINOUTS_BASE/robot_data_recorder (underscore, working tree —
# not yet bare). When the repo is bare-ified or published to GitHub, update
# the URL below.
#
# This script is invoked by the `pixi run sync-recorder` task.
set -e
mkdir -p src
BASE="${LEROBOT_SPINOUTS_BASE:-file:///home/koen/workspaces/spinouts}"
if [ -d "src/robot-data-recorder" ]; then
  echo "[sync-recorder] exists, skipping"
else
  echo "[sync-recorder] cloning from $BASE/robot_data_recorder"
  git clone "$BASE/robot_data_recorder" "src/robot-data-recorder" || true
fi
