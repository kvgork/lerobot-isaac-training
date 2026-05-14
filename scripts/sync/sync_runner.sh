#!/usr/bin/env bash
# scripts/sync/sync_runner.sh
# ===========================
# Opt-in: clone robot_data_runner into src/robot-data-runner/ for local
# development. Runner is a standalone hardware-deploy CLI — sibling of
# robot-data-recorder. Intentionally NOT a workspace dep.
#
# After this script:
#   pixi run -e train-policy pip install -e src/robot-data-runner
#
# Invoked by `pixi run sync-runner`.
set -e
mkdir -p src
BASE="${LEROBOT_SPINOUTS_BASE:-file:///home/koen/workspaces/spinouts}"
if [ -d "src/robot-data-runner" ]; then
  echo "[sync-runner] exists, skipping"
else
  echo "[sync-runner] cloning from $BASE/robot_data_runner"
  git clone "$BASE/robot_data_runner" "src/robot-data-runner" || true
fi
