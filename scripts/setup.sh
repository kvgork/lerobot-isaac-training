#!/usr/bin/env bash
# scripts/setup.sh
# ================
# Canonical first-time bootstrap for the lerobot-isaac-training workspace.
#
# This is the script you run after `git clone` to get a working development
# install. It:
#   1. Loads workspace env vars (scripts/setup_env.sh).
#   2. Clones the 6 sibling spinouts into src/<name>/ if missing
#      (scripts/sync/sync_siblings.sh — uses local bare repo first, then
#      GitHub HTTPS fallback).
#   3. Optionally clones robot-data-recorder when --recorder is passed.
#   4. Runs `pixi install` against the requested environment (default: default).
#
# After this script returns, all sibling packages are editable from src/ —
# edits to src/<name>/ reflect immediately in the active env, no reinstall.
#
# Usage:
#   bash scripts/setup.sh                     # default env
#   bash scripts/setup.sh -e sim              # sim env (Isaac Lab)
#   bash scripts/setup.sh -e full --recorder  # everything + recorder sibling
#   bash scripts/setup.sh --dry-run           # print steps, do nothing
set -eu

ENV_NAME="default"
INCLUDE_RECORDER=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    -e|--env)        ENV_NAME="$2"; shift 2 ;;
    --recorder)      INCLUDE_RECORDER=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,25p' "$0"; exit 0 ;;
    *)
      echo "setup.sh: unknown arg: $1" >&2
      echo "use --help for usage" >&2
      exit 2 ;;
  esac
done

WS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS_ROOT"

_log() { echo "[setup] $*"; }
_run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[setup] DRY-RUN: $*"
  else
    "$@"
  fi
}

_log "workspace = $WS_ROOT"
_log "env       = $ENV_NAME"
_log "recorder  = $INCLUDE_RECORDER"

# Step 1: env vars
if [ -f scripts/setup_env.sh ]; then
  _log "loading scripts/setup_env.sh"
  # shellcheck source=/dev/null
  if [ "$DRY_RUN" -eq 0 ]; then . scripts/setup_env.sh; fi
fi

# Step 2: clone siblings
_log "ensuring 6 sibling packages in src/ ..."
_run bash scripts/sync/sync_siblings.sh

# Step 3: optional recorder
if [ "$INCLUDE_RECORDER" -eq 1 ]; then
  _log "ensuring robot-data-recorder in src/ ..."
  _run bash scripts/sync/sync_recorder.sh
fi

# Step 4: pixi install
_log "pixi install -e $ENV_NAME ..."
_run pixi install -e "$ENV_NAME"

# Step 5: ensure SO-101 USD assets are present in src/ for editable lookup
# (resolve_usd_path() reads pkg_root/assets/usd/so101.usd in the src/ tree).
# The installed wheel inside .pixi/envs ships them; on fresh editable installs
# we mirror them into src/ so env construction doesn't crash.
USD_SRC_DIR="src/lerobot-isaac-env/assets/usd"
USD_WHEEL_DIR=".pixi/envs/${ENV_NAME}/lib/python3.12/assets/usd"
if [ -d "$USD_WHEEL_DIR" ] && [ ! -f "${USD_SRC_DIR}/Payload/Contents.usda" ]; then
  _log "copying SO-101 USD assets from wheel cache to ${USD_SRC_DIR}"
  _run cp -n "${USD_WHEEL_DIR}/so101.usd" "${USD_SRC_DIR}/so101.usd"
  _run cp -n "${USD_WHEEL_DIR}/so101_new_calib.usda" "${USD_SRC_DIR}/so101_new_calib.usda"
  _run rm -rf "${USD_SRC_DIR}/Payload"
  _run cp -r "${USD_WHEEL_DIR}/Payload" "${USD_SRC_DIR}/Payload"
else
  _log "SO-101 USD: ${USD_SRC_DIR}/Payload already present, skipping copy"
fi

_log "done."
_log "Activate with: pixi shell -e $ENV_NAME"
_log "Or run tasks:  pixi run -e $ENV_NAME <task>"
_log ""
_log "Edits to src/<sibling>/ take effect immediately (editable installs)."
_log "Push changes via: cd src/<sibling> && git push"
