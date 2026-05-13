#!/usr/bin/env bash
# Install lerobot-isaac-meta + 6 dep repos from local bare repos.
#
# TODO: swap file:// URLs for https://github.com/kvgork/<name>.git when repos are published.
#
# Usage:
#   bash scripts/install.sh                  # uses ~/workspaces/spinouts/
#   SPINOUTS_DIR=/path/to/dir bash scripts/install.sh
#
# Override with SPINOUTS_DIR for CI / containers / non-default layouts.
#
# Note: this script targets POST-SPINOUT install of meta. Inside the monorepo
# you do NOT need to run this — `pixi install` already gives editable workspace
# installs of all 8 packages.

set -euo pipefail

SPINOUTS_DIR="${SPINOUTS_DIR:-$HOME/workspaces/spinouts}"

if [[ ! -d "$SPINOUTS_DIR" ]]; then
  echo "ERROR: SPINOUTS_DIR does not exist: $SPINOUTS_DIR" >&2
  echo "Either create the bare repos there (see docs/runbook/00-install.md)" >&2
  echo "or set SPINOUTS_DIR to a directory containing the .git bare repos." >&2
  exit 1
fi

echo "Installing lerobot-isaac-meta with [post-spinout] extra from $SPINOUTS_DIR..."
echo ""

# Try the git+file:// install path first (canonical post-spinout install).
# Fallback to local editable meta if the bare repo for meta isn't published yet
# (meta currently lives in the monorepo only).
if [[ -d "$SPINOUTS_DIR/lerobot-isaac-meta.git" ]]; then
  pip install "git+file://$SPINOUTS_DIR/lerobot-isaac-meta.git@main[post-spinout]"
else
  echo "Note: lerobot-isaac-meta.git not found in $SPINOUTS_DIR — falling back to local source."
  pip install -e "$(dirname "$0")/../packages/lerobot-isaac-meta[post-spinout]"
fi

echo ""
echo "Done. Recorder is standalone — install separately if needed:"
echo "  pip install git+file://$SPINOUTS_DIR/robot-data-recorder.git@main"
