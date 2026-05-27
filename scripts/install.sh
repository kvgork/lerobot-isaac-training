#!/usr/bin/env bash
# Install lerobot-isaac-meta + its 6 sibling packages from public GitHub repos.
#
# This is the plain-pip install path (no pixi). It installs meta with the
# `post-spinout` extra, which pulls the 6 siblings from github.com/kvgork/<name>.
#
# Usage:
#   bash scripts/install.sh            # install into the active environment
#
# Inside the monorepo you do NOT need this — `bash scripts/setup.sh` (or
# `pixi install`) gives editable installs of all packages from src/.
#
# Recorder (robot-data-recorder) is a standalone hardware-tier package and is
# NOT pulled here. Install it separately when needed.

set -euo pipefail

META_DIR="$(cd "$(dirname "$0")/.." && pwd)/packages/lerobot-isaac-meta"

echo "Installing lerobot-isaac-meta[post-spinout] (siblings pulled from GitHub)..."
echo ""

pip install "${META_DIR}[post-spinout]"

echo ""
echo "Done. Siblings resolved from https://github.com/kvgork/<name>."
