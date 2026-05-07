#!/usr/bin/env bash
set -euo pipefail
# Smoke-test that lerobot-isaac-configs can be subtree-split and stand alone.

PKG="${1:-lerobot-isaac-configs}"
TMPDIR=$(mktemp -d -t spinout-smoke-XXXX)
trap "rm -rf '$TMPDIR'" EXIT

cd "$(dirname "$0")/.."
WORKSPACE_ROOT=$(pwd)

# Subtree split
git subtree split --prefix="packages/$PKG" -b "spinout/$PKG-$$" 2>&1
git clone -b "spinout/$PKG-$$" "$WORKSPACE_ROOT" "$TMPDIR/$PKG"
git branch -D "spinout/$PKG-$$"

# In the spun-out tree:
cd "$TMPDIR/$PKG"
[ -f pyproject.toml ] || { echo "FAIL: no pyproject.toml in spinout"; exit 1; }
[ -f pixi.toml ] || { echo "FAIL: no pixi.toml in spinout"; exit 1; }
[ -f README.md ] || { echo "FAIL: no README.md in spinout"; exit 1; }
[ -d "src/${PKG//-/_}" ] || { echo "FAIL: no src/$(echo $PKG | tr - _)"; exit 1; }

# Run package tests
python3 -m pytest tests/ -q --no-header 2>&1 | tail -3

echo "PASS: $PKG spinout smoke test"
