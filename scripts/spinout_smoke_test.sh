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
# Determine src package dir name from pyproject.toml [tool.hatch.build.targets.wheel].packages,
# falling back to PKG//-/_ if not declared.
SRC_PKG=$(python3 -c "
import sys, tomllib
with open('pyproject.toml','rb') as f:
    d = tomllib.load(f)
pkgs = d.get('tool',{}).get('hatch',{}).get('build',{}).get('targets',{}).get('wheel',{}).get('packages')
if pkgs and pkgs[0].startswith('src/'):
    print(pkgs[0][4:])
else:
    print('$PKG'.replace('-','_'))
" 2>/dev/null || echo "${PKG//-/_}")

[ -d "src/$SRC_PKG" ] || { echo "FAIL: no src/$SRC_PKG (pyproject-declared or fallback)"; exit 1; }

# Run package tests
python3 -m pytest tests/ -q --no-header 2>&1 | tail -3

echo "PASS: $PKG spinout smoke test (src/$SRC_PKG)"
