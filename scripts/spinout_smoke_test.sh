#!/usr/bin/env bash
set -euo pipefail
# Smoke-test that a package can be subtree-split and stand alone.
#
# After subtree-split we install via the package's dormant pixi.toml and run
# pytest inside that environment. This mirrors what a downstream consumer
# would do after cloning the spun-out repo.

PKG="${1:-lerobot-isaac-configs}"
TMPDIR=$(mktemp -d -t spinout-smoke-XXXX)
trap "rm -rf '$TMPDIR'" EXIT

cd "$(dirname "$0")/.."
WORKSPACE_ROOT=$(pwd)

# --- timing helper ------------------------------------------------------------
_t0=$(date +%s)
_step() {
    local _now=$(date +%s)
    local _dt=$((_now - _t0))
    echo "[+${_dt}s] $1"
}

# --- 1. subtree split ---------------------------------------------------------
# Locate the package: live workspace member (packages/) vs archived (archive/packages/).
if [ -d "$WORKSPACE_ROOT/packages/$PKG" ]; then
    PKG_PREFIX="packages/$PKG"
elif [ -d "$WORKSPACE_ROOT/archive/packages/$PKG" ]; then
    PKG_PREFIX="archive/packages/$PKG"
else
    echo "FAIL: package $PKG not found in packages/ or archive/packages/" >&2
    exit 1
fi
_step "subtree split: $PKG_PREFIX"
git subtree split --prefix="$PKG_PREFIX" -b "spinout/$PKG-$$" 2>&1 >/dev/null
git clone -q -b "spinout/$PKG-$$" "$WORKSPACE_ROOT" "$TMPDIR/$PKG"
git branch -D "spinout/$PKG-$$" >/dev/null

# In the spun-out tree:
cd "$TMPDIR/$PKG"

# --- 2. structural checks -----------------------------------------------------
_step "structural checks"
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

# --- 3. pixi install in the spun-out tree -------------------------------------
# Detect whether the package's pixi.toml defines a `default` environment.
HAS_DEFAULT_ENV=$(python3 -c "
import tomllib
with open('pixi.toml','rb') as f:
    d = tomllib.load(f)
print('1' if 'default' in d.get('environments', {}) else '0')
" 2>/dev/null || echo "0")

_step "pixi install (env-aware)"
if [ "$HAS_DEFAULT_ENV" = "1" ]; then
    pixi install -e default >/dev/null
else
    pixi install >/dev/null
fi

# --- 4. pytest inside the pixi env -------------------------------------------
_step "pytest (excluding GPU-only markers)"
PYTEST_MARKERS='not requires_isaaclab and not requires_lerobot and not requires_dreamerv3'
if [ "$HAS_DEFAULT_ENV" = "1" ]; then
    pixi run -e default pytest tests/ -q --no-header -m "$PYTEST_MARKERS" 2>&1 | tail -5
else
    pixi run pytest tests/ -q --no-header -m "$PYTEST_MARKERS" 2>&1 | tail -5
fi

_step "done"
echo "PASS: $PKG spinout smoke test (src/$SRC_PKG)"
