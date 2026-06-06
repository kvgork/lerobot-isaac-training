#!/usr/bin/env bash
# =============================================================================
# install_train_deps.sh — Install heavy training deps into the relevant
# pixi envs (train-policy, train-dreamer, train-lewm).
#
# Why this is a separate script
# -----------------------------
# `pixi install` resolves conda + workspace pypi-deps, but two heavyweight
# training libraries cannot be co-resolved by pixi:
#
#   1. lerobot[smolvla]    — gymnasium pin
#   2. sheeprl             — gymnasium pin AND Python <3.12 metadata pin
#
# So pixi.toml leaves `feature.lerobot` / `feature.dreamerv3` / `feature.leworldmodel`
# empty by design and this script pip-installs them per-env after `pixi install`.
#
# Usage
# -----
#   bash scripts/install_train_deps.sh                  # install all 3 envs
#   bash scripts/install_train_deps.sh --policy         # only train-policy
#   bash scripts/install_train_deps.sh --dreamer        # only train-dreamer
#   bash scripts/install_train_deps.sh --lewm           # only train-lewm
#   LEROBOT_EXTRAS=all  bash scripts/install_train_deps.sh   # override extras
#
# Exit codes
# ----------
#   0 — success (or already installed)
#   1 — pip install failure
#   2 — env missing (run `pixi install -e <env>` first)
# =============================================================================
set -uo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE_ROOT"

# smolvla → policy backbone; feetech → SO-101 servo SDK (scservo_sdk) needed by
# the hardware deploy path (lerobot-isaac-deploy session). Without `feetech`,
# robot setup fails with `No module named 'scservo_sdk'` at the dry-run loop.
LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-smolvla,feetech}"
SHEEPRL_GIT_URL="${SHEEPRL_GIT_URL:-git+https://github.com/Eclectic-Sheep/sheeprl.git}"

DO_POLICY=true
DO_DREAMER=true
DO_LEWM=true
case "${1:-}" in
  --policy)  DO_DREAMER=false; DO_LEWM=false ;;
  --dreamer) DO_POLICY=false; DO_LEWM=false ;;
  --lewm)    DO_POLICY=false; DO_DREAMER=false ;;
  --help|-h)
    sed -n '2,30p' "${BASH_SOURCE[0]}" | grep "^#" | sed 's/^# \?//'
    exit 0
    ;;
esac

# --- color helpers -----------------------------------------------------------
GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; YELLOW='\033[0;33m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*" >&2; }

_env_python() {
    local env="$1"
    local py=".pixi/envs/${env}/bin/python"
    if [ ! -x "$py" ]; then
        error "pixi env '$env' not installed at $py"
        error "  Run: pixi install -e $env"
        return 2
    fi
    echo "$py"
}

_have_module() {
    local py="$1" mod="$2"
    "$py" -c "import $mod" >/dev/null 2>&1
}

# --- 1. train-policy : lerobot[smolvla] --------------------------------------
if [ "$DO_POLICY" = true ]; then
    py=$(_env_python "train-policy") || exit $?
    if _have_module "$py" "lerobot"; then
        ver=$("$py" -c "import lerobot; print(getattr(lerobot, '__version__', '?'))" 2>/dev/null)
        success "train-policy : lerobot already present (version=$ver)"
    else
        info "train-policy : installing lerobot[${LEROBOT_EXTRAS}]..."
        "$py" -m pip install "lerobot[${LEROBOT_EXTRAS}]" || {
            error "lerobot install failed in train-policy"
            exit 1
        }
        success "train-policy : lerobot installed"
    fi
fi

# --- 2. train-dreamer : sheeprl from git (--ignore-requires-python on Py3.12) -
if [ "$DO_DREAMER" = true ]; then
    py=$(_env_python "train-dreamer") || exit $?
    if _have_module "$py" "sheeprl"; then
        success "train-dreamer : sheeprl already present"
    else
        # sheeprl pins Python <3.12 in metadata but is known to work on 3.12 in
        # practice for the dreamer_v3 algorithm. We install from the upstream
        # git master and pass --ignore-requires-python so pip doesn't reject the
        # metadata pin. If sheeprl publishes a 3.12-compatible release, switch
        # back to `pip install "sheeprl[dreamer]>=0.6"`.
        py_minor=$("$py" -c "import sys; print(sys.version_info.minor)")
        info "train-dreamer : installing sheeprl from git (py3.${py_minor})..."
        EXTRA_FLAGS=""
        if [ "$py_minor" -ge 12 ]; then
            EXTRA_FLAGS="--ignore-requires-python"
            warn "  Python 3.${py_minor} bypasses sheeprl metadata pin (<3.12). Verify runtime."
        fi
        "$py" -m pip install $EXTRA_FLAGS "$SHEEPRL_GIT_URL" || {
            error "sheeprl install failed in train-dreamer"
            exit 1
        }
        success "train-dreamer : sheeprl installed"
    fi
fi

# --- 3. train-lewm : lerobot (for HF LeWorldModel + train_world_model) -------
if [ "$DO_LEWM" = true ]; then
    py=$(_env_python "train-lewm") || exit $?
    if _have_module "$py" "lerobot"; then
        ver=$("$py" -c "import lerobot; print(getattr(lerobot, '__version__', '?'))" 2>/dev/null)
        success "train-lewm : lerobot already present (version=$ver)"
    else
        info "train-lewm : installing lerobot (no extras — HF model pulled on first use)..."
        "$py" -m pip install "lerobot" || {
            error "lerobot install failed in train-lewm"
            exit 1
        }
        success "train-lewm : lerobot installed"
    fi
fi

echo ""
success "All requested training-env deps installed."
echo ""
echo "Verify:"
echo "  pixi run -e train-policy  python -c 'import lerobot; print(lerobot.__version__)'"
echo "  pixi run -e train-dreamer python -c 'import sheeprl; print(sheeprl.__file__)'"
echo "  pixi run -e train-lewm    python -c 'import lerobot; print(lerobot.__version__)'"
