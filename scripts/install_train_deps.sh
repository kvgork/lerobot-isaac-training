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

# lerobot 0.6.0 split the install into extras: `training` pulls the trainer deps
# (implicit in 0.5.x), `vla_jepa` pulls the world-model policy backbone (Qwen3-VL
# + qwen-vl-utils). smolvla → policy backbone; feetech → SO-101 servo SDK
# (scservo_sdk) needed by the hardware deploy path (lerobot-isaac-deploy). Without
# `feetech`, robot setup fails with `No module named 'scservo_sdk'` at dry-run.
# fastwam / lingbot_va extras are intentionally NOT default — those world-model
# policies need >>10 GB VRAM; add them on capable HW via
#   LEROBOT_EXTRAS=training,smolvla,feetech,vla_jepa,fastwam bash scripts/install_train_deps.sh
LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-training,smolvla,feetech,vla_jepa}"
# Minimum lerobot version. 0.6.0 is the first release shipping the world-model
# policies (vla_jepa/fastwam/lingbot_va); below this the script upgrades.
LEROBOT_MIN_VERSION="${LEROBOT_MIN_VERSION:-0.6.0}"
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

_lerobot_version() {
    local py="$1"
    "$py" -c "import lerobot;print(getattr(lerobot,'__version__','?'))" 2>/dev/null || echo "none"
}

# Return 0 iff lerobot is installed AND >= LEROBOT_MIN_VERSION.
_lerobot_current_ok() {
    local py="$1"
    "$py" - "$LEROBOT_MIN_VERSION" <<'PY' 2>/dev/null
import sys
want = sys.argv[1]
try:
    import lerobot
    try:
        from packaging.version import Version as V
    except Exception:  # packaging absent — fall back to a naive numeric compare
        def V(s):
            return tuple(int(p) for p in s.split("+")[0].split(".") if p.isdigit())
    sys.exit(0 if V(getattr(lerobot, "__version__", "0")) >= V(want) else 1)
except Exception:
    sys.exit(1)
PY
}

# --- 1. train-policy : lerobot[smolvla] --------------------------------------
if [ "$DO_POLICY" = true ]; then
    py=$(_env_python "train-policy") || exit $?
    if _lerobot_current_ok "$py"; then
        success "train-policy : lerobot $(_lerobot_version "$py") OK (>= ${LEROBOT_MIN_VERSION})"
    else
        info "train-policy : installing/upgrading lerobot[${LEROBOT_EXTRAS}] (>= ${LEROBOT_MIN_VERSION}; was $(_lerobot_version "$py"))..."
        "$py" -m pip install -U "lerobot[${LEROBOT_EXTRAS}]>=${LEROBOT_MIN_VERSION}" || {
            error "lerobot install failed in train-policy"
            exit 1
        }
        success "train-policy : lerobot upgraded to $(_lerobot_version "$py")"
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
    # sheeprl pins opencv-python==4.8.0.* (built against numpy 1.x). This env runs
    # numpy>=2, so that wheel ABI-breaks at import ("numpy.core.multiarray failed
    # to import") and sheeprl won't load. Force a numpy-2-compatible opencv build.
    # The pip "sheeprl requires opencv==4.8.0.*" warning is expected and benign —
    # sheeprl runs fine on opencv>=4.10 at runtime.
    cv_ver=$("$py" -c "import cv2,sys;sys.stdout.write(cv2.__version__)" 2>/dev/null || echo "broken")
    case "$cv_ver" in
        4.8.*|3.*|broken)
            info "train-dreamer : upgrading opencv ($cv_ver) -> numpy>=2-compatible (>=4.10)..."
            "$py" -m pip install -U "opencv-python>=4.10" >/dev/null 2>&1 \
                && success "train-dreamer : opencv upgraded" \
                || warn "  opencv upgrade failed; sheeprl may fail to import under numpy>=2" ;;
        *) success "train-dreamer : opencv $cv_ver OK (numpy>=2 compatible)" ;;
    esac
fi

# --- 3. train-lewm : lerobot 0.6.0 (world-model policies via lerobot-train) --
# Historically the home of `lerobot.scripts.train_world_model` (never shipped in
# 0.5.x). lerobot 0.6.0 instead ships world models as POLICIES (vla_jepa etc.)
# trained through the ordinary `lerobot-train` CLI, so this env installs the same
# extras as train-policy. The predictive `le_world_model` adapter target still
# falls back to the in-process `_lewm_minimal` trainer (upstream never resurrected
# a standalone WM script) — see src/.../targets/wm_leworldmodel.py.
if [ "$DO_LEWM" = true ]; then
    py=$(_env_python "train-lewm") || exit $?
    if _lerobot_current_ok "$py"; then
        success "train-lewm : lerobot $(_lerobot_version "$py") OK (>= ${LEROBOT_MIN_VERSION})"
    else
        info "train-lewm : installing/upgrading lerobot[${LEROBOT_EXTRAS}] (>= ${LEROBOT_MIN_VERSION}; was $(_lerobot_version "$py"))..."
        "$py" -m pip install -U "lerobot[${LEROBOT_EXTRAS}]>=${LEROBOT_MIN_VERSION}" || {
            error "lerobot install failed in train-lewm"
            exit 1
        }
        success "train-lewm : lerobot upgraded to $(_lerobot_version "$py")"
    fi
fi

echo ""
success "All requested training-env deps installed."
echo ""
echo "Verify:"
echo "  pixi run -e train-policy  python -c 'import lerobot; print(lerobot.__version__)'"
echo "  pixi run -e train-dreamer python -c 'import sheeprl; print(sheeprl.__file__)'"
echo "  pixi run -e train-lewm    python -c 'import lerobot; print(lerobot.__version__)'"
