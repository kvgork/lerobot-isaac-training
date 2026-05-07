#!/usr/bin/env bash
# =============================================================================
# install_lerobot.sh — Install the LeRobot library from PyPI or source.
#
# Default: pip install lerobot[all]
#
# For SmolVLA support specifically, the relevant extra is [smolvla]:
#   pip install lerobot[smolvla]
# The [all] extra includes smolvla + all other optional dependencies.
#
# USAGE:
#   # Standard install from PyPI:
#   bash scripts/install_lerobot.sh
#
#   # Editable install from a cloned source repo:
#   LEROBOT_SRC=/path/to/cloned/lerobot bash scripts/install_lerobot.sh --editable
#
# ENVIRONMENT VARIABLES:
#   LEROBOT_SRC     — path to cloned lerobot source (used with --editable)
#   LEROBOT_EXTRAS  — pip extras to install (default: all)
#                     e.g. LEROBOT_EXTRAS=smolvla bash scripts/install_lerobot.sh
#
# EXIT CODES:
#   0 — success (or already installed)
#   1 — pip install failure
# =============================================================================
set -euo pipefail

LEROBOT_SRC="${LEROBOT_SRC:-}"
LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-all}"
EDITABLE=false

# ── Parse flags ───────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "${arg}" in
        --editable|-e)
            EDITABLE=true
            ;;
        --help|-h)
            sed -n '2,30p' "${BASH_SOURCE[0]}" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "[WARN] Unknown argument: ${arg}" >&2
            ;;
    esac
done

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*" >&2; }

# ── 1. Check: already installed? ─────────────────────────────────────────────
if python3 -c "import lerobot" 2>/dev/null; then
    VER=$(python3 -c "import lerobot; print(getattr(lerobot, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    success "LeRobot already installed (version: ${VER}). Nothing to do."
    echo ""
    echo "  To upgrade: pip install --upgrade lerobot[${LEROBOT_EXTRAS}]"
    echo "  To force re-install: pip uninstall lerobot -y && bash scripts/install_lerobot.sh"
    exit 0
fi

# ── 2. Install ────────────────────────────────────────────────────────────────
if [[ "${EDITABLE}" == "true" ]]; then
    # Editable install from source
    if [[ -z "${LEROBOT_SRC}" ]]; then
        # Default: clone from GitHub
        LEROBOT_SRC="/tmp/lerobot-src"
        if [[ ! -d "${LEROBOT_SRC}/.git" ]]; then
            info "Cloning LeRobot source to ${LEROBOT_SRC}..."
            git clone --depth 1 https://github.com/huggingface/lerobot.git "${LEROBOT_SRC}" || {
                error "Failed to clone LeRobot source."
                exit 1
            }
        else
            info "Using existing clone at ${LEROBOT_SRC}."
        fi
    fi

    if [[ ! -d "${LEROBOT_SRC}" ]]; then
        error "LEROBOT_SRC does not exist: ${LEROBOT_SRC}"
        exit 1
    fi

    info "Installing LeRobot in editable mode from ${LEROBOT_SRC}..."
    info "  Extras: [${LEROBOT_EXTRAS}]"
    pip install -e "${LEROBOT_SRC}[${LEROBOT_EXTRAS}]" || {
        error "Editable install failed."
        exit 1
    }
else
    # Standard PyPI install
    info "Installing LeRobot from PyPI with extras [${LEROBOT_EXTRAS}]..."
    info "  Command: pip install lerobot[${LEROBOT_EXTRAS}]"
    info "  (Note: use LEROBOT_EXTRAS=smolvla for SmolVLA-only install)"
    pip install "lerobot[${LEROBOT_EXTRAS}]" || {
        error "pip install failed."
        error "Common fixes:"
        error "  - Check Python version (requires >=3.10)"
        error "  - Try: pip install --upgrade pip && bash scripts/install_lerobot.sh"
        error "  - For minimal install: LEROBOT_EXTRAS=smolvla bash scripts/install_lerobot.sh"
        exit 1
    }
fi

# ── 3. Verify ─────────────────────────────────────────────────────────────────
if python3 -c "import lerobot" 2>/dev/null; then
    VER=$(python3 -c "import lerobot; print(getattr(lerobot, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    success "LeRobot installed (version: ${VER})."
else
    error "pip reported success but 'import lerobot' failed."
    error "Check your Python environment / virtual environment activation."
    exit 1
fi

# ── 4. Success banner ─────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          LeRobot installed successfully!             ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Verify with:"
echo "    python3 -c \"import lerobot; print(lerobot.__version__)\""
echo ""
echo "  Next steps (if Isaac Lab already installed):"
echo "    bash scripts/download_so101_usd.sh"
echo "    bash scripts/verify_install.sh"
echo ""
