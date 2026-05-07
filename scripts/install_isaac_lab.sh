#!/usr/bin/env bash
# =============================================================================
# install_isaac_lab.sh — Install Isaac Sim + Isaac Lab via pip (Path A).
#
# Omniverse Launcher was deprecated Oct 2025. Isaac Sim 4.5+ ships as Python
# wheels on https://pypi.nvidia.com — this script installs both via pip in the
# active Python environment (intended to be invoked inside `pixi shell -e full`
# or any env with python ≥3.10 + pip).
#
# USAGE:
#   pixi shell -e full        # or activate your venv
#   bash scripts/install_isaac_lab.sh
#
# ENVIRONMENT OVERRIDES:
#   ISAAC_SIM_VERSION  — pinned isaacsim wheel version (default: 4.5.0)
#   ISAAC_LAB_VERSION  — pinned isaaclab wheel version (default: 2.1.0)
#   ISAAC_PYPI_INDEX   — NVIDIA wheel index URL (default: https://pypi.nvidia.com)
#   SKIP_ISAAC_SIM     — set to 1 to skip Isaac Sim install (assume already present)
#
# EXIT CODES:
#   0 — success (or already installed)
#   1 — prerequisite missing (python / pip / GPU driver)
#   2 — pip install failure
# =============================================================================
set -euo pipefail

# Auto-accept Omniverse / Isaac Sim EULA + privacy consent so non-interactive
# install / import works.
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"

# Isaac Sim wheel ↔ Python ABI matrix (mid-2026):
#   isaacsim 4.x → Python 3.10
#   isaacsim 5.x → Python 3.11
#   isaacsim 6.x → Python 3.12   ← default for this workspace (lerobot ≥0.5 requires py3.12)
ISAAC_SIM_VERSION="${ISAAC_SIM_VERSION:-6.0.0.0}"
ISAAC_LAB_VERSION="${ISAAC_LAB_VERSION:-}"   # empty = let pip pick latest compat with isaacsim
ISAAC_PYPI_INDEX="${ISAAC_PYPI_INDEX:-https://pypi.nvidia.com}"
SKIP_ISAAC_SIM="${SKIP_ISAAC_SIM:-0}"

# ── Helpers ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*" >&2; }

# ── 1. Pre-flight ─────────────────────────────────────────────────────────────
info "Pre-flight checks..."

if ! command -v python3 >/dev/null 2>&1; then
    error "python3 not found in PATH."; exit 1
fi
PY_VER=$(python3 -c "import sys; print('%d.%d' % sys.version_info[:2])")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info[0])")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info[1])")
if [ "$PY_MAJOR" -lt 3 ] || [ "$PY_MINOR" -lt 10 ]; then
    error "Python 3.10+ required (found ${PY_VER})."; exit 1
fi
success "Python ${PY_VER} OK."

if ! python3 -m pip --version >/dev/null 2>&1; then
    error "pip not available."; exit 1
fi
success "pip OK."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn  "nvidia-smi not found — Isaac Sim requires NVIDIA GPU + driver ≥535."
else
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    success "NVIDIA driver ${DRIVER} detected."
fi

# ── 2. Check existing installs ────────────────────────────────────────────────
if python3 -c "import isaaclab" 2>/dev/null; then
    VER=$(python3 -c "import isaaclab; print(getattr(isaaclab,'__version__','unknown'))" 2>/dev/null || echo unknown)
    success "Isaac Lab already importable (version: ${VER}). Nothing to do."
    echo ""
    echo "  To force re-install:"
    echo "    pip uninstall -y isaaclab isaacsim"
    echo "    bash scripts/install_isaac_lab.sh"
    exit 0
fi

# ── 3. Install Isaac Sim ──────────────────────────────────────────────────────
if [ "${SKIP_ISAAC_SIM}" = "1" ]; then
    if ! python3 -c "import isaacsim" 2>/dev/null; then
        error "SKIP_ISAAC_SIM=1 but isaacsim is not importable. Aborting."
        exit 1
    fi
    info "Skipping Isaac Sim install (SKIP_ISAAC_SIM=1)."
else
    info "Installing Isaac Sim ${ISAAC_SIM_VERSION} from ${ISAAC_PYPI_INDEX} ..."
    info "This downloads ~10 GB of wheels (~10-15 min on a fast pipe)."
    python3 -m pip install --upgrade pip
    python3 -m pip install \
        "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}" \
        --extra-index-url "${ISAAC_PYPI_INDEX}" || {
        error "Failed to install isaacsim ${ISAAC_SIM_VERSION}."
        error "Common causes:"
        error "  - Wrong Python version (Isaac Sim 4.5+ requires Python 3.10 / 3.11 / 3.12)"
        error "  - Network restrictions on ${ISAAC_PYPI_INDEX}"
        error "  - pip too old (run: python3 -m pip install --upgrade pip)"
        exit 2
    }
    success "Isaac Sim ${ISAAC_SIM_VERSION} installed."
fi

# ── 4. Install Isaac Lab (clone + pip install -e .) ──────────────────────────
# NOTE: As of mid-2026, isaaclab is NOT yet published as wheels on pypi.nvidia.com.
# Only isaacsim is. So we still clone IsaacLab and pip-install in editable mode.
ISAAC_LAB_DIR="${ISAAC_LAB_DIR:-${HOME}/IsaacLab}"
ISAAC_LAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
ISAAC_LAB_BRANCH="${ISAAC_LAB_BRANCH:-main}"   # main tracks Isaac Sim 6.0+; pin tag if needed

if [[ -d "${ISAAC_LAB_DIR}/.git" ]]; then
    info "Isaac Lab repo already at ${ISAAC_LAB_DIR}. Pulling latest ${ISAAC_LAB_BRANCH} ..."
    git -C "${ISAAC_LAB_DIR}" fetch origin || true
    git -C "${ISAAC_LAB_DIR}" checkout "${ISAAC_LAB_BRANCH}" || true
    git -C "${ISAAC_LAB_DIR}" pull --ff-only origin "${ISAAC_LAB_BRANCH}" || true
else
    info "Cloning Isaac Lab (${ISAAC_LAB_BRANCH}) to ${ISAAC_LAB_DIR} ..."
    git clone --branch "${ISAAC_LAB_BRANCH}" --depth 1 "${ISAAC_LAB_REPO}" "${ISAAC_LAB_DIR}" || {
        error "Failed to clone Isaac Lab from ${ISAAC_LAB_REPO}."
        exit 2
    }
    success "Cloned Isaac Lab to ${ISAAC_LAB_DIR}."
fi

# Symlink Isaac Sim into Isaac Lab (Isaac Lab v2+ expects either pip-installed isaacsim
# or a `_isaac_sim` symlink in the lab dir)
if [[ ! -e "${ISAAC_LAB_DIR}/_isaac_sim" ]]; then
    ISAAC_SIM_LOC=$(python3 -c "import isaacsim, os; print(os.path.dirname(isaacsim.__file__))")
    info "Symlinking pip-installed isaacsim → ${ISAAC_LAB_DIR}/_isaac_sim"
    ln -s "${ISAAC_SIM_LOC}" "${ISAAC_LAB_DIR}/_isaac_sim" || true
fi

info "Running Isaac Lab installer (./isaaclab.sh --install ${ISAAC_LAB_LIB:-none}) ..."
# Default to '-i none' (core only) to avoid robomimic→egl_probe→libEGL build failure.
# To install learning frameworks set ISAAC_LAB_LIB=all (requires libegl1-mesa-dev + libgles2-mesa-dev apt packages).
pushd "${ISAAC_LAB_DIR}" > /dev/null
./isaaclab.sh --install "${ISAAC_LAB_LIB:-none}" || {
    error "Isaac Lab install script failed (lib=${ISAAC_LAB_LIB:-none})."
    error "If you need rsl_rl/rl_games/robomimic learning frameworks, first run:"
    error "  sudo apt install -y libegl1-mesa-dev libgles2-mesa-dev"
    error "Then re-run with: ISAAC_LAB_LIB=all bash scripts/install_isaac_lab.sh"
    popd > /dev/null
    exit 2
}
popd > /dev/null
success "Isaac Lab installed from source at ${ISAAC_LAB_DIR} (lib=${ISAAC_LAB_LIB:-none})."

# ── 5. Verify ─────────────────────────────────────────────────────────────────
info "Verifying installation..."
if python3 -c "import isaacsim" 2>/dev/null; then
    success "isaacsim importable."
else
    error "isaacsim install reported success but import fails."; exit 2
fi
if python3 -c "import isaaclab" 2>/dev/null; then
    VER=$(python3 -c "import isaaclab; print(getattr(isaaclab,'__version__','unknown'))" 2>/dev/null || echo unknown)
    success "isaaclab importable (version: ${VER})."
else
    error "isaaclab install reported success but import fails."; exit 2
fi

# ── 6. Banner ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Isaac Sim + Isaac Lab installed successfully!     ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Verify with:"
echo "    python3 -c \"import isaaclab; print(isaaclab.__version__)\""
echo ""
echo "  Next steps:"
echo "    bash scripts/install_lerobot.sh        # if not already done"
echo "    bash scripts/download_so101_usd.sh     # convert SO-101 URDF to USD"
echo "    bash scripts/verify_install.sh         # final 6/6 verify"
echo ""
