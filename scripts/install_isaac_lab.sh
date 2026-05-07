#!/usr/bin/env bash
# =============================================================================
# install_isaac_lab.sh — Clone and install Isaac Lab into the system.
#
# PREREQUISITE (MANDATORY — this script will NOT install Isaac Sim for you):
#   Isaac Sim must be installed before running this script.
#   Install via NVIDIA Omniverse Launcher or the standalone package:
#     https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html
#   Tested with: Isaac Sim 4.2.x (required for Isaac Lab v2.1.0)
#
# USAGE:
#   bash scripts/install_isaac_lab.sh
#
# ENVIRONMENT OVERRIDES:
#   ISAAC_LAB_DIR   — where to clone Isaac Lab (default: ~/IsaacLab)
#
# EXIT CODES:
#   0 — success (or already installed)
#   1 — prerequisite missing (Isaac Sim not found / python not importable)
#   2 — clone or install failure
# =============================================================================
set -euo pipefail

# ── Pinned version ────────────────────────────────────────────────────────────
# Pin to a known-good Isaac Lab release. Change ISAAC_LAB_TAG to upgrade.
# Latest stable as of 2026-05: v2.1.0
ISAAC_LAB_TAG="${ISAAC_LAB_TAG:-v2.1.0}"
ISAAC_LAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
ISAAC_LAB_DIR="${ISAAC_LAB_DIR:-${HOME}/IsaacLab}"

# ── Helpers ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*" >&2; }

# ── 1. Check: is Isaac Lab already importable? ────────────────────────────────
if python3 -c "import isaaclab" 2>/dev/null; then
    INSTALLED_VER=$(python3 -c "import isaaclab; print(getattr(isaaclab, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    success "Isaac Lab already installed (version: ${INSTALLED_VER}). Nothing to do."
    echo ""
    echo "  To force a re-install, uninstall first:"
    echo "    pip uninstall isaaclab -y"
    echo "  then re-run this script."
    exit 0
fi

# ── 2. Check Isaac Sim prerequisite ──────────────────────────────────────────
info "Checking Isaac Sim prerequisite..."
if ! python3 -c "import isaacsim" 2>/dev/null; then
    error "Isaac Sim Python bindings not found."
    error ""
    error "Isaac Sim must be installed BEFORE Isaac Lab."
    error "Installation guide:"
    error "  https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html"
    error ""
    error "After installing Isaac Sim, add its Python path to your environment, e.g.:"
    error "  export PYTHONPATH=\${PYTHONPATH}:/path/to/isaac-sim/exts/isaacsim.pip.torch.2.5.0/pip/torch"
    error ""
    error "Then re-run this script."
    exit 1
fi
success "Isaac Sim found."

# ── 3. Clone Isaac Lab ────────────────────────────────────────────────────────
if [[ -d "${ISAAC_LAB_DIR}/.git" ]]; then
    info "Isaac Lab directory already exists at ${ISAAC_LAB_DIR}. Skipping clone."
    info "Checking out pinned tag ${ISAAC_LAB_TAG}..."
    git -C "${ISAAC_LAB_DIR}" fetch --tags || {
        error "Failed to fetch tags in ${ISAAC_LAB_DIR}."
        exit 2
    }
    git -C "${ISAAC_LAB_DIR}" checkout "${ISAAC_LAB_TAG}" || {
        error "Failed to checkout tag ${ISAAC_LAB_TAG} in ${ISAAC_LAB_DIR}."
        exit 2
    }
else
    info "Cloning Isaac Lab (${ISAAC_LAB_TAG}) to ${ISAAC_LAB_DIR}..."
    git clone --branch "${ISAAC_LAB_TAG}" --depth 1 "${ISAAC_LAB_REPO}" "${ISAAC_LAB_DIR}" || {
        error "Failed to clone Isaac Lab from ${ISAAC_LAB_REPO}."
        error "Check network connectivity and that the tag '${ISAAC_LAB_TAG}' exists."
        exit 2
    }
    success "Cloned Isaac Lab to ${ISAAC_LAB_DIR}."
fi

# ── 4. Run Isaac Lab installer ────────────────────────────────────────────────
info "Running Isaac Lab installer (./isaaclab.sh --install)..."
info "This may take several minutes — it installs Python extensions and dependencies."

pushd "${ISAAC_LAB_DIR}" > /dev/null
./isaaclab.sh --install || {
    error "Isaac Lab install script failed."
    error "Check the output above for details."
    popd > /dev/null
    exit 2
}
popd > /dev/null

# ── 5. Verify ─────────────────────────────────────────────────────────────────
info "Verifying installation..."
if python3 -c "import isaaclab" 2>/dev/null; then
    VER=$(python3 -c "import isaaclab; print(getattr(isaaclab, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
    success "Isaac Lab installed successfully (version: ${VER})."
else
    error "Isaac Lab installed but 'import isaaclab' still fails."
    error "You may need to activate the Isaac Lab conda/venv environment:"
    error "  source ${ISAAC_LAB_DIR}/isaaclab.sh --conda  # if using conda"
    error "  source ${ISAAC_LAB_DIR}/.venv/bin/activate   # if using venv"
    exit 2
fi

# ── 6. Success banner ─────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Isaac Lab installed successfully!            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Verify with:"
echo "    python3 -c \"import isaaclab; print(isaaclab.__version__)\""
echo ""
echo "  Next step:"
echo "    bash scripts/install_lerobot.sh"
echo "    bash scripts/download_so101_usd.sh"
echo ""
