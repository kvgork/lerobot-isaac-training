#!/usr/bin/env bash
# =============================================================================
# download_so101_usd.sh — Clone SO-ARM100 repo and convert SO-101 URDF to USD.
#
# This script:
#   1. Clones TheRobotStudio/SO-ARM100 to a temp directory.
#   2. Locates the SO-101 URDF in Simulation/SO101/.
#   3. Runs Isaac Lab's convert_urdf.py to produce so101.usd.
#   4. Places the USD at packages/lerobot-isaac-env/assets/usd/so101.usd.
#
# PREREQUISITE:
#   Isaac Lab must be installed (run scripts/install_isaac_lab.sh first).
#
# ENVIRONMENT OVERRIDES:
#   ISAAC_LAB_DIR   — Isaac Lab install dir (default: ~/IsaacLab)
#   SO_ARM100_DIR   — where to clone SO-ARM100 (default: /tmp/SO-ARM100)
#
# EXIT CODES:
#   0 — success (or USD already exists)
#   1 — Isaac Lab not found
#   2 — SO-ARM100 clone failure
#   3 — URDF conversion failure
# =============================================================================
set -euo pipefail

ISAAC_LAB_DIR="${ISAAC_LAB_DIR:-${HOME}/IsaacLab}"
SO_ARM100_DIR="${SO_ARM100_DIR:-/tmp/SO-ARM100}"
SO_ARM100_REPO="https://github.com/TheRobotStudio/SO-ARM100.git"

# Resolve workspace root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Output path for the USD asset
USD_OUTPUT_DIR="${WORKSPACE_ROOT}/packages/lerobot-isaac-env/assets/usd"
USD_OUTPUT="${USD_OUTPUT_DIR}/so101.usd"

# Isaac Lab convert_urdf script path
CONVERT_URDF="${ISAAC_LAB_DIR}/scripts/tools/convert_urdf.py"

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

# ── 1. Check: USD already exists? ────────────────────────────────────────────
if [[ -f "${USD_OUTPUT}" ]]; then
    success "SO-101 USD already exists at ${USD_OUTPUT}. Nothing to do."
    ls -lh "${USD_OUTPUT}"
    exit 0
fi

# ── 2. Check Isaac Lab is installed ──────────────────────────────────────────
info "Checking Isaac Lab installation..."
if ! python3 -c "import isaaclab" 2>/dev/null; then
    error "Isaac Lab Python package not found."
    error "Run scripts/install_isaac_lab.sh first."
    exit 1
fi

if [[ ! -f "${CONVERT_URDF}" ]]; then
    error "Isaac Lab convert_urdf.py not found at: ${CONVERT_URDF}"
    error "Check that ISAAC_LAB_DIR points to the correct installation:"
    error "  ISAAC_LAB_DIR=${ISAAC_LAB_DIR}"
    error "Expected: ${CONVERT_URDF}"
    # Try to find it under the Isaac Lab dir
    FOUND=$(find "${ISAAC_LAB_DIR}" -name "convert_urdf.py" 2>/dev/null | head -1 || true)
    if [[ -n "${FOUND}" ]]; then
        warn "Found convert_urdf.py at: ${FOUND}"
        warn "Set CONVERT_URDF env var or update ISAAC_LAB_DIR."
    fi
    exit 1
fi
success "Isaac Lab found. convert_urdf.py at: ${CONVERT_URDF}"

# ── 3. Clone SO-ARM100 ────────────────────────────────────────────────────────
if [[ -d "${SO_ARM100_DIR}/.git" ]]; then
    info "SO-ARM100 already cloned at ${SO_ARM100_DIR}. Skipping clone."
else
    info "Cloning SO-ARM100 to ${SO_ARM100_DIR}..."
    git clone --depth 1 "${SO_ARM100_REPO}" "${SO_ARM100_DIR}" || {
        error "Failed to clone SO-ARM100 from ${SO_ARM100_REPO}."
        error "Check network connectivity."
        exit 2
    }
    success "Cloned SO-ARM100 to ${SO_ARM100_DIR}."
fi

# ── 4. Locate SO-101 URDF ─────────────────────────────────────────────────────
info "Locating SO-101 URDF in ${SO_ARM100_DIR}/Simulation/SO101/..."

# Primary expected path
URDF_PATH="${SO_ARM100_DIR}/Simulation/SO101/so101.urdf"

if [[ ! -f "${URDF_PATH}" ]]; then
    # Fallback: search for any .urdf inside Simulation/SO101/
    URDF_PATH=$(find "${SO_ARM100_DIR}/Simulation/SO101" -name "*.urdf" 2>/dev/null | head -1 || true)
    if [[ -z "${URDF_PATH}" ]]; then
        # Wider search under Simulation/
        URDF_PATH=$(find "${SO_ARM100_DIR}/Simulation" -name "*.urdf" 2>/dev/null | head -1 || true)
    fi
fi

if [[ -z "${URDF_PATH}" ]] || [[ ! -f "${URDF_PATH}" ]]; then
    error "Could not find SO-101 URDF in ${SO_ARM100_DIR}/Simulation/"
    error "Directory contents:"
    ls -la "${SO_ARM100_DIR}/Simulation/" 2>/dev/null || true
    exit 3
fi

success "Found URDF: ${URDF_PATH}"

# ── 5. Prepare output directory ──────────────────────────────────────────────
info "Creating output directory: ${USD_OUTPUT_DIR}"
mkdir -p "${USD_OUTPUT_DIR}"

# ── 6. Convert URDF → USD ─────────────────────────────────────────────────────
info "Converting URDF to USD..."
info "  Input:  ${URDF_PATH}"
info "  Output: ${USD_OUTPUT}"
info "  Script: ${CONVERT_URDF}"
info "(This runs Isaac Sim headless — may take 1-3 minutes on first run.)"

python3 "${CONVERT_URDF}" \
    --input "${URDF_PATH}" \
    --output "${USD_OUTPUT}" \
    --headless \
    || {
        error "URDF conversion failed."
        error "Check the output above for Isaac Sim/Isaac Lab error messages."
        error "Common issues:"
        error "  - Isaac Sim not in PATH / PYTHONPATH"
        error "  - GPU not available or driver mismatch"
        error "  - Missing mesh files referenced in URDF"
        exit 3
    }

# ── 7. Verify output ──────────────────────────────────────────────────────────
if [[ ! -f "${USD_OUTPUT}" ]]; then
    error "Conversion reported success but ${USD_OUTPUT} not found."
    exit 3
fi

success "USD conversion complete."
ls -lh "${USD_OUTPUT}"

# ── 8. Success banner ─────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        SO-101 USD asset created successfully!        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Asset location:"
echo "    ${USD_OUTPUT}"
echo ""
echo "  Next step:"
echo "    bash scripts/verify_install.sh"
echo ""
