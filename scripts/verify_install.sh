#!/usr/bin/env bash
# =============================================================================
# verify_install.sh — Smoke-test the full LeRobot + Isaac Lab installation.
#
# Runs 6 checks and prints PASS / FAIL for each.
# Exit code = number of failed checks (0 means all pass).
#
# Run from the workspace root:
#   bash scripts/verify_install.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
TOTAL=6

pass() { echo -e "  ${GREEN}[PASS]${NC} $*"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo -e "  ${RED}[FAIL]${NC} $*"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

section() {
    echo ""
    echo -e "${CYAN}${BOLD}── $* ──${NC}"
}

run_check() {
    local label="$1"
    local cmd="$2"
    local output
    if output=$(eval "${cmd}" 2>&1); then
        pass "${label}: ${output}"
    else
        fail "${label}"
        echo "       Command: ${cmd}"
        echo "       Output:  ${output}"
    fi
}

# ── Header ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}LeRobot + Isaac Lab Install Verification${NC}"
echo "Workspace: ${WORKSPACE_ROOT}"
echo "Date:      $(date)"
echo ""

# ── Check 1: Isaac Lab ────────────────────────────────────────────────────────
section "Check 1/6 — Isaac Lab"
run_check "isaaclab version" \
    "python3 -c \"import isaaclab; print(isaaclab.__version__)\""

# ── Check 2: LeRobot ─────────────────────────────────────────────────────────
section "Check 2/6 — LeRobot"
run_check "lerobot version" \
    "python3 -c \"import lerobot; print(lerobot.__version__)\""

# ── Check 3: lerobot-isaac-env package ───────────────────────────────────────
section "Check 3/6 — lerobot-isaac-env (SO101EnvCfg)"
run_check "SO101EnvCfg constructs" \
    "python3 -c \"from lerobot_isaac_env import SO101EnvCfg; cfg = SO101EnvCfg(); print(cfg)\""

# ── Check 4: lerobot-isaac-adapters CLI ──────────────────────────────────────
section "Check 4/6 — lerobot-isaac-adapters (--dry_run)"
run_check "dry_run flag works" \
    "python3 -c \"
from lerobot_isaac_adapters.train import _build_parser
ns = _build_parser().parse_args(['--target_arch=smolvla', '--dataset=test', '--dry_run'])
assert ns.dry_run, 'dry_run should be True'
print('dry_run OK')
\""

# ── Check 5: lerobot-isaac-synthetic (replay_with_randomization) ─────────────
section "Check 5/6 — lerobot-isaac-synthetic (replay_runner)"
run_check "replay_with_randomization importable" \
    "python3 -c \"
from lerobot_isaac_synthetic.isaac_dr.replay_runner import replay_with_randomization
print('synthetic OK')
\""

# ── Check 6: SO-101 USD asset ─────────────────────────────────────────────────
section "Check 6/6 — SO-101 USD asset"
USD_PATH="${WORKSPACE_ROOT}/packages/lerobot-isaac-env/assets/usd/so101.usd"
if [[ -f "${USD_PATH}" ]]; then
    SIZE=$(du -h "${USD_PATH}" | cut -f1)
    pass "so101.usd exists (${SIZE}): ${USD_PATH}"
    PASS_COUNT=$((PASS_COUNT + 1))
else
    fail "so101.usd NOT FOUND at: ${USD_PATH}"
    echo "       Run:  bash scripts/download_so101_usd.sh"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
if [[ ${FAIL_COUNT} -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}  ${PASS_COUNT}/${TOTAL} checks passed — all systems GO!${NC}"
else
    echo -e "${RED}${BOLD}  ${PASS_COUNT}/${TOTAL} checks passed — ${FAIL_COUNT} FAILED${NC}"
    echo ""
    echo "  Troubleshooting:"
    echo "    Isaac Lab missing?  → bash scripts/install_isaac_lab.sh"
    echo "    LeRobot missing?    → bash scripts/install_lerobot.sh"
    echo "    USD missing?        → bash scripts/download_so101_usd.sh"
    echo "    Package not found?  → pixi install  (installs workspace packages)"
fi
echo "════════════════════════════════════════════════════"
echo ""

exit ${FAIL_COUNT}
