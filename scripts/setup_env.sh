#!/usr/bin/env bash
# setup_env.sh — Source this file to export the workspace environment variables.
#
# Usage:
#   source scripts/setup_env.sh
#
# Idempotent: re-sourcing only updates values, never duplicates.
# Honours pre-existing exports (does not overwrite if already set by user).
#
# Variables exported:
#   LEROBOT_ISAAC_WORKSPACE   — absolute path to this workspace root
#   CLAUDE_CODE_ROOT          — absolute path to the claude_code repo
#   LEROBOT_CLAUDE_CODE_ROOT  — alias of CLAUDE_CODE_ROOT (legacy specific name)
#   VAULT_ROOT                — optional Obsidian/Markdown vault root (only if exists)

# --- workspace root --------------------------------------------------------
# Detect from the location of this script (scripts/ → ..)
_THIS_FILE="${BASH_SOURCE[0]:-$0}"
_THIS_DIR="$(cd "$(dirname "${_THIS_FILE}")" && pwd)"
_WORKSPACE_ROOT="$(cd "${_THIS_DIR}/.." && pwd)"

if [ -z "${LEROBOT_ISAAC_WORKSPACE:-}" ]; then
    export LEROBOT_ISAAC_WORKSPACE="${_WORKSPACE_ROOT}"
fi

# --- claude_code repo root -------------------------------------------------
# Search order: existing env var → conventional ~/tools/claude_code → sibling dir
if [ -z "${CLAUDE_CODE_ROOT:-}" ]; then
    if [ -n "${LEROBOT_CLAUDE_CODE_ROOT:-}" ] && [ -d "${LEROBOT_CLAUDE_CODE_ROOT}" ]; then
        export CLAUDE_CODE_ROOT="${LEROBOT_CLAUDE_CODE_ROOT}"
    elif [ -d "${HOME}/tools/claude_code" ]; then
        export CLAUDE_CODE_ROOT="${HOME}/tools/claude_code"
    elif [ -d "$(dirname "${_WORKSPACE_ROOT}")/claude_code" ]; then
        export CLAUDE_CODE_ROOT="$(cd "$(dirname "${_WORKSPACE_ROOT}")/claude_code" && pwd)"
    else
        echo "[setup_env] WARNING: CLAUDE_CODE_ROOT could not be auto-detected." >&2
        echo "[setup_env]   Clone https://github.com/<your-org>/claude_code and export:" >&2
        echo "[setup_env]     export CLAUDE_CODE_ROOT=/abs/path/to/claude_code" >&2
    fi
fi

# Mirror to the legacy specific-name variable.
if [ -n "${CLAUDE_CODE_ROOT:-}" ] && [ -z "${LEROBOT_CLAUDE_CODE_ROOT:-}" ]; then
    export LEROBOT_CLAUDE_CODE_ROOT="${CLAUDE_CODE_ROOT}"
fi

# --- vault root (optional) -------------------------------------------------
if [ -z "${VAULT_ROOT:-}" ]; then
    if [ -d "${HOME}/Documents/Vaults/Local" ]; then
        export VAULT_ROOT="${HOME}/Documents/Vaults/Local"
    fi
fi

# --- summary ---------------------------------------------------------------
echo "[setup_env] LEROBOT_ISAAC_WORKSPACE  = ${LEROBOT_ISAAC_WORKSPACE}"
echo "[setup_env] CLAUDE_CODE_ROOT         = ${CLAUDE_CODE_ROOT:-<unset>}"
echo "[setup_env] LEROBOT_CLAUDE_CODE_ROOT = ${LEROBOT_CLAUDE_CODE_ROOT:-<unset>}"
echo "[setup_env] VAULT_ROOT               = ${VAULT_ROOT:-<unset, optional>}"
