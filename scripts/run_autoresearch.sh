#!/usr/bin/env bash
# =============================================================================
# run_autoresearch.sh — Workspace wrapper for the autoresearch loop.
#
# Resolves a program by short name (matches programs/<name>.md or
# programs/lerobot-policy-<name>.md / programs/wm-<name>.md), prints the
# canonical /autoresearch invocation, and either:
#   - executes it as a foreground command, or
#   - prints the command for manual paste into Claude Code.
#
# Why a wrapper:
#   - canonicalises program-name aliases (`diffusion` → lerobot-policy-diffusion.md)
#   - injects PYTHONNOUSERSITE=1 so user-site pyarrow doesn't shadow the
#     pixi env's pyarrow during the executor's metric-extraction parquet read
#   - sets a session_id consistent with .agent-state/<session>/autoresearch/
#   - falls back to scripts/_run_autoresearch_smoke.sh when --bash is passed
#     (deterministic, no LLM proposer; same on-disk schema)
#
# Usage
# -----
#   bash scripts/run_autoresearch.sh --program diffusion
#   bash scripts/run_autoresearch.sh --program dreamerv3 --max-experiments 5
#   bash scripts/run_autoresearch.sh --program lewm --print-only
#   bash scripts/run_autoresearch.sh --program diffusion --bash    # deterministic fallback
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

PROGRAM=""
MAX_EXP=""
SESSION_ID="${SESSION_ID:-$(date +%Y%m%d-%H%M%S)-autoresearch}"
PRINT_ONLY=0
USE_BASH_FALLBACK=0

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | grep "^#" | sed 's/^# \?//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --program)         PROGRAM="$2"; shift 2 ;;
        --max-experiments) MAX_EXP="$2"; shift 2 ;;
        --session-id)      SESSION_ID="$2"; shift 2 ;;
        --print-only)      PRINT_ONLY=1; shift ;;
        --bash)            USE_BASH_FALLBACK=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *)                 echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

[ -n "$PROGRAM" ] || { echo "ERROR: --program required" >&2; usage; exit 2; }

# --- resolve program path ---------------------------------------------------
PROGRAM_PATH=""
for candidate in \
    "programs/${PROGRAM}.md" \
    "programs/lerobot-policy-${PROGRAM}.md" \
    "programs/wm-${PROGRAM}.md" \
    "${PROGRAM}"; do
    if [ -f "$WORKSPACE/$candidate" ] || [ -f "$candidate" ]; then
        PROGRAM_PATH="$(realpath "${WORKSPACE}/${candidate}" 2>/dev/null || realpath "${candidate}")"
        break
    fi
done

if [ -z "$PROGRAM_PATH" ] || [ ! -f "$PROGRAM_PATH" ]; then
    echo "ERROR: could not find program for alias '$PROGRAM'" >&2
    echo "Available programs:" >&2
    ls "$WORKSPACE/programs/"*.md 2>/dev/null | grep -v _domain_knowledge.md | grep -v README.md | sed 's|.*/||' | sed 's/^/  - /' >&2
    exit 3
fi

echo "Resolved program: $PROGRAM_PATH"
echo "Session ID:       $SESSION_ID"

# --- deterministic bash fallback --------------------------------------------
if [ "$USE_BASH_FALLBACK" = 1 ]; then
    echo "Running deterministic bash fallback (no LLM proposer)..."
    export SESSION_ID
    exec bash "$WORKSPACE/scripts/_run_autoresearch_smoke.sh"
fi

# --- canonical /autoresearch invocation -------------------------------------
CMD="/autoresearch $PROGRAM_PATH --type ml_model"
[ -n "$MAX_EXP" ] && CMD="$CMD --iterations $MAX_EXP"

cat <<EOF

To run inside Claude Code, paste:

    $CMD

The orchestrator agent will:
  1. Parse the program (including the \`domain_knowledge\` ref card).
  2. Spawn autoresearch-ml-proposer-worker / autoresearch-ml-executor-worker.
  3. Persist .agent-state/${SESSION_ID}/autoresearch/<slug>/{history,best,plateau,program}.{jsonl,json}

The dashboard's Autoresearch tab auto-picks up the artefacts once written.

For a fully autonomous bash-only loop (no LLM proposer), re-run with --bash.

EOF

if [ "$PRINT_ONLY" = 1 ]; then
    exit 0
fi

# --- envvar export so the executor inherits a clean parquet path ------------
export PYTHONNOUSERSITE=1
export LEROBOT_ISAAC_AUTORESEARCH_SESSION="$SESSION_ID"

echo "Tip: this wrapper cannot itself invoke /autoresearch (slash commands are"
echo "executed by Claude Code, not bash). Copy the command above into the"
echo "Claude Code prompt, or pass --bash to run the deterministic fallback."
