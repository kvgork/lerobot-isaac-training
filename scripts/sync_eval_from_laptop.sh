#!/usr/bin/env bash
# sync_eval_from_laptop.sh — pull closed-loop eval JSONs back to the desktop.
#
# Usage (run on DESKTOP):
#   bash scripts/sync_eval_from_laptop.sh [--host laptop] [--remote-base ~/workspaces/lerobot-isaac-deploy]
#
# Mirrors:
#   laptop:<base>/outputs/eval/*.json  →  desktop:./outputs/eval/
#
# Run after a closed-loop eval session on the laptop. Dashboard's
# Evaluation tab picks up the new JSONs on next refresh.
set -uo pipefail

LAPTOP_HOST="${LAPTOP_HOST:-laptop}"
LAPTOP_BASE="${LAPTOP_BASE:-\$HOME/workspaces/lerobot-isaac-deploy}"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)         LAPTOP_HOST="$2"; shift 2 ;;
        --remote-base)  LAPTOP_BASE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$WORKSPACE/outputs/eval"

echo "Pulling eval JSONs from $LAPTOP_HOST:$LAPTOP_BASE/outputs/eval/"
rsync -av --human-readable --progress --include='*.json' --exclude='*' \
    "$LAPTOP_HOST:$LAPTOP_BASE/outputs/eval/" \
    "$WORKSPACE/outputs/eval/"

echo ""
echo "Refreshing dashboard..."
"$WORKSPACE/.pixi/envs/dashboard/bin/python" -m lerobot_isaac_dashboard.report \
    --workspace "$WORKSPACE" \
    --output-dir "$WORKSPACE/outputs/eval-after-laptop-sync" \
    2>&1 | tail -3
echo ""
echo "Open http://localhost:8501 (Evaluation tab) to see the new closed-loop runs."
