#!/usr/bin/env bash
# sync_ckpt_to_laptop.sh — push a trained checkpoint + dataset meta to the laptop.
#
# Usage (run on DESKTOP):
#   bash scripts/sync_ckpt_to_laptop.sh [--host laptop] [--remote-base ~/workspaces/lerobot-isaac-deploy] [--run-dir outputs/long-train-2026-05-14-diffusion-dreamerv3-4h]
#
# Defaults pull from env: LAPTOP_HOST, LAPTOP_BASE, RUN_DIR.
#
# Transfers:
#   - <run-dir>/policy-diffusion/checkpoints/<latest>/pretrained_model/ → laptop:<base>/checkpoints/<run-name>/<latest>/pretrained_model/
#   - datasets/<repo>/                                                  → laptop:<base>/datasets/<repo>/      (incremental)
#   - <run-dir>/dashboard/manifest.json                                  → laptop:<base>/checkpoints/<run-name>/manifest.json
#
# Uses rsync over SSH. Configure your ~/.ssh/config so `LAPTOP_HOST` resolves.
set -uo pipefail

LAPTOP_HOST="${LAPTOP_HOST:-laptop}"
LAPTOP_BASE="${LAPTOP_BASE:-\$HOME/workspaces/lerobot-isaac-deploy}"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR=""
DATASET_DIR="datasets/kvgork/so101-pickplace1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)         LAPTOP_HOST="$2"; shift 2 ;;
        --remote-base)  LAPTOP_BASE="$2"; shift 2 ;;
        --run-dir)      RUN_DIR="$2"; shift 2 ;;
        --dataset)      DATASET_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$RUN_DIR" ]; then
    # Pick the most recent long-train run dir
    RUN_DIR="$(ls -td "$WORKSPACE/outputs/long-train-"* 2>/dev/null | head -1)"
fi
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "no --run-dir given and no outputs/long-train-* found" >&2
    exit 2
fi
RUN_NAME="$(basename "$RUN_DIR")"

# Pick latest checkpoint (highest numbered or 'last')
LATEST_CKPT="$(ls -d "$RUN_DIR/policy-diffusion/checkpoints/"*/pretrained_model 2>/dev/null | sort | tail -1)"
if [ -z "$LATEST_CKPT" ] || [ ! -d "$LATEST_CKPT" ]; then
    echo "no pretrained_model under $RUN_DIR/policy-diffusion/checkpoints/" >&2
    exit 3
fi
CKPT_PARENT_NAME="$(basename "$(dirname "$LATEST_CKPT")")"   # e.g. 0024000

echo "Pushing:"
echo "  run        $RUN_NAME"
echo "  ckpt       $CKPT_PARENT_NAME"
echo "  dataset    $DATASET_DIR"
echo "  to         $LAPTOP_HOST:$LAPTOP_BASE"
echo ""

# 1. Checkpoint
rsync -av --human-readable --progress \
    "$LATEST_CKPT/" \
    "$LAPTOP_HOST:$LAPTOP_BASE/checkpoints/$RUN_NAME/$CKPT_PARENT_NAME/pretrained_model/"

# 2. Manifest + run metadata (small; rsync the whole dashboard/ dir)
rsync -av --human-readable \
    "$RUN_DIR/dashboard/manifest.json" \
    "$LAPTOP_HOST:$LAPTOP_BASE/checkpoints/$RUN_NAME/manifest.json"

# 3. Dataset meta only by default (data parquet is large; skip unless --full-dataset)
rsync -av --human-readable --progress \
    "$WORKSPACE/$DATASET_DIR/meta/" \
    "$LAPTOP_HOST:$LAPTOP_BASE/$DATASET_DIR/meta/"

echo ""
echo "Done. On the laptop:"
echo "    pixi shell -e deploy"
echo "    robot-data-run-check \\"
echo "        --policy-path $LAPTOP_BASE/checkpoints/$RUN_NAME/$CKPT_PARENT_NAME/pretrained_model \\"
echo "        --dataset-root $LAPTOP_BASE/$DATASET_DIR"
