#!/usr/bin/env bash
# =============================================================================
# sim_deploy.sh — Closed-loop sim deploy entry script.
#
# Wraps `lerobot_isaac_deploy.sim.IsaacSceneSession.run()` so a trained
# policy can be evaluated against an Isaac Sim USD scene WITHOUT hardware.
#
# Status: Phase 3 of plans/2026-05-23-sim-deploy-pipeline.md (scaffold).
# The Isaac Sim runtime backend is Phase 2 future work — until it lands,
# this script supports the synthetic-marker stub for CI / wiring tests.
#
# Knobs (flags or env):
#   --policy-path <path>          (required)
#   --usd <path>                  default: assets/sim_scenes/so101_workspace.usd
#   --dataset-root <path>         default: datasets/kvgork/so101-pickplace1
#   --n-episodes <int>            default: 10
#   --max-steps <int>             default: 600
#   --rate-hz <float>             default: 30
#   --success-criterion <name>    default: pickplace_basket
#   --output-dir <path>           default: outputs/sim_deploy/<ts>
#   --dr-config <path>            optional
#   --pixi-env <name>             default: sim (for Isaac Sim) or train-dreamer (for synthetic)
#   --dry-run                     echo resolved args + exit
#
# Usage examples:
#   # Smoke (synthetic fixture, no Isaac Sim required)
#   bash scripts/sim_deploy.sh \
#       --policy-path src/lerobot-isaac-deploy/tests/fixtures/dreamerv3_synthetic/ \
#       --n-episodes 1
#
#   # Real run (when Phase 2 lands)
#   bash scripts/sim_deploy.sh \
#       --policy-path outputs/autoresearch-lerobot-policy-smolvla-lora/trial_12/checkpoints/merged/pretrained_model \
#       --usd assets/sim_scenes/so101_workspace.usd
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

POLICY_PATH=""
USD_PATH="assets/sim_scenes/so101_workspace.usd"
DATASET_ROOT="datasets/kvgork/so101-pickplace1"
N_EPISODES=10
MAX_STEPS=600
RATE_HZ=30
SUCCESS_CRITERION="pickplace_basket"
OUTPUT_DIR=""
DR_CONFIG=""
PIXI_ENV=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --policy-path)        POLICY_PATH="$2"; shift 2 ;;
        --usd)                USD_PATH="$2"; shift 2 ;;
        --dataset-root)       DATASET_ROOT="$2"; shift 2 ;;
        --n-episodes)         N_EPISODES="$2"; shift 2 ;;
        --max-steps)          MAX_STEPS="$2"; shift 2 ;;
        --rate-hz)            RATE_HZ="$2"; shift 2 ;;
        --success-criterion)  SUCCESS_CRITERION="$2"; shift 2 ;;
        --output-dir)         OUTPUT_DIR="$2"; shift 2 ;;
        --dr-config)          DR_CONFIG="$2"; shift 2 ;;
        --pixi-env)           PIXI_ENV="$2"; shift 2 ;;
        --dry-run)            DRY_RUN=1; shift ;;
        -h|--help)            sed -n '2,30p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

[ -n "$POLICY_PATH" ] || { echo "ERROR: --policy-path required" >&2; exit 2; }

# Auto-pick pixi env: synthetic ckpts can run in any env; real Isaac Sim needs `sim`.
if [ -z "$PIXI_ENV" ]; then
    if [ -f "$POLICY_PATH/synthetic_marker.json" ]; then
        PIXI_ENV="train-dreamer"
    else
        PIXI_ENV="sim"
    fi
fi

PY="$WORKSPACE/.pixi/envs/$PIXI_ENV/bin/python"
[ -x "$PY" ] || { echo "ERROR: pixi env '$PIXI_ENV' not installed at $PY" >&2; exit 2; }

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="outputs/sim_deploy/$(date +%Y%m%dT%H%M%S)"
fi

echo "[sim_deploy] policy=$POLICY_PATH"
echo "[sim_deploy] usd=$USD_PATH"
echo "[sim_deploy] dataset=$DATASET_ROOT"
echo "[sim_deploy] n_episodes=$N_EPISODES max_steps=$MAX_STEPS rate=${RATE_HZ}Hz"
echo "[sim_deploy] success=$SUCCESS_CRITERION"
echo "[sim_deploy] output=$OUTPUT_DIR"
echo "[sim_deploy] pixi_env=$PIXI_ENV"

[ "$DRY_RUN" = "1" ] && { echo "[sim_deploy] dry-run — exiting"; exit 0; }

mkdir -p "$OUTPUT_DIR"

PYTHONPATH="$WORKSPACE/src/lerobot-isaac-deploy/src:${PYTHONPATH:-}" "$PY" - <<PY
from pathlib import Path
from lerobot_isaac_deploy.sim import IsaacSceneSession

session = IsaacSceneSession(
    usd_path=Path("$USD_PATH"),
    policy_path=Path("$POLICY_PATH"),
    dataset_root=Path("$DATASET_ROOT"),
    n_episodes=$N_EPISODES,
    max_steps=$MAX_STEPS,
    rate_hz=$RATE_HZ,
    success_criterion="$SUCCESS_CRITERION",
    output_dir=Path("$OUTPUT_DIR"),
    dr_config=Path("$DR_CONFIG") if "$DR_CONFIG" else None,
)
out_path = session.run()
print(f"[sim_deploy] rollout summary written → {out_path}")
PY
rc=$?

if [ "$rc" -ne 0 ]; then
    echo "[sim_deploy] FAILED rc=$rc" >&2
    exit "$rc"
fi

echo "[sim_deploy] done"
ls -la "$OUTPUT_DIR"
