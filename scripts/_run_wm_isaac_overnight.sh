#!/usr/bin/env bash
# =============================================================================
# _run_wm_isaac_overnight.sh — Overnight DreamerV3 training in Isaac Lab.
#
# Replaces the HDF5-replay-env path. The actor head now has agency (actions
# affect physics) AND a reward gradient (progress_reward shapes EE→object
# distance via lerobot_isaac_env.rewards).
#
# Single-trial production train (NOT an HP sweep). The first overnight
# validates the full stack (sheeprl + IsaacSO101Env + DreamerV3 + reward) —
# multi-trial HP sweep follows once we know baseline numbers.
#
# Persists state to the autoresearch on-disk schema so the dashboard's
# Autoresearch tab picks it up.
#
# Knobs:
#   STEPS                   default 50000  (~30 Hz wall ~50 ms/step → ~40 min)
#   BATCH_SIZE              default 16     (↑ from 4 per 2026-05-23 perf v6;
#                                          fits 7 GB VRAM at num_envs=2)
#   LR                      default 1e-4
#   IMAGE_SIZE              default 64
#   MAX_EPISODE_STEPS       default 300    (10 s at 30 Hz)
#   NUM_ENVS                default 2      (↑ from 1; Isaac Lab vectorized)
#   DISCRETE_SIZE           default 32     (world model capacity)
#   STOCHASTIC_SIZE         default 32
#   REPLAY_RATIO            default 2      (↑ from 1; more grad-steps per env-step)
#   PRECISION               default bf16-mixed  (NEW; Ampere bf16 AMP)
#   CHECKPOINT_EVERY        default 10000
#   SECONDS_PER_EXP         default 21600  (6 h ceiling)
#   SESSION_ID              default wm-isaac-<ts>
#
# 2026-05-23 perf-tuning rationale:
#   Steady-state diagnostic (PID 510972 dmon × 30) showed SM 44.6 % mean,
#   clock 1950/2100 MHz, power 145 W / 320 W TDP, VRAM 6.9 GB / 9.85 GB,
#   main thread R (running). NOT env-rollout bound (despite an initial v5
#   misdiagnosis on boot-phase numbers). Real bottleneck = Class B (batch +
#   precision). Expected combined wall-clock 2-3× faster, SM 44 % → 70-80 %.
#   Full write-up: 05-Wiki/sources/2026-05-23-wm-gpu-perf-experiment.md §v6.
#
# Output:
#   outputs/wm-isaac-prod/                                    Hydra run dir
#   logs/runs/dreamer_v3/isaac_so101/wm-isaac-prod-<ts>/...   sheeprl ckpts
#   .agent-state/<session>/autoresearch/wm-isaac-prod/*.json
# =============================================================================
set -uo pipefail

WORKSPACE="${WORKSPACE:-${LEROBOT_ISAAC_WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$WORKSPACE"

SESSION_ID="${SESSION_ID:-wm-isaac-$(date +%Y%m%d-%H%M%S)}"
SLUG="wm-isaac-prod"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"          # 2026-05-23 perf v6: ↑ from 4
LR="${LR:-1e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
NUM_ENVS="${NUM_ENVS:-2}"               # 2026-05-23 perf v6: ↑ from 1
DISCRETE_SIZE="${DISCRETE_SIZE:-32}"
STOCHASTIC_SIZE="${STOCHASTIC_SIZE:-32}"
REPLAY_RATIO="${REPLAY_RATIO:-2}"       # 2026-05-23 perf v6: ↑ from 1
PRECISION="${PRECISION:-bf16-mixed}"    # 2026-05-23 perf v6: NEW
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
SECONDS_PER_EXP="${SECONDS_PER_EXP:-21600}"
# Extra Hydra overrides appended verbatim to the sheeprl cmd. Used to
# tune actor entropy / exploration without code edits. Space-separated.
EXTRA_HYDRA="${EXTRA_HYDRA:-}"
DRY_RUN="${DRY_RUN:-0}"

PY="$WORKSPACE/.pixi/envs/sim/bin/python"
AR_DIR="$WORKSPACE/.agent-state/$SESSION_ID/autoresearch/$SLUG"
OUT_DIR="$WORKSPACE/outputs/wm-isaac-prod-$SESSION_ID"
PROGRAM="$AR_DIR/program.json"

[ -x "$PY" ] || { echo "ERROR: sim python not found at $PY" >&2; exit 2; }

# sheeprl is in train-dreamer, not sim, by default. Probe and add path if needed.
"$PY" -c "import sheeprl, lerobot_isaac_env, lerobot_isaac_adapters" 2>/dev/null || {
    echo "WARN: sheeprl/lerobot_isaac_env/lerobot_isaac_adapters missing in sim env"
    echo "      run: pixi run -e sim pip install sheeprl-from-git && pip install -e <siblings>"
    echo "      OR launch via train-dreamer env if Isaac Lab is also there"
    # Try train-dreamer (has sheeprl + adapters; lerobot_isaac_env via path).
    PY="$WORKSPACE/.pixi/envs/train-dreamer/bin/python"
    "$PY" -c "import sheeprl, lerobot_isaac_env, lerobot_isaac_adapters" 2>/dev/null \
        || { echo "ERROR: neither sim nor train-dreamer env has all 3 packages" >&2; exit 2; }
    echo "      → falling back to train-dreamer python"
}

mkdir -p "$AR_DIR" "$OUT_DIR"

cat > "$PROGRAM" <<EOF
{
  "name": "$SLUG",
  "metric": {"name": "episode_return", "direction": "maximize"},
  "budget": {"seconds_per_experiment": $SECONDS_PER_EXP, "max_experiments": 1, "plateau_limit": 1},
  "target_arch": "dreamerv3",
  "env": "isaac_so101",
  "config": {
    "steps": $STEPS, "batch_size": $BATCH_SIZE, "lr": $LR,
    "image_size": $IMAGE_SIZE, "max_episode_steps": $MAX_EPISODE_STEPS,
    "num_envs": $NUM_ENVS,
    "discrete_size": $DISCRETE_SIZE, "stochastic_size": $STOCHASTIC_SIZE,
    "replay_ratio": $REPLAY_RATIO, "checkpoint_every": $CHECKPOINT_EVERY,
    "precision": "$PRECISION"
  },
  "session_id": "$SESSION_ID",
  "ts_start": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "[wm-isaac] session=$SESSION_ID slug=$SLUG"
echo "[wm-isaac] steps=$STEPS bs=$BATCH_SIZE lr=$LR num_envs=$NUM_ENVS image_size=$IMAGE_SIZE"
echo "[wm-isaac] capacity: D=$DISCRETE_SIZE S=$STOCHASTIC_SIZE replay_ratio=$REPLAY_RATIO precision=$PRECISION"
echo "[wm-isaac] output=$OUT_DIR"

# Build train cmd via the autoresearch wrapper so its metric_extractor
# guarantees a final stdout metric line.
CMD=(
    timeout "$SECONDS_PER_EXP"
    "$PY" -m lerobot_isaac_autoresearch.train_wrapper
        --target_arch dreamerv3
        --output_dir "$OUT_DIR"
        --steps "$STEPS"
        --batch_size "$BATCH_SIZE"
        --lr "$LR"
        --
        --env isaac_so101
        "env.num_envs=$NUM_ENVS"
        "env.image_size=$IMAGE_SIZE"
        "env.max_episode_steps=$MAX_EPISODE_STEPS"
        "env.headless=True"
        "fabric.accelerator=gpu"
        "fabric.devices=1"
        "fabric.precision=$PRECISION"
        "algo.world_model.discrete_size=$DISCRETE_SIZE"
        "algo.world_model.stochastic_size=$STOCHASTIC_SIZE"
        "algo.replay_ratio=$REPLAY_RATIO"
        "algo.total_steps=$STEPS"
        "checkpoint.every=$CHECKPOINT_EVERY"
)
# Append extra Hydra overrides (actor entropy, exploration, etc.).
if [ -n "$EXTRA_HYDRA" ]; then
    read -r -a EXTRA_TOKENS <<< "$EXTRA_HYDRA"
    CMD+=( "${EXTRA_TOKENS[@]}" )
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "[wm-isaac] (dry-run) command:"
    printf '  %s\n' "${CMD[@]}"
    exit 0
fi

PYTHONUNBUFFERED=1 "${CMD[@]}" > "$AR_DIR/train.log" 2>&1
rc=$?
echo "[wm-isaac] done rc=$rc"
echo "[wm-isaac] state: $AR_DIR"
echo "[wm-isaac] ckpts: logs/runs/dreamer_v3/isaac_so101/"
ls -la "$AR_DIR" 2>/dev/null
