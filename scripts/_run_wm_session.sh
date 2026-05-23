#!/usr/bin/env bash
# =============================================================================
# _run_wm_session.sh — Generic long world-model training session.
#
# Single-trial wrapper around lerobot_isaac_autoresearch.train_wrapper for a
# world-model arch (DreamerV3 or LeWorldModel). Persists state to the
# autoresearch on-disk schema so the dashboard's Autoresearch tab auto-discovers
# it (same pattern as scripts/_run_autoresearch_lora.sh).
#
# Why this script:
#   - The wm-* programs target multi-trial HP sweeps; this is the single-long-
#     session variant for budget-bounded training runs.
#   - lerobot 0.5.x does NOT ship `lerobot.scripts.train_world_model` →
#     LeWorldModel real training is BLOCKED. DreamerV3 is the only currently
#     working backend. The le_world_model arch path is kept here for the day
#     upstream lands the CLI.
#
# Usage:
#   bash scripts/_run_wm_session.sh [flags]
#
# Flags (long form preferred; env vars override defaults but flags override env):
#   --arch <dreamerv3|le_world_model>   Target world-model architecture (default: dreamerv3)
#   --dataset <path>                    LeRobotDataset root OR pre-bridged HDF5
#   --steps <int>                       Training steps (default: 500000)
#   --batch-size <int>                  Per-rank batch size (default: 8)
#   --lr <float>                        WM optimizer learning rate (default: 1e-4)
#   --seed <int>                        RNG seed (default: 42)
#   --image-size <int>                  Square image side passed to bridge (default: 64 dreamer / 96 lewm)
#   --window <int>                      Bridge window length (default: 16)
#   --stride <int>                      Bridge stride (default: 8)
#   --timeout <seconds>                 Hard wall timeout (default: 43200 = 12 h)
#   --poll <seconds>                    History.jsonl append cadence (default: 300)
#   --session-id <str>                  Override SESSION_ID (default: auto-generated)
#   --slug <str>                        Override autoresearch slug (default: wm-<arch>)
#   --output-dir <path>                 Override outputs/<session_id> default
#   --hdf5-cache <path>                 Bridge output path (default: outputs/wm_data/<dataset_name>_<arch>.hdf5)
#   --extra <"k=v k=v">                 Extra Hydra-style overrides forwarded after `--`
#   --pixi-env <name>                   Override pixi env (default: train-dreamer / train-lewm by arch)
#   --claude-code-root <path>           Override $CLAUDE_CODE_ROOT (default: $HOME/tools/claude_code)
#   --dry-run                           Echo bridge cmd + train cmd, no execution
#   --help                              Show this header and exit
#
# Env-var equivalents (same names but UPPER_SNAKE):
#   ARCH STEPS BATCH_SIZE LR SEED IMAGE_SIZE WINDOW STRIDE
#   SECONDS_PER_EXP METRIC_POLL_S SESSION_ID SLUG
#   DATASET HDF5_CACHE OUTPUT_DIR EXTRA PIXI_ENV CLAUDE_CODE_ROOT DRY_RUN
#
# On-disk artefacts:
#   .agent-state/<session_id>/autoresearch/<slug>/program.json
#   .agent-state/<session_id>/autoresearch/<slug>/history.jsonl
#   .agent-state/<session_id>/autoresearch/<slug>/best.json
#   .agent-state/<session_id>/autoresearch/<slug>/plateau.json
#   .agent-state/<session_id>/autoresearch/<slug>/train.log
#
# Examples:
#   # 12 h DreamerV3 default
#   bash scripts/_run_wm_session.sh
#
#   # Same but on a different dataset and longer steps
#   bash scripts/_run_wm_session.sh \
#       --dataset datasets/kvgork/so101-pickplace2 --steps 1000000
#
#   # LeWorldModel (BLOCKED upstream as of lerobot 0.5.x — kept for future)
#   bash scripts/_run_wm_session.sh --arch le_world_model --image-size 96
#
#   # Custom Hydra knobs for sheeprl
#   bash scripts/_run_wm_session.sh \
#       --extra "algo.replay_ratio=4 algo.world_model.discrete_size=64"
#
#   # Override pixi env (uncommon)
#   bash scripts/_run_wm_session.sh --pixi-env train-dreamer
# =============================================================================
set -uo pipefail

WORKSPACE="${WORKSPACE:-${LEROBOT_ISAAC_WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$WORKSPACE"

print_help() {
    sed -n '2,80p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'
}

# --- defaults via env -------------------------------------------------------
ARCH="${ARCH:-dreamerv3}"
STEPS="${STEPS:-500000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
IMAGE_SIZE="${IMAGE_SIZE:-}"
WINDOW="${WINDOW:-16}"
STRIDE="${STRIDE:-8}"
SECONDS_PER_EXP="${SECONDS_PER_EXP:-43200}"
METRIC_POLL_S="${METRIC_POLL_S:-300}"
SESSION_ID="${SESSION_ID:-}"
SLUG="${SLUG:-}"
DATASET="${DATASET:-datasets/kvgork/so101-pickplace1}"
HDF5_CACHE="${HDF5_CACHE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
EXTRA="${EXTRA:-}"
PIXI_ENV="${PIXI_ENV:-}"
CLAUDE_CODE_ROOT="${CLAUDE_CODE_ROOT:-$HOME/tools/claude_code}"
DRY_RUN="${DRY_RUN:-0}"

# --- flag parsing -----------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --arch)              ARCH="$2"; shift 2 ;;
        --dataset)           DATASET="$2"; shift 2 ;;
        --steps)             STEPS="$2"; shift 2 ;;
        --batch-size)        BATCH_SIZE="$2"; shift 2 ;;
        --lr)                LR="$2"; shift 2 ;;
        --seed)              SEED="$2"; shift 2 ;;
        --image-size)        IMAGE_SIZE="$2"; shift 2 ;;
        --window)            WINDOW="$2"; shift 2 ;;
        --stride)            STRIDE="$2"; shift 2 ;;
        --timeout)           SECONDS_PER_EXP="$2"; shift 2 ;;
        --poll)              METRIC_POLL_S="$2"; shift 2 ;;
        --session-id)        SESSION_ID="$2"; shift 2 ;;
        --slug)              SLUG="$2"; shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2"; shift 2 ;;
        --hdf5-cache)        HDF5_CACHE="$2"; shift 2 ;;
        --extra)             EXTRA="$2"; shift 2 ;;
        --pixi-env)          PIXI_ENV="$2"; shift 2 ;;
        --claude-code-root)  CLAUDE_CODE_ROOT="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=1; shift ;;
        -h|--help)           print_help; exit 0 ;;
        *) echo "unknown flag: $1" >&2; print_help; exit 2 ;;
    esac
done

# --- arch-derived defaults --------------------------------------------------
case "$ARCH" in
    dreamerv3)
        PIXI_ENV="${PIXI_ENV:-train-dreamer}"
        IMAGE_SIZE="${IMAGE_SIZE:-64}"
        METRIC_NAME="recon_loss"
        METRIC_DIR="minimize"
        METRIC_REGEX='recon_loss[=:][[:space:]]*[0-9.eE+\-]+'
        DEFAULT_EXTRA_HYDRA="algo.world_model.optimizer.lr=$LR algo.per_rank_batch_size=$BATCH_SIZE algo.total_steps=$STEPS seed=$SEED env.capture_video=False fabric.accelerator=gpu fabric.devices=1"
        ;;
    le_world_model)
        PIXI_ENV="${PIXI_ENV:-train-lewm}"
        IMAGE_SIZE="${IMAGE_SIZE:-96}"
        METRIC_NAME="pred_loss"
        METRIC_DIR="minimize"
        METRIC_REGEX='pred_loss[=:][[:space:]]*[0-9.eE+\-]+'
        DEFAULT_EXTRA_HYDRA=""
        echo "[wm] WARNING: le_world_model real training is BLOCKED in lerobot 0.5.x" >&2
        echo "[wm] WARNING: train_world_model script does not exist upstream; this run will fail" >&2
        ;;
    *)
        echo "ERROR: unsupported --arch '$ARCH' (expected: dreamerv3 | le_world_model)" >&2
        exit 2
        ;;
esac

SLUG="${SLUG:-wm-${ARCH}}"
SESSION_ID="${SESSION_ID:-${SLUG}-$(date +%Y%m%d-%H%M%S)}"
PY="$WORKSPACE/.pixi/envs/$PIXI_ENV/bin/python"

# Bridge cache name derived from dataset basename + arch.
DS_BASENAME=$(basename "$DATASET")
HDF5_CACHE="${HDF5_CACHE:-outputs/wm_data/${DS_BASENAME}_${ARCH}.hdf5}"

OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/outputs/$SESSION_ID}"
AR_DIR="$WORKSPACE/.agent-state/$SESSION_ID/autoresearch/$SLUG"
HISTORY="$AR_DIR/history.jsonl"
BEST="$AR_DIR/best.json"
PLATEAU="$AR_DIR/plateau.json"
PROGRAM="$AR_DIR/program.json"
TRAIN_LOG="$AR_DIR/train.log"

# --- pre-flight -------------------------------------------------------------
# Dataset can be a LeRobotDataset dir OR a pre-bridged HDF5 file.
if [ ! -e "$DATASET" ]; then
    echo "ERROR: dataset not found: $DATASET" >&2; exit 2
fi
[ -x "$PY" ] || { echo "ERROR: $PIXI_ENV python not found: $PY (pixi install -e $PIXI_ENV?)" >&2; exit 2; }

if [ "$ARCH" = "dreamerv3" ]; then
    "$PY" -c "import sheeprl" 2>/dev/null \
        || { echo "ERROR: sheeprl not installed in $PIXI_ENV. Run: bash scripts/install_train_deps.sh" >&2; exit 2; }
fi
[ -d "$CLAUDE_CODE_ROOT/skills/lerobot_world_model_bridge" ] \
    || { echo "ERROR: bridge skill not found at $CLAUDE_CODE_ROOT/skills/lerobot_world_model_bridge" >&2; exit 2; }

mkdir -p "$AR_DIR" "$OUTPUT_DIR" "$(dirname "$HDF5_CACHE")"

# --- bridge (idempotent, skipped when --dataset is already HDF5) -----------
if [[ "$DATASET" == *.h5 || "$DATASET" == *.hdf5 ]]; then
    HDF5_CACHE="$DATASET"
    echo "[wm] dataset is already HDF5 — skipping bridge"
elif [ ! -f "$HDF5_CACHE" ]; then
    echo "[wm] bridging $DATASET → $HDF5_CACHE (image_size=${IMAGE_SIZE}, window=${WINDOW}, stride=${STRIDE})"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[wm] (dry-run) skipping bridge"
    else
        PYTHONPATH="$CLAUDE_CODE_ROOT:${PYTHONPATH:-}" "$PY" - <<PY
import sys
from skills.lerobot_world_model_bridge.operations import lerobot_to_worldmodel
result = lerobot_to_worldmodel(
    dataset_path="$DATASET",
    output_path="$HDF5_CACHE",
    output_format="hdf5",
    image_size=($IMAGE_SIZE, $IMAGE_SIZE),
    window_size=$WINDOW,
    stride=$STRIDE,
    normalize_actions=True,
)
if not result.success:
    print(f"[wm] bridge failed: {result.error}", file=sys.stderr)
    sys.exit(3)
print(f"[wm] bridge done: {result.data}")
PY
        bridge_rc=$?
        [ "$bridge_rc" -eq 0 ] || { echo "ERROR: bridge step failed rc=$bridge_rc" >&2; exit 3; }
    fi
else
    echo "[wm] bridge cache HIT: $HDF5_CACHE"
fi

# --- program snapshot (dashboard reads this) --------------------------------
cat > "$PROGRAM" <<EOF
{
  "name": "$SLUG",
  "metric": {"name": "$METRIC_NAME", "direction": "$METRIC_DIR"},
  "budget": {
    "seconds_per_experiment": $SECONDS_PER_EXP,
    "max_experiments": 1,
    "plateau_limit": 1
  },
  "target_arch": "$ARCH",
  "pixi_env": "$PIXI_ENV",
  "dataset": "$DATASET",
  "hdf5_cache": "$HDF5_CACHE",
  "steps": $STEPS,
  "batch_size": $BATCH_SIZE,
  "lr": $LR,
  "seed": $SEED,
  "image_size": $IMAGE_SIZE,
  "window": $WINDOW,
  "stride": $STRIDE,
  "iterations": 1,
  "session_id": "$SESSION_ID",
  "ts_start": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "[wm] session=$SESSION_ID slug=$SLUG arch=$ARCH"
echo "[wm] steps=$STEPS bs=$BATCH_SIZE lr=$LR seed=$SEED timeout=${SECONDS_PER_EXP}s"
echo "[wm] dataset=$DATASET"
echo "[wm] hdf5=$HDF5_CACHE"
echo "[wm] state_dir=$AR_DIR"
echo "[wm] out_dir=$OUTPUT_DIR"
echo "[wm] pixi_env=$PIXI_ENV"

# --- build train cmd --------------------------------------------------------
# Forward DEFAULT_EXTRA_HYDRA + user-supplied EXTRA via the train_wrapper's `--`
# passthrough → arch-specific subprocess. The wrapper itself only consumes
# --target_arch / --dataset / --output_dir / --steps / --batch_size; the rest
# rides after `--`.
CMD=(
    timeout "$SECONDS_PER_EXP"
    "$PY" -m lerobot_isaac_autoresearch.train_wrapper
        --target_arch "$ARCH"
        --dataset "$HDF5_CACHE"
        --output_dir "$OUTPUT_DIR"
        --steps "$STEPS"
        --batch_size "$BATCH_SIZE"
        --
)
# Split DEFAULT_EXTRA_HYDRA + EXTRA into individual tokens.
read -r -a DEFAULT_EXTRA_TOKENS <<< "$DEFAULT_EXTRA_HYDRA"
read -r -a EXTRA_TOKENS <<< "$EXTRA"
CMD+=( "${DEFAULT_EXTRA_TOKENS[@]}" )
[ ${#EXTRA_TOKENS[@]} -gt 0 ] && CMD+=( "${EXTRA_TOKENS[@]}" )

if [ "$DRY_RUN" = "1" ]; then
    echo
    echo "[wm] (dry-run) command would be:"
    printf '  %s\n' "${CMD[@]}"
    : > "$HISTORY"
    exit 0
fi

# --- launch training in background, poll metric every N seconds -------------
: > "$HISTORY"
: > "$TRAIN_LOG"
start_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
start_s=$(date +%s)

echo "[wm] launching training subprocess (background)"
PYTHONPATH="$CLAUDE_CODE_ROOT:${PYTHONPATH:-}" \
PATH="$WORKSPACE/.pixi/envs/$PIXI_ENV/bin:$PATH" \
PYTHONUNBUFFERED=1 \
    "${CMD[@]}" > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "[wm] training PID=$TRAIN_PID"

on_exit() {
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[wm] caught signal — killing training PID=$TRAIN_PID"
        kill -INT "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
}
trap on_exit INT TERM

# --- metric polling loop ----------------------------------------------------
# Comparator factory: minimise vs maximise.
if [ "$METRIC_DIR" = "minimize" ]; then
    BETTER='exit(0 if float("$candidate") < float("$incumbent") else 1)'
else
    BETTER='exit(0 if float("$candidate") > float("$incumbent") else 1)'
fi

is_better() {
    # $1 = candidate; $2 = incumbent
    local candidate="$1"; local incumbent="$2"
    [ -z "$incumbent" ] && return 0
    "$PY" -c "exit(0 if float('$candidate') $( [ "$METRIC_DIR" = "minimize" ] && echo "<" || echo ">" ) float('$incumbent') else 1)"
}

best_metric=""
last_metric=""
iter=0
poll_until=$(( start_s + SECONDS_PER_EXP + 60 ))

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep "$METRIC_POLL_S"

    latest=$(grep -oE "$METRIC_REGEX" "$TRAIN_LOG" \
              | tail -1 | sed -E "s/${METRIC_NAME}[=:][[:space:]]*//")
    [ -z "$latest" ] && latest=""

    now_s=$(date +%s)
    elapsed=$(( now_s - start_s ))

    if [ -n "$latest" ] && [ "$latest" != "$last_metric" ]; then
        iter=$(( iter + 1 ))
        last_metric="$latest"

        "$PY" - <<PY >> "$HISTORY"
import json
print(json.dumps({
    "trial_index": $iter,
    "trial": $iter,
    "metric_name": "$METRIC_NAME",
    "metric_value": float("$latest"),
    "metric_kind": "${METRIC_NAME}_train",
    "config": {
        "arch": "$ARCH",
        "lr": float("$LR"),
        "batch_size": $BATCH_SIZE,
        "steps_target": $STEPS,
        "seed": $SEED,
        "image_size": $IMAGE_SIZE,
        "window": $WINDOW,
        "stride": $STRIDE,
        "dataset": "$DATASET"
    },
    "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "duration_s": $elapsed,
    "status": "running",
    "exit_code": -1,
}))
PY

        if is_better "$latest" "$best_metric"; then
            best_metric="$latest"
            "$PY" - <<PY > "$BEST"
import json
print(json.dumps({
    "trial": $iter,
    "metric_value": float("$latest"),
    "metric_kind": "${METRIC_NAME}_train",
    "config": {
        "arch": "$ARCH",
        "lr": float("$LR"),
        "batch_size": $BATCH_SIZE,
        "steps_target": $STEPS,
        "seed": $SEED,
        "image_size": $IMAGE_SIZE,
        "window": $WINDOW,
        "stride": $STRIDE
    }
}, indent=2))
PY
        fi

        "$PY" - <<PY > "$PLATEAU"
import json
print(json.dumps({
    "consecutive_non_improvements": 0,
    "plateau_limit": 1,
    "last_metric": float("$latest"),
    "best_metric": float("$best_metric"),
    "completed_trials": $iter,
    "planned_trials": 1,
    "elapsed_s": $elapsed,
    "budget_s": $SECONDS_PER_EXP
}, indent=2))
PY

        echo "[wm] poll iter=$iter elapsed=${elapsed}s ${METRIC_NAME}=$latest best=$best_metric"
    fi

    if [ "$now_s" -ge "$poll_until" ]; then
        echo "[wm] wall-clock budget exhausted — letting subprocess complete or be killed by timeout"
        break
    fi
done

wait "$TRAIN_PID" 2>/dev/null
train_rc=$?
end_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
dur=$(( $(date +%s) - start_s ))

status="ok"
if [ "$train_rc" -eq 124 ]; then status="timeout"
elif [ "$train_rc" -ne 0 ]; then status="error"
fi

latest=$(grep -oE "$METRIC_REGEX" "$TRAIN_LOG" \
          | tail -1 | sed -E "s/${METRIC_NAME}[=:][[:space:]]*//")
[ -z "$latest" ] && latest="${last_metric:-0.0}"
if [ -z "$best_metric" ]; then best_metric="$latest"; fi

iter=$(( iter + 1 ))
"$PY" - <<PY >> "$HISTORY"
import json
print(json.dumps({
    "trial_index": $iter,
    "trial": $iter,
    "metric_name": "$METRIC_NAME",
    "metric_value": float("$latest"),
    "metric_kind": "${METRIC_NAME}_final",
    "config": {
        "arch": "$ARCH",
        "lr": float("$LR"),
        "batch_size": $BATCH_SIZE,
        "steps_target": $STEPS,
        "seed": $SEED,
        "image_size": $IMAGE_SIZE,
        "window": $WINDOW,
        "stride": $STRIDE,
        "dataset": "$DATASET"
    },
    "ts": "$end_ts",
    "duration_s": $dur,
    "status": "$status",
    "exit_code": $train_rc,
}))
PY

if is_better "$latest" "$best_metric"; then
    best_metric="$latest"
fi
"$PY" - <<PY > "$BEST"
import json
print(json.dumps({
    "trial": $iter,
    "metric_value": float("$latest"),
    "best_metric_value": float("$best_metric"),
    "metric_kind": "${METRIC_NAME}_final",
    "config": {
        "arch": "$ARCH",
        "lr": float("$LR"),
        "batch_size": $BATCH_SIZE,
        "steps_target": $STEPS,
        "seed": $SEED,
        "image_size": $IMAGE_SIZE,
        "window": $WINDOW,
        "stride": $STRIDE
    },
    "status": "$status",
    "exit_code": $train_rc,
    "duration_s": $dur,
    "output_dir": "$OUTPUT_DIR"
}, indent=2))
PY

"$PY" - <<PY > "$PLATEAU"
import json
print(json.dumps({
    "consecutive_non_improvements": 0,
    "plateau_limit": 1,
    "last_metric": float("$latest"),
    "best_metric": float("$best_metric"),
    "completed_trials": $iter,
    "planned_trials": 1,
    "elapsed_s": $dur,
    "budget_s": $SECONDS_PER_EXP,
    "status": "$status"
}, indent=2))
PY

echo
echo "[wm] done"
echo "[wm] status=$status exit_code=$train_rc duration=${dur}s"
echo "[wm] final $METRIC_NAME=$latest best=$best_metric"
echo "[wm] state: $AR_DIR"
ls -la "$AR_DIR"
