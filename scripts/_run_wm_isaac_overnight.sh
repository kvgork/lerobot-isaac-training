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

# Orphan-kill trap. Trial 0 of wm-isaac-hp-v4 left a zombie python +
# Isaac kit process alive after the wrapper exited; trials 1-3 each
# died in 38s with SimulationContext-already-exists. Trap kills any
# Isaac-Sim-related descendants on exit.
cleanup_isaac_orphans() {
    local sig="${1:-EXIT}"
    echo "[wm-isaac] cleanup_isaac_orphans on $sig" >&2
    # Soft first, then hard. Targets only Isaac-specific processes.
    pkill -TERM -f "_wm_isaac_entry|lerobot_isaac_autoresearch.train_wrapper|lerobot_isaac_adapters.train" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "_wm_isaac_entry|lerobot_isaac_autoresearch.train_wrapper|lerobot_isaac_adapters.train" 2>/dev/null || true
    pkill -KILL -f "kit\\.app|carb\\.app" 2>/dev/null || true
    sleep 1
}
trap 'cleanup_isaac_orphans EXIT' EXIT
trap 'cleanup_isaac_orphans TERM; exit 143' TERM
trap 'cleanup_isaac_orphans INT;  exit 130' INT

# Run training subprocess in background so a watchdog can stream-monitor
# train.log for early-fatal patterns (Traceback / KeyError / "crashed too
# many times") and kill the subprocess as soon as failure is certain —
# instead of waiting the full $SECONDS_PER_EXP timeout. Saves hours of
# GPU on doomed trials.
TRAIN_LOG="$AR_DIR/train.log"
: > "$TRAIN_LOG"

PYTHONUNBUFFERED=1 "${CMD[@]}" > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "[wm-isaac] training PID=$TRAIN_PID (watchdog active)"

# Watchdog: ONLY two checks per trial, at T+60s and T+300s.
# - fatal: Traceback OR RuntimeError OR KeyError OR "crashed too many" in train.log → kill.
# - log-frozen: train.log mtime stale >120s at check time → kill (subprocess hung silently).
# After T+300s the subprocess is assumed past boot/fail-fast window and runs unattended
# under the outer `timeout $SECONDS_PER_EXP` cap.
FATAL_REGEX='Traceback|RuntimeError|KeyError|crashed too many'

watchdog_check() {
    local label="$1"; local stale_limit="$2"
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[wm-isaac] WATCHDOG[$label]: subprocess already exited" >&2
        return 1
    fi
    if grep -qE "$FATAL_REGEX" "$TRAIN_LOG" 2>/dev/null; then
        echo "[wm-isaac] WATCHDOG[$label]: fatal pattern in train.log → killing PID=$TRAIN_PID" >&2
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        sleep 5
        kill -KILL "$TRAIN_PID" 2>/dev/null || true
        watchdog_done=1
        return 1
    fi
    if [ -s "$TRAIN_LOG" ]; then
        local mtime_age=$(( $(date +%s) - $(stat -c%Y "$TRAIN_LOG") ))
        if [ "$mtime_age" -gt "$stale_limit" ]; then
            echo "[wm-isaac] WATCHDOG[$label]: train.log frozen ${mtime_age}s (>$stale_limit) → killing PID=$TRAIN_PID" >&2
            kill -TERM "$TRAIN_PID" 2>/dev/null || true
            sleep 5
            kill -KILL "$TRAIN_PID" 2>/dev/null || true
            watchdog_done=1
            return 1
        fi
    fi
    echo "[wm-isaac] WATCHDOG[$label]: OK" >&2
    return 0
}

watchdog_done=0
# T+60s check. Sleep + train_pid raced via `wait -n`; whichever ends first wakes us.
sleep 60 &
SLEEP_PID=$!
wait -n "$SLEEP_PID" "$TRAIN_PID" 2>/dev/null || true
if kill -0 "$TRAIN_PID" 2>/dev/null; then
    # Train still alive → sleep elapsed → run check.
    watchdog_check "T+60s" 60 || true
else
    kill -KILL "$SLEEP_PID" 2>/dev/null || true
fi

# T+300s check (only if subprocess + watchdog still relevant).
if kill -0 "$TRAIN_PID" 2>/dev/null && [ "$watchdog_done" = "0" ]; then
    sleep 240 &
    SLEEP_PID=$!
    wait -n "$SLEEP_PID" "$TRAIN_PID" 2>/dev/null || true
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
        watchdog_check "T+300s" 120 || true
    else
        kill -KILL "$SLEEP_PID" 2>/dev/null || true
    fi
fi

# Stage 3: background TB-scrape collapse watcher.
# Polls every 5 min from now (T+300s) onward. Trips when BOTH:
#   - last 3 Grads/actor samples all below GRADS_THRESHOLD (dead actor)
#   - last 3 Rewards/rew_avg samples within REW_FLAT_RANGE (stuck flat)
#   - latest policy_step >= MIN_STEP (don't kill before training has run long enough)
# On trip: kills TRAIN_PID + touches sentinel; main shell flags rc=137.
COLLAPSE_SENTINEL="$AR_DIR/.collapse-killed"
COLLAPSE_INTERVAL="${COLLAPSE_INTERVAL:-300}"
COLLAPSE_MIN_STEP="${COLLAPSE_MIN_STEP:-15000}"
COLLAPSE_GRADS_THRESHOLD="${COLLAPSE_GRADS_THRESHOLD:-0.001}"
COLLAPSE_REW_FLAT_RANGE="${COLLAPSE_REW_FLAT_RANGE:-0.02}"

# Scraper script (writes one line "step|rew|grads" or "skip|<reason>" to stdout).
TB_SCRAPE_PY="$AR_DIR/.tb_scrape.py"
cat > "$TB_SCRAPE_PY" <<'PY'
import glob, os, sys
try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception as e:
    print(f"skip|tb_import:{e}"); sys.exit(0)
runs = sorted(glob.glob("logs/runs/dreamer_v3/*/*/version_0"), key=os.path.getmtime, reverse=True)
if not runs:
    print("skip|no_run"); sys.exit(0)
ev = list(glob.glob(f"{runs[0]}/events.out.tfevents.*"))
if not ev:
    print("skip|no_events"); sys.exit(0)
ea = event_accumulator.EventAccumulator(runs[0], size_guidance={event_accumulator.SCALARS: 0})
ea.Reload()
tags = ea.Tags().get("scalars", [])
def last(tag):
    if tag in tags:
        s = ea.Scalars(tag)
        if s:
            return s[-1]
    return None
rew = last("Rewards/rew_avg")
grads = last("Grads/actor")
if rew is None or grads is None:
    print("skip|no_scalars"); sys.exit(0)
print(f"{rew.step}|{rew.value}|{grads.value}")
PY

(
    # Per-sample history kept as space-separated triples "step:rew:grads".
    samples=()
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        # Sleep with race against TRAIN_PID — exit early if subprocess dies.
        sleep "$COLLAPSE_INTERVAL" &
        SP=$!
        wait -n "$SP" "$TRAIN_PID" 2>/dev/null || true
        kill -KILL "$SP" 2>/dev/null || true
        if ! kill -0 "$TRAIN_PID" 2>/dev/null; then break; fi

        LINE=$("$PY" "$TB_SCRAPE_PY" 2>/dev/null)
        case "$LINE" in
            skip\|*)
                echo "[wm-isaac] COLLAPSE_WATCH: $LINE" >&2
                continue
                ;;
        esac
        STEP=$(echo "$LINE" | cut -d'|' -f1)
        REW=$(echo "$LINE" | cut -d'|' -f2)
        GRADS=$(echo "$LINE" | cut -d'|' -f3)
        samples+=("$STEP:$REW:$GRADS")
        # Trim history to last 3.
        if [ "${#samples[@]}" -gt 3 ]; then
            samples=("${samples[@]: -3}")
        fi
        echo "[wm-isaac] COLLAPSE_WATCH: step=$STEP rew=$REW grads=$GRADS (history n=${#samples[@]})" >&2

        # Need 3 samples + min step before tripping.
        [ "${#samples[@]}" -lt 3 ] && continue
        STEP_OK=$("$PY" -c "exit(0 if int(float('$STEP')) >= $COLLAPSE_MIN_STEP else 1)" && echo 1 || echo 0)
        [ "$STEP_OK" != "1" ] && continue

        # Check rule via python (float math).
        TRIP=$("$PY" - <<PY
samples = """${samples[0]}
${samples[1]}
${samples[2]}"""
grads = [float(s.split(":")[2]) for s in samples.strip().splitlines()]
rews  = [float(s.split(":")[1]) for s in samples.strip().splitlines()]
grads_all_dead = all(abs(g) < $COLLAPSE_GRADS_THRESHOLD for g in grads)
rew_range = max(rews) - min(rews)
rew_flat = rew_range < $COLLAPSE_REW_FLAT_RANGE
print("trip" if (grads_all_dead and rew_flat) else "ok")
PY
)
        if [ "$TRIP" = "trip" ]; then
            echo "[wm-isaac] COLLAPSE_WATCH: TRIP — grads<$COLLAPSE_GRADS_THRESHOLD for 3 samples + rew range<$COLLAPSE_REW_FLAT_RANGE → killing PID=$TRAIN_PID" >&2
            touch "$COLLAPSE_SENTINEL"
            kill -TERM "$TRAIN_PID" 2>/dev/null || true
            sleep 5
            kill -KILL "$TRAIN_PID" 2>/dev/null || true
            break
        fi
    done
) &
COLLAPSE_WATCHER_PID=$!

wait "$TRAIN_PID" 2>/dev/null
rc=$?
kill -KILL "$COLLAPSE_WATCHER_PID" 2>/dev/null || true
wait "$COLLAPSE_WATCHER_PID" 2>/dev/null || true
[ "$watchdog_done" = "1" ] && rc=137  # mark as watchdog-killed
[ -f "$COLLAPSE_SENTINEL" ] && rc=137 && echo "[wm-isaac] COLLAPSE_WATCH: trial killed on collapse rule"
echo "[wm-isaac] done rc=$rc"
echo "[wm-isaac] state: $AR_DIR"
echo "[wm-isaac] ckpts: logs/runs/dreamer_v3/isaac_so101/"
ls -la "$AR_DIR" 2>/dev/null
