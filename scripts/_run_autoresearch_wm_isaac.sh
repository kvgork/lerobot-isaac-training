#!/usr/bin/env bash
# =============================================================================
# _run_autoresearch_wm_isaac.sh — DreamerV3 + Isaac Lab HP sweep.
#
# Implements plans/2026-05-24-wm-isaac-hp-trials-1to9.md. Sweeps across
# 8 pre-encoded trials covering reward shape (sparse / hybrid / hybrid-small),
# actor entropy, init_std, min_std, replay_ratio, and max_episode_steps axes.
# Persists state to the autoresearch on-disk schema (history.jsonl, best.json,
# plateau.json) so the dashboard auto-discovers.
#
# Per-trial pipeline:
#   1. Build EXTRA_HYDRA from the trial pool entry.
#   2. Set LEROBOT_ISAAC_* env vars for the env-side knobs.
#   3. Call scripts/_run_wm_isaac_overnight.sh with per-trial knobs.
#   4. After completion, scrape Rewards/rew_avg from TensorBoard via
#      scripts/_scrape_tb_to_history.py.
#   5. Append row to history.jsonl + ratchet best.json (maximize).
#   6. Plateau-stop after PLATEAU_LIMIT consecutive non-improvers.
#
# Knobs (env-overridable):
#   SESSION_ID=wm-isaac-hp-<ts>
#   MAX_TRIALS=8
#   SKIP_TRIALS=0
#   STEPS_PER_TRIAL=80000          # env steps per trial (sparser reward → more steps)
#   SECONDS_PER_EXP=10800          # 3 h ceiling per trial
#   PLATEAU_LIMIT=4
#   RESUME_BEST_METRIC=""          # seed ratchet from prior sweep best
#   DRY_RUN=0
#   SKIP_DRYRUN_GATE=0             # set 1 to bypass pre-flight dry-run check
#
# Trial pool format: ENT|INIT_STD|MIN_STD|RR|MAX_EP|REWARD_SHAPE|ALGO|LABEL
#   ENT          — algo.actor.ent_coef (Hydra)
#   INIT_STD     — algo.actor.init_std (Hydra)
#   MIN_STD      — algo.actor.min_std (Hydra)
#   RR           — replay_ratio (env var forwarded to sheeprl)
#   MAX_EP       — env.max_episode_steps in env steps. Units: env steps at 30 Hz.
#                  max_ep=600 → 20 s per episode (prior default).
#                  max_ep=300 → 10 s per episode (trial 3 "short episodes").
#                  Plan table used seconds (max_ep=100 ≈ 600 steps, max_ep=50 ≈ 300
#                  steps) — we store env steps here for direct forwarding.
#   REWARD_SHAPE — sparse | hybrid | hybrid-small
#                  sparse       → LEROBOT_ISAAC_PROGRESS_WEIGHT=0 (terminal only)
#                  hybrid       → LEROBOT_ISAAC_PROGRESS_WEIGHT=1.0
#                  hybrid-small → LEROBOT_ISAAC_PROGRESS_WEIGHT=1.0 (same weight,
#                                 different ent/std config)
#   ALGO         — dreamer_v3 | ppo
#                  ppo is DEFERRED — logged as WARN + skipped (target_arch=ppo
#                  not implemented in lerobot_isaac_adapters.train).
#   LABEL        — human-readable tag; drives special per-trial logic
#                  (e.g. *object-at-home* → moves source_object spawn pos).
#
# Total budget: 8 trials × 3 h ≈ 24 h. Run as multi-day overnight.
# =============================================================================
set -uo pipefail

WORKSPACE="${WORKSPACE:-${LEROBOT_ISAAC_WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$WORKSPACE"

SESSION_ID="${SESSION_ID:-wm-isaac-hp-$(date +%Y%m%d-%H%M%S)}"
SLUG="wm-isaac-hp"
MAX_TRIALS="${MAX_TRIALS:-8}"
SKIP_TRIALS="${SKIP_TRIALS:-0}"
STEPS_PER_TRIAL="${STEPS_PER_TRIAL:-80000}"
SECONDS_PER_EXP="${SECONDS_PER_EXP:-10800}"
PLATEAU_LIMIT="${PLATEAU_LIMIT:-4}"
RESUME_BEST_METRIC="${RESUME_BEST_METRIC:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DRYRUN_GATE="${SKIP_DRYRUN_GATE:-0}"

PY="$WORKSPACE/.pixi/envs/sim/bin/python"
AR_DIR="$WORKSPACE/.agent-state/$SESSION_ID/autoresearch/$SLUG"
HISTORY="$AR_DIR/history.jsonl"
BEST="$AR_DIR/best.json"
PLATEAU="$AR_DIR/plateau.json"
PROGRAM="$AR_DIR/program.json"

mkdir -p "$AR_DIR"

# --- trial pool (8 configs) -------------------------------------------------
# Format: ENT|INIT_STD|MIN_STD|RR|MAX_EP|REWARD_SHAPE|ALGO|LABEL
# MAX_EP is in env steps (30 Hz). Plan "max_ep=100" ≈ 600 steps (20 s);
# plan "max_ep=50" ≈ 300 steps (10 s). ALGO=ppo is deferred (logged + skipped).
declare -a TRIAL_POOL=(
    "1e-2|2.0|0.3|0.5|600|sparse|dreamer_v3|t1-sparse-default"
    "3e-2|4.0|0.5|0.5|600|sparse|dreamer_v3|t2-sparse-high-init-std"
    "1e-2|2.0|0.3|0.5|300|sparse|dreamer_v3|t3-sparse-short-ep"
    "1e-2|2.0|0.3|0.1|600|sparse|dreamer_v3|t4-sparse-low-rr"
    "1e-2|2.0|0.3|0.5|600|hybrid|dreamer_v3|t5-hybrid-mid-progress"
    "1e-2|2.0|0.3|0.5|600|hybrid|dreamer_v3|t6-object-at-home"
    "n/a|n/a|n/a|n/a|600|sparse|ppo|t7-ppo-sparse-DEFERRED"
    "1e-1|4.0|0.5|0.5|600|hybrid-small|dreamer_v3|t8-extreme-entropy"
)
TOTAL_POOL=${#TRIAL_POOL[@]}
N=$(( MAX_TRIALS < TOTAL_POOL ? MAX_TRIALS : TOTAL_POOL ))

# --- program snapshot -------------------------------------------------------
cat > "$PROGRAM" <<EOF
{
  "name": "$SLUG",
  "metric": {"name": "rew_avg", "direction": "maximize", "tag": "Rewards/rew_avg"},
  "budget": {
    "seconds_per_experiment": $SECONDS_PER_EXP,
    "max_experiments": $N,
    "plateau_limit": $PLATEAU_LIMIT
  },
  "target_arch": "dreamerv3",
  "env": "isaac_so101",
  "iterations": $N,
  "session_id": "$SESSION_ID",
  "ts_start": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "[wm-hp] session=$SESSION_ID slug=$SLUG"
echo "[wm-hp] trials=$N skip=$SKIP_TRIALS timeout=${SECONDS_PER_EXP}s steps_per_trial=$STEPS_PER_TRIAL"
echo "[wm-hp] state_dir=$AR_DIR"

if [ "$DRY_RUN" != "1" ] && [ "$SKIP_TRIALS" = "0" ]; then
    : > "$HISTORY"
fi

best_metric=""
plateau_count=0
if [ -n "$RESUME_BEST_METRIC" ]; then
    best_metric="$RESUME_BEST_METRIC"
    echo "[wm-hp] resuming with seeded best_metric=$best_metric"
fi

# Maximize ratchet helper.
is_better() {
    local cand="$1"; local incum="$2"
    [ -z "$incum" ] && return 0
    "$PY" -c "exit(0 if float('$cand') > float('$incum') else 1)"
}

# --- dry-run gate (pre-flight WARN remediation 2026-05-24) -------------------
# Verify the resolved per-trial subprocess command carries the expected
# Hydra overrides for at least one sample. Skip when explicit DRY_RUN=1.
if [ "$DRY_RUN" != "1" ] && [ "$SKIP_DRYRUN_GATE" != "1" ]; then
    echo "[wm-hp] running dry-run gate (sample: trial 0)..."
    GATE_OUT=$(DRY_RUN=1 \
        LEROBOT_ISAAC_PROGRESS_WEIGHT=0.0 \
        STEPS=80000 \
        REPLAY_RATIO=0.5 \
        MAX_EPISODE_STEPS=600 \
        EXTRA_HYDRA="algo.actor.ent_coef=1e-2 algo.actor.init_std=2.0 algo.actor.min_std=0.3 algo.world_model.optimizer.lr=1e-4" \
        bash "$WORKSPACE/scripts/_run_wm_isaac_overnight.sh" 2>&1) || true

    missing=()
    for needle in \
        "algo.actor.ent_coef=1e-2" \
        "algo.actor.init_std=2.0" \
        "algo.actor.min_std=0.3" \
        "algo.replay_ratio=0.5" \
        "env.max_episode_steps=600"
    do
        if ! grep -qF "$needle" <<< "$GATE_OUT"; then
            missing+=("$needle")
        fi
    done

    if [ "${#missing[@]}" -gt 0 ]; then
        echo "[wm-hp] DRY-RUN GATE FAILED — missing overrides:" >&2
        printf '  %s\n' "${missing[@]}" >&2
        echo "[wm-hp] aborting — fix arg forwarding before launching 24h compute" >&2
        echo "[wm-hp] gate output:" >&2
        printf '%s\n' "$GATE_OUT" | head -50 >&2
        exit 3
    fi
    echo "[wm-hp] dry-run gate PASSED — all expected overrides reach sheeprl"
fi

# --- main loop --------------------------------------------------------------
for i in $(seq "$SKIP_TRIALS" $(( N - 1 ))); do
    IFS='|' read -r ENT_COEF INIT_STD MIN_STD REPLAY_RATIO MAX_EP REWARD_SHAPE ALGO LABEL <<< "${TRIAL_POOL[$i]}"

    echo
    echo "[wm-hp] trial=$i [$LABEL] ent=$ENT_COEF init_std=$INIT_STD min_std=$MIN_STD rr=$REPLAY_RATIO max_ep=$MAX_EP reward=$REWARD_SHAPE algo=$ALGO"

    # ALGO gate (defer PPO until target_arch=ppo is implemented in adapters).
    if [ "$ALGO" = "ppo" ]; then
        echo "[wm-hp] trial=$i [$LABEL] ALGO=ppo → DEFERRED (target_arch=ppo not implemented in lerobot_isaac_adapters.train)"
        echo "[wm-hp]   skipping; will be implemented in a follow-up sweep"
        continue
    fi

    # Build EXTRA_HYDRA for sheeprl. Only DreamerV3 knobs here.
    EXTRA_TOKENS=(
        "algo.actor.ent_coef=$ENT_COEF"
        "algo.actor.init_std=$INIT_STD"
        "algo.actor.min_std=$MIN_STD"
        "algo.world_model.optimizer.lr=1e-4"
    )

    # Env-side reward shaping via LEROBOT_ISAAC_PROGRESS_WEIGHT
    # (read by tasks/pick_and_place.py at module load).
    case "$REWARD_SHAPE" in
        sparse)       PROGRESS_W="0.0" ;;
        hybrid)       PROGRESS_W="1.0" ;;
        hybrid-small) PROGRESS_W="1.0" ;;
        *)            echo "[wm-hp] ERROR: unknown REWARD_SHAPE=$REWARD_SHAPE"; exit 2 ;;
    esac

    # Object-at-home curriculum (trial 6 label).
    case "$LABEL" in
        *object-at-home*)
            export LEROBOT_ISAAC_OBJECT_X="0.30"
            export LEROBOT_ISAAC_OBJECT_Y="0.05"
            export LEROBOT_ISAAC_OBJECT_Z="0.05"
            ;;
        *)
            unset LEROBOT_ISAAC_OBJECT_X LEROBOT_ISAAC_OBJECT_Y LEROBOT_ISAAC_OBJECT_Z
            ;;
    esac

    # Compose EXTRA_HYDRA string.
    EXTRA_HYDRA_STR="${EXTRA_TOKENS[*]}"

    trial_session="${SESSION_ID}-trial${i}"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  ENV: LEROBOT_ISAAC_PROGRESS_WEIGHT=$PROGRESS_W LEROBOT_ISAAC_OBJECT_X=${LEROBOT_ISAAC_OBJECT_X:-unset}"
        echo "  EXTRA_HYDRA=$EXTRA_HYDRA_STR"
        continue
    fi

    start_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    start_s=$(date +%s)

    # The single-trial runner already wires AppLauncher-first + sync_env;
    # we layer per-trial HP overrides via EXTRA_HYDRA + the existing
    # replay_ratio / precision / batch knobs.
    LEROBOT_ISAAC_PROGRESS_WEIGHT="$PROGRESS_W" \
    SESSION_ID="$trial_session" \
    STEPS="$STEPS_PER_TRIAL" \
    SECONDS_PER_EXP="$SECONDS_PER_EXP" \
    CHECKPOINT_EVERY="$(( STEPS_PER_TRIAL / 4 ))" \
    NUM_ENVS=1 \
    BATCH_SIZE=16 \
    REPLAY_RATIO="$REPLAY_RATIO" \
    PRECISION=bf16-mixed \
    MAX_EPISODE_STEPS="$MAX_EP" \
    LEROBOT_TRAIN_TIMEOUT="$SECONDS_PER_EXP" \
    EXTRA_HYDRA="$EXTRA_HYDRA_STR" \
        bash "$WORKSPACE/scripts/_run_wm_isaac_overnight.sh" > "$AR_DIR/trial_${i}_${LABEL}.log" 2>&1
    rc=$?
    dur=$(( $(date +%s) - start_s ))
    end_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    status="ok"
    [ "$rc" -eq 124 ] && status="timeout"
    [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && status="error"

    # --- TB scrape ----------------------------------------------------------
    # Locate this trial's sheeprl run dir (timestamped by trial start).
    # Pattern: logs/runs/dreamer_v3/isaac_so101/<YYYY-MM-DD_HH-MM-SS>_dreamer_v3_isaac_so101_<seed>
    # The newest matching dir created after start_s wins.
    TRIAL_RUN_DIR=$(find "$WORKSPACE/logs/runs/dreamer_v3/isaac_so101/" -maxdepth 1 -type d \
        -newer "$AR_DIR/trial_${i}_${LABEL}.log" 2>/dev/null | head -1)
    if [ -z "$TRIAL_RUN_DIR" ]; then
        # Fallback: the most recent dir, period.
        TRIAL_RUN_DIR=$(ls -dt "$WORKSPACE/logs/runs/dreamer_v3/isaac_so101/"*/ 2>/dev/null | head -1)
    fi

    # Scrape primary + secondary metrics.
    metric=""
    grads_actor=""
    post_entropy=""
    obs_loss=""
    policy_loss=""
    if [ -n "$TRIAL_RUN_DIR" ]; then
        readarray -t SCRAPE < <("$PY" - <<PY
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
run = "$TRIAL_RUN_DIR/version_0"
ev = next(Path(run).glob("**/events.out.tfevents.*"), None)
if ev is None:
    print(""); print(""); print(""); print(""); print("")
else:
    ea = event_accumulator.EventAccumulator(str(ev.parent), size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    for tag in ["Rewards/rew_avg", "Grads/actor", "State/post_entropy", "Loss/observation_loss", "Loss/policy_loss"]:
        if tag in tags and ea.Scalars(tag):
            print(ea.Scalars(tag)[-1].value)
        else:
            print("")
PY
        )
        metric="${SCRAPE[0]:-}"
        grads_actor="${SCRAPE[1]:-}"
        post_entropy="${SCRAPE[2]:-}"
        obs_loss="${SCRAPE[3]:-}"
        policy_loss="${SCRAPE[4]:-}"
    fi

    # Sentinel if scrape failed.
    [ -z "$metric" ] && metric="-9999"

    echo "[wm-hp] trial=$i [$LABEL] metric=$metric grads_actor=$grads_actor post_entropy=$post_entropy obs_loss=$obs_loss status=$status dur=${dur}s"

    # Forensic filter: actor must have non-collapsed grads to count as winner.
    actor_alive=$("$PY" -c "v='$grads_actor'; print(int(bool(v) and float(v) >= 0.05))" 2>/dev/null)
    [ -z "$actor_alive" ] && actor_alive="0"

    # Append history row.
    "$PY" - <<PY >> "$HISTORY"
import json
print(json.dumps({
    "trial_index": $i, "trial": $i,
    "label": "$LABEL",
    "metric_name": "rew_avg",
    "metric_value": float("$metric"),
    "metric_kind": "tb_rewards_rew_avg",
    "config": {
        "ent_coef": float("$ENT_COEF"),
        "init_std": float("$INIT_STD"),
        "min_std": float("$MIN_STD"),
        "replay_ratio": float("$REPLAY_RATIO"),
        "max_ep": int("$MAX_EP"),
        "reward_shape": "$REWARD_SHAPE",
        "algo": "$ALGO",
        "steps": $STEPS_PER_TRIAL
    },
    "forensics": {
        "grads_actor": float("$grads_actor" or 0.0),
        "post_entropy": float("$post_entropy" or 0.0),
        "obs_loss": float("$obs_loss" or 0.0),
        "policy_loss": float("$policy_loss" or 0.0),
        "actor_alive": bool($actor_alive)
    },
    "ts": "$start_ts", "end_ts": "$end_ts",
    "duration_s": $dur,
    "status": "$status",
    "exit_code": $rc,
    "trial_run_dir": "$TRIAL_RUN_DIR"
}))
PY

    # Ratchet best (maximize). Only count "alive actor" trials.
    if [ "$actor_alive" = "1" ] && is_better "$metric" "$best_metric"; then
        best_metric="$metric"
        "$PY" - <<PY > "$BEST"
import json
print(json.dumps({
    "trial": $i, "label": "$LABEL",
    "metric_value": float("$metric"),
    "metric_kind": "tb_rewards_rew_avg",
    "config": {
        "ent_coef": float("$ENT_COEF"),
        "init_std": float("$INIT_STD"),
        "min_std": float("$MIN_STD"),
        "replay_ratio": float("$REPLAY_RATIO"),
        "max_ep": int("$MAX_EP"),
        "reward_shape": "$REWARD_SHAPE",
        "algo": "$ALGO",
        "steps": $STEPS_PER_TRIAL
    },
    "forensics": {
        "grads_actor": float("$grads_actor" or 0.0),
        "post_entropy": float("$post_entropy" or 0.0)
    }
}, indent=2))
PY
        plateau_count=0
        echo "[wm-hp] NEW BEST: trial=$i metric=$metric"
    else
        plateau_count=$(( plateau_count + 1 ))
    fi

    "$PY" - <<PY > "$PLATEAU"
import json
print(json.dumps({
    "consecutive_non_improvements": $plateau_count,
    "plateau_limit": $PLATEAU_LIMIT,
    "last_metric": float("$metric"),
    "best_metric": float("$best_metric" or -9999.0),
    "completed_trials": $(( i + 1 )),
    "planned_trials": $N
}, indent=2))
PY

    if [ "$plateau_count" -ge "$PLATEAU_LIMIT" ]; then
        echo "[wm-hp] plateau_limit=$PLATEAU_LIMIT reached at trial=$i — stopping early"
        break
    fi
done

echo
echo "[wm-hp] done — best metric: $best_metric"
echo "[wm-hp] state: $AR_DIR"
ls -la "$AR_DIR"
