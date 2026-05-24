#!/usr/bin/env bash
# =============================================================================
# _run_autoresearch_wm_isaac.sh — DreamerV3 + Isaac Lab HP sweep.
#
# Implements plans/2026-05-23-wm-isaac-autoresearch-plan.md. Sweeps 4
# knobs (actor.ent_coef, replay_ratio, actor.min_std, world_model.optimizer.lr)
# across 10 pre-encoded trials, persists state to the autoresearch
# on-disk schema (history.jsonl, best.json, plateau.json) so the
# dashboard auto-discovers.
#
# Per-trial pipeline:
#   1. Build EXTRA_HYDRA from the trial pool entry.
#   2. Call scripts/_run_wm_isaac_overnight.sh with per-trial knobs.
#   3. After completion, scrape Rewards/rew_avg from TensorBoard via
#      scripts/_scrape_tb_to_history.py.
#   4. Append row to history.jsonl + ratchet best.json (maximize).
#   5. Plateau-stop after PLATEAU_LIMIT consecutive non-improvers.
#
# Knobs (env-overridable):
#   SESSION_ID=wm-isaac-hp-<ts>
#   MAX_TRIALS=10
#   SKIP_TRIALS=0
#   SECONDS_PER_EXP=10800      # 3 h ceiling per trial
#   PLATEAU_LIMIT=4
#   RESUME_BEST_METRIC=""      # seed ratchet from prior sweep best
#   DRY_RUN=0
#
# Total budget: 10 trials × 3 h = 30 h. Run as multi-day overnight.
# =============================================================================
set -uo pipefail

WORKSPACE="${WORKSPACE:-${LEROBOT_ISAAC_WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$WORKSPACE"

SESSION_ID="${SESSION_ID:-wm-isaac-hp-$(date +%Y%m%d-%H%M%S)}"
SLUG="wm-isaac-hp"
MAX_TRIALS="${MAX_TRIALS:-10}"
SKIP_TRIALS="${SKIP_TRIALS:-0}"
SECONDS_PER_EXP="${SECONDS_PER_EXP:-10800}"
PLATEAU_LIMIT="${PLATEAU_LIMIT:-4}"
RESUME_BEST_METRIC="${RESUME_BEST_METRIC:-}"
DRY_RUN="${DRY_RUN:-0}"

PY="$WORKSPACE/.pixi/envs/sim/bin/python"
AR_DIR="$WORKSPACE/.agent-state/$SESSION_ID/autoresearch/$SLUG"
HISTORY="$AR_DIR/history.jsonl"
BEST="$AR_DIR/best.json"
PLATEAU="$AR_DIR/plateau.json"
PROGRAM="$AR_DIR/program.json"

mkdir -p "$AR_DIR"

# --- trial pool (10 configs) ------------------------------------------------
# Format: ENT_COEF|REPLAY_RATIO|MIN_STD|WM_LR|STEPS|LABEL
declare -a TRIAL_POOL=(
    "3e-4|0.5|0.1|1e-4|60000|baseline"
    "1e-2|0.5|0.1|1e-4|60000|high-entropy"
    "3e-3|0.5|0.3|1e-4|60000|mid-ent-min_std"
    "1e-2|1.0|0.3|1e-4|60000|v8-config"
    "3e-4|0.25|0.1|1e-4|60000|low-replay"
    "3e-4|2.0|0.1|1e-4|30000|high-replay"
    "1e-2|0.5|0.5|1e-4|60000|high-ent-high-min_std"
    "1e-2|0.5|0.3|3e-5|60000|low-wm-lr"
    "1e-2|0.5|0.3|3e-4|60000|high-wm-lr"
    "1e-2|0.5|0.1|1e-4|60000|ablate-min_std"
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
echo "[wm-hp] trials=$N skip=$SKIP_TRIALS timeout=${SECONDS_PER_EXP}s"
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

# --- main loop --------------------------------------------------------------
for i in $(seq "$SKIP_TRIALS" $(( N - 1 ))); do
    IFS='|' read -r ENT_COEF REPLAY_RATIO MIN_STD WM_LR STEPS LABEL <<< "${TRIAL_POOL[$i]}"

    trial_session="${SESSION_ID}-trial${i}"

    echo
    echo "[wm-hp] trial=$i [$LABEL] ent=$ENT_COEF rr=$REPLAY_RATIO min_std=$MIN_STD wm_lr=$WM_LR steps=$STEPS"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  EXTRA_HYDRA=algo.actor.ent_coef=$ENT_COEF algo.actor.min_std=$MIN_STD algo.world_model.optimizer.lr=$WM_LR"
        continue
    fi

    start_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    start_s=$(date +%s)

    # The single-trial runner already wires AppLauncher-first + sync_env;
    # we layer per-trial HP overrides via EXTRA_HYDRA + the existing
    # replay_ratio / precision / batch knobs.
    SESSION_ID="$trial_session" \
    STEPS="$STEPS" \
    SECONDS_PER_EXP="$SECONDS_PER_EXP" \
    CHECKPOINT_EVERY="$(( STEPS / 4 ))" \
    NUM_ENVS=1 \
    BATCH_SIZE=16 \
    REPLAY_RATIO="$REPLAY_RATIO" \
    PRECISION=bf16-mixed \
    LEROBOT_TRAIN_TIMEOUT="$SECONDS_PER_EXP" \
    EXTRA_HYDRA="algo.actor.ent_coef=$ENT_COEF algo.actor.min_std=$MIN_STD algo.world_model.optimizer.lr=$WM_LR" \
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
        "replay_ratio": float("$REPLAY_RATIO"),
        "min_std": float("$MIN_STD"),
        "wm_lr": float("$WM_LR"),
        "steps": $STEPS
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
        "replay_ratio": float("$REPLAY_RATIO"),
        "min_std": float("$MIN_STD"),
        "wm_lr": float("$WM_LR"),
        "steps": $STEPS
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
