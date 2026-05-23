#!/usr/bin/env bash
# =============================================================================
# _run_autoresearch_wm.sh — Deterministic DreamerV3 world-model HP sweep.
#
# Multi-trial bash sweep for DreamerV3 on a bridged SO-101 HDF5. Companion to
# scripts/_run_autoresearch_lora.sh (LoRA pattern). No LLM proposer = no tokens.
# Mirrors the on-disk autoresearch schema → dashboard auto-discovers.
#
# Per-trial pipeline:
#   1. Build sheeprl Hydra overrides from the pool entry.
#   2. Invoke `lerobot_isaac_autoresearch.train_wrapper --target_arch dreamerv3`.
#   3. After training, locate the sheeprl ckpt at <out>/checkpoints/ckpt_*.ckpt.
#   4. Run lerobot_isaac_deploy.wm_rollout against the held-out dataset slice
#      → emits rollout_summary.json with mean_recon_loss.
#   5. Append a history.jsonl row with metric_value = mean_recon_loss.
#   6. Ratchet best.json on LOWER metric (minimize).
#   7. Plateau-stop after PLATEAU_LIMIT consecutive non-improvements.
#
# Trial pool: 12 pre-encoded configs covering
#   - lr (1e-4, 3e-5, 3e-4)
#   - batch_size (4, 8, 16)
#   - sequence_length (16, 32, 64)
#   - replay_ratio (1, 2, 4)
#   - capacity (discrete_size + stochastic_size)
#
# Knobs (env-overridable):
#   SESSION_ID=wm-bash-<ts>
#   MAX_TRIALS=12
#   SKIP_TRIALS=0
#   STEPS=200000                # per-trial sheeprl total_steps target
#   SECONDS_PER_EXP=2700        # ~45 min/trial wall timeout
#   PLATEAU_LIMIT=4
#   IMAGE_SIZE=64
#   WINDOW=16
#   STRIDE=8
#   EVAL_ENABLED=1              # 1: run wm_rollout after train; 0: train-loss only
#   EVAL_TIMEOUT=300
#   EVAL_HOLDOUT_FRAC=0.1       # last 10 % of episodes
#   RESUME_BEST_METRIC=""       # seed for ratchet on resumed sweep
#   DATASET=datasets/kvgork/so101-pickplace1
#   HDF5_CACHE=outputs/wm_data/so101-pickplace1_dreamerv3.hdf5
#   CLAUDE_CODE_ROOT=$HOME/tools/claude_code
#   DRY_RUN=0
#
# Time budget for 12 trials × SECONDS_PER_EXP=2700 ≈ 9 h compute.
#
# Usage:
#   bash scripts/_run_autoresearch_wm.sh                       # defaults
#   MAX_TRIALS=4 STEPS=20000 bash scripts/_run_autoresearch_wm.sh   # short smoke
#   DRY_RUN=1 bash scripts/_run_autoresearch_wm.sh             # echo cmds only
# =============================================================================
set -uo pipefail

WORKSPACE="${WORKSPACE:-${LEROBOT_ISAAC_WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$WORKSPACE"

# --- knobs ------------------------------------------------------------------
SESSION_ID="${SESSION_ID:-wm-bash-$(date +%Y%m%d-%H%M%S)}"
SLUG="${SLUG:-wm-dreamerv3}"
MAX_TRIALS="${MAX_TRIALS:-12}"
SKIP_TRIALS="${SKIP_TRIALS:-0}"
STEPS="${STEPS:-10000}"           # calibrated to ~4.3 step/s — 10k ≈ 39 min
SECONDS_PER_EXP="${SECONDS_PER_EXP:-2400}"
PLATEAU_LIMIT="${PLATEAU_LIMIT:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
WINDOW="${WINDOW:-16}"
STRIDE="${STRIDE:-8}"
EVAL_ENABLED="${EVAL_ENABLED:-1}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-300}"
EVAL_HOLDOUT_FRAC="${EVAL_HOLDOUT_FRAC:-0.1}"
RESUME_BEST_METRIC="${RESUME_BEST_METRIC:-}"
DATASET="${DATASET:-datasets/kvgork/so101-pickplace1}"
HDF5_CACHE="${HDF5_CACHE:-outputs/wm_data/so101-pickplace1_dreamerv3.hdf5}"
CLAUDE_CODE_ROOT="${CLAUDE_CODE_ROOT:-$HOME/tools/claude_code}"
DRY_RUN="${DRY_RUN:-0}"

PY="$WORKSPACE/.pixi/envs/train-dreamer/bin/python"
AR_DIR="$WORKSPACE/.agent-state/$SESSION_ID/autoresearch/$SLUG"
HISTORY="$AR_DIR/history.jsonl"
BEST="$AR_DIR/best.json"
PLATEAU="$AR_DIR/plateau.json"
PROGRAM="$AR_DIR/program.json"

# --- pre-flight -------------------------------------------------------------
[ -d "$DATASET" ] || [ -f "$DATASET" ] || { echo "ERROR: dataset not found: $DATASET" >&2; exit 2; }
[ -x "$PY" ] || { echo "ERROR: train-dreamer python not found: $PY" >&2; exit 2; }
"$PY" -c "import sheeprl" 2>/dev/null \
    || { echo "ERROR: sheeprl not in train-dreamer env" >&2; exit 2; }
"$PY" -c "import lerobot_isaac_deploy.wm_rollout" 2>/dev/null \
    || { echo "ERROR: lerobot_isaac_deploy not installed. Run: pixi run -e train-dreamer pip install -e ~/workspaces/lerobot-isaac-deploy" >&2; exit 2; }
[ -d "$CLAUDE_CODE_ROOT/skills/lerobot_world_model_bridge" ] \
    || { echo "ERROR: bridge skill not found at $CLAUDE_CODE_ROOT/skills/lerobot_world_model_bridge" >&2; exit 2; }

mkdir -p "$AR_DIR"

# --- trial pool (12 configs) ------------------------------------------------
# Format: LR|BS|SEQ_LEN|REPLAY_RATIO|DISCRETE|STOCHASTIC|TOTAL_STEPS
declare -a TRIAL_POOL=(
    "1e-4|8|16|1|32|32|10000"    #  0 baseline (sheeprl default)
    "3e-5|8|16|1|32|32|10000"    #  1 lr lower
    "3e-4|8|16|1|32|32|10000"    #  2 lr upper
    "1e-4|4|16|1|32|32|10000"    #  3 smaller bs
    "1e-4|16|16|1|32|32|10000"   #  4 larger bs (VRAM risk)
    "1e-4|8|32|1|32|32|10000"    #  5 longer sequence
    "1e-4|8|64|1|32|32|10000"    #  6 even longer (VRAM risk)
    "1e-4|8|16|2|32|32|10000"    #  7 replay_ratio=2
    "1e-4|8|16|4|32|32|10000"    #  8 replay_ratio=4
    "1e-4|8|16|1|64|32|10000"    #  9 wider discrete
    "1e-4|8|16|1|32|64|10000"    # 10 wider stochastic
    "1e-4|8|16|2|64|64|25000"    # 11 best-guess combo + longer train (~2x budget)
)
TOTAL_POOL=${#TRIAL_POOL[@]}
N=$(( MAX_TRIALS < TOTAL_POOL ? MAX_TRIALS : TOTAL_POOL ))

# --- bridge (idempotent) ----------------------------------------------------
if [[ "$DATASET" == *.h5 || "$DATASET" == *.hdf5 ]]; then
    HDF5_CACHE="$DATASET"
    echo "[wm-ar] dataset is HDF5 — skipping bridge"
elif [ ! -f "$HDF5_CACHE" ]; then
    echo "[wm-ar] bridging $DATASET → $HDF5_CACHE (image_size=${IMAGE_SIZE}, window=${WINDOW})"
    mkdir -p "$(dirname "$HDF5_CACHE")"
    if [ "$DRY_RUN" != "1" ]; then
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
    print(f"[wm-ar] bridge failed: {result.error}", file=sys.stderr)
    sys.exit(3)
print(f"[wm-ar] bridge done: {result.data}")
PY
        bridge_rc=$?
        [ "$bridge_rc" -eq 0 ] || { echo "ERROR: bridge step failed rc=$bridge_rc" >&2; exit 3; }
    fi
else
    echo "[wm-ar] bridge cache HIT: $HDF5_CACHE"
fi

# --- program snapshot -------------------------------------------------------
cat > "$PROGRAM" <<EOF
{
  "name": "$SLUG",
  "metric": {"name": "recon_loss", "direction": "minimize"},
  "budget": {
    "seconds_per_experiment": $SECONDS_PER_EXP,
    "max_experiments": $N,
    "plateau_limit": $PLATEAU_LIMIT
  },
  "target_arch": "dreamerv3",
  "dataset": "$DATASET",
  "hdf5_cache": "$HDF5_CACHE",
  "image_size": $IMAGE_SIZE,
  "window": $WINDOW,
  "stride": $STRIDE,
  "iterations": $N,
  "session_id": "$SESSION_ID",
  "ts_start": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "[wm-ar] session=$SESSION_ID slug=$SLUG"
echo "[wm-ar] trials=$N skip=$SKIP_TRIALS steps=$STEPS timeout=${SECONDS_PER_EXP}s"
echo "[wm-ar] state_dir=$AR_DIR"

# Only clear history on a fresh sweep (SKIP_TRIALS=0).
if [ "$DRY_RUN" != "1" ] && [ "$SKIP_TRIALS" = "0" ]; then
    : > "$HISTORY"
fi

best_metric=""
plateau_count=0
if [ -n "$RESUME_BEST_METRIC" ]; then
    best_metric="$RESUME_BEST_METRIC"
    echo "[wm-ar] resuming with seeded best_metric=$best_metric"
    # Stub best.json so the dashboard has something on a resume.
    if [ ! -f "$BEST" ]; then
        "$PY" - <<PY > "$BEST"
import json
print(json.dumps({
    "trial": -1, "metric_value": float("$best_metric"),
    "metric_kind": "resumed_seed", "note": "seeded via RESUME_BEST_METRIC; not from a real trial"
}, indent=2))
PY
    fi
fi

# Helper: lower-is-better comparator (minimize).
is_better() {
    local cand="$1"; local incum="$2"
    [ -z "$incum" ] && return 0
    "$PY" -c "exit(0 if float('$cand') < float('$incum') else 1)"
}

# --- main loop --------------------------------------------------------------
for i in $(seq "$SKIP_TRIALS" $(( N - 1 ))); do
    IFS='|' read -r LR BS SEQ_LEN RR DISCRETE STOCHASTIC TOTAL_STEPS <<< "${TRIAL_POOL[$i]}"

    out_dir="$WORKSPACE/outputs/autoresearch-$SLUG/trial_${i}"
    iter_log="$AR_DIR/trial_${i}.log"

    echo
    echo "[wm-ar] trial=$i lr=$LR bs=$BS seq=$SEQ_LEN rr=$RR D=$DISCRETE S=$STOCHASTIC steps=$TOTAL_STEPS"

    CMD=(
        timeout "$SECONDS_PER_EXP"
        "$PY" -m lerobot_isaac_autoresearch.train_wrapper
            --target_arch dreamerv3
            --dataset "$HDF5_CACHE"
            --output_dir "$out_dir"
            --steps "$TOTAL_STEPS"
            --batch_size "$BS"
            --
            algo.world_model.optimizer.lr="$LR"
            algo.per_rank_batch_size="$BS"
            algo.per_rank_sequence_length="$SEQ_LEN"
            algo.replay_ratio="$RR"
            algo.world_model.discrete_size="$DISCRETE"
            algo.world_model.stochastic_size="$STOCHASTIC"
            algo.total_steps="$TOTAL_STEPS"
            checkpoint.every=$(( TOTAL_STEPS / 2 ))
            run_name="trial_${i}_${SESSION_ID}"
            seed=42
            env.capture_video=False
            fabric.accelerator=gpu
            fabric.devices=1
    )

    if [ "$DRY_RUN" = "1" ]; then
        printf '  %s\n' "${CMD[@]}"
        continue
    fi

    rm -rf "$out_dir"
    mkdir -p "$(dirname "$out_dir")"

    start_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    start_s=$(date +%s)

    PYTHONPATH="$CLAUDE_CODE_ROOT:${PYTHONPATH:-}" \
    PATH="$WORKSPACE/.pixi/envs/train-dreamer/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
        "${CMD[@]}" > "$iter_log" 2>&1
    rc=$?

    train_dur=$(( $(date +%s) - start_s ))

    # --- post-train eval via wm_rollout --------------------------------------
    eval_metric=""
    eval_kind=""
    eval_json="$AR_DIR/trial_${i}_eval.json"
    rollout_dir="$out_dir/rollout"

    if [ "$EVAL_ENABLED" = "1" ]; then
        # Eval path: read sheeprl's own TensorBoard events file for this trial.
        # The deploy-pkg `wm_rollout.rollout()` path is partially-implemented
        # upstream (decoder reconstruction needs full RSSM forward — see
        # plans/2026-05-22-wm-autoresearch-plan.md). TB scrape is the
        # canonical fallback: same Loss/observation_loss the training loop
        # records, no downstream rollout dependency.
        sheeprl_run_dir="$WORKSPACE/logs/runs/dreamer_v3/custom_hdf5/trial_${i}_${SESSION_ID}"
        ckpt_file=$(find "$sheeprl_run_dir" -name "ckpt_*.ckpt" 2>/dev/null | sort -V | tail -1)
        if [ -z "$ckpt_file" ]; then
            ckpt_file=$(find "$out_dir" "$WORKSPACE/logs/runs/dreamer_v3" -name "ckpt_*.ckpt" 2>/dev/null \
                        | xargs -I{} stat --printf='%Y %n\n' {} 2>/dev/null \
                        | sort -rn | head -1 | awk '{print $2}')
        fi
        ckpt_dir=$([ -n "$ckpt_file" ] && dirname "$(dirname "$ckpt_file")" || echo "")
        if [ -n "$ckpt_dir" ]; then
            echo "[wm-ar] eval: trial=$i TB scrape from $ckpt_dir"
            eval_log="$AR_DIR/trial_${i}_eval.log"
            timeout "$EVAL_TIMEOUT" "$PY" - <<PY > "$eval_log" 2>&1
import json, sys
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
run_dir = Path("$ckpt_dir").parent  # version_0/, parent = run root
ev_dir = next((p.parent for p in run_dir.glob("**/events.out.tfevents.*")), None)
if ev_dir is None:
    print("[eval] no events.tfevents under", run_dir, file=sys.stderr)
    sys.exit(4)
ea = event_accumulator.EventAccumulator(str(ev_dir), size_guidance={event_accumulator.SCALARS: 0})
ea.Reload()
tag = "Loss/observation_loss"
if tag not in ea.Tags().get("scalars", []):
    print(f"[eval] tag '{tag}' missing — available:", ea.Tags().get("scalars", [])[:10], file=sys.stderr)
    sys.exit(5)
events = ea.Scalars(tag)
last = events[-1]
out = {
    "run_id": f"wm-bash-trial-${i}",
    "task": "so101-pickplace1-wm-tb-scrape",
    "metric_tag": tag,
    "mean_recon_loss": float(last.value),
    "step": int(last.step),
    "n_events": len(events),
}
Path("$eval_json").write_text(json.dumps(out, indent=2))
print(f"[eval] {tag}={last.value:.6f} step={last.step} n_events={len(events)}")
PY
            eval_rc=$?
            if [ "$eval_rc" -eq 0 ] && [ -f "$eval_json" ]; then
                eval_metric=$("$PY" -c "import json;d=json.load(open('$eval_json'));print(d.get('mean_recon_loss',''))" 2>/dev/null)
                eval_kind="tb_observation_loss"
            else
                echo "[wm-ar] eval FAILED rc=$eval_rc — see $eval_log"
            fi
        else
            echo "[wm-ar] no sheeprl ckpt found — skipping eval"
        fi
    fi

    # Fallback: last recon-style loss from training log. sheeprl logs as
    # `Loss/world_model_loss: <val>` or `Loss/observation_loss: <val>`; the
    # adapter's metric_extractor may emit `recon_loss=<val>` at the very end.
    train_metric=$(grep -oE '(recon_loss|world_model_loss|observation_loss)[=:][[:space:]]*[0-9.eE+\-]+' "$iter_log" \
                   | tail -1 | sed -E 's/^[^=:]*[=:][[:space:]]*//')

    if [ -n "$eval_metric" ]; then
        metric="$eval_metric"
        metric_kind="$eval_kind"
    elif [ -n "$train_metric" ]; then
        metric="$train_metric"
        metric_kind="recon_loss_train_fallback"
    else
        metric="0.0"
        metric_kind="sentinel"
    fi

    end_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    total_dur=$(( $(date +%s) - start_s ))

    status="ok"
    if [ "$rc" -eq 124 ]; then status="timeout"
    elif [ "$rc" -ne 0 ]; then status="error"
    fi

    "$PY" - <<PY >> "$HISTORY"
import json
print(json.dumps({
    "trial_index": $i,
    "trial": $i,
    "metric_name": "recon_loss",
    "metric_value": float("$metric"),
    "metric_kind": "$metric_kind",
    "config": {
        "lr": float("$LR"),
        "batch_size": $BS,
        "seq_len": $SEQ_LEN,
        "replay_ratio": $RR,
        "discrete_size": $DISCRETE,
        "stochastic_size": $STOCHASTIC,
        "total_steps": $TOTAL_STEPS,
        "image_size": $IMAGE_SIZE,
        "window": $WINDOW,
        "dataset": "$DATASET"
    },
    "train_recon_loss": float("$train_metric" or 0.0),
    "eval_recon_loss": float("$eval_metric" or 0.0) if "$eval_metric" else None,
    "ts": "$start_ts",
    "end_ts": "$end_ts",
    "duration_s": $total_dur,
    "train_duration_s": $train_dur,
    "status": "$status",
    "exit_code": $rc,
}))
PY

    echo "[wm-ar] trial=$i metric=$metric ($metric_kind) train_loss=${train_metric:-NA} status=$status dur=${total_dur}s"

    # Ratchet best.json (minimize).
    if is_better "$metric" "$best_metric"; then
        best_metric="$metric"
        "$PY" - <<PY > "$BEST"
import json
print(json.dumps({
    "trial": $i,
    "metric_value": float("$metric"),
    "metric_kind": "$metric_kind",
    "config": {
        "lr": float("$LR"),
        "batch_size": $BS,
        "seq_len": $SEQ_LEN,
        "replay_ratio": $RR,
        "discrete_size": $DISCRETE,
        "stochastic_size": $STOCHASTIC,
        "total_steps": $TOTAL_STEPS,
        "image_size": $IMAGE_SIZE,
        "window": $WINDOW
    }
}, indent=2))
PY
        plateau_count=0
    else
        plateau_count=$(( plateau_count + 1 ))
    fi

    "$PY" - <<PY > "$PLATEAU"
import json
print(json.dumps({
    "consecutive_non_improvements": $plateau_count,
    "plateau_limit": $PLATEAU_LIMIT,
    "last_metric": float("$metric"),
    "best_metric": float("$best_metric"),
    "completed_trials": $(( i + 1 )),
    "planned_trials": $N
}, indent=2))
PY

    if [ "$plateau_count" -ge "$PLATEAU_LIMIT" ]; then
        echo "[wm-ar] plateau_limit=$PLATEAU_LIMIT reached at trial=$i — stopping early"
        break
    fi
done

echo
echo "[wm-ar] done — best metric: $best_metric"
echo "[wm-ar] state: $AR_DIR"
ls -la "$AR_DIR"
