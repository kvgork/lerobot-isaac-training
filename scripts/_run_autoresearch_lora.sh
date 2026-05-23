#!/usr/bin/env bash
# =============================================================================
# _run_autoresearch_lora.sh — Deterministic LoRA rank sweep on SmolVLA.
#
# Purpose:
#   Bash-only autoresearch loop for the program
#   `programs/lerobot-policy-smolvla-lora.md`. No LLM proposer = no tokens.
#   Mirrors the on-disk artifact layout written by
#   `scripts/_run_autoresearch_smoke.sh` so the dashboard auto-discovers it.
#
# Trial pool: 16 pre-encoded configs covering
#   - rank ladder {16,32,64,128}
#   - alpha tying {alpha=r, alpha=2r}
#   - dropout ablation {0.0, 0.05, 0.10}
#   - lr sweep {1e-5, 3e-5, 5e-5}
#   - target_modules {attn_qv, attn_qkvo}
# Each trial finetunes from the prior best SmolVLA checkpoint (anchor).
#
# Knobs (env-overridable — stretch the budget without editing the script):
#   MAX_TRIALS=10              # how many configs to run (1..16)
#   STEPS=10000                # training steps per trial
#   SECONDS_PER_EXP=3600       # per-trial wall timeout
#   BATCH_SIZE=6               # halved automatically by wrapper on OOM
#   SESSION_ID=lora-bash-<ts>  # state dir slug
#   ANCHOR=<path>              # pretrained SmolVLA ckpt
#   DATASET=<path>             # LeRobotDataset root
#
# Time budget (rough, cached path per CLAUDE.md SmolVLA RTX 3080 table):
#   trial 0 = 16 min warmup + 1.6 sec/100 steps = ~16 + STEPS*0.016/60 min
#   trials 1..N = 6 s cache reload + same train rate
#   default (10 trials × 10k steps) ≈ 3.0–3.5 h
#   stretch (16 trials × 20k steps) ≈ 9.5–10 h
#
# Usage:
#   bash scripts/_run_autoresearch_lora.sh                    # defaults
#   MAX_TRIALS=16 STEPS=20000 bash scripts/_run_autoresearch_lora.sh
#   DRY_RUN=1 bash scripts/_run_autoresearch_lora.sh          # echo cmds only
# =============================================================================
set -uo pipefail

WORKSPACE="${WORKSPACE:-${LEROBOT_ISAAC_WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$WORKSPACE"

# --- knobs ------------------------------------------------------------------
SESSION_ID="${SESSION_ID:-lora-bash-$(date +%Y%m%d-%H%M%S)}"
SLUG="lerobot-policy-smolvla-lora"
MAX_TRIALS="${MAX_TRIALS:-10}"
STEPS="${STEPS:-5000}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
SECONDS_PER_EXP="${SECONDS_PER_EXP:-2400}"
BATCH_SIZE="${BATCH_SIZE:-6}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-4}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-300}"
EVAL_ENABLED="${EVAL_ENABLED:-1}"   # 1 by default — merge-and-save patch in place
MERGE_TIMEOUT="${MERGE_TIMEOUT:-300}"
SKIP_TRIALS="${SKIP_TRIALS:-0}"     # start the pool index at this value (resume from trial N)
PLATEAU_LIMIT="${PLATEAU_LIMIT:-6}" # was 3 — too tight for a 16-config exploration sweep
DRY_RUN="${DRY_RUN:-0}"

ANCHOR="${ANCHOR:-outputs/overnight-smolvla-2026-05-15T210257-anchor/policy-smolvla/checkpoints/last/pretrained_model}"
DATASET="${DATASET:-datasets/kvgork/so101-pickplace1}"
PY="$WORKSPACE/.pixi/envs/train-policy/bin/python"

AR_DIR="$WORKSPACE/.agent-state/$SESSION_ID/autoresearch/$SLUG"
HISTORY="$AR_DIR/history.jsonl"
BEST="$AR_DIR/best.json"
PLATEAU="$AR_DIR/plateau.json"
PROGRAM="$AR_DIR/program.json"

# --- pre-flight -------------------------------------------------------------
[ -d "$ANCHOR" ] || { echo "ERROR: anchor not found: $ANCHOR" >&2; exit 2; }
[ -d "$DATASET" ] || { echo "ERROR: dataset not found: $DATASET" >&2; exit 2; }
[ -x "$PY" ] || { echo "ERROR: train-policy python not found: $PY" >&2; exit 2; }
"$PY" -c "import peft; assert peft.__version__ >= '0.10', peft.__version__" 2>/dev/null \
    || { echo "ERROR: peft>=0.10 not installed in train-policy env" >&2; exit 2; }

mkdir -p "$AR_DIR"

# --- trial pool (16 configs) ------------------------------------------------
# Format: rank|alpha|dropout|target_modules|lr|warmup
declare -a TRIAL_POOL=(
    "64|64|0.05|attn_qv|3e-5|500"       # 0 baseline (HF default)
    "16|16|0.05|attn_qv|3e-5|500"       # 1 ladder low
    "32|32|0.05|attn_qv|3e-5|500"       # 2 ladder mid-low
    "128|128|0.05|attn_qv|3e-5|500"     # 3 ladder high
    "64|128|0.05|attn_qv|3e-5|500"      # 4 alpha=2r at baseline rank
    "128|256|0.05|attn_qv|3e-5|500"     # 5 alpha=2r at top rank
    "64|64|0.00|attn_qv|3e-5|500"       # 6 dropout=0
    "64|64|0.10|attn_qv|3e-5|500"       # 7 dropout=0.1
    "64|64|0.05|attn_qv|1e-5|500"       # 8 lr lower
    "64|64|0.05|attn_qv|5e-5|500"       # 9 lr upper
    "64|64|0.05|attn_qkvo|3e-5|500"     # 10 target_modules ablation
    "128|128|0.05|attn_qkvo|3e-5|500"   # 11 r=128 + attn_qkvo
    "32|64|0.05|attn_qv|3e-5|500"       # 12 alpha=2r at low rank
    "16|32|0.05|attn_qv|3e-5|500"       # 13 alpha=2r at lowest rank
    "64|64|0.05|attn_qv|3e-5|200"       # 14 warmup_steps=200
    "64|64|0.05|attn_qv|3e-5|1000"      # 15 warmup_steps=1000
)
TOTAL_POOL=${#TRIAL_POOL[@]}
N=$(( MAX_TRIALS < TOTAL_POOL ? MAX_TRIALS : TOTAL_POOL ))

# --- program snapshot (dashboard reads this) --------------------------------
cat > "$PROGRAM" <<EOF
{
  "name": "$SLUG",
  "metric": {"name": "pc_success", "direction": "maximize"},
  "budget": {
    "seconds_per_experiment": $SECONDS_PER_EXP,
    "max_experiments": $N,
    "plateau_limit": $PLATEAU_LIMIT
  },
  "target_arch": "smolvla",
  "dataset": "$DATASET",
  "anchor": "$ANCHOR",
  "steps": $STEPS,
  "batch_size_default": $BATCH_SIZE,
  "iterations": $N,
  "session_id": "$SESSION_ID",
  "ts_start": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "[lora-ar] session=$SESSION_ID"
echo "[lora-ar] trials=$N steps=$STEPS bs=$BATCH_SIZE timeout=${SECONDS_PER_EXP}s"
echo "[lora-ar] anchor=$ANCHOR"
echo "[lora-ar] state_dir=$AR_DIR"

if [ "$DRY_RUN" != "1" ] && [ "$SKIP_TRIALS" = "0" ]; then
    : > "$HISTORY"
fi

best_metric=""
plateau_count=0
# Seed best_metric from external env (e.g. carry-over from prior partial sweep).
if [ -n "${RESUME_BEST_METRIC:-}" ]; then
    best_metric="$RESUME_BEST_METRIC"
    echo "[lora-ar] resuming with seeded best_metric=$best_metric"
fi

# --- main loop --------------------------------------------------------------
for i in $(seq "$SKIP_TRIALS" $(( N - 1 ))); do
    IFS='|' read -r RANK ALPHA DROPOUT TARGET LR WARMUP <<< "${TRIAL_POOL[$i]}"

    out_dir="$WORKSPACE/outputs/autoresearch-$SLUG/trial_${i}"
    iter_log="$AR_DIR/trial_${i}.log"

    echo
    echo "[lora-ar] trial=$i rank=$RANK alpha=$ALPHA drop=$DROPOUT target=$TARGET lr=$LR warmup=$WARMUP"

    # Build command. cache_frames goes through wrapper's `extra` passthrough.
    # `--policy.pretrained_path` goes after `--` for lerobot-train remainder.
    CMD=(
        timeout "$SECONDS_PER_EXP"
        "$PY" -m lerobot_isaac_autoresearch.train_wrapper
            --target_arch smolvla
            --dataset "$DATASET"
            --output_dir "$out_dir"
            --steps "$STEPS"
            --batch_size "$BATCH_SIZE"
            --use_lora
            --lora_rank "$RANK"
            --lora_alpha "$ALPHA"
            --lora_dropout "$DROPOUT"
            --lora_target_modules "$TARGET"
            --cache_frames
            --lr "$LR"
            --
            --policy.pretrained_path="$ANCHOR"
            --policy.push_to_hub=false
            --save_freq="$SAVE_FREQ"
            --log_freq=200
    )

    if [ "$DRY_RUN" = "1" ]; then
        printf '  %s\n' "${CMD[@]}"
        continue
    fi

    # lerobot-train 0.5 refuses to overwrite existing output_dir.
    rm -rf "$out_dir"
    mkdir -p "$(dirname "$out_dir")"

    start_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    start_s=$(date +%s)

    PATH="$WORKSPACE/.pixi/envs/train-policy/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    LEROBOT_ISAAC_CACHE_RAM_GB=8 \
        "${CMD[@]}" > "$iter_log" 2>&1
    rc=$?

    end_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    dur=$(( $(date +%s) - start_s ))

    # Metric extraction priority:
    #   1. Open-loop eval pc_success on held-out episodes (real metric).
    #   2. pc_success line in stdout (rarely emitted, but covers future eval-in-loop).
    #   3. Training-loss proxy 1/(1+loss) (fallback when eval fails).
    #   4. Sentinel 0.0.
    last_pc=""
    eval_kind=""
    eval_json="$AR_DIR/trial_${i}_eval.json"

    # Find newest checkpoint dir (numeric step ID, descending). `last/` is a
    # symlink to the highest step but not always present on timeout.
    ckpt_dir=""
    if [ -d "$out_dir/checkpoints" ]; then
        for cand in $(ls -1 "$out_dir/checkpoints" 2>/dev/null | grep -E '^[0-9]+$' | sort -rn); do
            if [ -d "$out_dir/checkpoints/$cand/pretrained_model" ]; then
                ckpt_dir="$out_dir/checkpoints/$cand/pretrained_model"
                break
            fi
        done
    fi

    if [ -n "$ckpt_dir" ] && [ "$EVAL_ENABLED" = "1" ]; then
        # peft saved ckpts are not loadable by plain SmolVLA loader. Merge LoRA
        # delta into base weights first → produces a normal-shaped ckpt.
        merged_dir="$out_dir/checkpoints/merged/pretrained_model"
        merge_log="$AR_DIR/trial_${i}_merge.log"
        echo "[lora-ar] merge: trial=$i src=$ckpt_dir → $merged_dir"
        timeout "$MERGE_TIMEOUT" "$PY" scripts/_merge_lora_ckpt.py \
            --anchor "$ANCHOR" \
            --trial_ckpt "$ckpt_dir" \
            --dataset_root "$DATASET" \
            --lora_rank "$RANK" \
            --lora_alpha "$ALPHA" \
            --lora_dropout "$DROPOUT" \
            --lora_target_modules "$TARGET" \
            --out "$merged_dir" \
            > "$merge_log" 2>&1
        merge_rc=$?
        if [ "$merge_rc" -eq 0 ] && [ -d "$merged_dir" ]; then
            echo "[lora-ar] eval: trial=$i ckpt=$merged_dir"
            eval_log="$AR_DIR/trial_${i}_eval.log"
            timeout "$EVAL_TIMEOUT" "$PY" scripts/_open_loop_eval.py \
                --policy_path "$merged_dir" \
                --dataset_root "$DATASET" \
                --n_episodes "$EVAL_N_EPISODES" \
                --output_json "$eval_json" \
                --task_label "so101-pickplace1-lora-r${RANK}-a${ALPHA}-d${DROPOUT}" \
                --run_id "lora-bash-trial-${i}" \
                > "$eval_log" 2>&1
            eval_rc=$?
            if [ "$eval_rc" -eq 0 ] && [ -f "$eval_json" ]; then
                last_pc=$("$PY" -c "import json;print(json.load(open('$eval_json')).get('pc_success',''))" 2>/dev/null)
                eval_kind="open_loop_mse_merged"
            else
                echo "[lora-ar] eval FAILED rc=$eval_rc — see $eval_log"
            fi
        else
            echo "[lora-ar] merge FAILED rc=$merge_rc — see $merge_log"
        fi
    elif [ -z "$ckpt_dir" ]; then
        echo "[lora-ar] no checkpoint found for trial=$i — skipping eval"
    fi

    # Also keep loss-proxy as a secondary signal.
    last_loss=$(grep -oE 'loss:[0-9.eE+\-]+' "$iter_log" | tail -1 | sed 's/loss://')

    if [ -n "$last_pc" ] && [ "$last_pc" != "0.0" ]; then
        metric="$last_pc"
        metric_kind="$eval_kind"
    elif [ -n "$last_loss" ]; then
        metric=$("$PY" -c "print(round(1.0/(1.0+float('$last_loss')), 6))")
        metric_kind="loss_proxy"
    else
        metric="0.0"
        metric_kind="sentinel"
    fi

    status="ok"
    if [ "$rc" -eq 124 ]; then
        status="timeout"
    elif [ "$rc" -ne 0 ]; then
        status="error"
    fi

    raw_loss_lit="None"
    if [ -n "$last_loss" ]; then raw_loss_lit="$last_loss"; fi

    "$PY" - <<PY >> "$HISTORY"
import json
print(json.dumps({
    "trial_index": $i,
    "trial": $i,
    "metric_name": "pc_success",
    "metric_value": float("$metric"),
    "metric_kind": "$metric_kind",
    "config": {
        "lora_rank": $RANK,
        "lora_alpha": $ALPHA,
        "lora_dropout": $DROPOUT,
        "lora_target_modules": "$TARGET",
        "lr": float("$LR"),
        "warmup_steps": $WARMUP,
        "batch_size": $BATCH_SIZE,
        "steps": $STEPS,
        "anchor": "$ANCHOR"
    },
    "ts": "$start_ts",
    "end_ts": "$end_ts",
    "duration_s": $dur,
    "status": "$status",
    "raw_loss": $raw_loss_lit,
    "exit_code": $rc,
}))
PY

    echo "[lora-ar] trial=$i metric=$metric ($metric_kind) loss=${last_loss:-NA} status=$status duration=${dur}s"

    # Ratchet best.json.
    if [ -z "$best_metric" ] || "$PY" -c "exit(0 if float('$metric') > float('$best_metric') else 1)"; then
        best_metric="$metric"
        "$PY" - <<PY > "$BEST"
import json
print(json.dumps({
    "trial": $i,
    "metric_value": float("$metric"),
    "metric_kind": "$metric_kind",
    "config": {
        "lora_rank": $RANK,
        "lora_alpha": $ALPHA,
        "lora_dropout": $DROPOUT,
        "lora_target_modules": "$TARGET",
        "lr": float("$LR"),
        "warmup_steps": $WARMUP,
        "batch_size": $BATCH_SIZE,
        "steps": $STEPS
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

    # Plateau stop.
    if [ "$plateau_count" -ge "$PLATEAU_LIMIT" ]; then
        echo "[lora-ar] plateau_limit=$PLATEAU_LIMIT reached at trial=$i — stopping early"
        break
    fi
done

echo
echo "[lora-ar] done — best metric: $best_metric"
echo "[lora-ar] state: $AR_DIR"
ls -la "$AR_DIR"
