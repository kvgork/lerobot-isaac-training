#!/usr/bin/env bash
# =============================================================================
# _train_winning_lora.sh — Production training of the LoRA sweep winner.
#
# Trial 12 from session `wm-bash-20260522-185502` (LoRA sweep) found:
#   r=32  alpha=64  dropout=0.05  attn_qv  lr=3e-5  bs=6  warmup=500
#   pc_success(open_loop_mse) = 0.149  vs anchor 0.082  (+82 % rel)
#
# That run was capped at STEPS=5500 by the 40-min sweep timeout. The loss
# curve was still decreasing → leaving win on the table. This script trains
# the SAME config for STEPS=30000 with no sweep-timeout cap. Output ckpt
# is deploy-ready (auto-merged via _merge_lora_ckpt.py).
#
# Knobs:
#   STEPS=30000              ~5.5× the sweep cap; ~2.9 h @ 2.85 step/s cached
#   BATCH_SIZE=6             RTX 3080 fit, half-on-OOM retry by train_wrapper
#   LR=3e-5                  trial 12 value
#   LORA_RANK=32             trial 12 value
#   LORA_ALPHA=64            trial 12 value (alpha=2r)
#   LORA_DROPOUT=0.05        trial 12 value
#   LORA_TARGETS=attn_qv     trial 12 value
#   WARMUP=500               trial 12 value
#   SAVE_FREQ=5000           one ckpt per ~25 min of train
#   SECONDS_PER_EXP=21600    6 h ceiling (slack for OOM retry)
#   EVAL_AFTER=1             run open-loop eval on merged ckpt
#   AUTO_SYNC=0              if 1, calls li-deploy-sync-ckpt to the laptop
#
# Anchor: outputs/overnight-smolvla-2026-05-15T210257-anchor/...
#         (same as the LoRA sweep used)
#
# Output:
#   outputs/lora-prod-best/checkpoints/last/pretrained_model/        — peft-wrapped
#   outputs/lora-prod-best/checkpoints/merged/pretrained_model/      — deploy-ready
#   .agent-state/<session>/lora-prod-best/{program,history,best,plateau}.json
#
# Usage:
#   bash scripts/_train_winning_lora.sh
#   STEPS=50000 bash scripts/_train_winning_lora.sh   # longer overnight run
#   AUTO_SYNC=1 bash scripts/_train_winning_lora.sh   # rsync to laptop after eval
#   DRY_RUN=1 bash scripts/_train_winning_lora.sh
# =============================================================================
set -uo pipefail

WORKSPACE="${WORKSPACE:-${LEROBOT_ISAAC_WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$WORKSPACE"

# --- knobs ------------------------------------------------------------------
SESSION_ID="${SESSION_ID:-lora-prod-$(date +%Y%m%d-%H%M%S)}"
SLUG="lerobot-policy-smolvla-lora-prod"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-6}"
LR="${LR:-3e-5}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-attn_qv}"
WARMUP="${WARMUP:-500}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
SECONDS_PER_EXP="${SECONDS_PER_EXP:-21600}"
EVAL_AFTER="${EVAL_AFTER:-1}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-4}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-600}"
AUTO_SYNC="${AUTO_SYNC:-0}"
LAPTOP_HOST="${LAPTOP_HOST:-laptop}"
DRY_RUN="${DRY_RUN:-0}"

DATASET="${DATASET:-datasets/kvgork/so101-pickplace1}"
ANCHOR="${ANCHOR:-outputs/overnight-smolvla-2026-05-15T210257-anchor/policy-smolvla/checkpoints/last/pretrained_model}"
PY="$WORKSPACE/.pixi/envs/train-policy/bin/python"

OUT_DIR="$WORKSPACE/outputs/lora-prod-best"
AR_DIR="$WORKSPACE/.agent-state/$SESSION_ID/autoresearch/$SLUG"
HISTORY="$AR_DIR/history.jsonl"
BEST="$AR_DIR/best.json"
PROGRAM="$AR_DIR/program.json"
TRAIN_LOG="$AR_DIR/train.log"

# --- pre-flight -------------------------------------------------------------
[ -d "$DATASET" ] || { echo "ERROR: dataset not found: $DATASET" >&2; exit 2; }
[ -d "$ANCHOR" ]  || { echo "ERROR: anchor not found: $ANCHOR"  >&2; exit 2; }
[ -x "$PY" ]      || { echo "ERROR: train-policy python not found: $PY" >&2; exit 2; }
"$PY" -c "import peft; assert peft.__version__ >= '0.10', peft.__version__" 2>/dev/null \
    || { echo "ERROR: peft>=0.10 not installed in train-policy env" >&2; exit 2; }

mkdir -p "$AR_DIR"

cat > "$PROGRAM" <<EOF
{
  "name": "$SLUG",
  "metric": {"name": "pc_success", "direction": "maximize"},
  "budget": {"seconds_per_experiment": $SECONDS_PER_EXP, "max_experiments": 1, "plateau_limit": 1},
  "target_arch": "smolvla",
  "dataset": "$DATASET",
  "anchor": "$ANCHOR",
  "config": {
    "lora_rank": $LORA_RANK, "lora_alpha": $LORA_ALPHA,
    "lora_dropout": $LORA_DROPOUT, "lora_target_modules": "$LORA_TARGETS",
    "lr": $LR, "batch_size": $BATCH_SIZE, "warmup_steps": $WARMUP,
    "steps": $STEPS, "save_freq": $SAVE_FREQ
  },
  "session_id": "$SESSION_ID",
  "ts_start": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "[lora-prod] session=$SESSION_ID"
echo "[lora-prod] config: r=$LORA_RANK alpha=$LORA_ALPHA drop=$LORA_DROPOUT tgt=$LORA_TARGETS lr=$LR bs=$BATCH_SIZE warmup=$WARMUP"
echo "[lora-prod] steps=$STEPS timeout=${SECONDS_PER_EXP}s save_freq=$SAVE_FREQ"
echo "[lora-prod] anchor=$ANCHOR"
echo "[lora-prod] dataset=$DATASET"
echo "[lora-prod] out_dir=$OUT_DIR"
echo "[lora-prod] state_dir=$AR_DIR"

# --- build train cmd --------------------------------------------------------
CMD=(
    timeout "$SECONDS_PER_EXP"
    "$PY" -m lerobot_isaac_autoresearch.train_wrapper
        --target_arch smolvla
        --dataset "$DATASET"
        --output_dir "$OUT_DIR"
        --steps "$STEPS"
        --batch_size "$BATCH_SIZE"
        --use_lora
        --lora_rank "$LORA_RANK"
        --lora_alpha "$LORA_ALPHA"
        --lora_dropout "$LORA_DROPOUT"
        --lora_target_modules "$LORA_TARGETS"
        --cache_frames
        --lr "$LR"
        --
        --policy.pretrained_path="$ANCHOR"
        --policy.push_to_hub=false
        --save_freq="$SAVE_FREQ"
        --log_freq=200
)

if [ "$DRY_RUN" = "1" ]; then
    echo "[lora-prod] (dry-run) command would be:"
    printf '  %s\n' "${CMD[@]}"
    exit 0
fi

# clean output dir (lerobot-train refuses to overwrite)
rm -rf "$OUT_DIR"
mkdir -p "$(dirname "$OUT_DIR")"

start_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
start_s=$(date +%s)

# --- train ------------------------------------------------------------------
PATH="$WORKSPACE/.pixi/envs/train-policy/bin:$PATH" \
PYTHONUNBUFFERED=1 \
LEROBOT_ISAAC_CACHE_RAM_GB=8 \
    "${CMD[@]}" > "$TRAIN_LOG" 2>&1
rc=$?
train_dur=$(( $(date +%s) - start_s ))

status="ok"
[ "$rc" -eq 124 ] && status="timeout"
[ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && status="error"

echo "[lora-prod] train done rc=$rc status=$status duration=${train_dur}s"

# --- merge ------------------------------------------------------------------
MERGED_DIR="$OUT_DIR/checkpoints/merged/pretrained_model"
ckpt_dir=""
if [ -d "$OUT_DIR/checkpoints" ]; then
    # Pick the highest-step ckpt dir.
    for cand in $(ls -1 "$OUT_DIR/checkpoints" 2>/dev/null | grep -E '^[0-9]+$' | sort -rn); do
        if [ -d "$OUT_DIR/checkpoints/$cand/pretrained_model" ]; then
            ckpt_dir="$OUT_DIR/checkpoints/$cand/pretrained_model"
            break
        fi
    done
fi

if [ -n "$ckpt_dir" ]; then
    echo "[lora-prod] merging $ckpt_dir → $MERGED_DIR"
    "$PY" scripts/_merge_lora_ckpt.py \
        --anchor "$ANCHOR" \
        --trial_ckpt "$ckpt_dir" \
        --dataset_root "$DATASET" \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --lora_target_modules "$LORA_TARGETS" \
        --out "$MERGED_DIR" > "$AR_DIR/merge.log" 2>&1
    merge_rc=$?
    if [ "$merge_rc" -ne 0 ]; then
        echo "[lora-prod] MERGE FAILED rc=$merge_rc — see $AR_DIR/merge.log"
    else
        echo "[lora-prod] merge OK"
    fi
else
    echo "[lora-prod] no ckpt produced — skipping merge"
fi

# --- eval -------------------------------------------------------------------
metric=""
if [ "$EVAL_AFTER" = "1" ] && [ -d "$MERGED_DIR" ]; then
    eval_json="$AR_DIR/eval.json"
    echo "[lora-prod] eval: open-loop on $MERGED_DIR"
    timeout "$EVAL_TIMEOUT" "$PY" scripts/_open_loop_eval.py \
        --policy_path "$MERGED_DIR" \
        --dataset_root "$DATASET" \
        --n_episodes "$EVAL_N_EPISODES" \
        --output_json "$eval_json" \
        --task_label "so101-pickplace1-lora-prod" \
        --run_id "$SESSION_ID" > "$AR_DIR/eval.log" 2>&1
    eval_rc=$?
    if [ "$eval_rc" -eq 0 ] && [ -f "$eval_json" ]; then
        metric=$("$PY" -c "import json;d=json.load(open('$eval_json'));print(d['pc_success'])")
        echo "[lora-prod] eval pc_success=$metric"
    else
        echo "[lora-prod] eval FAILED rc=$eval_rc — see $AR_DIR/eval.log"
    fi
fi

# --- history + best ---------------------------------------------------------
end_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
dur=$(( $(date +%s) - start_s ))
"$PY" - <<PY >> "$HISTORY"
import json
print(json.dumps({
    "trial_index": 0, "trial": 0,
    "metric_name": "pc_success",
    "metric_value": float("$metric" or 0.0),
    "metric_kind": "open_loop_mse_merged" if "$metric" else "no_eval",
    "config": {
        "lora_rank": $LORA_RANK, "lora_alpha": $LORA_ALPHA,
        "lora_dropout": $LORA_DROPOUT, "lora_target_modules": "$LORA_TARGETS",
        "lr": float("$LR"), "batch_size": $BATCH_SIZE, "warmup_steps": $WARMUP,
        "steps": $STEPS, "anchor": "$ANCHOR"
    },
    "ts": "$start_ts", "end_ts": "$end_ts",
    "duration_s": $dur, "train_duration_s": $train_dur,
    "status": "$status", "exit_code": $rc,
    "merged_ckpt": "$MERGED_DIR"
}))
PY

if [ -n "$metric" ]; then
    "$PY" - <<PY > "$BEST"
import json
print(json.dumps({
    "trial": 0, "metric_value": float("$metric"),
    "metric_kind": "open_loop_mse_merged",
    "config": {
        "lora_rank": $LORA_RANK, "lora_alpha": $LORA_ALPHA,
        "lora_dropout": $LORA_DROPOUT, "lora_target_modules": "$LORA_TARGETS",
        "lr": float("$LR"), "batch_size": $BATCH_SIZE, "steps": $STEPS
    },
    "merged_ckpt": "$MERGED_DIR"
}, indent=2))
PY
fi

# --- optional auto-sync -----------------------------------------------------
if [ "$AUTO_SYNC" = "1" ] && [ -d "$MERGED_DIR" ]; then
    echo "[lora-prod] syncing $MERGED_DIR → $LAPTOP_HOST"
    # The merged ckpt is a regular lerobot policy dir; use the existing
    # li-deploy-sync-ckpt. run_dir is the trial root (its checkpoints/
    # contains the numbered ckpts AND merged/).
    "$PY" -m lerobot_isaac_deploy.cli sync-ckpt \
        --run-dir "$OUT_DIR" \
        --host "$LAPTOP_HOST" \
        --dataset-root "$DATASET" 2>&1 | tail -10
fi

echo
echo "[lora-prod] DONE — status=$status pc_success=$metric merged=$MERGED_DIR"
ls -la "$AR_DIR"
