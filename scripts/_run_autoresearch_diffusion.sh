#!/usr/bin/env bash
# Deterministic autoresearch loop for programs/lerobot-policy-diffusion.md
# Same on-disk schema as the LLM-driven orchestrator:
#   .agent-state/<session>/autoresearch/lerobot-policy-diffusion/
#     program.json   history.jsonl   best.json   plateau.json
# 3 trials by default (env: $TRIALS). Per-exp wall-clock = $SECONDS_PER_EXP.
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

SESSION="${SESSION_ID:-$(date +%Y%m%d-%H%M%S)-autoresearch-diffusion}"
SLUG="lerobot-policy-diffusion"
AR_DIR="$WORKSPACE/.agent-state/$SESSION/autoresearch/$SLUG"
mkdir -p "$AR_DIR"

DATASET="${DATASET:-$WORKSPACE/datasets/kvgork/so101-pickplace1}"
PY="$WORKSPACE/.pixi/envs/train-policy/bin/python"
SECONDS_PER_EXP="${SECONDS_PER_EXP:-1800}"
TRIALS="${TRIALS:-3}"

# Program snapshot
cat > "$AR_DIR/program.json" <<EOF
{
  "name": "lerobot-policy-diffusion",
  "domain": "lerobot_isaac",
  "metric": {"name": "pc_success", "direction": "maximize"},
  "budget": {"seconds_per_experiment": $SECONDS_PER_EXP, "max_experiments": $TRIALS, "plateau_limit": 3},
  "target_arch": "diffusion",
  "dataset": "datasets/kvgork/so101-pickplace1",
  "session_id": "$SESSION"
}
EOF

: > "$AR_DIR/history.jsonl"
best_metric=""
plateau=0

# Grounded mutation grid. Each trial exercises a different operator from
# programs/_domain_knowledge.md §9 so reruns don't repeat the same config.
declare -a CONFIGS=(
  '{"trial":0,"batch_size":8,"lr":1e-4,"steps":50000,"seed":42,"op":"baseline","extra":""}'
  '{"trial":1,"batch_size":8,"lr":5e-5,"steps":80000,"seed":1337,"op":"tune_hyperparams:lr_down","extra":""}'
  '{"trial":2,"batch_size":16,"lr":1e-4,"steps":50000,"seed":42,"op":"tune_hyperparams:batch_up","extra":""}'
  '{"trial":3,"batch_size":8,"lr":1e-4,"steps":50000,"seed":42,"op":"add_regularization:weight_decay","extra":"--optimizer.weight_decay=1e-4"}'
  '{"trial":4,"batch_size":8,"lr":3e-4,"steps":50000,"seed":42,"op":"tune_hyperparams:lr_up","extra":""}'
  '{"trial":5,"batch_size":8,"lr":1e-4,"steps":50000,"seed":7,"op":"tune_hyperparams:seed_swap","extra":""}'
)

for cfg in "${CONFIGS[@]:0:$TRIALS}"; do
    trial=$(echo "$cfg" | jq -r .trial)
    bs=$(echo "$cfg" | jq -r .batch_size)
    lr=$(echo "$cfg" | jq -r .lr)
    steps=$(echo "$cfg" | jq -r .steps)
    seed=$(echo "$cfg" | jq -r .seed)
    op=$(echo "$cfg" | jq -r .op)
    extra=$(echo "$cfg" | jq -r '.extra // ""')

    out_dir="$WORKSPACE/outputs/autoresearch-$SLUG/trial_$trial"
    rm -rf "$out_dir"
    mkdir -p "$(dirname "$out_dir")"

    echo "[ar diffusion trial=$trial] bs=$bs lr=$lr steps=$steps seed=$seed op=$op"
    iter_log="$AR_DIR/trial_${trial}.log"
    start_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    start_s=$(date +%s)
    # save_freq auto-scales: roughly one save at 40% of budget so eval has ckpt.
    save_freq=$(( SECONDS_PER_EXP * 25 / 30 ))
    [ "$save_freq" -lt 200 ] && save_freq=200

    PATH="$WORKSPACE/.pixi/envs/train-policy/bin:$PATH" \
    timeout "$SECONDS_PER_EXP" "$PY" -m lerobot_isaac_autoresearch.train_wrapper \
        --target_arch diffusion \
        --dataset "$DATASET" \
        --output_dir "$out_dir" \
        --steps "$steps" \
        --batch_size "$bs" \
        -- --optimizer.lr="$lr" --seed="$seed" --policy.device=cuda --save_freq="$save_freq" --log_freq=50 $extra \
        > "$iter_log" 2>&1
    rc=$?
    end_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    dur=$(( $(date +%s) - start_s ))

    # Run open-loop eval on the latest checkpoint we have so the metric is real.
    ckpt=$(find "$out_dir/checkpoints" -name pretrained_model -type d 2>/dev/null | sort | tail -1)
    pc=""
    mse=""
    if [ -n "${ckpt:-}" ] && [ -d "$ckpt" ]; then
        eval_json="$AR_DIR/trial_${trial}-eval.json"
        "$PY" "$WORKSPACE/scripts/_open_loop_eval.py" \
            --policy_path "$ckpt" \
            --dataset_root "$DATASET" \
            --n_episodes 3 \
            --output_json "$eval_json" \
            --task_label "autoresearch-diffusion-trial-$trial" \
            --run_id "${SESSION}-trial${trial}" \
            >> "$iter_log" 2>&1
        pc=$(jq -r .pc_success "$eval_json" 2>/dev/null)
        mse=$(jq -r ._metadata.mse "$eval_json" 2>/dev/null)
    fi
    if [ -z "$pc" ] || [ "$pc" = "null" ]; then pc="0.0"; fi

    status="ok"
    [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && status="error"
    [ "$rc" -eq 124 ] && status="timeout-ok"

    raw_loss=$(grep -oE 'loss:[0-9.]+' "$iter_log" | tail -1 | sed 's/loss://')
    [ -z "$raw_loss" ] && raw_loss=null

    "$WORKSPACE/.pixi/envs/default/bin/python" - <<PY >> "$AR_DIR/history.jsonl"
import json
print(json.dumps({
    "trial_index": $trial,
    "trial": $trial,
    "metric_name": "pc_success",
    "metric_value": float("$pc"),
    "config": {"batch_size": $bs, "lr": $lr, "steps": $steps, "seed": $seed, "operator": "$op"},
    "ts": "$start_ts",
    "end_ts": "$end_ts",
    "duration_s": $dur,
    "status": "$status",
    "raw_loss": $raw_loss,
    "mse": ${mse:-null},
    "exit_code": $rc,
    "ckpt": "$ckpt",
}))
PY

    echo "[ar diffusion trial=$trial] pc=$pc loss=$raw_loss mse=${mse:-NA} dur=${dur}s status=$status"

    if [ -z "$best_metric" ] || "$WORKSPACE/.pixi/envs/default/bin/python" -c "exit(0 if float('$pc') > float('$best_metric') else 1)"; then
        best_metric="$pc"
        "$WORKSPACE/.pixi/envs/default/bin/python" - <<PY > "$AR_DIR/best.json"
import json
print(json.dumps({
    "trial": $trial,
    "metric_value": float("$pc"),
    "config": {"batch_size": $bs, "lr": $lr, "steps": $steps, "seed": $seed},
    "operator": "$op",
}, indent=2))
PY
        plateau=0
    else
        plateau=$(( plateau + 1 ))
    fi

    "$WORKSPACE/.pixi/envs/default/bin/python" - <<PY > "$AR_DIR/plateau.json"
import json
print(json.dumps({
    "consecutive_non_improvements": $plateau,
    "plateau_limit": 3,
    "last_metric": float("$pc"),
    "best_metric": float("$best_metric"),
}, indent=2))
PY
done

echo "[ar diffusion] DONE — $AR_DIR"
ls -la "$AR_DIR"
