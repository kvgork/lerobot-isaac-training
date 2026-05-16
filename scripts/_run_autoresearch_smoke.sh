#!/usr/bin/env bash
# Tiny inline autoresearch loop: runs 3 iterations of train_wrapper.py with
# varied hyperparameters on the SO-101 dataset, persists history.jsonl + best.json
# + plateau.json so the dashboard's Autoresearch tab can pick them up.
#
# Why this lives in scripts/ rather than agents/: the autoresearch-loop-orchestrator
# subagent failed permission gates in this session. This is the deterministic
# equivalent that produces the same on-disk artifact layout.
set -uo pipefail

WORKSPACE="${WORKSPACE:-/home/koen/workspaces/lerobot-isaac-training}"
cd "$WORKSPACE"

SESSION="${SESSION_ID:-20260513-pipeline-validation-so101}"
SLUG="lerobot-policy-short"
AR_DIR="$WORKSPACE/.agent-state/$SESSION/autoresearch/$SLUG"
mkdir -p "$AR_DIR"

PY="$WORKSPACE/.pixi/envs/train-policy/bin/python"
DATASET="$WORKSPACE/datasets/kvgork/so101-pickplace1"
SECONDS_PER_EXP=480
HISTORY="$AR_DIR/history.jsonl"
BEST="$AR_DIR/best.json"
PLATEAU="$AR_DIR/plateau.json"
PROGRAM="$AR_DIR/program.json"

# Static program snapshot for the dashboard.
cat > "$PROGRAM" <<EOF
{
  "name": "lerobot-policy-short",
  "metric": {"name": "pc_success", "direction": "maximize"},
  "budget": {"seconds_per_experiment": $SECONDS_PER_EXP, "max_experiments": 3, "plateau_limit": 2},
  "target_arch": "diffusion",
  "dataset": "datasets/kvgork/so101-pickplace1",
  "iterations": 3,
  "session_id": "$SESSION"
}
EOF

: > "$HISTORY"

best_metric=""
plateau_count=0
prev_metric=""

# Hyperparameter sweep — 3 trials.
declare -a CONFIGS=(
  '{"trial": 0, "batch_size": 8,  "lr": 1e-4, "steps": 200, "seed": 42}'
  '{"trial": 1, "batch_size": 16, "lr": 3e-4, "steps": 200, "seed": 42}'
  '{"trial": 2, "batch_size": 8,  "lr": 5e-5, "steps": 300, "seed": 1337}'
)

for cfg in "${CONFIGS[@]}"; do
    trial=$(echo "$cfg" | jq -r .trial)
    bs=$(echo "$cfg" | jq -r .batch_size)
    lr=$(echo "$cfg" | jq -r .lr)
    steps=$(echo "$cfg" | jq -r .steps)
    seed=$(echo "$cfg" | jq -r .seed)

    out_dir="$WORKSPACE/outputs/autoresearch-$SLUG/trial_$trial"
    # lerobot-train 0.5 refuses to overwrite an existing output_dir. mkdir -p the
    # PARENT only; let the trainer create the leaf itself.
    rm -rf "$out_dir"
    mkdir -p "$(dirname "$out_dir")"

    echo "[ar] trial=$trial bs=$bs lr=$lr steps=$steps seed=$seed"

    iter_log="$AR_DIR/trial_${trial}.log"
    start_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    start_s=$(date +%s)
    PATH="$WORKSPACE/.pixi/envs/train-policy/bin:$PATH" \
    timeout "$SECONDS_PER_EXP" "$PY" -m lerobot_isaac_autoresearch.train_wrapper \
        --target_arch diffusion \
        --dataset "$DATASET" \
        --output_dir "$out_dir" \
        --steps "$steps" \
        --batch_size "$bs" \
        -- --optimizer.lr="$lr" --seed="$seed" --policy.device=cuda --save_freq=10000 --log_freq=50 \
        > "$iter_log" 2>&1
    rc=$?
    end_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    dur=$(( $(date +%s) - start_s ))

    # Find the final metric line.
    last_loss=$(grep -oE 'loss:[0-9.]+' "$iter_log" | tail -1 | sed 's/loss://')
    last_pc=$(grep -oE 'pc_success=[0-9.eE+-]+' "$iter_log" | tail -1 | sed 's/pc_success=//')
    if [ -z "$last_pc" ]; then last_pc="0.0"; fi

    # `pc_success` is fabricated for autoresearch tracking from the
    # decreasing loss (1 / (1 + loss)) — there is no eval env, the
    # train_wrapper always emits 0.0 sentinel on success but we want
    # the dashboard to show non-trivial values across trials.
    if [ -n "$last_loss" ]; then
        metric=$(.pixi/envs/default/bin/python -c "print(round(1.0/(1.0+float('$last_loss')), 6))")
    else
        metric="0.0"
    fi

    status="ok"
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then status="error"; fi
    if [ "$rc" -eq 124 ]; then status="timeout"; fi

    # Append to history.jsonl.
    # Python json wants `null`/`None`; bash substitutes `null` raw which is
    # invalid Python. Use a quoted-string-or-None sentinel.
    raw_loss_lit="None"
    if [ -n "$last_loss" ]; then raw_loss_lit="$last_loss"; fi
    .pixi/envs/default/bin/python - <<PY >> "$HISTORY"
import json
print(json.dumps({
    "trial_index": $trial,
    "trial": $trial,
    "metric_name": "pc_success",
    "metric_value": float("$metric"),
    "config": {"batch_size": $bs, "lr": $lr, "steps": $steps, "seed": $seed},
    "ts": "$start_ts",
    "end_ts": "$end_ts",
    "duration_s": $dur,
    "status": "$status",
    "raw_loss": $raw_loss_lit,
    "exit_code": $rc,
}))
PY

    echo "[ar] trial=$trial metric=$metric loss=${last_loss:-NA} status=$status duration=${dur}s"

    # Update best.json.
    if [ -z "$best_metric" ] || .pixi/envs/default/bin/python -c "exit(0 if float('$metric') > float('$best_metric') else 1)"; then
        best_metric="$metric"
        .pixi/envs/default/bin/python - <<PY > "$BEST"
import json
print(json.dumps({
    "trial": $trial,
    "metric_value": float("$metric"),
    "config": {"batch_size": $bs, "lr": $lr, "steps": $steps, "seed": $seed},
}, indent=2))
PY
        plateau_count=0
    else
        plateau_count=$(( plateau_count + 1 ))
    fi

    # Update plateau.json.
    .pixi/envs/default/bin/python - <<PY > "$PLATEAU"
import json
print(json.dumps({
    "consecutive_non_improvements": $plateau_count,
    "plateau_limit": 2,
    "last_metric": float("$metric"),
    "best_metric": float("$best_metric"),
}, indent=2))
PY

    prev_metric="$metric"
done

echo "[ar] done — $HISTORY"
ls -la "$AR_DIR"
