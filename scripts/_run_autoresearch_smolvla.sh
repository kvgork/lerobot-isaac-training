#!/usr/bin/env bash
# Deterministic autoresearch loop for programs/lerobot-policy-smolvla.md
# Mirrors _run_autoresearch_diffusion.sh but with SmolVLA-specific:
#   - --policy.load_vlm_weights=true in extra remainder (default config has
#     load_vlm_weights=False; without this the VLM backbone stays at random
#     init even though freeze_vision_encoder=True).
#   - narrower lr grid (smolvla hates lr > 1e-4)
#   - batch_size_default=4 (SmolVLM2-500M forward eats ~2GB on its own)
#   - 3600s budget per experiment (smolvla forward heavier than diffusion)
# Same on-disk schema as the LLM-driven orchestrator:
#   .agent-state/<session>/autoresearch/lerobot-policy-smolvla/
#     program.json   history.jsonl   best.json   plateau.json
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

SESSION="${SESSION_ID:-$(date +%Y%m%d-%H%M%S)-autoresearch-smolvla}"
SLUG="lerobot-policy-smolvla"
AR_DIR="$WORKSPACE/.agent-state/$SESSION/autoresearch/$SLUG"
mkdir -p "$AR_DIR"

DATASET="${DATASET:-$WORKSPACE/datasets/kvgork/so101-pickplace1}"
PY="$WORKSPACE/.pixi/envs/train-policy/bin/python"
SECONDS_PER_EXP="${SECONDS_PER_EXP:-3600}"
TRIALS="${TRIALS:-3}"
# Set CACHE_FRAMES=0 to disable the in-RAM cache wrapper (the ~7x speedup
# from plans/2026-05-15-dataloader-gpu-decode-plan.md approach A). Defaults
# to 1 because every trial otherwise wastes 75% of the budget on PNG decode.
CACHE_FRAMES="${CACHE_FRAMES:-1}"
ADAPTER_EXTRA=()
if [ "$CACHE_FRAMES" = 1 ]; then
    ADAPTER_EXTRA+=( "--cache_frames" )
fi
EXTRA_BASE="--policy.load_vlm_weights=true --policy.device=cuda --log_freq=50"

# Preflight: weights cached?
if [ ! -d "$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct" ]; then
    echo "[ar smolvla] WARN: SmolVLM2 weights not cached. Run:" >&2
    echo "  bash scripts/_run_smolvla_tonight.sh --prefetch-weights" >&2
    echo "first, or the first trial wastes ~5min on download inside the watchdog." >&2
fi

cat > "$AR_DIR/program.json" <<EOF
{
  "name": "lerobot-policy-smolvla",
  "domain": "lerobot_isaac",
  "metric": {"name": "pc_success", "direction": "maximize"},
  "budget": {"seconds_per_experiment": $SECONDS_PER_EXP, "max_experiments": $TRIALS, "plateau_limit": 3},
  "target_arch": "smolvla",
  "dataset": "datasets/kvgork/so101-pickplace1",
  "session_id": "$SESSION"
}
EOF

: > "$AR_DIR/history.jsonl"
best_metric=""
plateau=0

# Grounded mutation grid for SmolVLA. Operators from programs/_domain_knowledge.md
# § operators, narrowed for SmolVLA's lr-sensitivity. Baseline lr=3e-5 per
# the program's mutation_hints. Each trial exercises a different operator.
#
# steps: tuned 2026-05-16 from 10k → 50k after the first overnight run
# revealed trials finished in ~17 min instead of using the 100-min budget.
# 50k steps × 10 step/s (with disk-cached dataset) ≈ 83 min per trial → fills
# the SECONDS_PER_EXP=6000 budget cleanly.
declare -a CONFIGS=(
  '{"trial":0,"batch_size":4,"lr":3e-5,"steps":50000,"seed":42,"op":"baseline","extra":""}'
  '{"trial":1,"batch_size":4,"lr":1e-5,"steps":50000,"seed":1337,"op":"tune_hyperparams:lr_down","extra":""}'
  '{"trial":2,"batch_size":4,"lr":5e-5,"steps":50000,"seed":42,"op":"tune_hyperparams:lr_up","extra":""}'
  '{"trial":3,"batch_size":2,"lr":3e-5,"steps":50000,"seed":42,"op":"tune_hyperparams:batch_down","extra":""}'
  '{"trial":4,"batch_size":4,"lr":3e-5,"steps":50000,"seed":42,"op":"add_regularization:weight_decay","extra":"--optimizer.weight_decay=1e-4"}'
  '{"trial":5,"batch_size":4,"lr":3e-5,"steps":50000,"seed":7,"op":"tune_hyperparams:seed_swap","extra":""}'
  # 2026-05-16 additions — informed by rescue eval (lr=1e-5 beat baseline,
  # batch=2 lost). Test halfway lr + bigger batch with cached dataloader.
  '{"trial":6,"batch_size":4,"lr":2e-5,"steps":50000,"seed":42,"op":"tune_hyperparams:lr_mid","extra":""}'
  '{"trial":7,"batch_size":8,"lr":3e-5,"steps":50000,"seed":42,"op":"tune_hyperparams:batch_up","extra":""}'
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

    echo "[ar smolvla trial=$trial] bs=$bs lr=$lr steps=$steps seed=$seed op=$op"
    iter_log="$AR_DIR/trial_${trial}.log"
    start_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    start_s=$(date +%s)
    save_freq=$(( SECONDS_PER_EXP * 25 / 30 ))
    [ "$save_freq" -lt 200 ] && save_freq=200

    PATH="$WORKSPACE/.pixi/envs/train-policy/bin:$PATH" \
    timeout "$SECONDS_PER_EXP" "$PY" -m lerobot_isaac_autoresearch.train_wrapper \
        --target_arch smolvla \
        --dataset "$DATASET" \
        --output_dir "$out_dir" \
        --steps "$steps" \
        --batch_size "$bs" \
        "${ADAPTER_EXTRA[@]+"${ADAPTER_EXTRA[@]}"}" \
        -- --optimizer.lr="$lr" --seed="$seed" --save_freq="$save_freq" $EXTRA_BASE $extra \
        > "$iter_log" 2>&1
    rc=$?
    end_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    dur=$(( $(date +%s) - start_s ))

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
            --task_label "autoresearch-smolvla-trial-$trial" \
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

    echo "[ar smolvla trial=$trial] pc=$pc loss=$raw_loss mse=${mse:-NA} dur=${dur}s status=$status"

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

echo "[ar smolvla] DONE — $AR_DIR"
ls -la "$AR_DIR"
