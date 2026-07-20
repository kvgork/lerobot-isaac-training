#!/usr/bin/env bash
# =============================================================================
# run_autoresearch_policy.sh — GENERAL deterministic policy-autoresearch runner.
#
# Replaces the per-arch copy-paste runners (_run_autoresearch_{diffusion,act,...})
# with ONE arch-parametrized dispatcher. The loop skeleton lives in the shared
# claude_code autoresearch skill (run_deterministic.sh) — this file supplies only
# the workspace-specific bits (pixi env, train_wrapper, _open_loop_eval, dataset)
# and the per-arch mutation grid. Same on-disk schema as the LLM loop:
#   .agent-state/<session>/autoresearch/lerobot-policy-<arch>/{program,history,best,plateau}
#
# Usage:
#   bash scripts/run_autoresearch_policy.sh --arch act
#   bash scripts/run_autoresearch_policy.sh --arch diffusion --trials 6
#   TRIALS=8 SECONDS_PER_EXP=2700 bash scripts/run_autoresearch_policy.sh --arch act
#   bash scripts/run_autoresearch_policy.sh --arch act --print-cmd   # dry: echo trial-0 cmd, no train
#
# Add an arch: extend the `case "$ARCH"` grid block below. train/eval are generic
# (train_wrapper --target_arch $ARCH + _open_loop_eval), so most archs need only a grid.
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"
CLAUDE_CODE_ROOT="${CLAUDE_CODE_ROOT:-$HOME/tools/claude_code}"
ENGINE="$CLAUDE_CODE_ROOT/skills/autoresearch/run_deterministic.sh"

ARCH=""; PRINT_CMD=0; TRIALS_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)     ARCH="$2"; shift 2 ;;
    --trials)   TRIALS_OVERRIDE="$2"; shift 2 ;;
    --print-cmd) PRINT_CMD=1; shift ;;
    -h|--help)  sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ARCH" ] || { echo "ERROR: --arch required (act|diffusion|smolvla)" >&2; exit 2; }
[ -f "$ENGINE" ] || { echo "ERROR: engine not found: $ENGINE (set CLAUDE_CODE_ROOT)" >&2; exit 3; }

# --- workspace-specific knobs ------------------------------------------------
PY="$WORKSPACE/.pixi/envs/train-policy/bin/python"
ENV_BIN="$WORKSPACE/.pixi/envs/train-policy/bin"
DATASET="${DATASET:-$WORKSPACE/datasets/kvgork/so101-pickplace1}"
SECONDS_PER_EXP="${SECONDS_PER_EXP:-2700}"
SESSION="${SESSION_ID:-$(date +%Y%m%d-%H%M%S)-autoresearch-$ARCH}"
SEED_OFFSET="${SEED_OFFSET:-0}"   # tail-hop fresh seeds: added to every grid seed

# --- per-arch mutation grid (extend here to add an arch) ---------------------
declare -a CONFIGS
case "$ARCH" in
  diffusion)
    TRIALS="${TRIALS:-3}"
    CONFIGS=(
      '{"trial":0,"batch_size":8,"lr":1e-4,"steps":50000,"seed":42,"op":"baseline","extra":""}'
      '{"trial":1,"batch_size":8,"lr":5e-5,"steps":80000,"seed":1337,"op":"tune_hyperparams:lr_down","extra":""}'
      '{"trial":2,"batch_size":16,"lr":1e-4,"steps":50000,"seed":42,"op":"tune_hyperparams:batch_up","extra":""}'
      '{"trial":3,"batch_size":8,"lr":1e-4,"steps":50000,"seed":42,"op":"add_regularization:weight_decay","extra":"--optimizer.weight_decay=1e-4"}'
      '{"trial":4,"batch_size":8,"lr":3e-4,"steps":50000,"seed":42,"op":"tune_hyperparams:lr_up","extra":""}'
      '{"trial":5,"batch_size":8,"lr":1e-4,"steps":50000,"seed":7,"op":"tune_hyperparams:seed_swap","extra":""}'
    )
    ;;
  act)
    # Grid from programs/lerobot-policy-act.md (baseline lr 3e-4, chunk 50, kl 10;
    # operator #1 = lr; program warns high kl_weight regresses). n_action_steps==chunk_size
    # (lerobot n_action_steps<=chunk_size constraint).
    TRIALS="${TRIALS:-8}"
    CONFIGS=(
      '{"trial":0,"batch_size":4,"lr":3e-4,"steps":30000,"seed":42,"op":"baseline","extra":"--policy.chunk_size=50 --policy.n_action_steps=50 --policy.kl_weight=10"}'
      '{"trial":1,"batch_size":4,"lr":1e-4,"steps":30000,"seed":42,"op":"tune_hyperparams:lr_down","extra":"--policy.chunk_size=50 --policy.n_action_steps=50 --policy.kl_weight=10"}'
      '{"trial":2,"batch_size":4,"lr":5e-4,"steps":30000,"seed":42,"op":"tune_hyperparams:lr_up","extra":"--policy.chunk_size=50 --policy.n_action_steps=50 --policy.kl_weight=10"}'
      '{"trial":3,"batch_size":4,"lr":3e-4,"steps":30000,"seed":42,"op":"tune_hyperparams:chunk_up","extra":"--policy.chunk_size=100 --policy.n_action_steps=100 --policy.kl_weight=10"}'
      '{"trial":4,"batch_size":4,"lr":3e-4,"steps":30000,"seed":42,"op":"tune_hyperparams:kl_up","extra":"--policy.chunk_size=50 --policy.n_action_steps=50 --policy.kl_weight=20"}'
      '{"trial":5,"batch_size":8,"lr":3e-4,"steps":30000,"seed":1337,"op":"tune_hyperparams:batch_up+seed_swap","extra":"--policy.chunk_size=50 --policy.n_action_steps=50 --policy.kl_weight=10"}'
    )
    ;;
  smolvla)
    # SmolVLA needs the VLM backbone loaded (CLAUDE.md pitfall) + frame cache for throughput.
    TRIALS="${TRIALS:-4}"
    CONFIGS=(
      '{"trial":0,"batch_size":4,"lr":1e-4,"steps":20000,"seed":42,"op":"baseline","extra":"--policy.load_vlm_weights=true"}'
      '{"trial":1,"batch_size":4,"lr":5e-5,"steps":20000,"seed":42,"op":"tune_hyperparams:lr_down","extra":"--policy.load_vlm_weights=true"}'
      '{"trial":2,"batch_size":4,"lr":2e-4,"steps":20000,"seed":42,"op":"tune_hyperparams:lr_up","extra":"--policy.load_vlm_weights=true"}'
      '{"trial":3,"batch_size":4,"lr":1e-4,"steps":20000,"seed":7,"op":"tune_hyperparams:seed_swap","extra":"--policy.load_vlm_weights=true"}'
    )
    ;;
  *) echo "ERROR: unsupported --arch '$ARCH' (add a grid in the case block)" >&2; exit 2 ;;
esac
[ -n "$TRIALS_OVERRIDE" ] && TRIALS="$TRIALS_OVERRIDE"

# --- caller contract for the shared engine -----------------------------------
SLUG="lerobot-policy-$ARCH"
AR_DIR="$WORKSPACE/.agent-state/$SESSION/autoresearch/$SLUG"
AR_OUT_ROOT="${AR_OUT_ROOT:-$WORKSPACE/outputs/autoresearch-$SLUG}"   # overridable: tail hops must NOT clobber campaign checkpoints (engine rm -rfs trial dirs)
AR_METRIC_NAME="pc_success"
AR_METRIC_DIR="maximize"
AR_JQ_PY="$WORKSPACE/.pixi/envs/default/bin/python"
AR_PLATEAU_LIMIT=3
AR_PROGRAM_JSON=$(cat <<EOF
{"name":"$SLUG","domain":"lerobot_isaac","metric":{"name":"pc_success","direction":"maximize"},"budget":{"seconds_per_experiment":$SECONDS_PER_EXP,"max_experiments":$TRIALS,"plateau_limit":3},"target_arch":"$ARCH","dataset":"$DATASET","session_id":"$SESSION"}
EOF
)

# Run ONE training trial (generic across policy archs; arch knobs ride in .extra).
ar_run_trial() {
  local cfg="$1" out_dir="$2" iter_log="$3" bs lr steps seed extra save_freq
  bs=$(echo "$cfg" | jq -r .batch_size); lr=$(echo "$cfg" | jq -r .lr)
  steps=$(echo "$cfg" | jq -r .steps); seed=$(echo "$cfg" | jq -r .seed)
  seed=$(( seed + SEED_OFFSET ))
  extra=$(echo "$cfg" | jq -r '.extra // ""')
  # SECONDS_PER_EXP/4 (was *25/30): the old formula assumed >=1 step/s; diffusion
  # runs ~0.8 step/s, so trials timed out ~100 steps short of their FIRST checkpoint
  # (2026-07-17 diff-camp: 6 trials, zero ckpts, zero evals, all sentinel-0.0).
  # /4 guarantees multiple saves even at ~0.3 step/s; extra saves cost seconds.
  save_freq=$(( SECONDS_PER_EXP / 4 )); [ "$save_freq" -lt 200 ] && save_freq=200
  PATH="$ENV_BIN:$PATH" \
  timeout "$SECONDS_PER_EXP" "$PY" -m lerobot_isaac_autoresearch.train_wrapper \
    --target_arch "$ARCH" --dataset "$DATASET" --output_dir "$out_dir" \
    --steps "$steps" --batch_size "$bs" \
    -- --optimizer.lr="$lr" --seed="$seed" --policy.device=cuda \
       --save_freq="$save_freq" --log_freq=50 $extra \
    > "$iter_log" 2>&1
}

# Echo the trial metric (open-loop pc_success proxy) or "" if no checkpoint.
ar_eval_trial() {
  local cfg="$1" out_dir="$2" iter_log="$3" ckpt eval_json trial pc
  trial=$(echo "$cfg" | jq -r .trial)
  ckpt=$(find "$out_dir/checkpoints" -name pretrained_model -type d 2>/dev/null | sort | tail -1)
  { [ -n "$ckpt" ] && [ -d "$ckpt" ]; } || { echo ""; return 0; }
  eval_json="$AR_DIR/trial_${trial}-eval.json"
  "$PY" "$WORKSPACE/scripts/_open_loop_eval.py" --policy_path "$ckpt" \
    --dataset_root "$DATASET" --n_episodes 3 --output_json "$eval_json" \
    --task_label "ar-$ARCH-trial-$trial" --run_id "${SESSION}-trial${trial}" >> "$iter_log" 2>&1 || true
  pc=$(jq -r .pc_success "$eval_json" 2>/dev/null)
  { [ -z "$pc" ] || [ "$pc" = "null" ]; } && pc=""
  echo "$pc"
}

if [ "$PRINT_CMD" = "1" ]; then
  echo "arch=$ARCH slug=$SLUG trials=$TRIALS budget=${SECONDS_PER_EXP}s dataset=$DATASET"
  echo "engine=$ENGINE"
  echo "trial-0 config: ${CONFIGS[0]}"
  bs=$(echo "${CONFIGS[0]}" | jq -r .batch_size); lr=$(echo "${CONFIGS[0]}" | jq -r .lr)
  steps=$(echo "${CONFIGS[0]}" | jq -r .steps); seed=$(echo "${CONFIGS[0]}" | jq -r .seed)
  extra=$(echo "${CONFIGS[0]}" | jq -r '.extra // ""')
  echo "trial-0 cmd: timeout $SECONDS_PER_EXP $PY -m lerobot_isaac_autoresearch.train_wrapper --target_arch $ARCH --dataset $DATASET --output_dir $AR_OUT_ROOT/trial_0 --steps $steps --batch_size $bs -- --optimizer.lr=$lr --seed=$seed --policy.device=cuda --save_freq=<auto> --log_freq=50 $extra"
  exit 0
fi

export SESSION AR_DIR AR_OUT_ROOT AR_METRIC_NAME AR_METRIC_DIR AR_JQ_PY AR_PLATEAU_LIMIT \
  AR_PROGRAM_JSON AR_TRIALS ARCH DATASET PY ENV_BIN SECONDS_PER_EXP WORKSPACE
AR_TRIALS="$TRIALS"
echo "[run_autoresearch_policy] arch=$ARCH trials=$TRIALS budget=${SECONDS_PER_EXP}s → $AR_DIR"
# shellcheck source=/dev/null
source "$ENGINE"
run_autoresearch_loop
