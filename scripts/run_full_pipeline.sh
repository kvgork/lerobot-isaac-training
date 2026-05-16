#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh — Single-command end-to-end pipeline.
#
# Stages, in order:
#   1. Verify dataset + envs.
#   2. Generate synthetic data via Isaac DR replay.
#   3. Train LeRobot policy (timed).
#   4. Train DreamerV3 world model (timed).
#   5. Open-loop action-MSE evaluation on held-out source episodes.
#   6. Refresh dashboard report + snapshot.
#
# Usage
# -----
#   bash scripts/run_full_pipeline.sh                          # 30 min/train, diffusion
#   bash scripts/run_full_pipeline.sh --train-minutes 15       # 15 min/train
#   bash scripts/run_full_pipeline.sh --skip-synthetic         # use existing
#   bash scripts/run_full_pipeline.sh --dataset DIR            # alt dataset
#   bash scripts/run_full_pipeline.sh --target-arch smolvla    # SmolVLA instead of diffusion
#   bash scripts/run_full_pipeline.sh --target-arch act        # ACT
#
# Flags
#   --train-minutes N   Per-training watchdog cap. Default 30.
#   --n-synthetic N     Source episodes × variants for DR replay. Default 3 × 2.
#   --dataset DIR       Real LeRobotDataset root. Default datasets/kvgork/so101-pickplace1.
#   --run-dir DIR       Output root. Default outputs/full-pipeline-<ts>/.
#   --target-arch ARCH  Policy arch: diffusion | smolvla | act. Default diffusion.
#                       Drives output_dir name (policy-<arch>), batch_size default,
#                       and arch-specific remainder args (smolvla adds
#                       --policy.load_vlm_weights=true so the frozen VLM backbone
#                       loads pretrained weights instead of random init).
#   --policy-batch N    Override policy batch_size. Defaults: diffusion=8,
#                       smolvla=4, act=8. Drop to 2 on OOM.
#   --cache-frames      Enable in-RAM dataset cache (approach A). Adds ~16 min
#                       warmup but lifts SmolVLA throughput ~7x by removing
#                       PNG decode from the inner loop. Recommended for any
#                       single training run > 30 min. See plans/2026-05-15-
#                       dataloader-gpu-decode-plan.md.
#   --skip-synthetic    Reuse existing synthetic dataset at <run_dir>/synthetic/.
#   --skip-policy       Skip policy training.
#   --skip-worldmodel   Skip world-model training.
#   --skip-eval         Skip evaluation.
#   --skip-dashboard    Skip dashboard refresh.
#
# Exit codes
#   0  every stage that was attempted exited cleanly.
#   1  one or more stages failed; see stage logs under <run_dir>/logs/.
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

# --- defaults ---------------------------------------------------------------
TRAIN_MINUTES=30
N_SOURCE_EPISODES=3
N_VARIANTS=2
DATASET="datasets/kvgork/so101-pickplace1"
RUN_DIR="outputs/full-pipeline-$(date +%Y-%m-%d-%H%M%S)"
TARGET_ARCH="diffusion"
POLICY_BATCH=""           # resolved from TARGET_ARCH if empty
CACHE_FRAMES=0
SKIP_SYNTHETIC=0
SKIP_POLICY=0
SKIP_WORLDMODEL=0
SKIP_EVAL=0
SKIP_DASHBOARD=0

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | grep "^#" | sed 's/^# \?//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-minutes)    TRAIN_MINUTES="$2"; shift 2 ;;
        --n-synthetic)      N_SOURCE_EPISODES="$2"; shift 2 ;;
        --dataset)          DATASET="$2"; shift 2 ;;
        --run-dir)          RUN_DIR="$2"; shift 2 ;;
        --target-arch)      TARGET_ARCH="$2"; shift 2 ;;
        --policy-batch)     POLICY_BATCH="$2"; shift 2 ;;
        --cache-frames)     CACHE_FRAMES=1; shift ;;
        --skip-synthetic)   SKIP_SYNTHETIC=1; shift ;;
        --skip-policy)      SKIP_POLICY=1; shift ;;
        --skip-worldmodel)  SKIP_WORLDMODEL=1; shift ;;
        --skip-eval)        SKIP_EVAL=1; shift ;;
        --skip-dashboard)   SKIP_DASHBOARD=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        *)                  echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

TRAIN_SECONDS=$(( TRAIN_MINUTES * 60 ))
mkdir -p "$RUN_DIR/logs"
RUN_DIR="$(realpath "$RUN_DIR")"

# --- target-arch resolution -------------------------------------------------
case "$TARGET_ARCH" in
    diffusion) : "${POLICY_BATCH:=8}"  ; POLICY_EXTRA_REMAINDER=() ;;
    smolvla)   : "${POLICY_BATCH:=4}"  ; POLICY_EXTRA_REMAINDER=( "--policy.load_vlm_weights=true" ) ;;
    act)       : "${POLICY_BATCH:=8}"  ; POLICY_EXTRA_REMAINDER=() ;;
    *) echo "unsupported --target-arch: $TARGET_ARCH (use diffusion|smolvla|act)" >&2; exit 2 ;;
esac

# --- color helpers ----------------------------------------------------------
G='\033[0;32m'; R='\033[0;31m'; C='\033[0;36m'; Y='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${C}[$(date +%H:%M:%S) INFO]${NC} $*"; }
ok()    { echo -e "${G}[$(date +%H:%M:%S)  OK ]${NC} $*"; }
warn()  { echo -e "${Y}[$(date +%H:%M:%S) WARN]${NC} $*"; }
err()   { echo -e "${R}[$(date +%H:%M:%S) ERR ]${NC} $*" >&2; }

# --- stage state ------------------------------------------------------------
declare -A STAGE_STATUS
declare -A STAGE_DURATION

_stage_begin() { STAGE_T0=$(date +%s); info "── STAGE: $1 ──"; }
_stage_end() {
    local name="$1" rc="$2"
    local dur=$(( $(date +%s) - STAGE_T0 ))
    STAGE_DURATION[$name]="$dur"
    if [ "$rc" -eq 0 ]; then
        STAGE_STATUS[$name]="OK"
        ok "$name finished in ${dur}s"
    else
        STAGE_STATUS[$name]="FAIL(rc=$rc)"
        err "$name FAILED (rc=$rc) after ${dur}s — see $RUN_DIR/logs/$name.log"
    fi
}

# --- 1. preflight -----------------------------------------------------------
_stage_begin preflight
(
    set -e
    [ -d "$DATASET" ] || { err "dataset not found: $DATASET"; exit 2; }
    [ -d "$WORKSPACE/.pixi/envs/sim" ] || { err "pixi env 'sim' missing"; exit 2; }
    [ -d "$WORKSPACE/.pixi/envs/train-policy" ] || { err "pixi env 'train-policy' missing"; exit 2; }
    [ -d "$WORKSPACE/.pixi/envs/train-dreamer" ] || { err "pixi env 'train-dreamer' missing"; exit 2; }
    [ -d "$WORKSPACE/.pixi/envs/dashboard" ] || { err "pixi env 'dashboard' missing"; exit 2; }
    echo "dataset      : $DATASET"
    echo "run_dir      : $RUN_DIR"
    echo "target_arch  : $TARGET_ARCH"
    echo "policy_batch : $POLICY_BATCH"
    echo "extra remndr : ${POLICY_EXTRA_REMAINDER[*]:-(none)}"
    echo "train mins   : $TRAIN_MINUTES (watchdog = ${TRAIN_SECONDS}s)"
    echo "synthetic    : ${N_SOURCE_EPISODES} source eps × ${N_VARIANTS} variants"
) > "$RUN_DIR/logs/preflight.log" 2>&1
_stage_end preflight $?
cat "$RUN_DIR/logs/preflight.log" | tail -10
[ "${STAGE_STATUS[preflight]}" = "OK" ] || exit 1

# --- 2. synthetic data generation -------------------------------------------
SYNTH_DIR="$RUN_DIR/synthetic"
if [ "$SKIP_SYNTHETIC" = 1 ]; then
    info "skipping synthetic stage"
    STAGE_STATUS[synthetic]="SKIP"
else
    _stage_begin synthetic
    PYTHONNOUSERSITE=1 "$WORKSPACE/.pixi/envs/sim/bin/python" \
        -m lerobot_isaac_synthetic.isaac_dr.replay_runner \
        --source_dataset "$DATASET" \
        --n_variants "$N_VARIANTS" \
        --task pick \
        --output_path "$SYNTH_DIR" \
        --max_episodes "$N_SOURCE_EPISODES" \
        > "$RUN_DIR/logs/synthetic.log" 2>&1
    _stage_end synthetic $?
    # Surface into datasets/ so dashboard loaders pick it up.
    ln -sfn "$SYNTH_DIR" "$WORKSPACE/datasets/synthetic/$(basename "$RUN_DIR")"
    [ -d "$WORKSPACE/datasets/synthetic" ] || mkdir -p "$WORKSPACE/datasets/synthetic"
fi

# --- 3. policy training (timed) ---------------------------------------------
POLICY_DIR="$RUN_DIR/policy-$TARGET_ARCH"
if [ "$SKIP_POLICY" = 1 ]; then
    info "skipping policy training stage"
    STAGE_STATUS[policy_train]="SKIP"
else
    _stage_begin policy_train
    rm -rf "$POLICY_DIR"
    POLICY_SAVE_FREQ=$(( TRAIN_SECONDS * 25 / 10 / 3 ))
    [ "$POLICY_SAVE_FREQ" -lt 100 ] && POLICY_SAVE_FREQ=100

    # GPU monitor — runs in background, terminated on SIGTERM at stage end.
    GPU_METRICS="$RUN_DIR/system_metrics/policy_train/gpu_metrics.parquet"
    mkdir -p "$(dirname "$GPU_METRICS")"
    "$WORKSPACE/.pixi/envs/default/bin/python" "$WORKSPACE/scripts/_gpu_monitor.py" \
        --output "$GPU_METRICS" \
        --stage policy_train \
        --run-id "$(basename "$RUN_DIR")" \
        --interval-s 2 > "$RUN_DIR/logs/policy_train_gpu.log" 2>&1 &
    MON_PID=$!

    info "policy_train: arch=$TARGET_ARCH batch=$POLICY_BATCH cache=$CACHE_FRAMES extra=${POLICY_EXTRA_REMAINDER[*]:-(none)}"
    CACHE_ADAPTER_ARG=()
    [ "$CACHE_FRAMES" = 1 ] && CACHE_ADAPTER_ARG+=( "--cache_frames" )
    (
        export PATH="$WORKSPACE/.pixi/envs/train-policy/bin:$PATH"
        timeout --signal=SIGTERM "$TRAIN_SECONDS" \
            "$WORKSPACE/.pixi/envs/train-policy/bin/python" -m lerobot_isaac_adapters.train \
            --target_arch "$TARGET_ARCH" \
            --dataset "$DATASET" \
            --output_dir "$POLICY_DIR" \
            --steps 1000000 \
            --batch_size "$POLICY_BATCH" \
            --lr 1e-4 \
            --seed 42 \
            "${CACHE_ADAPTER_ARG[@]+"${CACHE_ADAPTER_ARG[@]}"}" \
            -- --policy.device=cuda --save_freq="$POLICY_SAVE_FREQ" --log_freq=50 \
            "${POLICY_EXTRA_REMAINDER[@]}"
    ) > "$RUN_DIR/logs/policy_train.log" 2>&1
    rc=$?
    kill -SIGTERM $MON_PID 2>/dev/null; wait $MON_PID 2>/dev/null
    [ "$rc" -eq 124 ] && rc=0
    _stage_end policy_train $rc
fi

# --- 4. world-model training (timed) ----------------------------------------
WM_DIR="$RUN_DIR/wm-dreamerv3"
if [ "$SKIP_WORLDMODEL" = 1 ]; then
    info "skipping world-model training stage"
    STAGE_STATUS[wm_train]="SKIP"
else
    _stage_begin wm_train
    # Need a 64x64 DreamerV3-shape HDF5 first — bridge the source dataset
    # via the lerobot_world_model_bridge skill into <run_dir>/bridge/.
    BRIDGE_DIR="$RUN_DIR/bridge"
    mkdir -p "$BRIDGE_DIR"
    HDF5_OUT="$BRIDGE_DIR/dreamerv3_data.hdf5"
    PYTHONPATH="${CLAUDE_CODE_ROOT:-/home/koen/tools/claude_code}:${PYTHONPATH:-}" \
        "$WORKSPACE/.pixi/envs/default/bin/python" - <<PY > "$RUN_DIR/logs/bridge.log" 2>&1
from skills.lerobot_world_model_bridge.operations import lerobot_to_worldmodel
r = lerobot_to_worldmodel(
    dataset_path="$DATASET",
    output_path="$HDF5_OUT",
    output_format="hdf5",
    image_size=(64, 64),
    window_size=16,
    stride=8,
    normalize_actions=True,
)
print("bridge:", r.success, r.error or "ok", r.data)
PY
    bridge_rc=$?
    if [ "$bridge_rc" -ne 0 ]; then
        _stage_end wm_train $bridge_rc
    else
        rm -rf "$WM_DIR"

        # GPU monitor for the WM run
        GPU_METRICS_WM="$RUN_DIR/system_metrics/wm_train/gpu_metrics.parquet"
        mkdir -p "$(dirname "$GPU_METRICS_WM")"
        "$WORKSPACE/.pixi/envs/default/bin/python" "$WORKSPACE/scripts/_gpu_monitor.py" \
            --output "$GPU_METRICS_WM" \
            --stage wm_train \
            --run-id "$(basename "$RUN_DIR")" \
            --interval-s 2 > "$RUN_DIR/logs/wm_train_gpu.log" 2>&1 &
        MON_PID=$!

        timeout --signal=SIGTERM "$TRAIN_SECONDS" \
            "$WORKSPACE/.pixi/envs/train-dreamer/bin/python" -m lerobot_isaac_adapters.train \
            --target_arch dreamerv3 \
            --dataset "$HDF5_OUT" \
            --output_dir "$WM_DIR" \
            --steps 1000000 \
            --batch_size 8 \
            --lr 1e-4 \
            --seed 42 \
            -- env.capture_video=False fabric.accelerator=gpu fabric.devices=1 \
            > "$RUN_DIR/logs/wm_train.log" 2>&1
        rc=$?
        kill -SIGTERM $MON_PID 2>/dev/null; wait $MON_PID 2>/dev/null
        [ "$rc" -eq 124 ] && rc=0
        _stage_end wm_train $rc
    fi
fi

# --- 5. evaluation ----------------------------------------------------------
if [ "$SKIP_EVAL" = 1 ]; then
    info "skipping eval stage"
    STAGE_STATUS[eval]="SKIP"
else
    _stage_begin eval
    EVAL_JSON="$WORKSPACE/outputs/eval/$(basename "$RUN_DIR")-policy.json"
    POLICY_CKPT="$POLICY_DIR/checkpoints/last/pretrained_model"
    if [ ! -d "$POLICY_CKPT" ]; then
        # Fallback to the last numbered checkpoint if `last/` symlink missing.
        POLICY_CKPT=$(find "$POLICY_DIR/checkpoints" -maxdepth 2 -name pretrained_model -type d 2>/dev/null | sort | tail -1)
    fi
    if [ -z "${POLICY_CKPT:-}" ] || [ ! -d "$POLICY_CKPT" ]; then
        warn "no policy checkpoint found under $POLICY_DIR/checkpoints — eval skipped"
        STAGE_STATUS[eval]="SKIP(no_ckpt)"
        STAGE_DURATION[eval]=0
    else
        mkdir -p "$(dirname "$EVAL_JSON")"
        "$WORKSPACE/.pixi/envs/train-policy/bin/python" \
            "$WORKSPACE/scripts/_open_loop_eval.py" \
            --policy_path "$POLICY_CKPT" \
            --dataset_root "$DATASET" \
            --n_episodes 3 \
            --output_json "$EVAL_JSON" \
            --task_label "$(basename "$DATASET")-open-loop-mse" \
            --run_id "$(basename "$RUN_DIR")" \
            > "$RUN_DIR/logs/eval.log" 2>&1
        _stage_end eval $?
        echo "eval json: $EVAL_JSON"
    fi
fi

# --- 6. dashboard refresh ---------------------------------------------------
if [ "$SKIP_DASHBOARD" = 1 ]; then
    info "skipping dashboard refresh"
    STAGE_STATUS[dashboard]="SKIP"
else
    _stage_begin dashboard
    "$WORKSPACE/.pixi/envs/dashboard/bin/python" -m lerobot_isaac_dashboard.report \
        --workspace "$WORKSPACE" \
        --output-dir "$RUN_DIR/dashboard" \
        > "$RUN_DIR/logs/dashboard.log" 2>&1
    rc=$?
    "$WORKSPACE/.pixi/envs/dashboard/bin/python" -m lerobot_isaac_dashboard.snapshots save \
        --workspace "$WORKSPACE" \
        --label "$(basename "$RUN_DIR")" \
        >> "$RUN_DIR/logs/dashboard.log" 2>&1
    _stage_end dashboard $rc
fi

# --- summary ---------------------------------------------------------------
echo
echo "================== RUN SUMMARY =================="
echo "run_dir: $RUN_DIR"
echo
printf '%-16s %-14s %s\n' "stage" "status" "duration"
echo "--------------------------------------------------"
overall_rc=0
for stage in preflight synthetic policy_train wm_train eval dashboard; do
    status="${STAGE_STATUS[$stage]:-NOT_RUN}"
    dur="${STAGE_DURATION[$stage]:-0}s"
    printf '%-16s %-14s %s\n' "$stage" "$status" "$dur"
    case "$status" in OK|SKIP*) ;; *) overall_rc=1 ;; esac
done
echo
if [ "$overall_rc" -eq 0 ]; then
    ok "pipeline complete — all attempted stages succeeded"
    ok "report:    $RUN_DIR/dashboard/report.html"
    ok "live dashboard URL: http://localhost:8501 (run \`pixi run -e dashboard dashboard\` if not running)"
else
    err "pipeline FAILED — inspect $RUN_DIR/logs/*.log"
fi
exit "$overall_rc"
