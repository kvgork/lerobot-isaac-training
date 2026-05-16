#!/usr/bin/env bash
# =============================================================================
# _smoke_train.sh — Short timed training probe to measure throughput.
#
# Purpose
# -------
# Before committing to a long training run, run this script to measure the
# actual steps/sec + peak VRAM for the chosen (arch, batch_size) pair on the
# current GPU. The output drives the watchdog budget for the real run.
#
# Usage
# -----
#   bash scripts/_smoke_train.sh                              # smolvla, 5 min
#   bash scripts/_smoke_train.sh --arch diffusion --batch 8   # diffusion smoke
#   bash scripts/_smoke_train.sh --duration-s 180             # 3 min smoke
#   bash scripts/_smoke_train.sh --arch smolvla --batch 2     # OOM rescue probe
#
# Flags
#   --arch ARCH          smolvla | diffusion | act. Default smolvla.
#   --batch N            batch_size. Defaults: diffusion=8, smolvla=4, act=8.
#   --duration-s N       Wall-clock budget in seconds. Default 300 (5 min).
#   --dataset DIR        LeRobotDataset root. Default datasets/kvgork/so101-pickplace1.
#   --run-dir DIR        Output root. Default outputs/smoke-<arch>-<ts>/.
#   --lr LR              learning rate. Default 1e-4.
#   --seed N             seed. Default 42.
#   --num-workers N      dataloader workers. Default 0 (lerobot CLI default).
#                        Bump to 4+ when `data_s` dominates `updt_s` in logs.
#   --cache-frames       Pre-decode every dataset row into RAM at train start.
#                        Adds ~30-60 s warmup but removes PNG decode from every
#                        subsequent step. Expect 3-4x steps/s on PNG-heavy
#                        LeRobotDataset. See approach A in plans/
#                        2026-05-15-dataloader-gpu-decode-plan.md.
#
# Output
# ------
#   <run_dir>/logs/train.log   raw training stdout
#   <run_dir>/logs/gpu.log     gpu_monitor stdout
#   <run_dir>/smoke_report.txt human-readable throughput summary
#
# Exit codes
#   0  smoke finished cleanly (watchdog hit counts as success).
#   1  preflight failed.
#   2  train crashed before reporting any steps (likely config or OOM).
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

ARCH="smolvla"
BATCH=""
DURATION_S=300
DATASET="datasets/kvgork/so101-pickplace1"
RUN_DIR=""
LR="1e-4"
SEED=42
NUM_WORKERS=""
CACHE_FRAMES=0

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | grep "^#" | sed 's/^# \?//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)        ARCH="$2"; shift 2 ;;
        --batch)       BATCH="$2"; shift 2 ;;
        --duration-s)  DURATION_S="$2"; shift 2 ;;
        --dataset)     DATASET="$2"; shift 2 ;;
        --run-dir)     RUN_DIR="$2"; shift 2 ;;
        --lr)          LR="$2"; shift 2 ;;
        --seed)        SEED="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --cache-frames) CACHE_FRAMES=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

# Arch-specific defaults
EXTRA_REMAINDER=()
case "$ARCH" in
    diffusion) : "${BATCH:=8}" ;;
    smolvla)   : "${BATCH:=4}" ; EXTRA_REMAINDER=( "--policy.load_vlm_weights=true" ) ;;
    act)       : "${BATCH:=8}" ;;
    *) echo "unsupported --arch: $ARCH (use diffusion|smolvla|act)" >&2; exit 2 ;;
esac

# Optional num_workers override (passes through as --num_workers=N to lerobot-train)
if [ -n "$NUM_WORKERS" ]; then
    EXTRA_REMAINDER+=( "--num_workers=$NUM_WORKERS" )
fi

[ -z "$RUN_DIR" ] && RUN_DIR="outputs/smoke-$ARCH-$(date +%Y-%m-%d-%H%M%S)"
mkdir -p "$RUN_DIR/logs"
RUN_DIR="$(realpath "$RUN_DIR")"

G='\033[0;32m'; R='\033[0;31m'; C='\033[0;36m'; Y='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${C}[$(date +%H:%M:%S) INFO]${NC} $*"; }
ok()    { echo -e "${G}[$(date +%H:%M:%S)  OK ]${NC} $*"; }
warn()  { echo -e "${Y}[$(date +%H:%M:%S) WARN]${NC} $*"; }
err()   { echo -e "${R}[$(date +%H:%M:%S) ERR ]${NC} $*" >&2; }

# --- preflight --------------------------------------------------------------
[ -d "$DATASET" ] || { err "dataset not found: $DATASET"; exit 1; }
[ -d "$WORKSPACE/.pixi/envs/train-policy" ] || { err "pixi env train-policy missing"; exit 1; }

info "smoke probe: arch=$ARCH batch=$BATCH duration=${DURATION_S}s lr=$LR cache_frames=$CACHE_FRAMES"
info "run_dir: $RUN_DIR"

# Adapter-level cache flag (separate from lerobot-train remainder)
ADAPTER_EXTRA=()
if [ "$CACHE_FRAMES" = 1 ]; then
    ADAPTER_EXTRA+=( "--cache_frames" )
fi

# --- GPU monitor (background) -----------------------------------------------
GPU_METRICS="$RUN_DIR/gpu_metrics.parquet"
"$WORKSPACE/.pixi/envs/default/bin/python" "$WORKSPACE/scripts/_gpu_monitor.py" \
    --output "$GPU_METRICS" \
    --stage smoke \
    --run-id "$(basename "$RUN_DIR")" \
    --interval-s 1 > "$RUN_DIR/logs/gpu.log" 2>&1 &
MON_PID=$!

# --- training run -----------------------------------------------------------
# Buffering note: subshell+redirect+SIGTERM previously dropped stdout when the
# watchdog fired. PYTHONUNBUFFERED=1 + python -u + stdbuf line-buffer flush
# every line so train.log captures everything up to the kill instant.
STAGE_T0=$(date +%s)
export PATH="$WORKSPACE/.pixi/envs/train-policy/bin:$PATH"
export PYTHONUNBUFFERED=1
TRAIN_LOG="$RUN_DIR/logs/train.log"
: > "$TRAIN_LOG"
timeout --signal=SIGTERM "$DURATION_S" \
    stdbuf -oL -eL \
    "$WORKSPACE/.pixi/envs/train-policy/bin/python" -u \
    -m lerobot_isaac_adapters.train \
    --target_arch "$ARCH" \
    --dataset "$DATASET" \
    --output_dir "$RUN_DIR/policy" \
    --steps 1000000 \
    --batch_size "$BATCH" \
    --lr "$LR" \
    "${ADAPTER_EXTRA[@]+"${ADAPTER_EXTRA[@]}"}" \
    --seed "$SEED" \
    -- --policy.device=cuda --save_freq=100000 --log_freq=20 \
    "${EXTRA_REMAINDER[@]+"${EXTRA_REMAINDER[@]}"}" \
    >> "$TRAIN_LOG" 2>&1
RC=$?
DUR=$(( $(date +%s) - STAGE_T0 ))
kill -SIGTERM "$MON_PID" 2>/dev/null; wait "$MON_PID" 2>/dev/null

# --- parse throughput -------------------------------------------------------
# tqdm-style line:  "Training:   0%|...| 235/1000000 [02:37<185:30:15, 1.49step/s]"
# Match last "<rate>step/s" line. step/s OR it/s (lerobot uses both depending on version)
LAST_STEP=$(grep -oE '[0-9]+/1000000' "$RUN_DIR/logs/train.log" | tail -1 | cut -d/ -f1)
LAST_RATE=$(grep -oE '[0-9.]+(step|it)/s\b' "$RUN_DIR/logs/train.log" | tail -1)
# Peak VRAM from gpu_metrics.parquet
PEAK_VRAM=$("$WORKSPACE/.pixi/envs/default/bin/python" - <<PY 2>/dev/null
try:
    import pandas as pd
    df = pd.read_parquet("$GPU_METRICS")
    if "memory_used_mb" in df.columns and len(df):
        print(f"{df['memory_used_mb'].max():.0f}")
    else:
        print("?")
except Exception:
    print("?")
PY
)

# Compose report
REPORT="$RUN_DIR/smoke_report.txt"
{
    echo "=================== SMOKE REPORT ==================="
    echo "arch         : $ARCH"
    echo "batch_size   : $BATCH"
    echo "duration     : ${DUR}s (budget ${DURATION_S}s)"
    echo "exit_code    : $RC ($([ "$RC" -eq 124 ] && echo "watchdog OK" || echo "$([ "$RC" -eq 0 ] && echo "clean" || echo "FAIL")"))"
    echo "last_step    : ${LAST_STEP:-(none)}"
    echo "last_rate    : ${LAST_RATE:-(none)}"
    echo "peak_vram_mb : $PEAK_VRAM"
    echo ""
    if [ -n "${LAST_STEP:-}" ] && [ "$LAST_STEP" -gt 0 ]; then
        # avg throughput across the run, not just last tqdm rate
        AVG_RATE=$("$WORKSPACE/.pixi/envs/default/bin/python" -c "print(f'{$LAST_STEP/$DUR:.2f}')")
        echo "avg_steps/s  : $AVG_RATE   (= last_step / duration)"
        echo ""
        echo "Time projections (using avg_steps/s = $AVG_RATE):"
        for tgt in 5000 10000 20000 30000; do
            HRS=$("$WORKSPACE/.pixi/envs/default/bin/python" -c "print(f'{$tgt/$AVG_RATE/3600:.2f}')")
            MIN=$("$WORKSPACE/.pixi/envs/default/bin/python" -c "print(f'{$tgt/$AVG_RATE/60:.0f}')")
            printf "  %5d steps  →  %sh  (%s min)\n" "$tgt" "$HRS" "$MIN"
        done
    else
        echo "WARNING: no step lines in log — train may have crashed before warmup."
        echo "         inspect $RUN_DIR/logs/train.log"
    fi
    echo "===================================================="
} | tee "$REPORT"

if [ -z "${LAST_STEP:-}" ] || [ "$LAST_STEP" -le 0 ]; then
    exit 2
fi
exit 0
