#!/usr/bin/env bash
# =============================================================================
# _run_smolvla_tonight.sh — Single-arch SmolVLA training run for the SO-101
# kvgork/so101-pickplace1 dataset. Mirrors run_full_pipeline.sh policy_train
# stage but with target_arch=smolvla, smaller batch_size, separate log dir.
#
# SmolVLA defaults (lerobot 0.5.1):
#   freeze_vision_encoder=True  train_expert_only=True
#   → only the expert head trains; SmolVLM2-500M backbone stays frozen.
# This is the cheap RTX-3080-compatible config. Batch 4 chosen vs diffusion's 8
# because SmolVLM forward pass uses ~2 GB on its own.
#
# Usage
# -----
#   bash scripts/_run_smolvla_tonight.sh                          # 30 min watchdog
#   bash scripts/_run_smolvla_tonight.sh --train-minutes 120      # 2h watchdog
#   bash scripts/_run_smolvla_tonight.sh --dataset DIR --batch 2  # smaller batch
#   bash scripts/_run_smolvla_tonight.sh --dry-run                # print cmd only
#
# Flags
#   --train-minutes N   Watchdog cap in minutes. Default 30.
#   --dataset DIR       LeRobotDataset root. Default datasets/kvgork/so101-pickplace1.
#   --run-dir DIR       Output root. Default outputs/smolvla-<ts>/.
#   --batch N           batch_size. Default 4.
#   --lr LR             learning rate. Default 1e-4 (matches SmolVLAConfig).
#   --seed N            seed. Default 42.
#   --no-pretrained     Skip loading the SmolVLM2 backbone from HF
#                       (--policy.load_vlm_weights=false). Default loads
#                       HuggingFaceTB/SmolVLM2-500M-Video-Instruct (frozen).
#   --finetune PATH     Resume from a SmolVLA checkpoint dir
#                       (--policy.pretrained_path=PATH). Mutually exclusive
#                       with --no-pretrained.
#   --prefetch-weights  Download SmolVLM2-500M-Video-Instruct from HF hub to
#                       the local cache and exit. Run this BEFORE the GPU is
#                       free so the actual training run doesn't waste budget
#                       on network IO. Idempotent.
#   --cache-frames      Pre-decode every dataset row into RAM at train start.
#                       Adds ~16 min warmup (parallel 4-worker decode for
#                       7.5k rows) but jumps per-step rate ~7× by removing
#                       PNG decode from the inner loop. Strongly recommended
#                       for runs >30 min. See approach A in plans/
#                       2026-05-15-dataloader-gpu-decode-plan.md.
#   --no-cache-frames   Disable the cache (default — set explicitly here so
#                       --cache-frames can be turned off again on a chain).
#   --dry-run           Print resolved subprocess command and exit 0.
#
# Output layout
#   <run_dir>/
#     logs/policy_train.log          training stdout/stderr
#     logs/policy_train_gpu.log      gpu_monitor stdout
#     logs/preflight.log             pre-flight + resolved config
#     policy-smolvla/checkpoints/    lerobot checkpoints
#     system_metrics/policy_train/   gpu_metrics.parquet
#
# Exit codes
#   0  training exited cleanly OR hit watchdog timeout (rc=124 normalized to 0).
#   1  preflight failed or training crashed.
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

TRAIN_MINUTES=30
DATASET="datasets/kvgork/so101-pickplace1"
RUN_DIR="outputs/smolvla-$(date +%Y-%m-%d-%H%M%S)"
BATCH=4
LR="1e-4"
SEED=42
DRY_RUN=0
LOAD_PRETRAINED=1
FINETUNE_PATH=""
PREFETCH=0
CACHE_FRAMES=0

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | grep "^#" | sed 's/^# \?//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-minutes)  TRAIN_MINUTES="$2"; shift 2 ;;
        --dataset)        DATASET="$2"; shift 2 ;;
        --run-dir)        RUN_DIR="$2"; shift 2 ;;
        --batch)          BATCH="$2"; shift 2 ;;
        --lr)             LR="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --no-pretrained)  LOAD_PRETRAINED=0; shift ;;
        --finetune)       FINETUNE_PATH="$2"; shift 2 ;;
        --prefetch-weights) PREFETCH=1; shift ;;
        --cache-frames)   CACHE_FRAMES=1; shift ;;
        --no-cache-frames) CACHE_FRAMES=0; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

TRAIN_SECONDS=$(( TRAIN_MINUTES * 60 ))
mkdir -p "$RUN_DIR/logs"
RUN_DIR="$(realpath "$RUN_DIR")"

G='\033[0;32m'; R='\033[0;31m'; C='\033[0;36m'; Y='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${C}[$(date +%H:%M:%S) INFO]${NC} $*"; }
ok()    { echo -e "${G}[$(date +%H:%M:%S)  OK ]${NC} $*"; }
warn()  { echo -e "${Y}[$(date +%H:%M:%S) WARN]${NC} $*"; }
err()   { echo -e "${R}[$(date +%H:%M:%S) ERR ]${NC} $*" >&2; }

# --- preflight --------------------------------------------------------------
info "── STAGE: preflight ──"
{
    set -e
    [ -d "$DATASET" ] || { err "dataset not found: $DATASET"; exit 2; }
    [ -d "$WORKSPACE/.pixi/envs/train-policy" ] || { err "pixi env 'train-policy' missing"; exit 2; }

    "$WORKSPACE/.pixi/envs/train-policy/bin/python" -c \
        "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy" \
        || { err "SmolVLA not importable in train-policy env"; exit 2; }

    # Warn if SmolVLM2 weights aren't cached — first launch will download ~2GB
    if [ ! -d "$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct" ]; then
        echo "WARN: SmolVLM2-500M-Video-Instruct not in HF cache. First train"
        echo "      step will block ~2GB download. Run --prefetch-weights first"
        echo "      to warm the cache."
    fi

    echo "arch          : smolvla"
    echo "dataset       : $DATASET"
    echo "run_dir       : $RUN_DIR"
    echo "train mins    : $TRAIN_MINUTES (watchdog = ${TRAIN_SECONDS}s)"
    echo "batch_size    : $BATCH"
    echo "lr            : $LR"
    echo "seed          : $SEED"
    echo "pretrained vlm: $LOAD_PRETRAINED"
    echo "finetune path : ${FINETUNE_PATH:-(none)}"
    echo "cache_frames  : $CACHE_FRAMES"
    echo "dry_run       : $DRY_RUN"
} > "$RUN_DIR/logs/preflight.log" 2>&1
PRE_RC=$?
cat "$RUN_DIR/logs/preflight.log"
[ "$PRE_RC" -eq 0 ] || { err "preflight failed"; exit 1; }
ok "preflight done"

# --- policy train -----------------------------------------------------------
POLICY_DIR="$RUN_DIR/policy-smolvla"
rm -rf "$POLICY_DIR"
SAVE_FREQ=$(( TRAIN_SECONDS * 25 / 10 / 3 ))
[ "$SAVE_FREQ" -lt 100 ] && SAVE_FREQ=100

# Build remainder. SmolVLA recipe in lerobot 0.5.1:
#   --policy.load_vlm_weights=true   pulls SmolVLM2-500M-Video-Instruct from HF
#                                    (default config has this False, which leaves
#                                    the VLM with random init — useless).
#   defaults to freeze_vision_encoder=True + train_expert_only=True so only the
#   action expert trains; VLM is frozen pretrained weights.
#   --policy.pretrained_path=DIR     resume from a SmolVLA checkpoint dir.
REMAINDER=( "--policy.device=cuda" "--save_freq=$SAVE_FREQ" "--log_freq=50" )
if [ -n "$FINETUNE_PATH" ]; then
    [ -d "$FINETUNE_PATH" ] || { err "--finetune path not a dir: $FINETUNE_PATH"; exit 2; }
    REMAINDER+=( "--policy.pretrained_path=$FINETUNE_PATH" )
elif [ "$LOAD_PRETRAINED" = 1 ]; then
    REMAINDER+=( "--policy.load_vlm_weights=true" )
fi

ADAPTER_EXTRA=()
if [ "$CACHE_FRAMES" = 1 ]; then
    ADAPTER_EXTRA+=( "--cache_frames" )
fi

CMD=(
    "$WORKSPACE/.pixi/envs/train-policy/bin/python" -m lerobot_isaac_adapters.train
    --target_arch smolvla
    --dataset "$DATASET"
    --output_dir "$POLICY_DIR"
    --steps 1000000
    --batch_size "$BATCH"
    --lr "$LR"
    --seed "$SEED"
    "${ADAPTER_EXTRA[@]+"${ADAPTER_EXTRA[@]}"}"
    --
    "${REMAINDER[@]}"
)

if [ "$PREFETCH" = 1 ]; then
    info "── PREFETCH weights ──"
    "$WORKSPACE/.pixi/envs/train-policy/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
import sys
repo = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
print(f"[prefetch] downloading {repo} …", flush=True)
try:
    p = snapshot_download(repo_id=repo)
    print(f"[prefetch] cached at {p}")
except Exception as exc:
    print(f"[prefetch] FAILED: {exc}", file=sys.stderr)
    sys.exit(1)
PY
    rc=$?
    [ "$rc" -eq 0 ] && ok "weights cached" || { err "prefetch failed"; exit "$rc"; }
    exit 0
fi

if [ "$DRY_RUN" = 1 ]; then
    info "── DRY-RUN ──"
    # --dry_run is an adapter-level flag — must land BEFORE the -- separator so
    # it isn't forwarded to lerobot-train. Splice it in just before "--".
    DRY_CMD=()
    for tok in "${CMD[@]}"; do
        if [ "$tok" = "--" ]; then
            DRY_CMD+=( --dry_run "--" )
        else
            DRY_CMD+=( "$tok" )
        fi
    done
    printf '%q ' "${DRY_CMD[@]}"; echo
    "${DRY_CMD[@]}"
    exit $?
fi

info "── STAGE: policy_train (smolvla) ──"
GPU_METRICS="$RUN_DIR/system_metrics/policy_train/gpu_metrics.parquet"
mkdir -p "$(dirname "$GPU_METRICS")"
"$WORKSPACE/.pixi/envs/default/bin/python" "$WORKSPACE/scripts/_gpu_monitor.py" \
    --output "$GPU_METRICS" \
    --stage policy_train \
    --run-id "$(basename "$RUN_DIR")" \
    --interval-s 2 > "$RUN_DIR/logs/policy_train_gpu.log" 2>&1 &
MON_PID=$!

STAGE_T0=$(date +%s)
(
    export PATH="$WORKSPACE/.pixi/envs/train-policy/bin:$PATH"
    timeout --signal=SIGTERM "$TRAIN_SECONDS" "${CMD[@]}"
) > "$RUN_DIR/logs/policy_train.log" 2>&1
rc=$?
kill -SIGTERM "$MON_PID" 2>/dev/null; wait "$MON_PID" 2>/dev/null
DUR=$(( $(date +%s) - STAGE_T0 ))
# rc=124 → watchdog timeout, expected for budget-capped runs
[ "$rc" -eq 124 ] && { warn "watchdog hit ${TRAIN_SECONDS}s — treating as OK"; rc=0; }

if [ "$rc" -eq 0 ]; then
    ok "policy_train finished in ${DUR}s — checkpoints under $POLICY_DIR/checkpoints/"
else
    err "policy_train FAILED rc=$rc after ${DUR}s — see $RUN_DIR/logs/policy_train.log"
fi

echo
echo "================== RUN SUMMARY =================="
echo "run_dir : $RUN_DIR"
echo "arch    : smolvla"
echo "duration: ${DUR}s (cap ${TRAIN_SECONDS}s)"
echo "rc      : $rc"
echo "log     : $RUN_DIR/logs/policy_train.log"
echo "ckpts   : $POLICY_DIR/checkpoints/"
exit "$rc"
