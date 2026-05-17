#!/usr/bin/env bash
# =============================================================================
# _run_tonight_smolvla_12h.sh — Overnight SmolVLA pipeline (12-h budget).
#
# Anchor + AR + auto re-rank + dashboard, in one launch. AR may plateau-stop
# or natural-exit early; this script fills the freed budget by running the
# full re-rank eval on EVERY checkpoint (anchor + 8 AR trials) and writing
# a winner summary so the morning hand-off is one file lookup.
#
# Stages
# ------
#   STAGE 1  ANCHOR        90  min   (skippable, default skip — re-uses 2026-05-15 anchor)
#   STAGE 2  AR SWEEP     ≤720 min   8 trials × 50k steps × disk-cached dataloader
#   STAGE 3  RE-RANK        ≤60 min  open-loop eval EVERY ckpt → ranked JSON
#   STAGE 4  DASHBOARD       5 min   static report + snapshot
#
# Budget: launch at any hour; finishes within 12 h or earlier on plateau-stop.
#
# Usage
# -----
#   bash scripts/_run_tonight_smolvla_12h.sh                # default (no anchor)
#   bash scripts/_run_tonight_smolvla_12h.sh --with-anchor  # add 90-min fresh anchor
#   bash scripts/_run_tonight_smolvla_12h.sh --dry-run      # print steps only
#
# Flags
#   --with-anchor          Train a fresh anchor (default: skip and reuse the
#                          2026-05-15 anchor at outputs/overnight-smolvla-*-anchor/).
#   --anchor-minutes N     Watchdog for STAGE 1 anchor. Default 90.
#   --ar-trials N          AR trial count. Default 8 (CONFIGS 0–7 in the AR script).
#   --ar-seconds N         Per-trial budget in seconds. Default 6000.
#   --reference-anchor DIR Existing anchor run-dir to include in the re-rank.
#                          Default: outputs/overnight-smolvla-2026-05-15T210257-anchor/.
#   --skip-rerank          Skip STAGE 3.
#   --skip-dashboard       Skip STAGE 4.
#   --dry-run              Print resolved commands and exit.
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

ANCHOR_MIN=90
AR_TRIALS=8                  # 6 original + 2 added 2026-05-16 (lr_mid + batch_up)
AR_SEC_PER_TRIAL=6000        # 100 min per trial
WITH_ANCHOR=0
SKIP_RERANK=0
SKIP_DASHBOARD=0
DRY_RUN=0
LAUNCH_TS="$(date +%Y-%m-%dT%H%M%S)"
SESSION_PREFIX="overnight-smolvla-${LAUNCH_TS}"
REFERENCE_ANCHOR="$WORKSPACE/outputs/overnight-smolvla-2026-05-15T210257-anchor"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-anchor)        WITH_ANCHOR=1; shift ;;
        --anchor-minutes)     ANCHOR_MIN="$2"; shift 2 ;;
        --ar-trials)          AR_TRIALS="$2"; shift 2 ;;
        --ar-seconds)         AR_SEC_PER_TRIAL="$2"; shift 2 ;;
        --reference-anchor)   REFERENCE_ANCHOR="$2"; shift 2 ;;
        --skip-rerank)        SKIP_RERANK=1; shift ;;
        --skip-dashboard)     SKIP_DASHBOARD=1; shift ;;
        --dry-run)            DRY_RUN=1; shift ;;
        -h|--help)            sed -n '2,30p' "${BASH_SOURCE[0]}" | grep "^#" | sed 's/^# \?//'; exit 0 ;;
        *)                    echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

G='\033[0;32m'; R='\033[0;31m'; C='\033[0;36m'; Y='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${C}[$(date +%H:%M:%S) INFO]${NC} $*"; }
ok()    { echo -e "${G}[$(date +%H:%M:%S)  OK ]${NC} $*"; }
warn()  { echo -e "${Y}[$(date +%H:%M:%S) WARN]${NC} $*"; }
err()   { echo -e "${R}[$(date +%H:%M:%S) ERR ]${NC} $*" >&2; }

DATASET="$WORKSPACE/datasets/kvgork/so101-pickplace1"
TRAIN_PY="$WORKSPACE/.pixi/envs/train-policy/bin/python"
DEFAULT_PY="$WORKSPACE/.pixi/envs/default/bin/python"
OPEN_EVAL="$WORKSPACE/scripts/_open_loop_eval.py"

# Banner
info "12-h SmolVLA overnight plan — A + B combined"
info "  anchor          : $([ "$WITH_ANCHOR" = 1 ] && echo "fresh ${ANCHOR_MIN} min" || echo "REUSE $REFERENCE_ANCHOR")"
info "  ar trials       : $AR_TRIALS × ${AR_SEC_PER_TRIAL}s (max $((AR_TRIALS * AR_SEC_PER_TRIAL / 60)) min)"
info "  rerank          : $([ "$SKIP_RERANK" = 1 ] && echo skip || echo ON)"
info "  dashboard       : $([ "$SKIP_DASHBOARD" = 1 ] && echo skip || echo ON)"
info "  cache_dir       : $WORKSPACE/outputs/cache_storage/"
info "  session         : $SESSION_PREFIX"

# preflight
WEIGHTS_DIR="$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
[ -d "$WEIGHTS_DIR" ] || { err "prefetch SmolVLM2 weights: bash scripts/_run_smolvla_tonight.sh --prefetch-weights"; exit 2; }
[ -d "$DATASET" ] || { err "dataset missing: $DATASET"; exit 2; }
if [ "$WITH_ANCHOR" = 0 ]; then
    [ -d "$REFERENCE_ANCHOR/policy-smolvla/checkpoints/last/pretrained_model" ] \
        || { err "reference anchor not found at $REFERENCE_ANCHOR (run with --with-anchor or set --reference-anchor)"; exit 2; }
fi
ok "preflight clean"

ANCHOR_DIR="$WORKSPACE/outputs/${SESSION_PREFIX}-anchor"
AR_OUT_ROOT="$WORKSPACE/outputs/autoresearch-lerobot-policy-smolvla"
RERANK_DIR="$WORKSPACE/outputs/eval/${SESSION_PREFIX}-rerank"
WINNER_JSON="$RERANK_DIR/winner.json"

if [ "$DRY_RUN" = 1 ]; then
    info "── DRY-RUN ──"
    [ "$WITH_ANCHOR" = 1 ] && cat <<EOF
# STAGE 1 — ANCHOR (fresh, ${ANCHOR_MIN} min)
bash scripts/_run_smolvla_tonight.sh --cache-frames --train-minutes ${ANCHOR_MIN} --run-dir ${ANCHOR_DIR}
EOF
    cat <<EOF
# STAGE 2 — AR SWEEP (${AR_TRIALS} × ${AR_SEC_PER_TRIAL}s)
SESSION_ID=${SESSION_PREFIX}-ar SECONDS_PER_EXP=${AR_SEC_PER_TRIAL} TRIALS=${AR_TRIALS} CACHE_FRAMES=1 \\
    bash scripts/_run_autoresearch_smolvla.sh
EOF
    [ "$SKIP_RERANK" = 0 ] && cat <<EOF
# STAGE 3 — RE-RANK (all anchor + AR ckpts via open-loop eval)
mkdir -p ${RERANK_DIR}
# loops each ckpt → scripts/_open_loop_eval.py → ${RERANK_DIR}/<name>.json
EOF
    [ "$SKIP_DASHBOARD" = 0 ] && cat <<EOF
# STAGE 4 — dashboard report + snapshot
.pixi/envs/dashboard/bin/python -m lerobot_isaac_dashboard.report --workspace . --output-dir outputs/${SESSION_PREFIX}-dashboard
.pixi/envs/dashboard/bin/python -m lerobot_isaac_dashboard.snapshots save --workspace . --label ${SESSION_PREFIX}-final
EOF
    exit 0
fi

# --- STAGE 1: anchor (optional) ---------------------------------------------
if [ "$WITH_ANCHOR" = 1 ]; then
    info "── STAGE 1: ANCHOR (${ANCHOR_MIN} min) ──"
    bash "$WORKSPACE/scripts/_run_smolvla_tonight.sh" \
        --cache-frames --train-minutes "$ANCHOR_MIN" --run-dir "$ANCHOR_DIR"
    rc=$?
    if [ "$rc" -ne 0 ]; then warn "anchor rc=$rc — continuing to AR"; else ok "anchor complete"; fi
else
    info "── STAGE 1: SKIPPED (reusing $REFERENCE_ANCHOR) ──"
    ANCHOR_DIR="$REFERENCE_ANCHOR"
fi

# --- STAGE 2: AR sweep ------------------------------------------------------
info "── STAGE 2: AR SWEEP (${AR_TRIALS} trials × ${AR_SEC_PER_TRIAL}s each) ──"
STAGE2_T0=$(date +%s)
SESSION_ID="${SESSION_PREFIX}-ar" \
SECONDS_PER_EXP="$AR_SEC_PER_TRIAL" \
TRIALS="$AR_TRIALS" \
CACHE_FRAMES=1 \
bash "$WORKSPACE/scripts/_run_autoresearch_smolvla.sh"
ar_rc=$?
STAGE2_DUR=$(( $(date +%s) - STAGE2_T0 ))
ok "AR finished rc=$ar_rc in $((STAGE2_DUR / 60))min (early-stop if < $((AR_TRIALS * AR_SEC_PER_TRIAL / 60))min)"

# --- STAGE 3: re-rank EVERY checkpoint --------------------------------------
if [ "$SKIP_RERANK" = 1 ]; then
    info "── STAGE 3: SKIPPED ──"
else
    info "── STAGE 3: RE-RANK (open-loop eval on every ckpt) ──"
    mkdir -p "$RERANK_DIR"

    # Anchor: re-eval the deepest checkpoint.
    ANCHOR_CKPT="$ANCHOR_DIR/policy-smolvla/checkpoints/last/pretrained_model"
    if [ -d "$ANCHOR_CKPT" ]; then
        info "rerank: anchor"
        "$TRAIN_PY" "$OPEN_EVAL" \
            --policy_path "$ANCHOR_CKPT" \
            --dataset_root "$DATASET" \
            --n_episodes 3 \
            --output_json "$RERANK_DIR/anchor.json" \
            --task_label "${SESSION_PREFIX}-rerank-anchor" \
            --run_id "${SESSION_PREFIX}-rerank-anchor" 2>&1 | grep -E "result:|error|frame=$" | tail -3
    fi

    # Each AR trial: take the latest checkpoint dir, eval it.
    for t in $(seq 0 $((AR_TRIALS - 1))); do
        TRIAL_CKPT=$(find "$AR_OUT_ROOT/trial_$t/checkpoints" -name pretrained_model -type d 2>/dev/null | sort | tail -1)
        if [ -z "${TRIAL_CKPT:-}" ] || [ ! -d "$TRIAL_CKPT" ]; then
            warn "rerank: trial_$t — no ckpt found, skipping"
            continue
        fi
        info "rerank: trial_$t"
        "$TRAIN_PY" "$OPEN_EVAL" \
            --policy_path "$TRIAL_CKPT" \
            --dataset_root "$DATASET" \
            --n_episodes 3 \
            --output_json "$RERANK_DIR/trial_${t}.json" \
            --task_label "${SESSION_PREFIX}-rerank-trial-$t" \
            --run_id "${SESSION_PREFIX}-rerank-trial-$t" 2>&1 | grep -E "result:|error|frame=$" | tail -3
    done

    # Pick the winner (highest pc_success).
    "$DEFAULT_PY" - <<PY > "$WINNER_JSON"
import json, glob, pathlib, sys

candidates = []
for fp in glob.glob("${RERANK_DIR}/*.json"):
    if pathlib.Path(fp).name == "winner.json":
        continue
    try:
        d = json.load(open(fp))
    except Exception:
        continue
    pc = d.get("pc_success")
    if pc is None or pc != pc:  # NaN check
        continue
    candidates.append((float(pc), fp, d))

if not candidates:
    print(json.dumps({"error": "no valid eval JSONs found"}, indent=2))
    sys.exit(0)

candidates.sort(reverse=True)
top_pc, top_fp, top_d = candidates[0]
print(json.dumps({
    "winner_run_id": top_d.get("run_id"),
    "winner_eval_json": top_fp,
    "winner_pc_success": top_pc,
    "winner_mse": top_d.get("_metadata", {}).get("mse"),
    "winner_policy_path": top_d.get("_metadata", {}).get("policy_path"),
    "ranking": [
        {"run_id": d.get("run_id"), "pc_success": pc, "mse": d.get("_metadata", {}).get("mse")}
        for pc, _, d in candidates
    ],
}, indent=2))
PY
    ok "winner written → $WINNER_JSON"
    cat "$WINNER_JSON" | head -12
fi

# --- STAGE 4: dashboard ------------------------------------------------------
if [ "$SKIP_DASHBOARD" = 1 ]; then
    info "── STAGE 4: SKIPPED ──"
else
    info "── STAGE 4: dashboard report + snapshot ──"
    "$WORKSPACE/.pixi/envs/dashboard/bin/python" -m lerobot_isaac_dashboard.report \
        --workspace "$WORKSPACE" \
        --output-dir "$WORKSPACE/outputs/${SESSION_PREFIX}-dashboard" >/dev/null
    "$WORKSPACE/.pixi/envs/dashboard/bin/python" -m lerobot_isaac_dashboard.snapshots save \
        --workspace "$WORKSPACE" --label "${SESSION_PREFIX}-final" >/dev/null
    ok "dashboard refreshed + snapshot saved"
fi

echo
ok "12-hour pipeline complete. Session: $SESSION_PREFIX"
echo "  AR        $AR_OUT_ROOT/trial_*/checkpoints/"
echo "  Anchor    $ANCHOR_DIR/policy-smolvla/checkpoints/"
echo "  Re-rank   $RERANK_DIR/"
echo "  Winner    $WINNER_JSON"
echo
echo "Next step (laptop, after syncing $WINNER_JSON + ckpt):"
echo "  cd ~/workspaces/lerobot-isaac-deploy"
echo "  pixi run sync-ckpt -- --from desktop --winner $WINNER_JSON"
echo "  pixi run session   -- --winner <local-winner.json> --execute"
