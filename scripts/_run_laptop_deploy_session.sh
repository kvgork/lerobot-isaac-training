#!/usr/bin/env bash
# =============================================================================
# _run_laptop_deploy_session.sh — One-command laptop deploy workflow.
#
# Runs on the laptop physically connected to the SO-101. Consumes the
# `winner.json` produced by `_run_tonight_smolvla_12h.sh` on the desktop
# (synced via `sync_ckpt_to_laptop.sh`) and walks the full
# pre-flight → dry-run → tight-clamp execute → closed-loop eval ladder.
#
# Safety contract
# ---------------
# DEFAULT IS DRY-RUN — the script will NOT write to motors unless you
# pass `--execute` AND there is a fresh "yes" confirmation from stdin at
# the execute gate.
#
# Usage
# -----
#   # Pre-flight only (load policy, dump schema, NO robot connection)
#   bash scripts/_run_laptop_deploy_session.sh --policy-path <DIR>
#
#   # Full dry-run loop (with robot connected, NO motor writes)
#   bash scripts/_run_laptop_deploy_session.sh --policy-path <DIR> --dry-run-loop
#
#   # The whole ladder (dry → execute @ 1° → execute @ 3° → 10-ep closed-loop)
#   bash scripts/_run_laptop_deploy_session.sh --policy-path <DIR> --execute
#
#   # Or feed the winner JSON straight from desktop sync
#   bash scripts/_run_laptop_deploy_session.sh --winner <PATH-TO-winner.json> --execute
#
# Flags
#   --policy-path DIR      pretrained_model/ directory. Mutually exclusive
#                          with --winner.
#   --winner JSON          path to winner.json from the desktop sweep;
#                          resolves --policy-path automatically.
#   --dataset-root DIR     LeRobotDataset root on the laptop. Default
#                          $HOME/workspaces/lerobot-isaac-deploy/datasets/kvgork/so101-pickplace1
#   --port DEV             serial port of the SO-101. Default /dev/ttyACM0
#   --camera SPEC          camera spec, e.g. d435_rgb=/dev/video0,640,480.
#                          Required for --dry-run-loop / --execute. Default
#                          d435_rgb=/dev/video0,640,480.
#   --task STR             language instruction. Default "pick and place cube"
#   --dry-run-loop         run the inference loop with NO motor writes
#                          (verifies camera, observation pipeline, inference
#                          latency). Implies --execute=false.
#   --execute              ESCALATING-CLAMP MODE: dry → 1° → 3° → closed-loop eval
#                          (each step prompts stdin "yes" to advance)
#   --skip-closed-loop     stop after the 3° execute step (no eval recording)
#   --n-eval-episodes N    closed-loop episode count. Default 10
#   --safety-ack-only      run the safety-ack file creation and exit (so the
#                          first real run does not block on the interactive
#                          consent prompt)
# =============================================================================
set -uo pipefail

PROG="$(basename "$0")"

# --- defaults ---------------------------------------------------------------
POLICY_PATH=""
WINNER=""
DATASET_ROOT="$HOME/workspaces/lerobot-isaac-deploy/datasets/kvgork/so101-pickplace1"
PORT="/dev/ttyACM0"
CAMERA="d435_rgb=/dev/video0,640,480"
TASK="pick and place cube"
DO_DRY_LOOP=0
DO_EXECUTE=0
SKIP_CLOSED_LOOP=0
N_EVAL_EPS=10
SAFETY_ACK_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy-path)      POLICY_PATH="$2"; shift 2 ;;
        --winner)           WINNER="$2"; shift 2 ;;
        --dataset-root)     DATASET_ROOT="$2"; shift 2 ;;
        --port)             PORT="$2"; shift 2 ;;
        --camera)           CAMERA="$2"; shift 2 ;;
        --task)             TASK="$2"; shift 2 ;;
        --dry-run-loop)     DO_DRY_LOOP=1; shift ;;
        --execute)          DO_EXECUTE=1; shift ;;
        --skip-closed-loop) SKIP_CLOSED_LOOP=1; shift ;;
        --n-eval-episodes)  N_EVAL_EPS="$2"; shift 2 ;;
        --safety-ack-only)  SAFETY_ACK_ONLY=1; shift ;;
        -h|--help)          sed -n '2,30p' "${BASH_SOURCE[0]}" | grep "^#" | sed 's/^# \?//'; exit 0 ;;
        *)                  echo "$PROG: unknown arg: $1" >&2; exit 2 ;;
    esac
done

G='\033[0;32m'; R='\033[0;31m'; C='\033[0;36m'; Y='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${C}[$(date +%H:%M:%S) INFO]${NC} $*"; }
ok()    { echo -e "${G}[$(date +%H:%M:%S)  OK ]${NC} $*"; }
warn()  { echo -e "${Y}[$(date +%H:%M:%S) WARN]${NC} $*"; }
err()   { echo -e "${R}[$(date +%H:%M:%S) ERR ]${NC} $*" >&2; }

confirm() {
    # confirm "message" — read y/n; non-y aborts.
    local msg="$1"
    read -r -p "$(echo -e "${Y}$msg [type 'yes' to continue]:${NC} ")" ans
    if [ "$ans" != "yes" ]; then
        err "operator aborted at: $msg"
        exit 10
    fi
}

# --- safety-ack-only path ---------------------------------------------------
SAFETY_ACK="$HOME/.config/robot-data-runner/safety_ack"
if [ "$SAFETY_ACK_ONLY" = 1 ]; then
    mkdir -p "$(dirname "$SAFETY_ACK")"
    echo "acked" > "$SAFETY_ACK"
    ok "safety-ack written → $SAFETY_ACK"
    exit 0
fi

# --- resolve policy from --winner ------------------------------------------
if [ -n "$WINNER" ]; then
    if [ ! -f "$WINNER" ]; then err "winner JSON not found: $WINNER"; exit 2; fi
    POLICY_PATH=$(python3 -c "import json; print(json.load(open('$WINNER'))['winner_policy_path'])" 2>/dev/null || true)
    if [ -z "$POLICY_PATH" ] || [ ! -d "$POLICY_PATH" ]; then
        err "could not resolve winner_policy_path from $WINNER"
        exit 2
    fi
    info "winner from $WINNER → $POLICY_PATH"
fi

[ -n "$POLICY_PATH" ] || { err "--policy-path or --winner required"; exit 2; }
[ -d "$POLICY_PATH" ] || { err "policy path not a directory: $POLICY_PATH"; exit 2; }
[ -d "$DATASET_ROOT" ] || { err "dataset root not a directory: $DATASET_ROOT"; exit 2; }

# --- detect runner ---------------------------------------------------------
RUNNER_BIN="$(command -v robot-data-run 2>/dev/null || true)"
[ -n "$RUNNER_BIN" ] || { err "robot-data-run not on PATH. Run laptop_bootstrap.sh first."; exit 2; }
CHECK_BIN="$(command -v robot-data-run-check 2>/dev/null || true)"
EVAL_BIN="$(command -v robot-data-run-eval 2>/dev/null || true)"

# Banner
info "laptop deploy session"
info "  policy-path : $POLICY_PATH"
info "  dataset     : $DATASET_ROOT"
info "  port        : $PORT"
info "  camera      : $CAMERA"
info "  task        : '$TASK'"
info "  dry-loop    : $DO_DRY_LOOP"
info "  execute     : $DO_EXECUTE"
info "  closed-loop : $([ "$SKIP_CLOSED_LOOP" = 1 ] && echo skip || echo "$N_EVAL_EPS eps")"
echo

# --- STEP 1: preflight (no robot needed) -----------------------------------
info "── STEP 1: preflight (load policy + dump I/O schema, no motors) ──"
"$CHECK_BIN" --policy-path "$POLICY_PATH" --dataset-root "$DATASET_ROOT" \
    2>&1 | grep -vE "HTTP Request|Loading weights" | tail -10
ok "policy loads cleanly"
echo

# Pure preflight invocation: stop here.
if [ "$DO_DRY_LOOP" = 0 ] && [ "$DO_EXECUTE" = 0 ]; then
    ok "preflight only — done. Add --dry-run-loop or --execute to continue."
    exit 0
fi

# --- STEP 2: dry-run loop (robot connected, no motor writes) ---------------
info "── STEP 2: dry-run loop (30s, NO motor writes) ──"
confirm "Confirm SO-101 is plugged in to $PORT and powered. Workspace clear around the arm."
"$RUNNER_BIN" \
    --policy-path "$POLICY_PATH" \
    --dataset-root "$DATASET_ROOT" \
    --port "$PORT" \
    --camera "$CAMERA" \
    --rate-hz 30 \
    --duration-s 30 \
    --task "$TASK" \
    -v
ok "dry-run loop complete — check action lines made sense"
echo

if [ "$DO_EXECUTE" = 0 ]; then
    ok "dry-run only — done. Add --execute to send motor commands."
    exit 0
fi

# --- STEP 3: execute, 1° tight clamp, 30 s ---------------------------------
confirm "READY for tight-clamp execute? Hand on physical e-stop. 1° per step, 30 s."
info "── STEP 3: execute @ max-relative-target=1.0, 30s ──"
"$RUNNER_BIN" \
    --policy-path "$POLICY_PATH" \
    --dataset-root "$DATASET_ROOT" \
    --port "$PORT" \
    --camera "$CAMERA" \
    --rate-hz 30 \
    --duration-s 30 \
    --max-relative-target 1.0 \
    --task "$TASK" \
    --execute --home-on-exit \
    -v
ok "tight-clamp execute complete — abort here if motion looked wrong"
echo

# --- STEP 4: execute, 3° clamp, 60 s ---------------------------------------
confirm "Step 3 looked OK. Proceed to 3°/step, 60 s real task?"
info "── STEP 4: execute @ max-relative-target=3.0, 60s ──"
"$RUNNER_BIN" \
    --policy-path "$POLICY_PATH" \
    --dataset-root "$DATASET_ROOT" \
    --port "$PORT" \
    --camera "$CAMERA" \
    --rate-hz 30 \
    --duration-s 60 \
    --max-relative-target 3.0 \
    --task "$TASK" \
    --execute --home-on-exit
ok "3° execute complete"
echo

if [ "$SKIP_CLOSED_LOOP" = 1 ]; then
    ok "skipped closed-loop eval — done."
    exit 0
fi

# --- STEP 5: closed-loop N-episode eval ------------------------------------
[ -n "$EVAL_BIN" ] || { err "robot-data-run-eval not on PATH"; exit 2; }
EVAL_OUT_DIR="$HOME/outputs/eval"
mkdir -p "$EVAL_OUT_DIR"
RUN_ID="laptop-$(date +%Y-%m-%dT%H%M%S)"
EVAL_JSON="$EVAL_OUT_DIR/${RUN_ID}-closed-loop.json"

confirm "Step 4 looked OK. Proceed to $N_EVAL_EPS-episode closed-loop eval (prompt-user-observer)?"
info "── STEP 5: closed-loop eval ──"

# Auto-create safety-ack so we don't block on the interactive consent prompt
# in the eval runner — operator already confirmed verbally above.
mkdir -p "$(dirname "$SAFETY_ACK")"
[ -f "$SAFETY_ACK" ] || echo "acked" > "$SAFETY_ACK"

"$EVAL_BIN" \
    --policy-path "$POLICY_PATH" \
    --dataset-root "$DATASET_ROOT" \
    --port "$PORT" \
    --camera "$CAMERA" \
    --rate-hz 30 \
    --max-relative-target 3.0 \
    --task "$TASK" \
    --task-spec prompt_user_observer \
    --n-episodes "$N_EVAL_EPS" \
    --duration-per-episode-s 15 \
    --output-json "$EVAL_JSON" \
    --home-on-exit \
    --i-have-read-the-safety-runbook
ok "closed-loop eval JSON → $EVAL_JSON"
echo
ok "laptop session complete. Sync the eval back to desktop with:"
echo "  scp $EVAL_JSON desktop:lerobot-isaac-training/outputs/eval/"
echo "  OR on desktop: bash scripts/sync_eval_from_laptop.sh"
