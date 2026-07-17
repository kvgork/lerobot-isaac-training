#!/usr/bin/env bash
# =============================================================================
# gpu_campaign_ext2_diffusion.sh — campaign extension #2: re-run the diffusion AR.
#
# Why: hop 5 (diff-camp, 2026-07-17) trained 6 full trials but banked NOTHING —
# save_freq=SECONDS_PER_EXP*25/30 (=1500 steps) assumed >=1 step/s while
# diffusion runs ~0.8 step/s, so every trial timed out just short of its first
# checkpoint (no ckpt -> no eval -> sentinel 0.0). The re-run uses the fixed
# dispatcher copy (run_autoresearch_policy_fixed.sh, save_freq=SECONDS_PER_EXP/4).
# The original dispatcher is NOT edited while the ACT hop still executes it;
# merge the fix back at the attended wrap.
#
# Runs after BOTH the main supervisor and the smolvla_A2 extension exit,
# so it is the last job in the extended chain.
#
# Usage:
#   setsid bash scripts/gpu_campaign_ext2_diffusion.sh <supervisor_pid> <ext1_pid> \
#     > outputs/gpu_campaign/ext2_diffusion.log 2>&1 < /dev/null &
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

SUP_PID="${1:?usage: gpu_campaign_ext2_diffusion.sh <supervisor_pid> <ext1_pid>}"
EXT1_PID="${2:?usage: gpu_campaign_ext2_diffusion.sh <supervisor_pid> <ext1_pid>}"
CAMP_DIR="$WORKSPACE/outputs/gpu_campaign"; mkdir -p "$CAMP_DIR"
STATE_DIR="$WORKSPACE/.agent-state/gpu-campaign"; mkdir -p "$STATE_DIR"
EVENTS="$STATE_DIR/events.jsonl"
POLL_S="${POLL_S:-120}"
STALE_KILL_S=2400

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
emit(){
  printf '{"ts":"%s","hop":"%s","event":"%s","detail":"%s"}\n' \
    "$(ts)" "$1" "$2" "${3:-}" >> "$EVENTS"
  echo "[campaign-ext2 $(ts)] $1 :: $2 ${3:+:: $3}"
}

# Process liveness only (cmdline-guarded against pid reuse) — never events.jsonl.
alive() { # pid pattern
  [ -d "/proc/$1" ] && grep -aq "$2" "/proc/$1/cmdline" 2>/dev/null
}

emit diffusion_E2 wait "waiting for supervisor $SUP_PID + ext1 $EXT1_PID to exit"
while alive "$SUP_PID" gpu_campaign || alive "$EXT1_PID" ext_smolvla; do
  sleep "$POLL_S"
done

emit diffusion_E2 launch "SESSION_ID=diff-camp2 TRIALS=6 SECONDS_PER_EXP=1800 bash scripts/run_autoresearch_policy_fixed.sh --arch diffusion (rerun: save_freq fix)"
log="$CAMP_DIR/diffusion_E2.log"
setsid bash -c "SESSION_ID=diff-camp2 TRIALS=6 SECONDS_PER_EXP=1800 bash scripts/run_autoresearch_policy_fixed.sh --arch diffusion" > "$log" 2>&1 < /dev/null &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  sleep 60
  if [ -f "$log" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if [ "$age" -gt "$STALE_KILL_S" ]; then
      emit diffusion_E2 stale-kill "log idle ${age}s > ${STALE_KILL_S}s — presumed wedged"
      kill -TERM -"$pid" 2>/dev/null; sleep 20; kill -KILL -"$pid" 2>/dev/null
      break
    fi
  fi
done
wait "$pid" 2>/dev/null; rc=$?
emit diffusion_E2 exit "rc=$rc"
