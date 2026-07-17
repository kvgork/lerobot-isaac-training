#!/usr/bin/env bash
# =============================================================================
# gpu_campaign_ext3_act.sh — campaign extension #3: finish the ACT sweep.
#
# Why: the supervisor stale-killed hop 6 (act_F) mid-sweep on 2026-07-17 —
# its STALE_KILL (2700 s) equalled SECONDS_PER_EXP, but the dispatcher only
# writes a stdout line per FINISHED trial, so the hop log is legitimately
# quiet for a full trial duration. 3/8 trials banked; this reruns trials 3-7
# into the SAME session (act-camp) via the engine's AR_TRIAL_START offset
# (claude_code autoresearch engine, requires the resume commit).
#
# Runs last: after supervisor, ext1 (smolvla_A2) and ext2 (diffusion_E2).
#
# Usage:
#   setsid bash scripts/gpu_campaign_ext3_act.sh <sup_pid> <ext1_pid> <ext2_pid> \
#     > outputs/gpu_campaign/ext3_act.log 2>&1 < /dev/null &
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

SUP_PID="${1:?sup pid}"; EXT1_PID="${2:?ext1 pid}"; EXT2_PID="${3:?ext2 pid}"
CAMP_DIR="$WORKSPACE/outputs/gpu_campaign"; mkdir -p "$CAMP_DIR"
STATE_DIR="$WORKSPACE/.agent-state/gpu-campaign"; mkdir -p "$STATE_DIR"
EVENTS="$STATE_DIR/events.jsonl"
POLL_S="${POLL_S:-120}"
# Trial 2700 s + eval ≤300 s legit stdout quiet gap → 3900 with margin.
STALE_KILL_S=3900

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
emit(){
  printf '{"ts":"%s","hop":"%s","event":"%s","detail":"%s"}\n' \
    "$(ts)" "$1" "$2" "${3:-}" >> "$EVENTS"
  echo "[campaign-ext3 $(ts)] $1 :: $2 ${3:+:: $3}"
}
alive() { [ -d "/proc/$1" ] && grep -aq "$2" "/proc/$1/cmdline" 2>/dev/null; }

emit act_F2 wait "waiting for sup $SUP_PID + ext1 $EXT1_PID + ext2 $EXT2_PID"
while alive "$SUP_PID" gpu_campaign.sh || alive "$EXT1_PID" ext_smolvla || alive "$EXT2_PID" ext2_diffusion; do
  sleep "$POLL_S"
done

emit act_F2 launch "AR_TRIAL_START=3 TRIALS=5 SESSION_ID=act-camp SECONDS_PER_EXP=2700 run_autoresearch_policy_fixed.sh --arch act (finish trials 3-7)"
log="$CAMP_DIR/act_F2.log"
setsid bash -c "AR_TRIAL_START=3 TRIALS=5 SESSION_ID=act-camp SECONDS_PER_EXP=2700 bash scripts/run_autoresearch_policy_fixed.sh --arch act" > "$log" 2>&1 < /dev/null &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  sleep 60
  if [ -f "$log" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if [ "$age" -gt "$STALE_KILL_S" ]; then
      emit act_F2 stale-kill "log idle ${age}s > ${STALE_KILL_S}s — presumed wedged"
      kill -TERM -"$pid" 2>/dev/null; sleep 20; kill -KILL -"$pid" 2>/dev/null
      break
    fi
  fi
done
wait "$pid" 2>/dev/null; rc=$?
emit act_F2 exit "rc=$rc"
