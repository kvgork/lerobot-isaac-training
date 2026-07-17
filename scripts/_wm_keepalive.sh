#!/usr/bin/env bash
# =============================================================================
# _wm_keepalive.sh — liveness forwarder for the WM hop (hop 7) of gpu_campaign.
#
# Problem: the supervisor stale-kills wm_offline_G at 2700 s of stdout silence,
# but the WM AR script writes trial output to per-trial iter logs — its stdout
# quiet gap is SECONDS_PER_EXP (2400 s) + eval (≤300 s) ≈ the threshold, so a
# slow eval false-kills the hop (exactly what killed act_F at trial 3).
#
# Fix WITHOUT touching the running supervisor: every 10 min, if the newest WM
# trial log grew recently (real liveness), append a heartbeat line to the
# supervisor-watched hop log so its mtime stays fresh. If the trial log
# freezes (real hang / masked crash), no heartbeat is written and the
# supervisor's stale-kill still fires — the backstop is preserved, only the
# false-positive is removed.
#
# Usage: setsid bash scripts/_wm_keepalive.sh <supervisor_pid> <wm_session_dir> \
#          > outputs/gpu_campaign/wm_keepalive.log 2>&1 < /dev/null &
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

SUP_PID="${1:?supervisor pid}"
WM_DIR="${2:?wm session dir (.agent-state/wm-bash-...)}"
HOP_LOG="$WORKSPACE/outputs/gpu_campaign/wm_offline_G.log"
FRESH_S=1200   # trial log must have grown within this window to count as alive
SLEEP_S=600    # heartbeat cadence (threshold is 2700 s)

alive() { [ -d "/proc/$SUP_PID" ] && grep -aq gpu_campaign.sh "/proc/$SUP_PID/cmdline" 2>/dev/null; }

while alive; do
  newest=$(find "$WM_DIR" -name 'trial_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
  if [ -n "$newest" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
    if [ "$age" -lt "$FRESH_S" ] && [ -f "$HOP_LOG" ]; then
      echo "[wm-keepalive $(date -u +%Y-%m-%dT%H:%M:%SZ)] trial log $(basename "$newest") active ${age}s ago" >> "$HOP_LOG"
    fi
  fi
  sleep "$SLEEP_S"
done
