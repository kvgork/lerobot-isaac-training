#!/usr/bin/env bash
# =============================================================================
# gpu_campaign_ext_smolvla.sh — one-shot campaign extension: re-run hop 1.
#
# Why: the original smolvla_A hop (2026-07-16T14:51Z) exited rc=0 after ~2 min —
# every AR trial died in ~4 s because lerobot 0.6.0 broke the --cache_frames
# monkey-patch (cli_train_cached exit 2; fixed in adapters c018aeb). The
# supervisor correctly advanced the chain, so the SmolVLA sweep must be re-run
# AFTER the chain finishes (plan rule: "if the chain exhausts before 72 h,
# extend the top offline-scoreable sweep").
#
# Waits for the main supervisor to finish (its "campaign done" event OR its
# process disappearing), then runs the SmolVLA sweep with the same stale-kill
# backstop the supervisor uses.
#
# Usage (setsid-detached, like the supervisor):
#   setsid bash scripts/gpu_campaign_ext_smolvla.sh <supervisor_pid> \
#     > outputs/gpu_campaign/ext_smolvla.log 2>&1 < /dev/null &
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

SUP_PID="${1:?usage: gpu_campaign_ext_smolvla.sh <supervisor_pid>}"
CAMP_DIR="$WORKSPACE/outputs/gpu_campaign"; mkdir -p "$CAMP_DIR"
STATE_DIR="$WORKSPACE/.agent-state/gpu-campaign"; mkdir -p "$STATE_DIR"
EVENTS="$STATE_DIR/events.jsonl"
POLL_S="${POLL_S:-120}"
# Must exceed the job's longest LEGIT quiet gap on ITS OWN stdout: the AR stage
# only prints a line per finished trial, so the gap is SECONDS_PER_EXP (5000 s)
# + eval — 3600 here false-killed the ACT hop's analogue (2026-07-17).
STALE_KILL_S=6300

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
emit(){
  printf '{"ts":"%s","hop":"%s","event":"%s","detail":"%s"}\n' \
    "$(ts)" "$1" "$2" "${3:-}" >> "$EVENTS"
  echo "[campaign-ext $(ts)] $1 :: $2 ${3:+:: $3}"
}

sup_running() {
  # PID must still exist AND still be the gpu_campaign process (guards pid reuse).
  [ -d "/proc/$SUP_PID" ] && grep -aq gpu_campaign "/proc/$SUP_PID/cmdline" 2>/dev/null
}

# Wait on process liveness ONLY. Do NOT grep events.jsonl for "campaign done" —
# dry-runs append to the same file, so a stale done-event fires the rerun
# immediately while the chain is still running (bug hit 2026-07-16).
emit smolvla_A2 wait "waiting for supervisor pid $SUP_PID to exit"
while sup_running; do
  sleep "$POLL_S"
done

emit smolvla_A2 launch "bash scripts/_run_tonight_smolvla_12h.sh --ar-seconds 5000 (rerun of failed hop 1, adapters cache fix c018aeb)"
log="$CAMP_DIR/smolvla_A2.log"
setsid bash -c "bash scripts/_run_tonight_smolvla_12h.sh --ar-seconds 5000" > "$log" 2>&1 < /dev/null &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  sleep 60
  if [ -f "$log" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if [ "$age" -gt "$STALE_KILL_S" ]; then
      emit smolvla_A2 stale-kill "log idle ${age}s > ${STALE_KILL_S}s — presumed wedged"
      kill -TERM -"$pid" 2>/dev/null; sleep 20; kill -KILL -"$pid" 2>/dev/null
      break
    fi
  fi
done
wait "$pid" 2>/dev/null; rc=$?
emit smolvla_A2 exit "rc=$rc"
