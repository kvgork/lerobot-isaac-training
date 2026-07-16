#!/usr/bin/env bash
# =============================================================================
# gpu_campaign.sh — unattended serial-GPU campaign supervisor.
#
# Runs the 3-day autonomous job chain back-to-back on the single RTX 3080,
# NEVER idling: on each job's exit (clean, crash, or hard stale-log kill) it
# advances a FIXED fallback chain and evaluates scripted gates. This is the
# missing piece the 2026-07-15 plan critique flagged — the existing training
# watchdogs are REPORT-ONLY ([[watchdog-report-only]]); this supervisor ACTS
# (it kills a wedged job and launches the next), so a crash at hour 4 does not
# idle the GPU until a human returns.
#
# Design (plan: plans/2026-07-15-three-day-autonomous-gpu-plan.md, Phase 0a):
#   - SWEEPS LEAD: offline-MSE-scoreable policy sweeps fill the chain.
#   - RESIDUAL is a GATED STRETCH: it runs only if the C1 ee-descent fix passes
#     scripts/_residual_smoke_gate.sh (a scripted exit 0/1) — else it is skipped
#     and the chain continues on sweeps (no GPU wasted on an un-grasping base).
#   - Each job self-terminates on its OWN wall budget; the per-job STALE_KILL
#     here is a BACKSTOP for the masked-crash + Isaac-teardown-hang failure mode
#     ([[wm-isaac-stall-resolved]]) — sized ABOVE each job's longest legit quiet
#     gap (e.g. residual learning_starts collection) so it never false-fires.
#
# Usage — detach the WHOLE campaign once (the plan's unattended window):
#   setsid bash scripts/gpu_campaign.sh > outputs/gpu_campaign/campaign.log 2>&1 < /dev/null &
#   tail -f outputs/gpu_campaign/campaign.log        # monitor
#   cat .agent-state/gpu-campaign/events.jsonl       # machine-readable transitions
#
# Dry-run the full chain (no GPU, prints every hop + gate) — the Phase-0 exit gate:
#   bash scripts/gpu_campaign.sh --dry-run
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

CAMP_DIR="$WORKSPACE/outputs/gpu_campaign"; mkdir -p "$CAMP_DIR"
STATE_DIR="$WORKSPACE/.agent-state/gpu-campaign"; mkdir -p "$STATE_DIR"
EVENTS="$STATE_DIR/events.jsonl"
POLL_S="${POLL_S:-60}"                 # how often to check liveness + log staleness
DRY_RUN=0; [ "${1:-}" = "--dry-run" ] && DRY_RUN=1

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
emit(){ # hop event [detail]
  printf '{"ts":"%s","hop":"%s","event":"%s","detail":"%s"}\n' \
    "$(ts)" "$1" "$2" "${3:-}" >> "$EVENTS"
  echo "[campaign $(ts)] $1 :: $2 ${3:+:: $3}"
}

# run_job NAME STALE_KILL_S CMD_STRING
# Launches CMD in its OWN session (setsid → own process group) so the whole
# subtree (python → Isaac → children) can be killed together on a stale-kill.
run_job() {
  local name="$1" stale="$2" cmd="$3"
  local log="$CAMP_DIR/${name}.log"
  if [ "$DRY_RUN" = "1" ]; then emit "$name" "dry-run" "$cmd"; return 0; fi
  emit "$name" "launch" "$cmd"
  setsid bash -c "$cmd" > "$log" 2>&1 < /dev/null &
  local pid=$!                                   # setsid session leader = process-group id
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_S"
    if [ -f "$log" ]; then
      local age=$(( $(date +%s) - $(stat -c %Y "$log") ))
      if [ "$age" -gt "$stale" ]; then
        emit "$name" "stale-kill" "log idle ${age}s > ${stale}s — presumed wedged"
        kill -TERM -"$pid" 2>/dev/null; sleep 20; kill -KILL -"$pid" 2>/dev/null
        break
      fi
    fi
  done
  wait "$pid" 2>/dev/null; local rc=$?
  emit "$name" "exit" "rc=$rc"
  return $rc
}

# run_gate NAME CMD_STRING — foreground; its exit code decides the branch.
run_gate() {
  local name="$1" cmd="$2"
  if [ "$DRY_RUN" = "1" ]; then emit "$name" "dry-run-gate" "$cmd"; return 0; fi
  emit "$name" "gate-start" "$cmd"
  bash -c "$cmd"; local rc=$?
  emit "$name" "gate-result" "rc=$rc"
  return $rc
}

emit campaign start "dry_run=$DRY_RUN poll=${POLL_S}s"

# --- HOP 1: SmolVLA orchestrated sweep (turnkey, offline-scoreable) ----------
# --ar-seconds 5000 keeps 8 trials inside the 12 h hop cap (8×6000 s = 13.3 h overshoots; plan hop-1 note)
run_job smolvla_A     3600 "bash scripts/_run_tonight_smolvla_12h.sh --ar-seconds 5000" || true

# --- HOP 2: C1 residual ee-descent gate (probe→apply-scale→regen→smoke→grep) --
# Gate PASS (rc=0) → run the residual full stretch; FAIL → skip, keep sweeping.
if run_gate c1_residual_gate "bash scripts/_residual_smoke_gate.sh"; then
  # --- HOP 3: residual-RL full (GATED STRETCH). decay 15000 < ~23k reachable
  #     in the 13h cap so the pure-RL retention phase actually runs (plan 0c).
  run_job residual_B 3600 "LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS=15000 bash scripts/launch_residual_rl.sh" || true
else
  emit residual_B skip "C1 gate rc!=0 — ee-descent fix not validated; continuing sweep chain (no GPU wasted)"
fi

# --- HOP 4: LoRA rank sweep (offline-scoreable) ------------------------------
run_job lora_C        2700 "MAX_TRIALS=16 STEPS=20000 bash scripts/_run_autoresearch_lora.sh" || true

# --- HOP 5: diffusion AR baseline (offline-scoreable) — GENERAL dispatcher ---
run_job diffusion_E   2400 "SESSION_ID=diff-camp TRIALS=6 SECONDS_PER_EXP=1800 bash scripts/run_autoresearch_policy.sh --arch diffusion" || true

# --- HOP 6: ACT sweep (offline-scoreable) — GENERAL dispatcher ---------------
# (run_autoresearch_policy.sh --arch <x> replaces the per-arch _run_autoresearch_*.sh
#  copies; loop engine is the shared claude_code autoresearch skill.)
run_job act_F         2700 "SESSION_ID=act-camp TRIALS=8 SECONDS_PER_EXP=2700 bash scripts/run_autoresearch_policy.sh --arch act" || true

# --- HOP 7: WM-offline DreamerV3 AR (low-value tail-fill) --------------------
run_job wm_offline_G  2700 "MAX_TRIALS=12 STEPS=200000 bash scripts/_run_autoresearch_wm.sh" || true

emit campaign done "chain complete — see outputs/gpu_campaign/*.log + .agent-state/gpu-campaign/events.jsonl"
