#!/usr/bin/env bash
# =============================================================================
# _residual_smoke_gate.sh — C1 ee-descent gate (machine-checkable exit 0/1).
#
# Chains the C1 action-scale fix into a single scripted PASS/FAIL so the
# unattended supervisor (scripts/gpu_campaign.sh HOP 2) can branch without a human:
#   1. measure per-joint action scale (if outputs/action_scale.json absent)
#   2. export LEROBOT_ISAAC_ACTION_SCALE_JSON so env + scripted controller rescale
#   3. regenerate demos with the new scale (BC labels back in [-1,1])
#   4. run the residual smoke (64-step warmup floor so the residual is exercised)
#   5. parse the smoke's [script-dbg] lines → ee descended AND obj lifted AND phase
#      reached CARRY  ⇒  exit 0 (fix works → run full residual)  else exit 1 (skip).
#
# Task geometry mirrors launch_residual_rl.sh so the measured scale + demos match
# the run the gate authorises.
# =============================================================================
set -uo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$WORKSPACE"
SIMPY="$WORKSPACE/.pixi/envs/sim/bin/python"
PYJSON="$WORKSPACE/.pixi/envs/default/bin/python"
SCALE_JSON="$WORKSPACE/outputs/action_scale.json"
DEMO_OUT="${DEMO_OUT:-datasets/local/so101-sim-pickplace-demos-cup0-wide-v2}"
DEMO_EPISODES="${DEMO_EPISODES:-40}"
SMOKE_SESSION="c1-residual-smoke"
SMOKE_STDOUT="$WORKSPACE/outputs/gpu_campaign/c1_smoke.log"          # launcher wrapper stdout
# [script-dbg] / [residual-rl] go to the TRAINING log, NOT the wrapper stdout:
TRAIN_LOG="$WORKSPACE/.agent-state/$SMOKE_SESSION/autoresearch/wm-isaac-prod/train.log"
mkdir -p "$WORKSPACE/outputs/gpu_campaign"

# Residual task geometry (identical to launch_residual_rl.sh).
export LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_STAGED_REWARD=1 \
       LEROBOT_ISAAC_PLACE_CUP=1 LEROBOT_ISAAC_CARRY_Z=0.19 \
       LEROBOT_ISAAC_OBJECT_X=0.22 LEROBOT_ISAAC_OBJECT_Y=-0.06 LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1

echo "[c1-gate] $(date -u +%FT%TZ) START demo_out=$DEMO_OUT episodes=$DEMO_EPISODES"

# 1. measure (reuse an existing probe result if present)
if [ ! -f "$SCALE_JSON" ]; then
  echo "[c1-gate] measuring per-joint action scale..."
  LEROBOT_ISAAC_PLACE_REST_Z=-1 "$SIMPY" scripts/_gen_sim_demos.py --measure_scale \
    --obj_x 0.22 --obj_y -0.06 --grasp_z 0.106 || { echo "[c1-gate] FAIL: measure crashed"; exit 1; }
fi
[ -f "$SCALE_JSON" ] || { echo "[c1-gate] FAIL: no $SCALE_JSON"; exit 1; }
export LEROBOT_ISAAC_ACTION_SCALE_JSON="$SCALE_JSON"

# clamp_innocent=true ⇒ all |delta|<=0.5, the clamp never bit → Option A is a no-op and the
# ee-descent cause is elsewhere; do NOT claim the fix works (route to IK diagnostics instead).
CI=$("$PYJSON" -c "import json;print(json.load(open('$SCALE_JSON')).get('clamp_innocent'))" 2>/dev/null || echo "None")
echo "[c1-gate] clamp_innocent=$CI scale_json=$(tr -d '\n' < "$SCALE_JSON" | head -c 260)"
if [ "$CI" = "True" ]; then
  echo "[c1-gate] FAIL: clamp_innocent — action clamp was NOT the descent blocker; needs IK diagnosis, not rescale"
  exit 1
fi

# 3. regen demos with the new scale (skip if already built)
if [ ! -d "$DEMO_OUT" ]; then
  echo "[c1-gate] regenerating $DEMO_EPISODES demos at the new scale..."
  LEROBOT_ISAAC_PLACE_REST_Z=-1 "$SIMPY" scripts/_gen_sim_demos.py \
    --episodes "$DEMO_EPISODES" --out "$DEMO_OUT" \
    --obj_x 0.22 --obj_y -0.06 --grasp_z 0.106 || { echo "[c1-gate] FAIL: demo regen crashed"; exit 1; }
fi

# 4. residual smoke — warmup floor 64 so PlayerDV3.get_actions (the residual seam) runs
#    almost immediately: 12 demo episodes (5820 transitions) are pre-seeded into the replay
#    buffer, so the random-action warmup buys nothing — at 200 it held the phase machine in
#    APPROACH for the first 200 steps of ep-1 and displaced the arm (2026-07-21 trace: first
#    transition at t=209). 64 = num_envs(1) x seq_len(64) floor.
#    STEPS=1400 (>= learning_starts 64 + one full ~540-step pick->place) so LIFT/CARRY are reachable.
#    RESIDUAL_RL_DECAY_STEPS pinned to 1e7 pins script_frac≈1.0 for the SMOKE ONLY —
#    isolates the scripted base from actor-blend interference (ep-2 descent freeze suspect);
#    the full run uses its own decay.
rm -rf "$WORKSPACE/.agent-state/$SMOKE_SESSION" "$WORKSPACE/outputs/wm-isaac-prod-$SMOKE_SESSION"
echo "[c1-gate] running residual smoke (session=$SMOKE_SESSION) → train log $TRAIN_LOG ..."
LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS=10000000 \
  STEPS=1400 MAX_EPISODE_STEPS=700 SECONDS_PER_EXP=3600 SESSION_ID="$SMOKE_SESSION" \
  LEROBOT_ISAAC_DEMO_DATASET="$DEMO_OUT" \
  EXTRA_HYDRA='algo.actor.ent_coef=1e-3 algo.horizon=25 algo.world_model.kl_free_nats=1.0 algo.mlp_keys.encoder=[state] algo.learning_starts=64' \
  bash scripts/launch_residual_rl.sh > "$SMOKE_STDOUT" 2>&1 || true

# 5. verdict from the [script-dbg] trace in the TRAINING log (NOT the wrapper stdout).
[ -f "$TRAIN_LOG" ] || { echo "[c1-gate] FAIL: no train log at $TRAIN_LOG (smoke crashed?)"; tail -20 "$SMOKE_STDOUT"; exit 1; }
"$PYJSON" - "$TRAIN_LOG" <<'PY'
import re, sys
log = open(sys.argv[1], errors="ignore").read()
dbg = re.findall(r"\[script-dbg\] phase=(\w+) obj_lifted=(\w+) oz=([\-0-9.]+) ez=([\-0-9.]+)", log)
if not dbg:
    print("[c1-gate] FAIL: no [script-dbg] lines — residual never engaged (check learning_starts / crash)")
    sys.exit(1)
phases = {d[0] for d in dbg}
min_ez = min(float(d[3]) for d in dbg)
max_oz = max(float(d[2]) for d in dbg)
lifted = any(d[1] == "True" for d in dbg)
reached_lift = bool({"LIFT", "CARRY", "LOWER", "RELEASE"} & phases)
descended = min_ez < 0.15          # was stuck hovering at ez~0.30; grasp_z target is 0.106
# PASS bar (2026-07-22): reached_lift became VACUOUS once per-phase caps force-advance
# unconditionally (round-2 formal PASS with max_oz=0.008). Require the PHYSICAL outcome:
# die actually lifted (max_oz > 0.07, matches demo-gen's post-hoc bar) AND the machine
# entered CARRY with it (transition prints capture oz at CARRY entry).
ok = descended and max_oz > 0.07 and ("CARRY" in phases)
print(f"[c1-gate] phases={sorted(phases)} min_ez={min_ez:.3f} max_oz={max_oz:.3f} "
      f"lifted={lifted} reached_lift={reached_lift} descended={descended}")
print("[c1-gate] VERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
