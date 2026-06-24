#!/usr/bin/env bash
# =============================================================================
# launch_warmstart_full.sh — v1 FULL warm-start DreamerV3 carry-place launch.
#
# Recreated 2026-06-24 (was in an ephemeral scratchpad/, never committed).
# Source plans:
#   - plans/2026-06-23-world-model-training-plan.md  §3 Stage 2 launch block
#                                                     + the "Knobs" line (wm-vla playbook)
#   - plans/2026-06-23-carryplace-cup-campaign.md     ("the 2 plateaus" v1 row + Keep line)
#
# This is the v1 FULL config: die 0.18, fixed base, staged reward, FULL carry
# (no easy-cup curriculum), demo seeding + DreamerFD BC, plus the wm-vla
# playbook exploration knobs (actor ent_coef 1e-3, replay_ratio 16, horizon 25,
# kl_free 1.0, demo_ratio 0.5) that BOTH prior v1 plateaus omitted.
#
# It wraps scripts/_run_wm_isaac_overnight.sh — that launcher consumes the
# STEPS / BATCH_SIZE / SESSION_ID / EXTRA_HYDRA shell-env knobs (see its header)
# and forwards EXTRA_HYDRA verbatim onto the sheeprl command. The
# LEROBOT_ISAAC_* env vars are read by scripts/_wm_isaac_entry.py + the Isaac
# env at scene-build / patch-arm time.
#
# Run (multi-hour GPU; do NOT run unless you mean it):
#   bash scripts/launch_warmstart_full.sh
#
# Pre-flight: kill stray GPU procs (nvidia-smi --query-compute-apps); batch 8
# (NOT 16) on the 10 GB RTX 3080.
# =============================================================================
set -euo pipefail

# --- run identity / budget --------------------------------------------------
SESSION_ID="${SESSION_ID:-wm-warmstart-full-v1}"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-8}"            # 10 GB 3080: 8 fits; drop to 4 on OOM
SECONDS_PER_EXP="${SECONDS_PER_EXP:-46800}"   # 13 h hard ceiling — launcher runs: timeout "$SECONDS_PER_EXP" (default 6 h). NOT LEROBOT_TRAIN_TIMEOUT (dead in this launcher).

# --- task geometry: v1 FULL config (die 0.18, full carry) -------------------
# die 0.18 == OBJECT_SCALE 0.267 (16 mm die); object at (0.18, 0.05).
LEROBOT_ISAAC_OBJECT_SCALE="${LEROBOT_ISAAC_OBJECT_SCALE:-0.267}"
LEROBOT_ISAAC_OBJECT_X="${LEROBOT_ISAAC_OBJECT_X:-0.18}"
LEROBOT_ISAAC_OBJECT_Y="${LEROBOT_ISAAC_OBJECT_Y:-0.05}"
LEROBOT_ISAAC_FIX_BASE="${LEROBOT_ISAAC_FIX_BASE:-1}"
LEROBOT_ISAAC_STAGED_REWARD="${LEROBOT_ISAAC_STAGED_REWARD:-1}"

# --- warm-start: demo seeding + DreamerFD BC actor loss ---------------------
LEROBOT_ISAAC_DEMO_DATASET="${LEROBOT_ISAAC_DEMO_DATASET:-datasets/local/so101-sim-pickplace-demos}"
LEROBOT_ISAAC_BC_WEIGHT="${LEROBOT_ISAAC_BC_WEIGHT:-1.0}"

# --- observation: the actor must SEE the object pose the demos/script use ----
LEROBOT_ISAAC_INCLUDE_OBJECT_POSE="${LEROBOT_ISAAC_INCLUDE_OBJECT_POSE:-1}"

# --- wm-vla playbook exploration knobs (the point of the recreation) --------
# Verified key paths against sheeprl dreamer_v3.yaml (train-dreamer env):
#   algo.actor.ent_coef        (default 3e-4 → 1e-3, beats reach+grip entropy-collapse)
#   algo.replay_ratio          (default 1    → 16)
#   algo.horizon               (default 15   → 25)
#   algo.world_model.kl_free_nats (default 1.0; set explicitly per playbook "kl_free 1.0")
#   algo.mlp_keys.encoder=[state]  (so the object_pose state vec is encoded)
ACTOR_ENT_COEF="${ACTOR_ENT_COEF:-1e-3}"
REPLAY_RATIO="${REPLAY_RATIO:-16}"
HORIZON="${HORIZON:-25}"
KL_FREE_NATS="${KL_FREE_NATS:-1.0}"
# NOTE: "demo_ratio 0.5" from the playbook is NOT a sheeprl hydra key — sheeprl
# has no demo/RLPD mixing knob. In THIS repo the demo contribution is the
# DreamerFD BC actor loss (LEROBOT_ISAAC_BC_WEIGHT, above) + the replay-buffer
# seed; there is no separate per-batch demo_ratio to set. Captured here as a
# shell var for documentation only — it is intentionally NOT forwarded to
# sheeprl. TODO: if a true demo-mix ratio is ever wanted, plumb it through the
# BC patch in scripts/_wm_isaac_entry.py (_patch_bc_actor_loss), not hydra.
DEMO_RATIO="${DEMO_RATIO:-0.5}"   # documentation only; not a sheeprl key

# --- assemble the EXTRA_HYDRA overrides forwarded verbatim by the launcher ---
# (space-separated hydra key=value tokens; the launcher read -a splits them.)
EXTRA_HYDRA="${EXTRA_HYDRA:-\
algo.actor.ent_coef=${ACTOR_ENT_COEF} \
algo.replay_ratio=${REPLAY_RATIO} \
algo.horizon=${HORIZON} \
algo.world_model.kl_free_nats=${KL_FREE_NATS} \
algo.mlp_keys.encoder=[state]}"

echo "[launch-full] session=$SESSION_ID steps=$STEPS batch=$BATCH_SIZE"
echo "[launch-full] die 0.18 (scale=$LEROBOT_ISAAC_OBJECT_SCALE) at (${LEROBOT_ISAAC_OBJECT_X},${LEROBOT_ISAAC_OBJECT_Y}) fix_base=$LEROBOT_ISAAC_FIX_BASE staged=$LEROBOT_ISAAC_STAGED_REWARD"
echo "[launch-full] demo=$LEROBOT_ISAAC_DEMO_DATASET bc_weight=$LEROBOT_ISAAC_BC_WEIGHT object_pose=$LEROBOT_ISAAC_INCLUDE_OBJECT_POSE"
echo "[launch-full] EXTRA_HYDRA=$EXTRA_HYDRA"
echo "[launch-full] (demo_ratio=$DEMO_RATIO is documentation-only; not forwarded)"

export SESSION_ID STEPS BATCH_SIZE SECONDS_PER_EXP \
  LEROBOT_ISAAC_OBJECT_SCALE LEROBOT_ISAAC_OBJECT_X LEROBOT_ISAAC_OBJECT_Y \
  LEROBOT_ISAAC_FIX_BASE LEROBOT_ISAAC_STAGED_REWARD \
  LEROBOT_ISAAC_DEMO_DATASET LEROBOT_ISAAC_BC_WEIGHT \
  LEROBOT_ISAAC_INCLUDE_OBJECT_POSE EXTRA_HYDRA

exec bash scripts/_run_wm_isaac_overnight.sh
