#!/usr/bin/env bash
# =============================================================================
# launch_warmstart_cur.sh — easy-cup CURRICULUM warm-start DreamerV3 launch.
#
# Recreated 2026-06-24 (was in an ephemeral scratchpad/, never committed).
# Source plans:
#   - plans/2026-06-23-carryplace-cup-campaign.md  "RECOMMENDED next: easy-cup
#                                                   curriculum" (Stage 0..N) +
#                                                   the Keep + ADD-counter-levers lines
#   - plans/2026-06-23-world-model-training-plan.md §3 Stage 2 (the launch seam
#                                                   + wm-vla playbook knobs)
#
# The lever that ever worked (cur1, 2026-06-16: 30% places) was an easy-enough
# start that exploration STUMBLES into a place. This template hardens the task
# on TWO axes — cup height + carry distance — one stage at a time, resuming each
# stage's checkpoint.
#
# HOW TO USE: edit the PER-STAGE block below for the stage you are launching,
# then: bash scripts/launch_warmstart_cur.sh   (multi-hour GPU — do NOT run
# unless you mean it). Resume a prior stage via RESUME_FROM=<ckpt>.
#
# Wraps scripts/_run_wm_isaac_overnight.sh (consumes STEPS/BATCH_SIZE/SESSION_ID
# /EXTRA_HYDRA env). LEROBOT_ISAAC_* are read by scripts/_wm_isaac_entry.py +
# the Isaac env. PLACE_CUP_HEIGHT is read at SCENE BUILD, so it is an env knob,
# NOT a sheeprl override.
#
# ---------------------------------------------------------------------------
# CURRICULUM STAGE MAP (edit ONE stage's block at the top, then launch):
#
#   Stage 0  cup 0.03, die IN/at the cup  (OBJECT_X=0.22 OBJECT_Y=-0.13)
#            -> trivial carry; agent discovers lift -> release-in-cup.
#            Regenerate matched demos FIRST:
#              LEROBOT_ISAAC_PLACE_CUP_HEIGHT=0.03 \
#                python scripts/_gen_sim_demos.py --obj_x 0.22 --obj_y -0.13
#            (cup height is an ENV knob read at scene build, NOT a script flag.)
#            RESUME_FROM= (empty — train from scratch).
#
#   Stage 1  cup 0.03, die ~6 cm out  (OBJECT_X=0.22 OBJECT_Y=-0.05)
#            -> short carry. RESUME_FROM=<Stage-0 ckpt>.
#
#   Stage 2..N  raise PLACE_CUP_HEIGHT 0.03 -> 0.07 AND push the die out toward
#            (0.18, 0.05), resuming each prior ckpt. Bump STEPS each stage.
#            Resume pattern: EXTRA_HYDRA gets checkpoint.resume_from=<ckpt>
#            (handled automatically below when RESUME_FROM is set).
# ---------------------------------------------------------------------------
# =============================================================================
set -euo pipefail

# ======================= EDIT THIS PER-STAGE BLOCK ==========================
# Stage-0 defaults (cup 0.03 + die-in-cup, trivial carry):
STAGE="${STAGE:-0}"
OBJECT_X="${OBJECT_X:-0.22}"
OBJECT_Y="${OBJECT_Y:--0.13}"
PLACE_CUP_HEIGHT="${PLACE_CUP_HEIGHT:-0.03}"
RESUME_FROM="${RESUME_FROM:-}"           # sheeprl ckpt .ckpt path; empty = from scratch
# ============================================================================

# --- run identity / budget --------------------------------------------------
SESSION_ID="${SESSION_ID:-wm-warmstart-cur-stage${STAGE}}"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-8}"            # 10 GB 3080: 8 fits; drop to 4 on OOM
SECONDS_PER_EXP="${SECONDS_PER_EXP:-46800}"   # 13 h hard ceiling — launcher runs: timeout "$SECONDS_PER_EXP" (default 6 h). NOT LEROBOT_TRAIN_TIMEOUT (dead in this launcher).

# --- task geometry (per-stage; die 0.18 == OBJECT_SCALE 0.267) --------------
LEROBOT_ISAAC_OBJECT_SCALE="${LEROBOT_ISAAC_OBJECT_SCALE:-0.267}"
LEROBOT_ISAAC_OBJECT_X="${LEROBOT_ISAAC_OBJECT_X:-$OBJECT_X}"
LEROBOT_ISAAC_OBJECT_Y="${LEROBOT_ISAAC_OBJECT_Y:-$OBJECT_Y}"
LEROBOT_ISAAC_FIX_BASE="${LEROBOT_ISAAC_FIX_BASE:-1}"
LEROBOT_ISAAC_STAGED_REWARD="${LEROBOT_ISAAC_STAGED_REWARD:-1}"

# --- cup geometry: easy-cup curriculum axis (read at scene build) -----------
LEROBOT_ISAAC_PLACE_CUP="${LEROBOT_ISAAC_PLACE_CUP:-1}"
LEROBOT_ISAAC_PLACE_CUP_HEIGHT="${LEROBOT_ISAAC_PLACE_CUP_HEIGHT:-$PLACE_CUP_HEIGHT}"

# --- place bonus (so a discovered place gets a positive reward spike) -------
# Verified knob name: LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT (env grep 2026-06-24).
LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT="${LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT:-1.0}"

# --- warm-start: demo seeding + DreamerFD BC actor loss ---------------------
# Per-stage: regenerate matched demos at the stage's cup-height + obj x/y and
# point LEROBOT_ISAAC_DEMO_DATASET at them. Default uses the base demos dir.
LEROBOT_ISAAC_DEMO_DATASET="${LEROBOT_ISAAC_DEMO_DATASET:-datasets/local/so101-sim-pickplace-demos}"
LEROBOT_ISAAC_BC_WEIGHT="${LEROBOT_ISAAC_BC_WEIGHT:-1.0}"

# --- observation: actor must SEE the object pose ----------------------------
LEROBOT_ISAAC_INCLUDE_OBJECT_POSE="${LEROBOT_ISAAC_INCLUDE_OBJECT_POSE:-1}"

# --- wm-vla playbook exploration counter-levers (BOTH v1 plateaus omitted) --
# Verified key paths against sheeprl dreamer_v3.yaml (train-dreamer env):
#   algo.actor.ent_coef (3e-4 -> 1e-3), algo.replay_ratio (1 -> 16),
#   algo.horizon (15 -> 25), algo.world_model.kl_free_nats (1.0),
#   algo.mlp_keys.encoder=[state]  (encode the object_pose state vector).
ACTOR_ENT_COEF="${ACTOR_ENT_COEF:-1e-3}"
REPLAY_RATIO="${REPLAY_RATIO:-16}"
HORIZON="${HORIZON:-25}"
KL_FREE_NATS="${KL_FREE_NATS:-1.0}"
# "demo_ratio 0.5" is NOT a sheeprl key — in this repo the demo contribution is
# the DreamerFD BC loss (LEROBOT_ISAAC_BC_WEIGHT) + replay seed; documentation
# only, NOT forwarded to sheeprl. TODO: plumb a real demo-mix ratio through the
# BC patch in scripts/_wm_isaac_entry.py if ever wanted.
DEMO_RATIO="${DEMO_RATIO:-0.5}"   # documentation only; not a sheeprl key

# --- assemble EXTRA_HYDRA (playbook knobs + optional resume) ----------------
PLAYBOOK_HYDRA="\
algo.actor.ent_coef=${ACTOR_ENT_COEF} \
algo.replay_ratio=${REPLAY_RATIO} \
algo.horizon=${HORIZON} \
algo.world_model.kl_free_nats=${KL_FREE_NATS} \
algo.mlp_keys.encoder=[state]"

# Resume pattern: cur1->cur2 used EXTRA_HYDRA=checkpoint.resume_from=<ckpt>
# (dreamerv3-carryplace-launch-gotchas). torch.load weights_only=False is
# already patched in _wm_isaac_entry.py so the buffer/cfg pickle loads.
EXTRA_HYDRA="${EXTRA_HYDRA:-$PLAYBOOK_HYDRA}"
# Honor RESUME_FROM even when EXTRA_HYDRA was set externally — the :- fallback
# above would otherwise silently drop the resume token and train from scratch:
if [ -n "$RESUME_FROM" ] && [[ "$EXTRA_HYDRA" != *checkpoint.resume_from=* ]]; then
  EXTRA_HYDRA="${EXTRA_HYDRA} checkpoint.resume_from=${RESUME_FROM}"
fi

echo "[launch-cur] STAGE=$STAGE session=$SESSION_ID steps=$STEPS batch=$BATCH_SIZE"
echo "[launch-cur] cup_height=$LEROBOT_ISAAC_PLACE_CUP_HEIGHT die at (${LEROBOT_ISAAC_OBJECT_X},${LEROBOT_ISAAC_OBJECT_Y})"
echo "[launch-cur] place_success_weight=$LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT bc_weight=$LEROBOT_ISAAC_BC_WEIGHT demo=$LEROBOT_ISAAC_DEMO_DATASET"
echo "[launch-cur] resume_from=${RESUME_FROM:-<none>}"
echo "[launch-cur] EXTRA_HYDRA=$EXTRA_HYDRA"
echo "[launch-cur] (demo_ratio=$DEMO_RATIO is documentation-only; not forwarded)"

export SESSION_ID STEPS BATCH_SIZE SECONDS_PER_EXP \
  LEROBOT_ISAAC_OBJECT_SCALE LEROBOT_ISAAC_OBJECT_X LEROBOT_ISAAC_OBJECT_Y \
  LEROBOT_ISAAC_FIX_BASE LEROBOT_ISAAC_STAGED_REWARD \
  LEROBOT_ISAAC_PLACE_CUP LEROBOT_ISAAC_PLACE_CUP_HEIGHT \
  LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT \
  LEROBOT_ISAAC_DEMO_DATASET LEROBOT_ISAAC_BC_WEIGHT \
  LEROBOT_ISAAC_INCLUDE_OBJECT_POSE EXTRA_HYDRA

exec bash scripts/_run_wm_isaac_overnight.sh
