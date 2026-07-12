#!/usr/bin/env bash
# =============================================================================
# launch_residual_rl.sh — residual-RL on the scripted-grasp base (DreamerV3, sim).
#
# Created 2026-07-12. The campaign conclusion (see memory scripted-grasp-infeasible /
# carryplace-*): pure BC fails by compounding error and pure RL fails by sparse-reward
# discovery on the knife-edge sim grasp, while the scripted analytical-IK controller
# does ~67-80%. So learn a RESIDUAL on the working scripted base instead of the whole
# visuomotor grasp from scratch.
#
# LAUNCH-READY FIX (2026-07-12): the scripted base used to STALL in APPROACH under the
# residual blend (its reactive gate `aligned = xy_to_tgt < 0.015` never cleared once
# the action is blended + clamped to [-1,1]). It now uses per-phase HARD step caps
# (lerobot_isaac_adapters.scripted_grasp_phases.next_phase) so every phase force-
# advances and the machine can never stall. Caps sum ≈ 540 steps ⇒ MAX_EPISODE_STEPS
# is 700 here so a full pick→place fits inside one episode.
#
# The residual is blended at PlayerDV3.get_actions BEFORE rb.add (eval-guarded) in
# scripts/_wm_isaac_entry.py — see [[sheeprl-action-override-buffer-seam]]. w0 is the
# script fraction at step 0 (1.0 = pure scripted warmup); it decays w0→0 over
# RESIDUAL_RL_DECAY_STEPS (DAgger/residual handoff: script-dominant early, policy late).
#
# Run (multi-hour GPU on the Isaac host; needs the `sim`/train-dreamer stack):
#   bash scripts/launch_residual_rl.sh
# Smoke first (short). IMPORTANT: the residual patch lives on PlayerDV3.get_actions,
# which sheeprl only calls AFTER algo.learning_starts (default 1024) — before that it
# collects with random action_space.sample(). So a smoke MUST either run past 1024
# steps OR lower learning_starts, else get_actions (and the residual) is never exercised
# and you just measure random prefill. Lower it for a fast smoke:
#   STEPS=700 MAX_EPISODE_STEPS=700 SECONDS_PER_EXP=2400 SESSION_ID=residual-smoke \
#     EXTRA_HYDRA='algo.actor.ent_coef=1e-3 algo.horizon=25 algo.world_model.kl_free_nats=1.0 algo.mlp_keys.encoder=[state] algo.learning_starts=200' \
#     bash scripts/launch_residual_rl.sh
# Then confirm in the train log: [residual-rl] scripted-grasp controller INITIALISED +
# ENGAGED, and [script-dbg] phase=... advancing beyond APPROACH (DESCEND/CLOSE/LIFT/
# CARRY, obj_lifted=True), with reward climbing above the ~-61 random-policy floor.
# (The real run keeps the default learning_starts=1024.)
# =============================================================================
set -euo pipefail

# --- run identity / budget --------------------------------------------------
SESSION_ID="${SESSION_ID:-residual-rl-v1}"
STEPS="${STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-8}"                 # 10 GB 3080: 8 fits; drop to 4 on OOM
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-700}" # caps sum ~540 → need >600 for a full place
NUM_ENVS="${NUM_ENVS:-1}"                     # residual supports single-env only (one scripted
                                              # action can't broadcast; also the wm-isaac num_envs bug)
REPLAY_RATIO="${REPLAY_RATIO:-4}"             # NOT 16 — wall-clock-fatal on online num_envs=1 Isaac
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"  # early ckpt (the cup0 no-ckpt bug: never reached 10000)
SECONDS_PER_EXP="${SECONDS_PER_EXP:-46800}"   # 13 h ceiling (launcher: timeout "$SECONDS_PER_EXP")
# inner train_wrapper timeout must not preempt the outer cap + kill the checkpoint (cup0 bug):
LEROBOT_TRAIN_TIMEOUT="${LEROBOT_TRAIN_TIMEOUT:-$SECONDS_PER_EXP}"

# --- RESIDUAL RL: blend the scripted base into the executed action -----------
LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT="${LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT:-1.0}"          # w0 (pure scripted at step 0)
LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS="${LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS:-30000}"  # w0→0 handoff window

# --- task geometry: real place into the cup (matches the cup0-wide demos) ----
LEROBOT_ISAAC_OBJECT_SCALE="${LEROBOT_ISAAC_OBJECT_SCALE:-0.267}"   # 16 mm die
LEROBOT_ISAAC_OBJECT_FRICTION="${LEROBOT_ISAAC_OBJECT_FRICTION:-3.0}"
LEROBOT_ISAAC_FIX_BASE="${LEROBOT_ISAAC_FIX_BASE:-1}"
LEROBOT_ISAAC_STAGED_REWARD="${LEROBOT_ISAAC_STAGED_REWARD:-1}"
LEROBOT_ISAAC_PLACE_CUP="${LEROBOT_ISAAC_PLACE_CUP:-1}"
LEROBOT_ISAAC_CARRY_Z="${LEROBOT_ISAAC_CARRY_Z:-0.19}"             # clears the 7 cm rim; 0.22 breaks the grasp
LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT="${LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT:-50.0}"  # salient terminal place bonus
LEROBOT_ISAAC_OBJECT_X="${LEROBOT_ISAAC_OBJECT_X:-0.22}"
LEROBOT_ISAAC_OBJECT_Y="${LEROBOT_ISAAC_OBJECT_Y:--0.06}"

# --- observation: the actor + WM must SEE the object pose the script uses -----
LEROBOT_ISAAC_INCLUDE_OBJECT_POSE="${LEROBOT_ISAAC_INCLUDE_OBJECT_POSE:-1}"

# --- optional demo seeding + DreamerFD BC actor loss (helps, not required) ----
LEROBOT_ISAAC_DEMO_DATASET="${LEROBOT_ISAAC_DEMO_DATASET:-datasets/local/so101-sim-pickplace-demos-cup0-wide}"
LEROBOT_ISAAC_BC_WEIGHT="${LEROBOT_ISAAC_BC_WEIGHT:-1.0}"

# --- exploration knobs (wm-vla playbook) + encode the object_pose state -------
ACTOR_ENT_COEF="${ACTOR_ENT_COEF:-1e-3}"
HORIZON="${HORIZON:-25}"
KL_FREE_NATS="${KL_FREE_NATS:-1.0}"
EXTRA_HYDRA="${EXTRA_HYDRA:-\
algo.actor.ent_coef=${ACTOR_ENT_COEF} \
algo.horizon=${HORIZON} \
algo.world_model.kl_free_nats=${KL_FREE_NATS} \
algo.mlp_keys.encoder=[state]}"

echo "[launch-residual] session=$SESSION_ID steps=$STEPS batch=$BATCH_SIZE ep_len=$MAX_EPISODE_STEPS"
echo "[launch-residual] residual w0=$LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT decay=$LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS replay_ratio=$REPLAY_RATIO"
echo "[launch-residual] cup place: obj (${LEROBOT_ISAAC_OBJECT_X},${LEROBOT_ISAAC_OBJECT_Y}) carry_z=$LEROBOT_ISAAC_CARRY_Z place_bonus=$LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT"
echo "[launch-residual] demo=$LEROBOT_ISAAC_DEMO_DATASET bc_weight=$LEROBOT_ISAAC_BC_WEIGHT object_pose=$LEROBOT_ISAAC_INCLUDE_OBJECT_POSE"
echo "[launch-residual] EXTRA_HYDRA=$EXTRA_HYDRA"

export SESSION_ID STEPS BATCH_SIZE MAX_EPISODE_STEPS NUM_ENVS REPLAY_RATIO \
  CHECKPOINT_EVERY SECONDS_PER_EXP LEROBOT_TRAIN_TIMEOUT \
  LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS \
  LEROBOT_ISAAC_OBJECT_SCALE LEROBOT_ISAAC_OBJECT_FRICTION LEROBOT_ISAAC_FIX_BASE \
  LEROBOT_ISAAC_STAGED_REWARD LEROBOT_ISAAC_PLACE_CUP LEROBOT_ISAAC_CARRY_Z \
  LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT LEROBOT_ISAAC_OBJECT_X LEROBOT_ISAAC_OBJECT_Y \
  LEROBOT_ISAAC_INCLUDE_OBJECT_POSE LEROBOT_ISAAC_DEMO_DATASET LEROBOT_ISAAC_BC_WEIGHT \
  EXTRA_HYDRA

exec bash scripts/_run_wm_isaac_overnight.sh
