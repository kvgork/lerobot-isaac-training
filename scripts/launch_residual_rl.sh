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
#
# Resume: set RESUME_FROM=<abs path to a .ckpt> to continue a prior run (mirrors
# scripts/launch_warmstart_cur.sh). CAUTION — sheeprl resume is NOT a full config
# override: only algo.total_steps and algo.learning_starts are re-read from the
# CLI/hydra args on resume; every other hydra key (batch_size, replay_ratio,
# checkpoint.every, max_episode_steps, ...) is re-read from the CHECKPOINT's own
# saved config.yaml, so those CANNOT be changed by re-launching with different env
# vars. Also: the residual-weight decay clock (LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT /
# _DECAY_STEPS) and the BC-weight clock (LEROBOT_ISAAC_BC_WEIGHT) are PER-PROCESS
# step counters that reset to 0 on resume — they do NOT know the checkpoint's
# global_step. So continuing a run whose script->policy handoff already completed
# needs LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS=1 (decay immediately) with WEIGHT left
# at 1.0 (never set WEIGHT=0 directly — that disarms the scripted base in every
# phase, not just post-handoff) and LEROBOT_ISAAC_BC_WEIGHT=0.0 (BC already handed
# off). Finally: SESSION_ID must be a NEW id on resume — the overnight wrapper
# truncates .agent-state/<id>/.../train.log on start, so reusing the old SESSION_ID
# destroys the prior run's log history.
#
# SIBLING SHA PINNING (2026-08-02): a 13h GPU run once resumed a checkpoint after
# src/lerobot-isaac-env had been switched from feature/wm-isaac-env to main. main
# lacks place_termination/FIX_BASE/staged-reward — six exported LEROBOT_ISAAC_* env
# vars silently became no-ops and the run could never register a success. Every
# launch now snapshots all src/*/ sibling SHAs (+ the workspace root) to a manifest
# next to its outputs, and a RESUME refuses to continue unless the resumed run's own
# manifest matches (see the block below the resume re-inject).
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
RESUME_FROM="${RESUME_FROM:-}"                 # sheeprl .ckpt path; empty = fresh run
EXTRA_HYDRA_APPEND="${EXTRA_HYDRA_APPEND:-}"   # extra hydra tokens APPENDED to the playbook default

# --- RESIDUAL RL: blend the scripted base into the executed action -----------
LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT="${LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT:-1.0}"          # w0 (pure scripted at step 0)
LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS="${LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS:-30000}"  # w0→0 handoff window

# C1 per-joint action scale — MANDATORY for any residual run (2026-07-24 root cause
# of v2-v5's zero grips): the seam clamps the executed action to [-1,1]; without
# this json the wrapper/env fall back to scale 0.5 and the wrist needs |action|
# 3.4-4.2 -> clamp saturates -> underdriven wrist -> DLS drifts the ee up, grasp
# never seats. The GATE always exported it (why every smoke could pass while every
# directly-launched full run failed). Fail loudly rather than launch an invalid run.
_LAUNCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEROBOT_ISAAC_ACTION_SCALE_JSON="${LEROBOT_ISAAC_ACTION_SCALE_JSON:-$_LAUNCH_ROOT/outputs/action_scale.json}"
[ -f "$LEROBOT_ISAAC_ACTION_SCALE_JSON" ] || {
  echo "[launch-residual] FATAL: action scale json missing: $LEROBOT_ISAAC_ACTION_SCALE_JSON" >&2
  echo "[launch-residual] run: LEROBOT_ISAAC_PLACE_REST_Z=-1 .pixi/envs/sim/bin/python scripts/_gen_sim_demos.py --measure_scale --obj_x 0.22 --obj_y -0.06 --grasp_z 0.106" >&2
  exit 3
}
export LEROBOT_ISAAC_ACTION_SCALE_JSON

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

# --- EXTRA_HYDRA clobber guard + resume re-inject --------------------------
# EXTRA_HYDRA is a ${VAR:-default} FULL REPLACEMENT: a caller who sets it to add
# one token silently drops every playbook knob. Dropping algo.mlp_keys.encoder=[state]
# disables the object_pose observation the scripted base needs. Warn loudly; prefer
# EXTRA_HYDRA_APPEND, which keeps the default intact.
for _tok in algo.actor.ent_coef algo.horizon algo.world_model.kl_free_nats algo.mlp_keys.encoder; do
  case "$EXTRA_HYDRA" in
    *"$_tok"*) ;;
    *) echo "[launch-residual] WARN: caller EXTRA_HYDRA replaced the playbook default and is missing '$_tok' — use EXTRA_HYDRA_APPEND instead" >&2 ;;
  esac
done
if [ -n "$EXTRA_HYDRA_APPEND" ]; then
  EXTRA_HYDRA="$EXTRA_HYDRA $EXTRA_HYDRA_APPEND"
fi
# Resume re-inject (mirrors scripts/launch_warmstart_cur.sh:115-120).
if [ -n "$RESUME_FROM" ]; then
  [ -f "$RESUME_FROM" ] || {
    echo "[launch-residual] FATAL: RESUME_FROM is not a file: $RESUME_FROM" >&2
    exit 4
  }
  case "$EXTRA_HYDRA" in
    *checkpoint.resume_from=*) ;;
    *) EXTRA_HYDRA="$EXTRA_HYDRA checkpoint.resume_from=$(readlink -f "$RESUME_FROM")" ;;
  esac
fi

# --- sibling SHA-pin manifest: record, and fail closed on resume mismatch ---
# Snapshot every editable sibling checkout under src/*/ (branch + SHA) plus the
# workspace root itself, so a resumed run can be checked against the sibling
# state that PRODUCED the checkpoint it is resuming from. See the incident note
# at the top of this file.
_MANIFEST_DIR="$_LAUNCH_ROOT/outputs/wm-isaac-prod-$SESSION_ID"
mkdir -p "$_MANIFEST_DIR"
_MANIFEST_FILE="$_MANIFEST_DIR/sibling_manifest.txt"
_MANIFEST_TMP="$_MANIFEST_FILE.tmp"
: > "$_MANIFEST_TMP"
for _d in "$_LAUNCH_ROOT"/src/*/; do
  [ -d "$_d" ] || continue
  git -C "$_d" rev-parse --git-dir >/dev/null 2>&1 || continue
  _name="$(basename "$_d")"
  _branch="$(git -C "$_d" rev-parse --abbrev-ref HEAD 2>/dev/null || echo UNKNOWN)"
  _sha="$(git -C "$_d" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
  printf '%s %s %s\n' "$_name" "$_branch" "$_sha" >> "$_MANIFEST_TMP"
done
sort -o "$_MANIFEST_TMP" "$_MANIFEST_TMP"
_sibling_count="$(wc -l < "$_MANIFEST_TMP" | tr -d '[:space:]')"
_ws_branch="$(git -C "$_LAUNCH_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo UNKNOWN)"
_ws_sha="$(git -C "$_LAUNCH_ROOT" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
_ws_line="workspace $_ws_branch $_ws_sha"
if [ -n "$(git -C "$_LAUNCH_ROOT" status --porcelain 2>/dev/null)" ]; then
  _ws_line="$_ws_line DIRTY"
fi
printf '%s\n' "$_ws_line" >> "$_MANIFEST_TMP"
mv "$_MANIFEST_TMP" "$_MANIFEST_FILE"
echo "[launch-residual] sibling manifest: $_sibling_count repos -> $_MANIFEST_FILE"

# NOTE: the sheeprl run dir (logs/runs/dreamer_v3/isaac_so101/<hydra-timestamp>_.../)
# is assigned by hydra INSIDE the subprocess this script execs into below — its
# name is not knowable here, so a second manifest copy cannot be placed there at
# launch time. Only the outputs/ copy above is written; _run_wm_isaac_overnight.sh
# (or a post-run step) would need to copy it into the run dir once known.

# Fail closed on resume: the resumed run must carry a manifest that matches ours.
if [ -n "$RESUME_FROM" ]; then
  _resolved_ckpt="$(readlink -f "$RESUME_FROM")"
  _ckpt_dir="$(dirname "$_resolved_ckpt")"          # .../version_0/checkpoint
  _resumed_run_dir="$(dirname "$(dirname "$_ckpt_dir")")"  # .../<run_dir>
  _resumed_manifest="$_resumed_run_dir/sibling_manifest.txt"
  if [ ! -f "$_resumed_manifest" ]; then
    echo "[launch-residual] WARNING: resumed run predates sibling-manifest pinning — cannot verify sibling SHAs" >&2
    echo "[launch-residual] resumed run: $RESUME_FROM (expected manifest at $_resumed_manifest)" >&2
    echo "[launch-residual] current sibling manifest:" >&2
    cat "$_MANIFEST_FILE" >&2
    if [ "${ALLOW_UNPINNED_RESUME:-0}" = "1" ]; then
      echo "[launch-residual] WARNING: ALLOW_UNPINNED_RESUME=1 set — proceeding without verification" >&2
    else
      echo "[launch-residual] FATAL: refusing to resume an unpinned run without sibling verification" >&2
      echo "[launch-residual] to override: ALLOW_UNPINNED_RESUME=1 RESUME_FROM=$RESUME_FROM bash scripts/launch_residual_rl.sh" >&2
      exit 5
    fi
  elif _diff_out="$(diff -u "$_resumed_manifest" "$_MANIFEST_FILE")"; then
    echo "[launch-residual] sibling manifest MATCHES resumed run — OK"
  else
    echo "[launch-residual] FATAL: sibling manifest MISMATCH vs resumed run $_resumed_manifest" >&2
    echo "$_diff_out" >&2
    if [ "${ALLOW_SIBLING_DRIFT:-0}" = "1" ]; then
      echo "[launch-residual] WARNING: ALLOW_SIBLING_DRIFT=1 set — proceeding despite sibling SHA drift (run may not reproduce the checkpoint's behavior)" >&2
    else
      echo "[launch-residual] to override: ALLOW_SIBLING_DRIFT=1 RESUME_FROM=$RESUME_FROM bash scripts/launch_residual_rl.sh" >&2
      exit 5
    fi
  fi
fi

echo "[launch-residual] session=$SESSION_ID steps=$STEPS batch=$BATCH_SIZE ep_len=$MAX_EPISODE_STEPS"
echo "[launch-residual] residual w0=$LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT decay=$LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS replay_ratio=$REPLAY_RATIO"
echo "[launch-residual] cup place: obj (${LEROBOT_ISAAC_OBJECT_X},${LEROBOT_ISAAC_OBJECT_Y}) carry_z=$LEROBOT_ISAAC_CARRY_Z place_bonus=$LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT"
echo "[launch-residual] demo=$LEROBOT_ISAAC_DEMO_DATASET bc_weight=$LEROBOT_ISAAC_BC_WEIGHT object_pose=$LEROBOT_ISAAC_INCLUDE_OBJECT_POSE"
echo "[launch-residual] resume_from=${RESUME_FROM:-<none>}"
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
