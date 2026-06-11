# Scripted pick-place grasp — finish-it plan (2026-06-11)

Goal: get `scripts/_scripted_pickplace.py` to actually grasp+lift+carry+place the 16 mm die, so it
can generate **sim demos** for the demo-buffer warm-start (the carry→place unlock). Best done with
a hand eyeballing the sim (GUI/video) — the grasp is a fine-alignment problem that's slow to tune blind.

## Current state (works)
- Env: fixed-base SO-101 (`LEROBOT_ISAAC_FIX_BASE=1`), 16 mm die (`LEROBOT_ISAAC_OBJECT_SCALE=0.267`),
  object (0.22,0.05) rests z≈0.008, target bin (0.22,−0.13).
- IK: `DifferentialIKController`, **pose** mode. Adaptive jacobian indexing already handles
  fixed-base (row=ee_idx−1, cols=arm_ids, no +6 offset). Position+orientation tracking works.
- Gripper descends to gripper_link z≈0.052 with `GRASP_QUAT=[-0.8593,-0.0507,0.507,-0.0434]`
  (downward) → **fingertips reach the die, it gets nudged**. Action map: `a=(q_des−q_default)/0.5`,
  gripper dim: +1 close / −1 open (closes toward upper limit).

## The blocker (what to fix)
Grasp **capture** fails: the 5-DOF arm can't hit exact xy AND full-down orientation, so DLS
compromises (~2 cm xy error) → the gripper lands off-center and **pushes** the die instead of
straddling+gripping it. Die nudges (0.05→0.056 y) but never lifts.

## What to try (ordered, eyeball the sim)
1. **Watch it** — render a GUI/video of one rollout (`headless=False` standalone, or sheeprl
   capture_video) to SEE the gripper-vs-die alignment at the close. The numbers can't show whether
   the jaws straddle the die.
2. **xy-center the grasp:** re-read the LIVE die pose right before the descend (it may have moved);
   command the gripper to the die xy, and add a small empirical xy offset to cancel the DLS
   undershoot (gripper landed at (0.230,0.030) when commanded (0.22,0.05) → bias by +(−0.01,+0.02)).
3. **Relax orientation weight** so position dominates: try `ik_method="dls"` with a larger
   `lambda_val`, or command a less-extreme downward quat (the 5-DOF arm can hit position better if
   orientation isn't fully constrained). A partial downward tilt may suffice if the jaw opening
   straddles the 16 mm die.
4. **Lower + dwell:** descend gripper_link to ~0.04, dwell 30+ steps before closing; close over
   50+ steps; verify with the grip-physics check (object z rises with the arm).
5. **If still failing:** widen the gripper open (more negative gripper action) on approach so the
   jaws clear the die, then close; or shrink the die slightly (`OBJECT_SCALE` 0.20) to fit the jaw.

## Verify success
Object xy reaches within 6 cm of the target bin AND object z rises >0.05 during carry (not just
nudged). `scripts/_scripted_pickplace.py` already prints the per-phase obj/ee trace + SUCCESS.

## Then — generate demos
Once a rollout SUCCEEDs: run the controller ~30× with small object-pose jitter (vary OBJECT_X/Y
within reach) → record via `lerobot_isaac_adapters.isaac_data_recorder.record_episodes()` →
sim LeRobotDataset under `datasets/local/so101-sim-pickplace-demos/`. These feed the demo-buffer
(`plans/2026-06-11-demo-warmstart-plan.md`).

## Files
- `scripts/_scripted_pickplace.py` — the controller (IK works, grasp capture WIP)
- `scripts/_graspose_probe.py`, `_orient_probe.py`, `_grip_physics_probe.py`, `_reach_down_probe.py` — diagnostics
- Alternative that sidesteps the grasp entirely: RFCL reverse-curriculum (object-in-bin state-resets),
  see the demo-warmstart plan.
