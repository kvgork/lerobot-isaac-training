# Scripted grasp — manual investigation plan (2026-06-13)

**Goal:** get the SO-101 to physically grasp+lift the die in sim so `scripts/_scripted_pickplace.py`
can generate pick→place **demos** for the DreamerV3 warm-start
(`plans/2026-06-11-demo-warmstart-plan.md` Stage 1). Automated tuning is exhausted — this
plan is for **hands-on GUI investigation** (the grasp is a fine-alignment problem the
numbers can't fully see).

---

## TL;DR of where we are

A physical grasp **does not work yet** after extensive automated tuning (240 trials, 0
grips). The arm/pose/height are all correct; the failure is the **single moving jaw
shoving the free die out of the gap as it closes**. The decision: either crack it
manually (this plan, Part C) or take a sanctioned **sidestep** (Part E) to unblock the
warm-start.

---

## Part A — What is CONFIRMED working / true (don't re-investigate)

| Fact | Evidence |
|------|----------|
| Straight-down orientation IS reachable | `GRASP_QUAT=[1,0,0,0]` (identity) → gripper_link local **−z** points world **−Z**. `_grasp_joint_diag.py`: down_dot=1.0, **no joint near its limit**. |
| Position tracking is exact | diag: target `[0.18,0.05,0.146]` → reached `[0.18,0.05,0.145]`. |
| The earlier "28° tilt / can't go vertical" was WRONG | artifact of the coarse `_approach_axis_probe.py` grid + a sign-label bug. Vertical is fine at die x≤~0.18. |
| Gripper sign | `+1` (joint→+0.5) = **OPEN**; `−1` (joint→−0.1745) = **CLOSE**. (Old docstring was reversed.) |
| Gripper closed stop | joint **−0.1745 rad** is a hard lower limit (commanding −3× doesn't go further). |
| Grasp height | gripper_link `z≈0.106–0.108` puts fingertips at the settled 16 mm die (rest z=**0.008**, NOT the 0.048 spawn — die settles). Fingertip offset ≈ 0.10 below gripper_link. |
| Colliders exist on BOTH jaws | USD `Payload/Physics.usda`: moving_jaw + fixed (sts3215 / wrist_roll_follower) all `convexHull`. GUI collider-viz confirmed "look good". |
| Straight-down **forward reach limit** ≈ ee_x **0.218** | beyond that, DLS returns the boundary pose; die default 0.22 is at/over the limit → use die x ≈ 0.16–0.18. |
| Single moving jaw | only `moving_jaw_so101_v1_link` actuates, closing against the fixed `gripper_frame`/`wrist_roll_follower`. Not two symmetric fingers. |

## Part B — What FAILS and the mechanism

- **240 trials, 0 grips.** `_grasp_pose_sweep.py`: 48 poses (roll 0/45/90/135° × bias_x∈
  {−0.04,−0.02,0,0.02} × bias_y∈{−0.02,0,0.02}) × {16 mm die, 24 mm die, PhysX friction 2.0}.
  Die **never lifts** (max lift_z=0.011 = rest).
- **Mechanism:** at correct height + straight-down, the closing moving jaw **contacts the
  die and pushes it out of the gap** (push up to 3.7 cm) before pinning it against the
  fixed finger. The die ends up **outside the moving finger** (your GUI observation).
- Friction (→2.0), bigger die (24 mm), and slow ramped close did **not** fix it.

**Working hypothesis:** the convexHull finger geometry + single-sided closing sweep can't
trap a small free cube — the contact normal pushes the cube laterally faster than the jaw
traps it. This is an **asset/contact** problem, not control.

---

## Part C — Manual GUI investigation checklist (ordered, do in the viewport)

Run the controller in **hold mode** so the sim stays live and you can rotate the camera +
toggle collision viz. Base command (vary the flags):

```bash
LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 \
  LEROBOT_ISAAC_OBJECT_X=0.18 LEROBOT_ISAAC_OBJECT_Y=0.05 LEROBOT_ISAAC_STAGED_REWARD=1 \
  .pixi/envs/sim/bin/python scripts/_scripted_pickplace.py \
    --gui --grasp_z 0.106 --bias_x 0.0 --close_steps 80 --hold 3000 \
    --out outputs/scripted-pickplace-hold.json
```

Collision viz: viewport **eye icon → Show By Type → Physics → Colliders → All**.

1. **Watch the close frame-by-frame.** Does the moving jaw's leading edge hit the die
   FIRST (one-sided) and flick it, or do both faces meet the die together? If one-sided →
   the die must sit deeper toward the fixed finger before closing.
2. **Measure the CLOSED gap vs die.** With `--hold`, drive to closed (the script closes
   before holding). Eyeball / use collider viz: is the gap at full-close (joint −0.1745)
   **larger than 16 mm**? If yes, the jaw can't pinch a 16 mm cube at all → need a die
   sized to the gap, or extend the joint close range (Part D-3).
3. **Find where the die must sit to be trapped.** Manually nudge `--bias_x` / `--bias_y`
   (try ±0.01 … ±0.06, both axes — the gap axis is ~34° off x, so a diagonal bias may be
   needed) until, at the hold, the die sits **between** the fixed and moving finger, not
   outside. Note the winning bias.
4. **Watch the descend.** Does a fingertip clip the die before the gripper bottoms (die
   nudges during descend)? If so, the die starts off-center under the gripper — adjust
   bias so the descend lands the gap around the die.
5. **Try opening WIDER on approach.** The "open" action is +1 → joint +0.5, but the joint
   max is +1.745. The gripper is only partly open. Test a wider open by editing the
   controller's `GRIP_OPEN` toward +3 (unclipped → joint clamps at +1.745) so the jaws
   clear the die on the way down, then close.

## Part D — Asset/physics levers (if Part C shows a geometry problem)

1. **Finger collision approximation.** `assets/usd/Payload/Physics.usda` lines ~305–319:
   moving_jaw uses `physics:approximation = "convexHull"`. A convex hull **fills the
   finger's concavity**, so the gripping face may bulge. Try `convexDecomposition` (or
   `sdf`/`triangleMesh` for the static parts) so the real concave jaw surface grips. Edit
   the `.usda`, re-test. (Back up `Physics.usda` first.)
2. **Friction at the jaw, not just the die.** We bumped die friction to 2.0 (no help).
   Also set the **gripper finger** material friction high (robot articulation material) —
   combined friction uses both surfaces. Runtime via `robot.root_physx_view`
   `get/set_material_properties` (see `_grasp_pose_sweep.py` for the die pattern).
3. **Extend the gripper close range.** Joint lower limit is −0.1745 rad. If the closed gap
   is too wide, lower the limit in the articulation cfg (`so101_articulation.py`, actuator
   / joint limits) so the jaws close further — but verify the fingers don't interpenetrate
   (unphysical). The real SO-101 jaws touch at full close; the sim limit may be wrong.
4. **Contact offset / rest offset.** PhysX contact offset on the finger colliders may be
   too small for stable small-object contact. Bump via `collision_props` /
   `RigidBodyPropertiesCfg` on the gripper bodies.

## Part D-tooling — scripts built this session (reuse these)

| Script | What it does |
|--------|--------------|
| `scripts/_scripted_pickplace.py` | the controller. Flags: `--gui --grasp_z --bias_x --bias_y --dwell --close_steps --hold`. `GRASP_QUAT=[1,0,0,0]` straight-down; `GRIP_OPEN=+1`/`GRIP_CLOSE=−1`; ramped close. |
| `scripts/_grasp_pose_sweep.py` | headless 48-pose sweep (roll × bias), scored by die lift_z. Settles die before reading height; sets die friction (`LEROBOT_ISAAC_OBJECT_FRICTION`). os._exit (no teardown hang). |
| `scripts/_grasp_joint_diag.py` | prints arm joint pos vs limits + down_dot at the grasp pose. |
| `scripts/_gripper_physics_probe.py` | gripper joint limits + open/closed travel + action-cap vs joint-stop test. |
| `scripts/_jaw_width_probe.py` | jaw actuation + gap geometry (note: link origins sit at the hinge — misleading for the contact gap; the ledger warns link-frame ≠ contact point). |
| `scripts/_approach_axis_probe.py` | (coarse, partly superseded) finds gripper finger axis = local −z. |

**Probe discipline:** all probes use `os._exit(0)` to skip Isaac's hanging `app.close()`
teardown (the WM-Isaac pitfall — without it a finished probe sits at 115% CPU "running"
for an hour). Read die z **after a settle loop**, not at reset (spawn z 0.048 ≠ settled 0.008).

---

## Part E — Sidesteps (if manual grasp stays infeasible) — RECOMMENDED to unblock

The grasp is **not required** to generate warm-start demos. Two sanctioned routes:

1. **Kinematic-weld demos (fastest, ~1 controller edit).** After the jaws close at the
   die, set the die root pose to track the EE (`obj.write_root_pose_to_sim`) through
   lift→carry→release. Produces clean pick→place (obs, action) demos. The WM/RL learns the
   correct trajectory — which is the entire point of demos. Faked contact only.
2. **RFCL reverse-curriculum.** Reset the env with the die already in-gripper / in-bin via
   `write_root_state_to_sim`, train RL backward (in-bin → further out). No grasp needed;
   needs a state-reset wrapper. Most principled per the demo-warmstart plan.

**Decision rule:** if Part C (≤~1 hr manual) doesn't yield a holding grasp, go to E-1
(weld) to unblock Stages 2–4; revisit a real grasp later as asset work (Part D) only if
sim-physical demos are required.

---

## Part F — Once a grasp (or weld) works → downstream (unchanged)

- **Stage 2 demo-gen:** run controller ~30–50× with object-pose jitter (`OBJECT_X/Y`
  within straight-down reach r≲0.20) → LeRobotDataset at
  `datasets/local/so101-sim-pickplace-demos/`. NOTE wiring gap: `record_episodes()` takes
  a stateless `policy_fn`; the controller is phased/stateful → needs a stateful-closure
  adapter or a dedicated demo-recorder that drops non-SUCCESS rollouts.
- **Stage 3 warm-start:** seed sheeprl DreamerV3 replay buffer (user order 1→3→2).
- **Stage 4 verify:** `scripts/_sim_eval.py` → pc_success.
- **Task re-home:** demos used die x≈0.16–0.18 (straight-down reach). Move the RL task
  object/target to match (bin at the same reach), and re-home `_TARGET_POS` inward too.

## Related
- `plans/2026-06-11-scripted-grasp-plan.md` (original) · `plans/2026-06-11-demo-warmstart-plan.md`
  · `plans/2026-06-10-data-collection-and-plateau-break-plan.md`
- memory: `scripted-grasp-infeasible.md`, `so101-sim-reach-envelope.md`
