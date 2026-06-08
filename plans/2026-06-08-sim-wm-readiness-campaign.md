# Plan: Sim WM readiness campaign (2026-06-08)

A sequenced GPU campaign to get a *meaningful* sim world-model + pick-place policy:
**Step 1 staged per-term sanity → Step 2 camera pose calibration → Step 3 staged-reward
DreamerV3 run**. Steps are strictly ordered: 1 unblocks 3; 2 makes any sim result trustworthy.
Single RTX 3080 → sequential.

Cross-refs (do not duplicate): `plans/2026-06-08-staged-reward-tuning-plan.md` (full reward
tuning detail), `plans/2026-06-07-good-world-model-plan.md` (WM path + Fix 2),
`plans/2026-06-08-fix2-isaac-vectorization-plan.md` (num_envs>1, separate track).

---

## Step 1 — Staged per-term reward sanity  (~20 min GPU)

**Goal:** confirm the staged terms (registered, committed `cedfd19`) actually *fire when the
object moves* — not just that they exist.

**Run:** short online DreamerV3, num_envs=1, staged on:
```
STEPS=2500 NUM_ENVS=1 SECONDS_PER_EXP=1200 LEROBOT_ISAAC_STAGED_REWARD=1 \
SESSION_ID=staged-sanity-<ts> bash scripts/_run_wm_isaac_overnight.sh
```
**Instrument:** read the run's TB scalars for per-term reward (`Reward/grasp`, `Reward/lift`,
`Reward/place`, `Reward/progress`). Optionally set `PROGRESS_REWARD_DEBUG=1` for the EE/obj
distance print.

**Pass criteria:**
- `progress` non-zero from step 0 (reach gradient present).
- `grasp` ≈ 0 until EE is near the object; spikes when close (tighten `std` if it fires too
  early/wide).
- `lift` ≈ 0 until the cube rises; **manually sanity-check**: it must be *capable* of going
  positive (e.g. confirm `obj_z` reads sensibly). If it never moves during exploration that's
  fine (actor hasn't grasped) — the test is *capability*, not learned behaviour.
- `place` stays 0 until lifted (gate works).
- No crash; 6 reward terms present (already confirmed).

**Output:** a note in the run dir + decide initial weight ladder for Step 3.

---

## Step 2 — Camera pose calibration  (~30–60 min GPU + iterate)

**Why:** the policy trained on REAL D435 frames; the sim `d435_rgb` prim pose is **uncalibrated**
(env CLAUDE.md TODO, never done). If the sim wrist-cam view ≠ the real wrist-cam view, (a) no
real-trained policy transfers, and (b) the sim WM learns the wrong visuals. This is the
sim2real lever — it makes Step 3 *and* the sim eval (`scripts/_sim_eval.py`) trustworthy.

Note: real dataset camera is named `overhead` but is physically the **wrist D435** (recorder
default name); sim `d435_rgb` is the wrist D435 — they correspond, just renamed.

**Procedure:**
1. Render one sim frame:
   ```
   .pixi/envs/sim/bin/python -c "<AppLauncher cameras=on; make_env(pick_and_place,
   enable_cameras=True); reset; save obs['policy']['d435_rgb'][0] as PNG>"
   ```
2. Compare side-by-side to `datasets/local/so101-pickplace-new` row 0
   (`observation.images.overhead`) at the robot home pose.
3. Tune the D435 camera prim offset (translate/rotate) in `so101_env_cfg.py`
   (`d435_camera` CameraCfg, prim path `…/wrist_link/d435`, H-FOV ~69.4°) until the gripper
   jaws + workspace land in the same image region as the real frame.
4. Iterate render→compare→adjust (each render ~1 boot).

**Pass criteria:** gripper jaws appear in the same image region; workspace/object roughly
aligned. Document the final offset; commit the env-cfg change.

**Scope note:** full visual match also needs domain randomization (lighting/texture) for real
transfer — out of scope here; this step only fixes the *geometric* pose so sim views are
plausible.

---

## Step 3 — Staged-reward DreamerV3 run  (multi-hour GPU)

**Depends on:** Step 1 (sane terms + weight ladder) and ideally Step 2 (trustworthy visuals).

**Goal:** first WM/policy trained with the full reach→grasp→lift→place shaping; produce a
**non-zero, rising success rate** (the first evidence of task completion, not just reaching).

**Run:** per `plans/2026-06-08-staged-reward-tuning-plan.md` Steps 1–3 — set the weight ladder
from Step 1, `LEROBOT_ISAAC_STAGED_REWARD=1`, num_envs=1 (or num_envs>1 once Fix 2 lands —
that is the throughput unlock for a practical run length), `SECONDS_PER_EXP` ≥ 4–6 h.

**Track:** TB per-term rewards, episode return, and **success rate** (fraction hitting
`success_termination`). Plateau-detect on success rate.

**Caveat — success criterion:** `success_termination` is currently **EE-to-object distance
(reach)**, not object-in-bin (place). For a true pick-place success metric, update the
termination to object-in-target before/with this run, then it scores placement. Decide:
reach-success (as-is) vs place-success (small change + re-verify).

---

## Sequencing + estimates
| Step | GPU time | Gate to next |
|------|----------|--------------|
| 1 staged sanity | ~20 min | terms fire sanely → set weights |
| 2 camera calib | ~30–60 min iterate | sim view ≈ real view |
| 3 staged run | 4–6 h+ | success rate rises → tune / iterate |

Fix 2 (num_envs>1) is a parallel throughput track — land it to make Step 3 practical at
num_envs>1, but Step 3 works at num_envs=1 (slower). Run order: **1 → 2 → 3**.
