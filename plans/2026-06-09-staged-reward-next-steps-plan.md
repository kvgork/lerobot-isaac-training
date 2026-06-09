# Staged-reward — autonomous next-steps plan (2026-06-09 eve)

**Context:** executing `plans/2026-06-08-staged-reward-tuning-plan.md`; results in
`plans/2026-06-09-staged-reward-tuning-results.md`. PC left on → I run this autonomously
overnight. This file = the plan you can check; my autonomous choices follow the decision gates
below (pre-authorised). No live GUI viewer needed; I render offscreen videos as artifacts.

## Where we are (live)
- **Run #3b** `20260609-staged-closure-b` — DreamerV3 online RL, num_envs=1, headless, staged
  ladder grasp 3 / **closure 4** / lift 8 / place 8 / progress 10, reachable object (0.22).
- Reward trajectory: −62.7 (300) → −18.3 (2.4k) → **−14.6 (3.9k)** — broke past run #2's −17.6
  plateau. Closure reward appears to be working (grip→lift starting). Watching for sustained climb.
- Throughput ~83 steps/min; 25k-step / 6h cap → finishes ~late evening.

## Decision gate (read at ~10k, ~20k, and run end)
Classify run #3b by best sustained reward (smoothed over ≥3 samples):

| Signal | Meaning | Branch |
|--------|---------|--------|
| reward climbs well above −18 (toward 0/positive), sustained | grip→lift firing, maybe place | **A — it works** |
| reward hovers ~−14 to −18, no further climb | reaches+closes but lift inconsistent | **A-weak — tune** |
| reward back at ~−18 flat | closure didn't break it | **B — diagnose grip** |

## Branch A — pick is happening (auto-execute)
1. Let run #3b finish; record final reward + render a rollout **video** (offscreen, no window)
   from the last checkpoint → save `outputs/run3b-rollout.mp4`, surface to user.
2. **success_bonus non-dt-scaled** (queued #3): currently weight 5, Isaac dt-scales it → a
   completed place barely registers. Make it a large terminal bonus so success dominates the
   ladder. Code change in `rewards.py`/env cfg + commit. Then a run to chase real place success.
3. **Curriculum** (queued #2): lock reach+grasp+lift weights, ramp `place` weight; track
   `place_reward` firing + success-termination rate. Goal: first non-zero pick-AND-place success.
4. Update results doc + memory; this policy becomes the working baseline.

## Branch A-weak / B — diagnose grip (auto-execute cheap ones, prep heavy ones)
Cheap, auto-run in order until reward improves:
1. **Flip closure sign** — `LEROBOT_ISAAC_GRIPPER_CLOSED_HIGH=0` (maybe jaw opens, not closes).
   Quick rerun (~30-40 min to see if reward exceeds the prior best). Cheap.
2. **Widen grasp std** — `LEROBOT_ISAAC_GRASP_STD=0.08` if EE reaches but grasp/closure ~0.
3. **Closure proximity_std / weight** — tune so closure only pays on-object (not free-air).
Heavy (PREP + pause for user review before committing a big change):
4. **Contact-based grip** — if closing the jaw doesn't physically lift the cube, the blocker is
   sim contact (gripper effort_limit 10 N·m, jaw friction, cube mass/size). Options: add Isaac
   contact sensors + contact-gated grasp; raise jaw friction; shrink/lighten cube. This is the
   real manipulation-physics work — I'll diagnose + write a sub-plan, NOT commit blind.

## Infra queued (gated on a working pick; auto-prep)
- **Fix 2 (num_envs>1)** — throughput unlock (`plans/2026-06-08-fix2-isaac-vectorization-plan.md`).
  Needed before any real curriculum grind / HP sweep. Medium-high risk → implement + GPU-verify,
  do NOT commit unverified. I'll prep + smoke-test; flag before relying on it.
- **success_bonus non-dt-scale** (see Branch A.2).

## Autonomous execution rules (UPDATED 19:2x — full autonomy, user directive)
User: "work on the 3 steps, keep giving updates but don't ask for input. If something breaks
fix it and keep a ledger." → **No pause gates.** I execute everything autonomously, including
the previously-gated heavy items (contact-sensor changes, Fix 2 commit). Breaks → I fix in place
and log to `plans/2026-06-09-autonomous-fix-ledger.md`. Updates posted, no questions.

**The 3 steps (autonomous):**
1. **success_bonus non-dt-scale** — make terminal success reward dominate the ladder.
2. **Curriculum** — reach+grasp+lift locked, ramp place → real pick-and-place success rate.
3. **Fix 2 (num_envs>1)** — throughput unlock; implement + GPU-verify + commit.
Plus: branch-B grip diagnosis if run #3b regresses (= "fix what breaks").

- ✅ Everything: launch/kill/relaunch runs; tune knobs; edit reward/env/scene code; implement
  contact sensors + Fix 2; commit to `feature/wm-isaac-env` + workspace main; render videos.
- One 10GB GPU: one training run at a time; CPU code-work (steps' implementation) parallelised
  with GPU training, GPU-verify as runs free up.

## Where to check progress
- Live reward: `.agent-state/20260609-staged-closure-b/autoresearch/wm-isaac-prod/train.log`
  (`grep policy_step=`).
- Results + findings: `plans/2026-06-09-staged-reward-tuning-results.md` (I append each run).
- Commits: `src/lerobot-isaac-env` (`feature/wm-isaac-env`) + workspace `main`.
- Videos/checkpoints: `outputs/`.
- This plan: I tick the gate I took + link the run, so you can see what path I chose.

## Execution sequence (GPU is single — one task at a time)
Step 1 is CPU-done. Steps 2 & 3 need the GPU, busy with #3b until ~22:00. Order chosen to
compound: **Fix 2 first** (if it lands, the place-chase run goes N× faster).

**On #3b completion (monitor DONE event):**
1. Classify branch from final reward (gate table above); append verdict to results doc.
2. **Step 3 — Fix 2**: implement `IsaacSO101VectorEnv` (new file, num_envs>1 gated, single-env
   path untouched) + gymnasium.vector patch in `_wm_isaac_entry.py`. GPU-verify NUM_ENVS=2 smoke
   (shapes (2,…), grad step, close()→test() lifecycle survives). If it fails after a couple boot
   cycles → fall back to num_envs=1, log to ledger, proceed to step 2 anyway.
3. **Step 2 — place-chase / curriculum**: launch a run with closure + place_success enabled,
   num_envs = (4 if Fix 2 verified else 1). Recipe:
   ```
   export LEROBOT_ISAAC_STAGED_REWARD=1 LEROBOT_ISAAC_GRASP_WEIGHT=3 \
     LEROBOT_ISAAC_LIFT_WEIGHT=8 LEROBOT_ISAAC_PLACE_WEIGHT=8 \
     LEROBOT_ISAAC_CLOSURE_WEIGHT=4 LEROBOT_ISAAC_GRIPPER_CLOSED_HIGH=1 \
     LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT=1 LEROBOT_ISAAC_PLACE_SUCCESS_BONUS=5 \
     LEROBOT_ISAAC_OBJECT_X=0.22 LEROBOT_ISAAC_OBJECT_Y=0.05 LEROBOT_ISAAC_OBJECT_Z=0.05 \
     LEROBOT_ISAAC_TARGET_X=0.22 LEROBOT_ISAAC_TARGET_Y=-0.13 LEROBOT_ISAAC_TARGET_Z=0.01
   STEPS=40000 SECONDS_PER_EXP=21600 NUM_ENVS=<1|4> SESSION_ID=20260609-place-chase \
     bash scripts/_run_wm_isaac_overnight.sh
   ```
   Success metric: `place_reward`/`place_success` firing + reward going positive = object in bin.
4. Curriculum refinement (phased weight ramp) only if place plateaus and throughput allows.

## Progress log (I append as I go)
- 19:1x — run #3b at −14.6 (3.9k), broke −18. Plan saved. Continuing autonomously.
- 19:5x — STEP 1 done (place_success, 892f5d5). Steps 2/3 GPU-bound → queued behind #3b per
  sequence above. Run #3b climbing (Branch A trend). Not blind-implementing Fix 2 (stall-risk
  surface, needs GPU iteration). Monitor drives next action on #3b DONE.
- 19:54 — **BRANCH A CONFIRMED.** Run #3b reward −12.4 @6.6k, sustained climb above run #2's
  −17.6 hard plateau (−18.3→−17.2→−15.4→−14.6→−12.4). Closure-grasp reward (commit 3a0c0ab)
  broke the plateau → agent gripping+lifting. The closure-grasp hypothesis (next-step #1) works.
