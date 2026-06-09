# Staged-reward tuning — execution results (2026-06-09)

Executes `plans/2026-06-08-staged-reward-tuning-plan.md`. Deadline-bounded run (till 16:00).

## Constraints accepted
- **Fix 2 (num_envs>1) NOT implemented** — spec only, medium-high risk, "do NOT commit
  unverified". Ran at `num_envs=1` (the plan's explicit fallback). ~36 steps/min → one
  long run, no sweep within budget.
- **Joint** (all staged terms on from step 0), per plan Step 2 "try first". Curriculum deferred.

## Code change
`pick_and_place.py`: staged weights were hardcoded (grasp 2 / lift 5 / place 5). Made
env-tunable — `LEROBOT_ISAAC_{GRASP,LIFT,PLACE}_WEIGHT`, defaults preserve old values.
Commit `4ca2ae1` on `feature/wm-isaac-env`. Lets the ladder be balanced without re-editing code.

## Step 0 — instrument (PASS)
Standalone probe `scripts/_staged_reward_probe.py`: boots env num_envs=1 (no cameras → fast),
calls reward funcs directly each step (sheeprl does NOT forward Isaac per-term rewards to TB,
so direct-call is the only way to read the decomposition). `outputs/staged-probe.json`.

**6 terms register at runtime** ✓: success_bonus 5, action_penalty −0.01, progress 10,
grasp 2, lift 5, place 5.

Random-action decomposition (200 steps, EE dist 0.25–0.43 m, never near object):
| term | value | verdict |
|------|-------|---------|
| grasp | ~1e-11 (≈0) | fires only on true contact (std 0.04 ≈ 4 cm) ✓ |
| lift | 0.0 | object at rest ✓ |
| place | 0.0 | gate closed (not lifted) ✓ |
| env_rew | ≈ −0.25/step | progress dense reach gradient working |

Manual-lift sanity (teleport object):
| state | lift | place |
|-------|------|-------|
| lifted (z=0.18) + over target xy | 0.119 ✓ | **1.0** ✓ |
| lifted (z=0.18) + off target | 0.119 ✓ | ~0 ✓ |
| rest (z=0.05) + over target | 0.0 ✓ | 0.0 ✓ |

**Gating logic fully correct**: lift fires on real rise; place is lift-gated AND xy-targeted.

### Findings / tuning levers
1. Probe's direct `progress_reward` call errored on default object key `object` vs scene
   `source_object` — that's the *probe's* wrong default, NOT the env (env's registered progress
   term works: env_rew confirms). Cosmetic; left as-is.
2. `success_bonus` weight 5 (plan wanted 10) and Isaac dt-scales reward terms — a one-shot
   terminal success bonus is therefore small (5×1×dt). Deferred: success can't fire in a
   ~13k-step run, so immaterial today. **Follow-up:** make success_bonus large + non-dt-scaled
   so it dominates the ladder once the agent succeeds.
3. grasp `std=0.04` (≈4 cm) is tight. progress covers the dense reach down to contact, so OK;
   but if the agent reaches yet grasp never fires, widen std to 0.06–0.08.

## Step 1 — ladder weights
Plan's prescribed start: `progress 10, grasp 3, lift 8, place 8, success_bonus 5(→10 deferred)`.
Confirmed live in the run's RewardManager table (grasp 3.0 / lift 8.0 / place 8.0).

## Step 3 — run #1 (object at 0.51 m) → PLATEAU, root cause found
- Session `20260609-staged-reward`. STAGED_REWARD=1, ladder, num_envs=1.
- Boot clean (d435 cams, 6 terms). Throughput ~83 steps/min (2× the baseline estimate).
- reward_env_0 trajectory: −86.5 (902) → −57.8 (4.2k) → −56.6 (7.5k) → −55.3 (10.8k) →
  **−55.2 (14k, dead flat)**. Converged to the SAME reach plateau as the baseline.

### ROOT CAUSE (the real finding)
Reward math: −55/300 steps ≈ −0.18/step ⇒ progress term implies the EE hovers ~0.2 m from
the object, never reaching contact → grasp/lift/place never fire.

Empirical reach probe (`scripts/_reach_probe.py`, `outputs/reach-probe.json`): swept the
JointPositionAction grid, logged gripper_link world xy. **Max planar reach = 0.346 m**
(matches the `rewards.py:214` doc note "~0.4 m"). Object at (0.5, 0.1) = **0.51 m → 0.16 m
beyond reach**. Target bin (0.5, −0.2) = 0.54 m, also unreachable.

**The staged reward is correct + correctly gated (Step 0 proved it). The task was
geometrically impossible** — the object sat outside the arm's workspace. This is also why the
2026-05 baseline "never demonstrated a pick." No reward tuning could ever fix it; only the
object/target position. Run #1 killed at 14k.

## Step 3 — run #2 (object repositioned INTO reach)
- Session `20260609-staged-reach`. Object → (0.22, 0.05, 0.05) r=0.226; target → (0.22, −0.13,
  0.01) r=0.256 — both « 0.346 max reach, clear pick→place lateral separation.
- Same staged ladder (grasp 3 / lift 8 / place 8 / progress 10), num_envs=1, STEPS 25000,
  `SECONDS_PER_EXP=16000` (~4.4 h cap → ~14:30).
- Success signal watched: reward_env_0 climbing ABOVE the −55 reach plateau ⇒ grasp/lift/place
  are paying ⇒ the agent is actually picking.
- Output: `.agent-state/20260609-staged-reach/autoresearch/wm-isaac-prod/train.log`.

### Result — run #2 (clean timeout at ~4 h, 20 467 steps)
Learning curve (reward_env_0, single-env episode return):
| step | 300 | 3.3k | 6.3k | 9.3k | 12.3k | 15.4k | 18.4k | 20.5k |
|------|-----|------|------|------|-------|-------|-------|-------|
| rew  | −62.7 | −34.8 | −18.9 | −17.7 | −17.6 | −17.7 | −17.6 | −17.6 |

Rapid climb −62.7 → −18.9 over the first 6.3k steps, then **hard plateau at ≈ −17.6**.

**vs run #1 (unreachable): −55.2 → −17.6 = 3.1× improvement.** With a reachable object the
agent learns to reach to near-contact (progress saturates near 0) — the reach + grasp-proximity
stages demonstrably drive behaviour. But the plateau at −17.6 (≈ −0.06/step ⇒ EE hovering
~7 cm short) means it does **not** complete a sustained lift/place.

**Why no lift (the next blocker):** `grasp_reward` is a pure *proximity* Gaussian — reaching the
object earns the bonus but does **not** physically grip it, and `lift_reward` only pays when the
object's z actually rises. With no contact/closure-based grasp + no incentive to close the
gripper on the object, the cube never leaves the ground. This is the exact refinement
`rewards.py` flagged ("a true contact/closure-based grasp signal is a GPU-verify refinement").

## Success criteria (from plan)
- [x] Per-term rewards behave as designed (Step 0 — proven).
- [x] **Found + fixed the real blocker to "rising success": object geometry.** Run #1
  reproduced the baseline reach plateau (−55) and the reach probe proved the object was 0.16 m
  out of reach. Run #2 (reachable) lifted the plateau 3.1× and proved reach+grasp-proximity work.
- [ ] Non-zero success rate (object placed in bin): **not reached** — blocked by the lack of a
  contact/closure grasp, not by reward weights or geometry. This is the clean next step.

## Run #3 — closure grasp (next-step #1 implemented)
Added `grasp_closure_reward = proximity_gate × gripper_closedness` (rewards.py) — rewards
closing the jaw while near the object, to break run #2's "hover near object, jaw open" local
optimum. Wired into the staged block (opt-in `LEROBOT_ISAAC_CLOSURE_WEIGHT>0`). Gripper = joint 5,
limits [−0.17, 1.75]; `closed_high=True` (SO-101 angle increases to close). `lift_reward` stays
the true arbiter, so a wrong closure sign is unhelpful, never farmable.

Run `20260609-staged-closure`: ladder grasp 3 / **closure 4** / lift 8 / place 8, reachable
object (0.22), num_envs=1, 25k steps / 6 h cap. Boot confirmed **7 terms incl `grasp_closure`
@4.0**. First step −62.7. Success signal: reward breaking ABOVE run #2's −18 plateau.

Commits: rewards+wiring `3a0c0ab`, gripper probe `6f5beb3` (lerobot-isaac-env). Gripper
geometry proxy in the probe was inconclusive (jaw link frame at hinge) — direction taken from
SO-101 convention + made env-tunable.

### Result
_(pending — run #3 launched ~16:42, async)_

## Next steps (ranked)
1. **Contact/closure grasp** — replace the proximity `grasp_reward` with one gated on actual
   gripper–object contact (Isaac contact sensors) + reward gripper closure when in contact, so
   `lift` becomes achievable. This is the single blocker to a real pick.
2. **Curriculum** (plan Step 2) — phase 1 reach+grasp (lift/place weight 0) → add lift → add
   place, via `lerobot-curriculum-agent`. Now worth doing since joint plateaus at grasp.
3. **success_bonus** — make it large + non-dt-scaled so it dominates the ladder once a pick
   lands (currently weight 5, dt-scaled → tiny).
4. **grasp std** — now env-tunable (`LEROBOT_ISAAC_GRASP_STD`); widen to 0.06–0.08 if reach
   stalls short of contact.
5. **Fix 2 (num_envs>1)** — throughput unlock for the multi-hour runs the above needs.

## Key takeaway
The headline result is a **task-geometry bug, not a reward-tuning result**: staged shaping was
sound all along, but every prior pick attempt targeted an object 0.16 m outside the SO-101
reach envelope. `LEROBOT_ISAAC_OBJECT_X/Y` and `_TARGET_X/Y` must stay inside r≈0.30 m.
**Recommend: change the env's default object/target spawn to a reachable point** so future runs
don't silently inherit the unreachable geometry.
