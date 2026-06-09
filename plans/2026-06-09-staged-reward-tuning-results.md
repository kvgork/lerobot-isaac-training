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

### Result
_(pending — filled at analysis ~14:30–15:00)_

## Success criteria (from plan)
- [x] Per-term rewards behave as designed (Step 0).
- [~] Tuned run shows non-zero, rising success rate — run #1 disproved the OLD geometry;
  run #2 (reachable object) is the real test. _(pending)_

## Key takeaway
The headline result is a **task-geometry bug, not a reward-tuning result**: staged shaping was
sound all along, but every prior pick attempt targeted an object 0.16 m outside the SO-101
reach envelope. `LEROBOT_ISAAC_OBJECT_X/Y` and `_TARGET_X/Y` must stay inside r≈0.30 m.
**Recommend: change the env's default object/target spawn to a reachable point** so future runs
don't silently inherit the unreachable geometry.
