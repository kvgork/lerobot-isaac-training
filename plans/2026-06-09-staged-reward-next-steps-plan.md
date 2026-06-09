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

## Autonomous execution rules (what I'll do without asking)
- ✅ Launch/kill/relaunch training runs; tune env-var knobs (weights, std, object pos, closed_high);
  commit reward/config changes to `feature/wm-isaac-env`; commit docs/plans to workspace main;
  render offscreen videos; update results doc + memory.
- ⏸️ PAUSE for your review before: contact-sensor scene changes (Branch B.4), committing Fix 2,
  anything that rewrites the env scene structure or risks the boot path.
- One 10GB GPU: only one training run at a time; video render pauses training briefly then resumes.

## Where to check progress
- Live reward: `.agent-state/20260609-staged-closure-b/autoresearch/wm-isaac-prod/train.log`
  (`grep policy_step=`).
- Results + findings: `plans/2026-06-09-staged-reward-tuning-results.md` (I append each run).
- Commits: `src/lerobot-isaac-env` (`feature/wm-isaac-env`) + workspace `main`.
- Videos/checkpoints: `outputs/`.
- This plan: I tick the gate I took + link the run, so you can see what path I chose.

## Progress log (I append as I go)
- 19:1x — run #3b at −14.6 (3.9k), broke −18. Plan saved. Continuing autonomously.
