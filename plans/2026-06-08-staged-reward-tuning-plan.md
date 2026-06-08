# Plan: Tuning the staged pick-place reward for DreamerV3 (2026-06-08)

## Where we are
- Staged terms `grasp_reward` / `lift_reward` / `place_reward` are implemented + **verified to
  register** in the Isaac Reward Manager (6 active terms: success_bonus 5, action_penalty,
  progress 10, grasp 2, lift 5, place 5). Committed `cedfd19` (lerobot-isaac-env).
- They **run** each step without crashing. They are **NOT tuned** — the weights, the grasp
  proximity proxy, the lift threshold, and the target xy are best-guess config.
- Baseline (progress+success only) DreamerV3 run learned to *approach* (reward −84→−55 over
  ~8.7k steps) but never demonstrated a pick (no lift/place signal then).

## Goal
A DreamerV3 policy that actually picks + places, trained by online RL in the Isaac sim with
the staged shaping providing dense credit at each sub-goal.

## Approach — staged tuning, not one big run

### Step 0 — instrument (1 short run, ~20 min, num_envs=1)
Run with `LEROBOT_ISAAC_STAGED_REWARD=1` + `PROGRESS_REWARD_DEBUG`-style logging. Confirm via
TB per-term scalars (`Reward/grasp`, `Reward/lift`, `Reward/place`) that:
- `progress` fires from step 0 (reach gradient present).
- `grasp` fires only when EE is genuinely on the object (tighten `std` if it fires too early).
- `lift` is ~0 until the object actually rises (sanity: drop a manual lift, confirm it spikes).
- `place` stays 0 until lifted (gate works).
If `lift`/`place` never move during exploration, the actor hasn't reached grasp yet — that's
expected early; the point is to confirm the terms are *capable* of firing.

### Step 1 — weight balance
Isaac scales reward by `weight * dt`. With dt≈1/120 and `progress` weight 10 + distance_scale
0.4, per-step progress ≈ −0.21 at 1 m. Tune so the stage rewards form an increasing ladder:
reach < grasp < lift < place < success_bonus, and no single term saturates the return. Start:
`progress 10, grasp 3, lift 8, place 8, success_bonus 10`. Adjust after reading return
decomposition.

### Step 2 — curriculum vs joint (decide)
Two options:
- **Joint** (all terms on from step 0): simplest; relies on the ladder + gating. Try first.
- **Curriculum** (CurricuLLM-style): phase 1 reach+grasp (lift/place weight 0), phase 2 add
  lift, phase 3 add place. Use `lerobot-curriculum-agent`. More robust if joint plateaus at
  reach.

### Step 3 — multi-hour runs + success metric
- Run length: ≥4–6h (the baseline only reached 8.7k steps in 4h at num_envs=1 → **Fix 2
  vectorization makes this practical**; see plans/2026-06-07-good-world-model-plan.md §Fix 2).
- Success signal: the env's `success_termination` (object in target bin) → `success_bonus`.
  Verify the termination threshold matches `_TARGET_POS` + a reasonable radius.
- Track: TB per-term rewards, episode return, **success rate** (fraction of episodes hitting
  `success_termination`). Plateau-detect on success rate.

### Step 4 — reward-hacking guards
- `lift` could be hacked by knocking the cube up without grasping → `place` is lift-gated, and
  `grasp` proximity must precede; if hacking appears, gate `lift` on `grasp` proximity too.
- Cap each term (lift already capped at `max_height`); keep `action_penalty` on to avoid
  erratic motion.

## Dependencies / decisions
- **Fix 2 (num_envs>1)** is the throughput unlock — staged tuning needs many env steps; at
  num_envs=1 (~35 steps/min observed) a 50k-step run is ~24h. Land Fix 2 first, or accept
  long single-env runs.
- Open: joint vs curriculum (Step 2); the weight ladder (Step 1); success radius (Step 3).
- Sim2real is out of scope here — this is a *sim* WM/policy. Real transfer needs DR + the
  camera-pose calibration TODO (lerobot-isaac-env CLAUDE.md).

## Success criteria for this plan
- Per-term rewards behave as designed (Step 0). 
- A tuned run shows **non-zero, rising success rate** (object placed in bin) — the first
  evidence the staged shaping produces task completion, not just reaching.
