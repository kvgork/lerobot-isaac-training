# WM Isaac HP Sweep — REVISED Trial Pool

**Date:** 2026-05-24 (revised after lessons learned)
**Branch:** `feature/wm-isaac-env`
**Supersedes:** original trial pool in
`scripts/_run_autoresearch_wm_isaac.sh:TRIAL_POOL` and the trials-1to9
table previously in this file.

---

## Why this plan was rewritten

The original trial pool (10 configs sweeping `ent_coef ∈ {1e-4..1e-2}`,
`replay_ratio ∈ {0.25..2.0}`, `min_std ∈ {0.1..0.5}`, `wm.lr ∈ {3e-5..3e-4}`)
was DESIGNED before we knew that ALL 4 honest runs converged to the
same broken actor — including the default config (ent=3e-4) AND the
33×-bumped (ent=1e-2). See `plans/2026-05-24-wm-isaac-lessons.md` for
the run tally and forensic signature.

**Implication:** the original 10-trial pool falls entirely WITHIN the
collapse regime. Running it spends 30 h discovering the same result 9
more times. The pool needs new axes.

---

## Pre-requisite — wire `success` termination FIRST (≤ 1 day eng)

The cleanest exit path from the actor-collapse trap is a sparse
terminal reward that the actor cannot trivially satisfy by sitting at
home pose. That requires:

1. `src/lerobot-isaac-env/.../terminations.py` — implement
   `success_termination(env, dist_threshold=0.05, lift_threshold=0.02)`:
   ```python
   def success_termination(env, dist_threshold=0.05, lift_threshold=0.02,
                            robot_cfg=None, object_cfg=None):
       robot = env.scene[robot_cfg.name]
       obj = env.scene[object_cfg.name]
       ee_pos = robot.data.body_pos_w[:, gripper_idx, :]
       obj_pos = obj.data.root_pos_w
       dist = torch.norm(ee_pos - obj_pos, dim=-1)
       lifted = obj_pos[:, 2] > lift_threshold
       return (dist < dist_threshold) & lifted
   ```
2. `src/lerobot-isaac-env/.../so101_env_cfg.py:TerminationsCfg` —
   add `success` field referencing the new term function.
3. `src/lerobot-isaac-env/.../so101_env_cfg.py:RewardsCfg.success_bonus`
   — set to:
   ```python
   RewardTermCfg(
       func=mdp.is_terminated_term,
       params={"term_keys": ["success"]},
       weight=5.0,
   )
   ```
   Default `None` was a defensive setting before the term existed.

**Acceptance:** smoke run that scripts an arm directly to the cube
+ small lift triggers `success` → episode terminates with reward
spike → recorded in TB.

This is ~4 h work + 30 min smoke. NOT part of the sweep — it's the
gate.

---

## New trial pool (8 configs, ~24 h compute)

Each row tests a hypothesis that we know is UNDER-explored, NOT a
fine-grained sweep of axes we know converge.

| # | Label | reward shape | ent_coef | init_std | min_std | rr | max_ep | algo | hypothesis |
|---|-------|--------------|----------|----------|---------|-----|--------|------|-----------|
| 1 | sparse-success-default | terminal ONLY (no progress) | 1e-2 | 2.0 | 0.3 | 0.5 | 100 | dreamer_v3 | Pure sparse reward + entropy fix forces exploration. Reference experiment. |
| 2 | sparse + high init_std | terminal ONLY | 3e-2 | **4.0** | 0.5 | 0.5 | 100 | dreamer_v3 | Test claim that initial action variance is the missing exploration pressure. |
| 3 | sparse + short episodes | terminal ONLY | 1e-2 | 2.0 | 0.3 | 0.5 | **50** | dreamer_v3 | Shorter horizons → terminal signal closer to actor's planning horizon. |
| 4 | sparse + low replay_ratio | terminal ONLY | 1e-2 | 2.0 | 0.3 | **0.1** | 100 | dreamer_v3 | More env data per gradient step → actor sees diverse trajectories before WM over-commits. |
| 5 | hybrid: terminal + small progress | terminal=5 + progress(weight=1) | 1e-2 | 2.0 | 0.3 | 0.5 | 100 | dreamer_v3 | Compromise between v7/v8 (dense only) and trial 1 (sparse only). Maybe DreamerV3 needs SOME dense shaping. |
| 6 | object-at-home curriculum | terminal + progress(weight=1) | 1e-2 | 2.0 | 0.3 | 0.5 | 100 | dreamer_v3 | Move source_object spawn from (0.5, 0.1, 0.05) → (0.30, 0.05, 0.05). Home gripper IS the target. Trivially solvable; tests whether ANY policy can be learned. |
| 7 | PPO baseline (sparse) | terminal ONLY | n/a | n/a | n/a | n/a | 100 | **ppo** | DreamerV3-free reality check. PPO has no WM to over-fit → if PPO learns and DreamerV3 doesn't, the failure is WM-specific. |
| 8 | extreme entropy | terminal + small progress | **1e-1** | 4.0 | 0.5 | 0.5 | 100 | dreamer_v3 | Final attempt at the entropy axis. If even 1e-1 collapses, entropy alone cannot escape. |

**STEPS per trial:** 80000 (vs original 60000). The sparser reward
landscape means actor needs more env steps to find the first
non-trivial success. Wall time per trial ≈ 3 h at ~7 step/s.

**Wall total:** 8 × 3 h = 24 h. One full day + overnight.

---

## Order rationale

- Trials 1-3 are pure-sparse variants — the cheapest cleanest exit
  from collapse.
- Trial 4 isolates the replay_ratio axis.
- Trial 5 hedges: maybe sparse-only is too hard.
- Trial 6 is the "ground-truth" — if even this trivial setup fails to
  learn, the issue is in the env wiring, not RL.
- Trial 7 (PPO) breaks the "is it the algo or the env" question.
- Trial 8 caps the entropy axis.

**Order trials 6 + 7 first** if budget is tight — they're the
diagnostic anchors. Everything else is exploration around them.

---

## Acceptance per trial

| Reading | Verdict |
|---------|---------|
| `Rewards/rew_avg > 0` at any point | **WIN** — at least one episode succeeded |
| `Game/ep_len_avg < max_ep_steps` | **WIN** — episodes terminating early (= success or contact-terminal) |
| `Grads/actor ≥ 0.05` throughout | actor NOT collapsed; promising even if `rew_avg` is still 0 |
| `Grads/actor → 0` by step 10k AND `rew_avg = 0` | **collapse** (the known failure mode) — kill the trial after step 15k, don't waste full 3h |

The collapse signature is well-characterised now. Kill early on
trip — don't burn the wall.

---

## Resume command

After Pre-requisite (success termination) lands:

```bash
# Update _run_autoresearch_wm_isaac.sh TRIAL_POOL with the new 8 rows.
# Each row format: ENT|INIT_STD|MIN_STD|RR|MAX_EP|REWARD_SHAPE|ALGO|LABEL

cd ~/workspaces/lerobot-isaac-training
SESSION_ID=wm-isaac-hp-v2-$(date +%Y%m%d-%H%M%S) \
SECONDS_PER_EXP=10800 \
MAX_TRIALS=8 \
PLATEAU_LIMIT=4 \
  bash scripts/_run_autoresearch_wm_isaac.sh
```

**The sweep script will need a refactor** to take REWARD_SHAPE +
ALGO axes in the pool format (currently only Hydra-knob axes). Add as
follow-up — minimal change: keep current 4-axis format, encode
`REWARD_SHAPE` as `EXTRA_HYDRA` lines per trial (e.g.
`rewards.progress.weight=0` for "sparse only"), encode `ALGO` by
swapping `exp=dreamer_v3` → `exp=ppo` in the Hydra cfg via
EXTRA_HYDRA.

---

## Stop / pivot rules

- If **trial 6** (object-at-home, trivially solvable) STILL collapses
  → the issue is in the env wiring (observation space, action scaling,
  or scene physics), not RL. Stop sweeping. Audit `IsaacSO101Env`,
  `ActionsCfg`, `ObservationsCfg`. Likely root: action `scale=0.5` too
  small, OR `JointPositionActionCfg.use_default_offset` interaction
  with home pose, OR camera obs noise.
- If **trial 7** (PPO) succeeds and DreamerV3 trials don't → the
  failure is WM-specific. Either (a) drop DreamerV3 and use PPO as
  the actor backbone, or (b) deep-dive `Loss/world_model_loss` +
  `State/post_entropy` to understand WM over-confidence.
- If **all 8 trials collapse** → switch goal. Either give up on
  sim-to-real for SO-101 + DreamerV3 (use BC LoRA instead — which
  ALREADY works, see `outputs/lora-prod-best/`), or escalate to a
  fundamentally different exploration mechanism (RND, NoveltyD).

---

## What we keep no matter what the sweep finds

- The infrastructure pile (AppLauncher-first entry, sync_env, target_bin
  AssetBaseCfg, ee_body_name knob, sweep orchestrator). All correct.
- The LoRA-best ckpt (`outputs/lora-prod-best/`, pc_success=0.211
  open-loop, +158% rel vs anchor). THAT model is robot-deployable
  TODAY — no DreamerV3 success required.
- The lessons file (`plans/2026-05-24-wm-isaac-lessons.md`) — keeps
  growing as we learn more.

---

## Time budget estimates

| Pre-requisite | Sweep | Verify winner | Deploy test | Total |
|---------------|-------|---------------|-------------|-------|
| 4 h (success term + smoke) | 24 h | 4 h (alt-seed re-run) | 2 h (laptop deploy + DRY-RUN) | **~34 h** |

Spread across 2-3 days. NOT a single-overnight task.

---

## Cross-references

- Lessons (read this first): `plans/2026-05-24-wm-isaac-lessons.md`
- Parent plan: `plans/2026-05-23-wm-isaac-env-plan.md`
- Sweep script (needs TRIAL_POOL refactor):
  `scripts/_run_autoresearch_wm_isaac.sh`
- Single-trial runner: `scripts/_run_wm_isaac_overnight.sh`
- LoRA fallback (already works): `outputs/lora-prod-best/`,
  `plans/2026-05-22-lora-sweep-next-steps.md`
