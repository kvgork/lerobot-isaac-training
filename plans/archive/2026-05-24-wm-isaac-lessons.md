# WM Isaac — Lessons Learned

> **Status: REFERENCE (living doc).** Not an action plan — captures stable findings
> for future sweeps. Updated as new failure modes are observed.

**Date:** 2026-05-24
**Branch:** `feature/wm-isaac-env`
**Scope:** Stable artifact summarising what 2026-05-23 / 2026-05-24
multi-day arc taught us about training DreamerV3 on Isaac Lab SO-101
pick-place. Reference this BEFORE the next sweep attempt.

---

## Tally of runs

| Run | ent_coef | min_std | rr | bs | precision | reward signal | steps reached | rew_avg | actor verdict |
|-----|----------|---------|-----|----|-----------|---------------|---------------|---------|---------------|
| v5  | 3e-4 (default) | 0.1 | 1 | 4 | fp32 | weight=1, scale=1 | 37800 | -2.37 | collapsed |
| v6  | 3e-4 | 0.1 | 2 | 16 | bf16-mixed | weight=10, scale=0.4 | OOM (co-tenancy) | n/a | aborted |
| v7  | 3e-4 | 0.1 | 1 | 16 | bf16-mixed | weight=10, scale=0.4 | 15000 | -62.7 | collapsed (Grads/actor=0.009) |
| v8  | 1e-2 | 0.3 | 1 | 16 | bf16-mixed | weight=10, scale=0.4 | 36000 | -62.94 | collapsed (Grads/actor=0.000) |
| HP-trial0 (baseline, 1st) | 3e-4 | 0.1 | 0.5 | 16 | bf16-mixed | weight=10, scale=0.4 | 0 (zombie GPU OOM) | n/a | aborted |
| HP-trial0 (baseline, 2nd) | 3e-4 | 0.1 | 0.5 | 16 | bf16-mixed | weight=10, scale=0.4 | 26700 | -62.65 | collapsed |

**Net:** 4 honest runs, 4 actor collapses. The default ent_coef + 33×
ent_coef both reach the same broken attractor. The reward magnitude
(25× bump from v5 → v7) didn't help either.

---

## Confirmed pitfalls (do NOT re-trip)

### Boot / install order

1. **`pip install sheeprl` brings deps that break Isaac Sim's libgobject.**
   Fix: `pip install --no-deps sheeprl@git+https://github.com/Eclectic-Sheep/sheeprl.git`.
   Pinned in `bootstrap.py:pip_install_dreamerv3` (deploy pkg).
2. **AppLauncher MUST boot BEFORE sheeprl imports.** `python -m sheeprl`
   loses libgobject to hydra+lightning first → Isaac Sim's
   `libgpu.foundation.plugin.so` fails to load with
   `undefined symbol: g_string_copy`. Use
   `scripts/_wm_isaac_entry.py` which calls AppLauncher then dispatches
   to `sheeprl.cli.run`.
3. **`IsaacSO101Env._boot` MUST detect existing `omni.kit.app.get_app()`**
   and skip re-launching. Otherwise double-AppLauncher deadlocks waiting
   for kit extension reload.
4. **`sync_env: True` in `configs/env/isaac_so101.yaml` is mandatory.**
   sheeprl's default `AsyncVectorEnv` forks workers → fork can't
   re-init CUDA (since AppLauncher already claimed it) →
   `RuntimeError: Cannot re-initialize CUDA in forked subprocess`.
5. **`env.num_envs=1` only.** Isaac Lab's SimulationContext is a
   singleton; sheeprl trying to spawn 2 envs → second one fails with
   `Simulation context already exists. Cannot create a new one`.

### Scene cfg

6. **`target_bin` MUST be `AssetBaseCfg`, not `RigidObjectCfg(kinematic_enabled=True)`.**
   Isaac Sim 6.0 + PhysX 6.0 hang `sim.reset()` with
   `Failed to get a valid attached USD stage id for kinematic bodies`.
   Static marker is the correct semantics anyway.
7. **`success_bonus` requires a `success` termination term.** Default
   `TerminationsCfg` has only `time_out` → `is_terminated_term` raises
   `ValueError: success: []`. Until a real success-termination is wired
   in `terminations.py`, leave `RewardsCfg.success_bonus = None`.

### Process management

8. **`train_wrapper` `proc.kill()` does NOT propagate to the Isaac Sim
   kit subprocess.** Zombie kit processes outlive the parent kill →
   next run hits PhysX OOM. Mitigation: `kill -9 <pid>` explicit on the
   `_wm_isaac_entry.py` PID, then `pkill -9 -f kit\\.app|carb\\.app`.
   TODO: fix `train_wrapper.run_subprocess` to use
   `preexec_fn=os.setsid` + `os.killpg` so the whole process group dies.
9. **`LEROBOT_TRAIN_TIMEOUT` defaults to 4h hardcoded in
   `train_wrapper.py`.** Override via env var when bash timeout is
   longer — otherwise the wrapper kills sheeprl at 4h regardless of
   bash ceiling.

### Reward / metric

10. **`body_pos_w[:, -1, :]` was NOT the bug.** Last body is
    `moving_jaw_so101_v1_link` at extended offset, but at home pose
    distance to source_object ≈ 0.29 m — physically plausible. The
    "1 m" estimate was a dt-unwinding error: Isaac Lab's reward
    manager scales by `step_dt = decimation × sim_dt = 1/30`, not
    `sim_dt = 1/120`. Using the correct step_dt unwinds reward -0.21 to
    dist ≈ 0.25 m (home pose). Named `gripper_link` lookup is good
    hygiene but didn't change the actor's failure mode.
11. **Reward = -dist (no terminal +1) is too smooth for DreamerV3 on a
    static-init scene.** WM learns the dynamics fine
    (`Loss/observation_loss` drops 350×) and predicts reward (`reward_loss`
    plateaus at ~0.5), but actor finds a deterministic policy that
    plateaus at the random-baseline distance (~0.25 m at home pose,
    sometimes farther under random actions). Classic premature
    convergence on shallow landscape.
12. **Bumping reward weight 25× did NOT help.** v5 (weight=1) and v7+
    (weight=10) both produce the SAME actor (random-baseline
    distance). Stronger gradient on a wrong-fixed-point objective is
    still a wrong fixed point.

---

## DreamerV3 actor-collapse signature (forensic)

Every collapsed run shows:

- `Loss/policy_loss` drops to ≈ 0 (or negative if entropy bonus
  dominates) within ~5k–10k steps.
- `Grads/actor` drops 60×+ in same window, often to literal 0.
- `Grads/critic` follows.
- `State/post_entropy` drops 5×+ (WM over-confident).
- `Rewards/rew_avg` stays flat at random-baseline forever.
- `Loss/observation_loss` drops fine (WM dynamics ARE learning).
- `Loss/reward_loss` plateaus around 0.5 (WM-predicted reward never
  becomes a useful actor gradient).

If first 5k steps show those signs → it's collapsed. Don't waste 12 h.

---

## What's NOT yet tried (next-sweep candidates)

These remain potentially useful axes — none tested:

1. **Wider entropy.** `ent_coef ∈ {0.03, 0.1, 0.3}` — beyond the
   `0.01` we tested.
2. **Higher `actor.init_std` + `max_std`.** Forces initial exploration
   variance. Default `init_std=2.0` may not be enough for a 6-DOF
   continuous action space with bounded range.
3. **Sparse terminal reward.** Remove `progress_reward`, add a
   `success` termination + bonus. Counter-intuitive for sample
   efficiency but DreamerV3 sometimes prefers sparse + exploration
   over dense smooth gradients (the dense reward gives the actor a
   "low-energy fixed point" to converge to).
4. **`max_episode_steps=100`** (down from 300). Shorter episodes →
   actor sees terminal-state transitions more often → reward
   discounting cares about reaching cube within seconds.
5. **`replay_ratio = 0.1`** (lower than 0.25). More env data per
   gradient step. May give actor enough variety to escape its current
   policy fixed point.
6. **Sheeprl PPO instead of DreamerV3.** PPO doesn't have a world
   model to over-fit; sometimes more robust on hand-shaped rewards.
   `programs/wm-dreamerv3-isaac-hp.md` could spawn a sibling
   `programs/wm-ppo-isaac.md`.
7. **Curriculum: object closer to home pose.** Move source_object
   spawn to (0.30, 0.05, 0.05) → home-pose gripper IS the target.
   Actor that learns to stay still gets reward; learning gradient is
   non-trivial. Then move object outward across stages.
8. **Action scale.** `JointPositionActionCfg.scale = 0.5` may be too
   tight. Try `scale=1.0` (full ±π rad delta) — actor can reach more
   of the workspace per step but may overshoot.

---

## Recommended next experiment

**Single trial, ~3 h, in this order:**

1. Add `success` termination (gripper inside 5 cm of cube AND lifted
   > 2 cm) in `terminations.py`.
2. Re-enable `success_bonus` (RewardTermCfg, weight=5.0).
3. Optionally **remove `progress_reward`** entirely — go pure sparse.
4. Bump `ent_coef` to 0.03 + `init_std` to 4.0.
5. Drop `max_episode_steps` to 100.
6. Launch 100k steps.

If reward CURVE shows any non-zero `Rewards/rew_avg` (i.e. any episode
ever terminates via success) → real signal. Then continue with HP
sweep over remaining axes. If still flat → switch algorithm (PPO).

---

## Files that survived the arc (worth keeping)

- `programs/wm-dreamerv3-isaac-hp.md` — sweep program (will be useful
  again once the next iteration has a winning baseline to ratchet).
- `scripts/_run_autoresearch_wm_isaac.sh` — sweep orchestrator.
- `scripts/_run_wm_isaac_overnight.sh` — single-trial runner with the
  AppLauncher-first + sync_env + EXTRA_HYDRA wiring.
- `scripts/_wm_isaac_entry.py` — AppLauncher-first entry point.
- `src/lerobot-isaac-adapters/.../sheeprl_plugin/{isaac_env.py,configs/env/isaac_so101.yaml}`
  — the bridge layer (correct, just blocked by training-algo failure).
- `src/lerobot-isaac-env/.../rewards.py` — `progress_reward` with
  named EE body + `PROGRESS_REWARD_DEBUG` env var.
- `src/lerobot-isaac-env/.../tasks/pick_and_place.py` — wires
  progress_reward at weight=10, distance_scale=0.4, ee_body_name=gripper_link.

All infrastructure is correct. The blocker is in the RL training axis,
not the env / wrapper / boot ladder.

---

## Cross-references

- Parent plan: `plans/2026-05-23-wm-isaac-env-plan.md`
- Single-trial diagnoses: `plans/2026-05-23-wm-isaac-tonight.md`
- HP sweep plan (deferred): `plans/2026-05-23-wm-isaac-autoresearch-plan.md`
- Resume guide for trials 1-9 (now obsolete pending lessons here):
  `plans/2026-05-24-wm-isaac-hp-trials-1to9.md`
- Multi-dataset orthogonal axis: `plans/2026-05-23-multi-dataset-training.md`
