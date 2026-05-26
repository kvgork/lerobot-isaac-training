# WM Isaac-Lab Env Track (C) — Plan

> **Status: IMPLEMENTED (2026-05-26).** Env runs end-to-end on `feature/wm-isaac-env`;
> reward signal tuned in follow-up plans (`2026-05-24-wm-isaac-prereq.md`).

**Date:** 2026-05-23
**Branch:** `feature/wm-isaac-env`
**Parent context:** earlier WM autoresearch sweep (`wm-bash-20260522-211616`)
revealed a hard limit — `HDF5ReplayEnv.step()` ignores actions AND
hardcodes reward to 0. DreamerV3 actor head trained against that env
cannot learn task control. This track replaces the replay env with a
real Isaac Lab physics env so the actor finally has causal feedback +
a reward signal.

---

## Problem Statement

Current pipeline:

```
LeRobotDataset (parquet+mp4)
   ↓ lerobot_world_model_bridge
HDF5 windows (frames, states, actions, NO rewards)
   ↓ HDF5ReplayEnv  (sheeprl custom env)
DreamerV3 training:
   - World model learns latent dynamics from REPLAY transitions
     (next-state determined by recorded trajectory, NOT actor action)
   - Actor head receives 0 reward + can't influence env state
   - Result: actor is untrained for the task
```

Target pipeline:

```
Isaac Lab SO-101 pick-place env  (already wired in lerobot-isaac-env)
   ↓ wrapped as sheeprl-compatible gym.Env
   ↓ exposes 30 Hz step(action) — action MATTERS
   ↓ reward = shaped pick-place signal (Z-height, gripper distance, terminal)
DreamerV3 training:
   - World model learns dynamics under arbitrary actor actions
   - Actor head trained via imagined rollouts against task reward
   - Result: actor IS task-directed → real-robot deployable
```

---

## Existing Building Blocks

| Asset | Path | What it gives |
|-------|------|---------------|
| SO-101 Articulation cfg | `src/lerobot-isaac-env/.../so101_articulation.py` | `build_articulation_cfg()` returns ready-to-use Isaac Lab cfg |
| Pick-place env scaffold | `src/lerobot-isaac-env/.../tasks/pickplace.py` | Existing env stub (per CLAUDE.md Phase 1 impl green for cfg construction) |
| Domain randomisation | `src/lerobot-isaac-env/.../randomization.py` | DR config dataclass |
| sheeprl_plugin | `src/lerobot-isaac-adapters/.../sheeprl_plugin/` | Hydra-discoverable env registration |
| DreamerV3 train flow | `src/lerobot-isaac-adapters/.../targets/wm_dreamerv3.py` | Subprocess + metric extraction (works) |
| HP knobs | from prior sweep | `replay_ratio=2, D=64, S=64, lr=1e-4, steps=25k` was the winner |

**Reuse rule:** the sheeprl train cmd shape stays unchanged. Only the env
factory `cfg.env._target_` swaps from `hdf5_env.get_hdf5_env` to a new
`isaac_env.get_isaac_env`.

---

## Architecture

```
sheeprl_plugin/
├── hdf5_env.py            (existing — replay env, kept for back-compat)
├── isaac_env.py           (NEW — Isaac Lab gym wrapper)
└── configs/env/
    ├── custom_hdf5.yaml   (existing)
    └── isaac_so101.yaml   (NEW — Hydra cfg pointing at isaac_env.get_isaac_env)
```

`isaac_env.IsaacSO101Env` (NEW) signature:

```python
class IsaacSO101Env(gym.Env):
    def __init__(
        self,
        task: str = "pickplace",
        num_envs: int = 4,                  # batch dim — DreamerV3 fans out
        image_size: int = 64,
        rate_hz: float = 30.0,
        max_episode_steps: int = 600,
        headless: bool = True,
        device: str = "cuda",
        seed: int | None = None,
        dr_config: str | None = None,
    ): ...

    observation_space = gym.spaces.Dict({
        "rgb":   Box(0, 255, (3, 64, 64), uint8),   # wrist camera default
        "state": Box(-inf, inf, (6,), float32),     # joint positions
    })
    action_space = gym.spaces.Box(-1, 1, (6,), float32)  # joint position deltas

    def reset(self, seed=None, options=None) -> (obs, info): ...
    def step(self, action) -> (obs, reward, terminated, truncated, info):
        # action: target joint positions (or deltas — TBD)
        # reward: shaped pick-place signal (see Reward below)
        # terminated: object in basket OR contact-terminal
        # truncated: max_episode_steps reached
        ...
```

Reward function (shaped, anti-sparse):

```
r = 0
r += 0.1 * exp(-||gripper_pos - object_pos||)   # reach
r += 0.5 if gripper_closed_around_object         # grasp
r += 1.0 * object_z / basket_height_target       # lift
r += 5.0 if object_in_basket                     # terminal
r -= 0.01 if self_collision                      # safety penalty
```

---

## Phase Breakdown

### Phase C0 — Branch + plan (THIS COMMIT)

* `feature/wm-isaac-env` branched off `feature/sim-deploy`.
* This plan file + skeleton `isaac_env.py`.

### Phase C1 — Env wrapper skeleton (1 day)

Tasks:
1. Write `src/lerobot-isaac-adapters/.../sheeprl_plugin/isaac_env.py`
   with the gym.Env shape above. Soft-import isaaclab; raise clear
   error when missing.
2. Wire `_setup_scene()` that builds the Isaac Lab env via the same
   `SO101EnvCfg` already used by `lerobot-isaac-env`.
3. Implement `reset()` and `step()` calling Isaac Lab's
   `env.reset()` / `env.step()`.
4. Implement reward function inside `_compute_reward(obs, info)`.
5. Add Hydra cfg `configs/env/isaac_so101.yaml`.

Acceptance:
* `from lerobot_isaac_adapters.sheeprl_plugin.isaac_env import get_isaac_env; env = get_isaac_env()` returns a gym.Env.
* `env.reset()` and `env.step(env.action_space.sample())` both return non-trivial obs.
* `env.step()` reward is ≠ 0 for at least one frame in 100 steps.

### Phase C2 — sheeprl integration (0.5 day)

Tasks:
1. Update `wm_dreamerv3` target's bridge logic — for `--env isaac_so101`,
   SKIP the HDF5 bridge step (Isaac Lab env is live, no HDF5 needed).
2. Sheeprl cmd uses `env=isaac_so101` (Hydra config name).
3. Verify the env registers correctly via `pixi run -e sim python -c
   "import sheeprl; ..."`.

Acceptance:
* Smoke-train sheeprl exp=dreamer_v3 env=isaac_so101 for 100 steps
  completes without errors.
* `recon_loss` AND `reward_mean` lines emitted to TensorBoard (NOT
  hardcoded 0 like the HDF5 replay path).

### Phase C3 — Reward design + tuning (1 day)

Tasks:
1. Verify the reward function actually moves with the task (positive
   gradient between random arm motion vs scripted pick attempt).
2. Tune reward weights — sparse-only fails for DreamerV3 on small
   compute; shaping is mandatory.
3. Log per-episode reward distribution to the dashboard.

Acceptance:
* On a held-out scripted policy (e.g. linear interpolation toward
  object + close gripper), reward at terminal step > 0.5.
* On a random policy, mean reward ≤ 0.05.
* Reward gradient is informative (correlation between optimal-action
  distance and reward > 0.3).

### Phase C4 — DreamerV3 train (overnight)

Tasks:
1. Launch `scripts/_run_autoresearch_wm.sh` with the new env
   (`ARCH=dreamerv3 ENV=isaac_so101`).
2. Reuse the HP winner from the prior sweep:
   `replay_ratio=2 discrete=64 stochastic=64 steps=200k`.
3. Eval via in-env rollouts (DreamerV3 actor is now task-directed →
   `pc_success` from `IsaacSO101Env` rollouts is real).

Acceptance:
* Training runs to completion within the 12 h budget.
* Eval `pc_success` ≥ 0.10 (some episodes complete; vs 0.0 baseline
  with the replay-env path).

### Phase C5 — Deploy + robot test (0.5 day)

Tasks:
1. Sync the trained ckpt to the laptop via `li-deploy-sync-wm`.
2. Stage with `detect_policy_kind` → `dreamerv3` (already works).
3. Run `li-deploy-session --policy-path <ckpt> --dry-run` on real
   SO-101 to verify joint targets look sane.
4. `--execute --max-relative-target 1.0 --home-on-exit` for the actual
   test.

Acceptance:
* DRY-RUN: predicted joint targets stay in calibrated range, no NaN.
* EXECUTE: arm moves toward objects in the workspace. Success rate
  not required at this phase — just non-degenerate motion.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Isaac Lab boot is slow (~30 s) | Cache the SimulationApp across reset()s, only reset state |
| 30 Hz step rate not achievable | Drop image_size to 32×32, reduce `num_envs` to 1 |
| Reward shaping rewards exploit (arm wags near object forever) | Add max-step truncation + small step penalty |
| Sim-to-real gap | DR via the existing `SO101DomainRandomizationCfg` (lerobot-isaac-env already has it wired) |
| Pixi env conflicts | Isaac Lab lives in `train-dreamer` already? — verify; might need a new `train-dreamer-isaac` env that includes both sheeprl + isaaclab |

---

## Estimated Effort

| Phase | Time |
|-------|------|
| C0 (this commit) | 0.5 h |
| C1 (env wrapper) | 1 d |
| C2 (sheeprl wiring) | 0.5 d |
| C3 (reward design) | 1 d |
| C4 (DreamerV3 train) | 12 h compute |
| C5 (deploy + robot) | 0.5 d |
| **Total active eng** | **~3 working days** + 12 h overnight |

---

## Exit Criteria

The Isaac-Lab WM track is "landed" when ALL hold:

* `isaac_env.IsaacSO101Env` registered as a sheeprl env via Hydra config.
* DreamerV3 training emits non-zero `reward_mean` lines to TensorBoard.
* Trained actor produces non-degenerate joint targets on the real arm
  in DRY-RUN.
* Pixi env can be installed clean from `pixi install -e train-dreamer`
  + `bash scripts/install_train_deps.sh --dreamer --isaac-lab` (or a
  new combined `--wm-isaac` flag).
* A new autoresearch sweep targets `programs/wm-dreamerv3-isaac.md`
  (NEW — sister to the existing `wm-dreamerv3.md`).

---

## Out of Scope

* MuJoCo or any non-Isaac sim backend.
* Multi-task training (focus on pick-place only for first iteration).
* LeWorldModel — still BLOCKED upstream per `lerobot 0.5.x`.
* Mobile manipulation (SO-101 base is fixed in the existing cfg).
