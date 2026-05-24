# WM Isaac HP-Sweep Pre-Requisite Plan

**Date:** 2026-05-24
**Branch:** `feature/wm-isaac-env`
**Parent:** `plans/2026-05-24-wm-isaac-hp-trials-1to9.md` (revised
8-trial pool blocked on this work)
**Sibling lessons:** `plans/2026-05-24-wm-isaac-lessons.md`

**Goal:** unblock the sparse-reward / curriculum / PPO sweep by
landing the success-termination signal AND the sweep-script refactor.
Estimate **~6 h active eng + 30 min smoke + 1 h commit/push**.

---

## What's missing today

| Asset | Status | Why it blocks the sweep |
|-------|--------|-------------------------|
| `success_termination` in `terminations.py` | NOT IMPLEMENTED | Sparse-reward trials need an episode-terminal signal to ratchet on. Without it the actor sees 0 reward forever. |
| `TerminationsCfg.success` field | NOT WIRED | Reward manager's `is_terminated_term(term_keys=["success"])` raises `ValueError: success: []` if no such term exists. |
| `RewardsCfg.success_bonus` | hardcoded `None` (defensive) | Needs to point at `is_terminated_term({"term_keys":["success"]})` once the termination exists. |
| `TRIAL_POOL` format in sweep script | 6-field `ENT\|RR\|MIN_STD\|WM_LR\|STEPS\|LABEL` | Revised 8 trials need REWARD_SHAPE + ALGO axes → 8-field format. |
| Sweep script reward / algo dispatch | hardcoded `weight=10` reward, `exp=dreamer_v3` algo | Trial 1 needs `rewards.progress.weight=0` (pure sparse); trial 7 needs `exp=ppo`. |
| Trial 6 curriculum hook | object spawn fixed at `(0.5, 0.1, 0.05)` | Trial 6 needs spawn override to `(0.30, 0.05, 0.05)`. |

---

## Phase 1 — `success_termination` (~2 h)

### File: `src/lerobot-isaac-env/src/lerobot_isaac_env/terminations.py`

Implement:
```python
def success_termination(
    env: ManagerBasedRLEnv,
    *,
    dist_threshold: float = 0.05,    # 5 cm — within gripper reach
    lift_threshold: float = 0.02,    # 2 cm above table surface
    robot_cfg: Any = None,
    object_cfg: Any = None,
    ee_body_name: str = "gripper_link",
) -> torch.Tensor:
    """Task success: gripper close to object AND object lifted.

    Returns Bool tensor shape ``(num_envs,)``. Used as a
    TerminationTermCfg via:
        TerminationTermCfg(func=success_termination,
                           params={"dist_threshold":0.05, ...},
                           time_out=False)   # terminal, NOT truncation

    Truncation (`time_out=True`) means "episode hit max_steps".
    Termination (`time_out=False`) means "task done, reset early".
    Only the latter triggers a `success_bonus` RewardTermCfg via
    is_terminated_term(term_keys=["success"]).
    """
    _require_isaaclab()
    if robot_cfg is None: robot_cfg = SceneEntityCfg("robot")
    if object_cfg is None: object_cfg = SceneEntityCfg("source_object")
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    try:
        ee_idx = int(robot.find_bodies(ee_body_name)[0][0])
    except Exception:
        ee_idx = 0
    ee_pos = robot.data.body_pos_w[:, ee_idx, :]
    obj_pos = obj.data.root_pos_w
    dist = torch.norm(ee_pos - obj_pos, dim=-1)
    lifted = obj_pos[:, 2] > lift_threshold
    return (dist < dist_threshold) & lifted
```

**Tests** (`src/lerobot-isaac-env/tests/test_terminations.py`):
- Function importable without Isaac Lab installed.
- Function present in `__all__`.

**Acceptance:** `python -c "from lerobot_isaac_env.terminations import
success_termination"` works in any env.

---

## Phase 2 — wire `TerminationsCfg.success` + `RewardsCfg.success_bonus` (~1 h)

### File: `src/lerobot-isaac-env/src/lerobot_isaac_env/so101_env_cfg.py`

Two edits:

**1. `TerminationsCfg` (around line 366):**
```python
@configclass
@dataclass
class TerminationsCfg:
    time_out: Any = field(
        default_factory=lambda: (
            TerminationTermCfg(func=mdp.time_out, time_out=True)
            if _ISAACLAB_AVAILABLE and mdp is not None
            else None
        )
    )
    # Task-success termination (terminal, NOT truncation). Triggers
    # the `success_bonus` reward term via is_terminated_term.
    success: Any = field(
        default_factory=lambda: (
            TerminationTermCfg(
                func=_terminations_mod.success_termination,
                params={
                    "dist_threshold": 0.05,
                    "lift_threshold": 0.02,
                    "object_cfg": SceneEntityCfg("source_object"),
                    "ee_body_name": "gripper_link",
                },
                time_out=False,
            )
            if _ISAACLAB_AVAILABLE and mdp is not None
            else None
        )
    )
```

Add top-of-file import:
```python
from lerobot_isaac_env import terminations as _terminations_mod
```

**2. `RewardsCfg.success_bonus` (around line 333):**

Replace the `success_bonus: Any = None` placeholder with:
```python
success_bonus: Any = field(
    default_factory=lambda: (
        RewardTermCfg(
            func=mdp.is_terminated_term,
            params={"term_keys": ["success"]},
            weight=5.0,
        )
        if _ISAACLAB_AVAILABLE and mdp is not None
        and hasattr(mdp, "is_terminated_term")
        else None
    )
)
```

**Acceptance:** unit test that constructs `SO101EnvCfg()` without Isaac
Lab installed → `cfg.terminations.success is None` (graceful fallback)
+ `cfg.rewards.success_bonus is None`. Confirms scaffold mode still
works.

---

## Phase 3 — live smoke (~1 h)

Boot the env, drive a scripted policy that moves arm to cube + closes
gripper, verify `success` termination fires + `rew_avg` shows the +5
spike.

### Script: `scripts/_smoke_success_term.py`

```python
"""Smoke: scripted arm-to-cube + lift → verifies success_termination
fires + success_bonus contributes to reward."""
from isaaclab.app import AppLauncher
AppLauncher(headless=True, enable_cameras=True)

import numpy as np
import torch
from lerobot_isaac_adapters.sheeprl_plugin.isaac_env import IsaacSO101Env

env = IsaacSO101Env(num_envs=1, image_size=64, headless=True, device="cuda")
obs, info = env.reset(seed=42)

# Scripted policy: linear interpolation from home to over-cube,
# close gripper at end. ~50 steps total.
home_q = np.zeros(6, dtype=np.float32)
target_q = np.array([0.3, -0.5, 0.6, 0.0, 0.0, 1.0], dtype=np.float32)  # tune

terminated_at = None
rewards = []
for i in range(80):
    t = min(i / 50.0, 1.0)
    a = home_q + (target_q - home_q) * t
    obs, r, term, trunc, info = env.step(a)
    rewards.append(r)
    if term:
        terminated_at = i
        print(f"SUCCESS terminated at step {i} reward={r:.4f}")
        break

print(f"terminated_at={terminated_at} max_r={max(rewards):.4f} "
      f"mean_r={sum(rewards)/len(rewards):.4f}")
```

**Acceptance:**
- If `terminated_at is None` → tuning needed (target_q is wrong OR
  thresholds are wrong). Adjust + retry.
- If `terminated_at` fires AND `max(rewards) > 0.1` (= +5 weight × dt
  ≈ +0.17) → success_bonus is contributing. PASS.
- If terminates but `max(rewards) ≈ 0` → success_bonus not wired
  correctly; verify `RewardsCfg.success_bonus` is non-None at runtime.

---

## Phase 4 — sweep script refactor (~1.5 h)

### File: `scripts/_run_autoresearch_wm_isaac.sh`

Two changes:

**1. Pool format → 8 fields:**

```bash
declare -a TRIAL_POOL=(
    # ENT|INIT_STD|MIN_STD|RR|MAX_EP|REWARD_SHAPE|ALGO|LABEL
    "1e-2|2.0|0.3|0.5|100|sparse|dreamer_v3|sparse-success-default"
    "3e-2|4.0|0.5|0.5|100|sparse|dreamer_v3|sparse-high-init_std"
    "1e-2|2.0|0.3|0.5|50|sparse|dreamer_v3|sparse-short-eps"
    "1e-2|2.0|0.3|0.1|100|sparse|dreamer_v3|sparse-low-replay"
    "1e-2|2.0|0.3|0.5|100|hybrid|dreamer_v3|hybrid"
    "1e-2|2.0|0.3|0.5|100|hybrid-curriculum|dreamer_v3|object-at-home"
    "1e-2|2.0|0.3|0.5|100|sparse|ppo|ppo-baseline"
    "1e-1|4.0|0.5|0.5|100|hybrid|dreamer_v3|extreme-entropy"
)
```

Steps fixed at `STEPS=80000`.

**2. Build EXTRA_HYDRA + algo dispatch from REWARD_SHAPE + ALGO:**

```bash
IFS='|' read -r ENT INIT_STD MIN_STD RR MAX_EP REWARD_SHAPE ALGO LABEL <<< "${TRIAL_POOL[$i]}"

# Reward shape → Hydra overrides on lerobot_isaac_env's
# PickAndPlaceEnvCfg.rewards.progress.weight.
case "$REWARD_SHAPE" in
    sparse)
        # progress weight=0 → success_bonus is the only reward signal.
        REWARD_HYDRA="rewards.progress.weight=0.0"
        ;;
    hybrid)
        # progress weight=1 (down from default 10) + success_bonus.
        REWARD_HYDRA="rewards.progress.weight=1.0"
        ;;
    hybrid-curriculum)
        # Same as hybrid + move source_object spawn to near-home.
        REWARD_HYDRA="rewards.progress.weight=1.0 scene.source_object.init_state.pos=[0.30,0.05,0.05]"
        ;;
    *)
        REWARD_HYDRA=""
        ;;
esac

# Algo dispatch — swap exp=dreamer_v3 to exp=ppo when requested.
# sheeprl uses `exp=<name>` and reads from configs/exp/<name>.yaml.
# Verify ppo cfg exists at .pixi/envs/sim/lib/.../sheeprl/configs/exp/ppo.yaml.
ALGO_HYDRA="exp=$ALGO"

EXTRA_HYDRA="algo.actor.ent_coef=$ENT \
             algo.actor.init_std=$INIT_STD \
             algo.actor.min_std=$MIN_STD \
             algo.world_model.optimizer.lr=1e-4 \
             $REWARD_HYDRA \
             $ALGO_HYDRA"
```

NB: for PPO trials, `algo.world_model.optimizer.lr` is invalid — PPO
has no world model. Need conditional: drop the WM-LR Hydra override
when `ALGO=ppo`. Detect via `[ "$ALGO" = "ppo" ]` branch.

Also: pass `MAX_EP` via `env.max_episode_steps=$MAX_EP`.

**Acceptance:** `DRY_RUN=1 bash scripts/_run_autoresearch_wm_isaac.sh`
prints 8 cmds, each with the right per-trial EXTRA_HYDRA. Spot-check
trial 1 (sparse only — no `progress.weight`), trial 6 (curriculum
— object pos override), trial 7 (PPO — no WM-LR).

---

## Phase 5 — sweep script early-kill on actor collapse (~30 min)

Per `plans/2026-05-24-wm-isaac-lessons.md` §collapse signature, kill
a trial early if it trips the v7/v8 pattern. Saves wall budget.

Inside the sweep script's per-trial scrape block, ADD a forensic check
at step 15000 if the trial is still running:

```bash
# Mid-trial early-kill check (collapse detection).
# Skip if SKIP_EARLY_KILL=1.
if [ -z "${SKIP_EARLY_KILL:-}" ]; then
    # After 15k steps, peek at TB. If Grads/actor < 0.005 AND
    # rew_avg < -50 → known collapse. Send SIGINT to the trial.
    (
        sleep 1800   # ~30 min — when 15k steps is reached at ~7 step/s
        ...probe TB scalars, compare Grads/actor < 0.005, etc.
        if collapsed: pkill -INT -f "_wm_isaac_entry.*trial_${i}"
    ) &
    EARLY_KILL_PID=$!
fi
```

Sketch only — exact impl needs careful TB-reload semantics + race
control. Could also be a separate post-launch monitor script
(`scripts/_collapse_killer.sh`) launched alongside the sweep.

Defer if Phase 1-4 already over budget. Without it, trials still hit
their 3h timeout — just slower.

---

## Phase 6 — commit + push (~30 min)

Commits go to TWO repos:

**lerobot-isaac-env feature/wm-isaac-env:**
- `success_termination` impl in `terminations.py`
- `TerminationsCfg.success` wiring
- `RewardsCfg.success_bonus` re-enabled
- (optional) `tests/test_terminations.py` unit test

**lerobot-isaac-training feature/wm-isaac-env:**
- `scripts/_smoke_success_term.py` (new)
- `scripts/_run_autoresearch_wm_isaac.sh` (refactored pool +
  reward/algo dispatch)
- (optional) `scripts/_collapse_killer.sh` if Phase 5 lands
- `plans/2026-05-24-wm-isaac-prereq.md` (this file — marked DONE)

---

## Effort table

| Phase | Time | Blocks |
|-------|------|--------|
| 1 — success_termination impl | 2 h | sparse-reward trials |
| 2 — Cfg wiring | 1 h | reward signal during training |
| 3 — live smoke | 1 h | confidence the termination fires |
| 4 — sweep refactor | 1.5 h | new pool format support |
| 5 — early-kill (optional) | 0.5 h | wall budget efficiency |
| 6 — commit + push | 0.5 h | landing on remote |
| **Total** | **~6 h** | sweep launch |

Add 30-min buffer per phase for debugging → realistic ~9 h elapsed.
One workday with focus.

---

## Risks

| Risk | Mitigation |
|------|------------|
| `success_termination` thresholds wrong → never fires OR fires spuriously | Phase 3 smoke catches both. Tune `dist_threshold` / `lift_threshold` until scripted policy fires once. |
| `sheeprl PPO config` doesn't exist or has incompatible signature with our isaac_so101 env | Check `.pixi/envs/sim/lib/.../sheeprl/configs/exp/ppo.yaml` BEFORE Phase 4. If missing or env-incompatible → trial 7 (PPO baseline) is dropped from the pool. |
| Curriculum spawn override `scene.source_object.init_state.pos=[...]` Hydra path may not exist (depends on how PickAndPlaceEnvCfg sets it) | Verify by `DRY_RUN=1` + dumping the resolved cfg before launching trial 6. If the path is off, fall back to a code-level `scene.source_object_pos` knob on the env. |
| WM-LR Hydra override under `exp=ppo` raises Hydra structured-cfg error | Conditional branch in EXTRA_HYDRA build (Phase 4). |
| `success_bonus` weight=5 + step_dt=1/30 → +0.17 per terminating step, only 1 step per episode → tiny signal | Bump `weight` to 50 if Phase 3 smoke shows reward spike < 1.0. Pure-sparse trials need a LOUD signal. |

---

## Exit Criteria

Pre-req is done when ALL hold:

- `bash scripts/_smoke_success_term.py` shows scripted policy
  terminating with `terminated_at < 80` AND `max_reward > 0.5`.
- `DRY_RUN=1 bash scripts/_run_autoresearch_wm_isaac.sh` prints 8
  trials with correct per-trial Hydra overrides (reward shape, algo,
  episode steps).
- `pytest src/lerobot-isaac-env/tests/test_terminations.py` passes
  (graceful scaffold-mode fallback).
- Both repos pushed to `feature/wm-isaac-env`.

Then the sweep can launch as documented in
`plans/2026-05-24-wm-isaac-hp-trials-1to9.md`.
