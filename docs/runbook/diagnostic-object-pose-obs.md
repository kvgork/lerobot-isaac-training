# Diagnostic: actor object_pose obs

## Why

The 2026-05-24 autoresearch sweep (6 trials, DreamerV3 + Isaac Lab SO-101) showed
all trials collapse with an identical signature:

- `Grads/actor` → 0 within ~500 steps
- `rew_avg` ≈ -0.37 (sparse) or -2 to -6 (hybrid)

Root cause diagnosis: `PolicyObsGroupCfg` only exposed `joint_pos` + `joint_vel` +
`last_action`. The actor had **no information about object position**. The
`object_pose` function in `observations.py` existed but was privileged (critic-only)
and used a hardcoded `env.scene["object"]` key that didn't match any scene entity
in `PickAndPlaceEnvCfg` (which uses `"source_object"`).

Two bugs fixed in this change (2026-05-25):
1. `object_pose` now takes `object_name: str = "source_object"` kwarg — matches
   the `success_termination` fix from commit 811c2e2.
2. `PolicyObsGroupCfg.object_pose` field added, gated by
   `LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1`. Default OFF to preserve the existing
   6-dim state space for BC/LoRA consumers.

## How It Works

When `LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1` is set before the module is imported:

- `PolicyObsGroupCfg.object_pose` is populated with an `ObservationTermCfg`
  pointing to `observations.object_pose(env, object_name="source_object")`.
- `IsaacSO101Env._state_dim` expands from 6 to 13.
- `IsaacSO101Env.observation_space["state"]` becomes `Box(shape=(13,))`.
- `_translate_obs` concatenates `joint_pos[6]` + `object_pose[7]` into a 13-dim
  state vector.

The env var is read at **module import time** in both packages. Set it in the
environment before launching any Python process that imports these modules.

## Run (diagnostic sweep)

```bash
cd ~/workspaces/lerobot-isaac-training
LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1 \
LEROBOT_ISAAC_PROGRESS_WEIGHT=1.0 \
LEROBOT_ISAAC_OBJECT_X=0.30 \
LEROBOT_ISAAC_OBJECT_Y=0.05 \
LEROBOT_ISAAC_OBJECT_Z=0.05 \
SESSION_ID="wm-isaac-objpose-$(date +%Y%m%d-%H%M%S)" \
SECONDS_PER_EXP=10800 \
STEPS=80000 \
  bash scripts/_run_wm_isaac_overnight.sh
```

## Quick smoke check (no GPU needed)

Verify state dimension without Isaac Lab:

```bash
# With object_pose enabled → state_dim should be 13
LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1 \
  pixi run python -c "
from lerobot_isaac_adapters.sheeprl_plugin.isaac_env import IsaacSO101Env
e = IsaacSO101Env()
print('state_dim:', e._state_dim)
print('obs_space[state]:', e.observation_space['state'])
assert e._state_dim == 13, f'expected 13, got {e._state_dim}'
print('PASS')
"

# Without env var → state_dim should be 6
pixi run python -c "
from lerobot_isaac_adapters.sheeprl_plugin.isaac_env import IsaacSO101Env
e = IsaacSO101Env()
print('state_dim:', e._state_dim)
assert e._state_dim == 6, f'expected 6, got {e._state_dim}'
print('PASS')
"
```

Note: the module-level `_INCLUDE_OBJECT_POSE` flag is read once at import. In a
long-running Python process, changing the env var after import has no effect.
Use separate processes for the two checks above.

## Expected reading after 30-45 min

**Diagnostic passes** (missing-obs was the bug):
- `Rewards/rew_avg > 0` (progress reward accumulating)
- `Grads/actor` stays ≥ 0.05 throughout
- Collapse-watcher does NOT trip

**Diagnostic fails** (bug is deeper):
- Actor still collapses — check action scale, joint cfg, scene physics
- Continue audit per `plans/2026-05-24-wm-isaac-hp-trials-1to9.md` stop-rules

## Production path

Once confirmed that object_pose fixes the collapse:

1. Enable cameras (`enable_cameras=True` + `CameraCfg` wiring — see
   `docs/runbook/01-bootstrap.md §Camera obs`).
2. Return `PolicyObsGroupCfg` to cameras-only (drop `LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1`
   from production launch scripts).
3. Keep the object_pose opt-in as a permanent diagnostic escape hatch.

Object pose is privileged information not available on the real robot — using it
in production would create a sim-to-real gap. Cameras are the correct production
fix.
