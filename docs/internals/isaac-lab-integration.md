# Isaac Lab Integration — Internals

**Cross-references:** [ARCHITECTURE.md](../../ARCHITECTURE.md) | [data-pipeline.md](./data-pipeline.md)
**Package:** `packages/lerobot-isaac-env/`
**External docs:** https://isaac-sim.github.io/IsaacLab/

---

## ManagerBasedRLEnv Architecture

Isaac Lab's Manager-Based RL environment separates concerns into independent "managers",
each responsible for one aspect of the MDP. `SO101EnvCfg` is a subclass of
`ManagerBasedRLEnvCfg` and wires all managers together:

```
SO101EnvCfg (ManagerBasedRLEnvCfg subclass)
  |
  +-- scene:        SceneCfg          -- USD stage: robot, objects, lights, cameras
  +-- observations: ObservationsCfg   -- MDP observation functions -> obs dict
  +-- actions:      ActionsCfg        -- MDP action processors -> actuator commands
  +-- rewards:      RewardsCfg        -- MDP reward functions -> scalar per step
  +-- terminations: TerminationsCfg   -- MDP done conditions -> bool per env
  +-- events:       EventCfg          -- DR event functions -> called at reset/interval
```

Each manager holds a dict of named "terms" — Python functions decorated with `@configclass`
that receive the env and return values.

---

## MDP Terms (Key Functions)

### Observation Terms

Defined in `packages/lerobot-isaac-env/src/lerobot_isaac_env/observations.py`:

| Term | Output Shape | LeRobot Column | Notes |
|------|-------------|----------------|-------|
| `joint_pos` | `(num_envs, 6)` | `observation.state[:6]` | radians, not normalized |
| `joint_vel` | `(num_envs, 6)` | `observation.state[6:]` | rad/s |
| `wrist_cam_rgb` | `(num_envs, H, W, 3)` | `observation.images.wrist` | uint8, 30 Hz |
| `overhead_cam_rgb` | `(num_envs, H, W, 3)` | `observation.images.overhead` | uint8 |
| `object_pose` | `(num_envs, 7)` | not in LeRobot; internal use | position + quaternion |

The concatenation `[joint_pos, joint_vel]` produces `observation.state` with shape `(12,)`,
matching the real SO-101 recording format exactly.

### Action Terms

Defined in `packages/lerobot-isaac-env/src/lerobot_isaac_env/actions.py`:

| Term | Input | Mapping |
|------|-------|---------|
| `JointPositionAction` | `(num_envs, 6)` float32 | direct joint position targets in radians |

Actions are NOT normalized internally. LeRobot convention uses raw radians; Isaac Lab's
actuator model (`ImplicitActuatorCfg`) converts targets to torques via PD control.

### Reward Terms

Defined in `packages/lerobot-isaac-env/src/lerobot_isaac_env/rewards.py`:

| Term | Value | Condition |
|------|-------|-----------|
| `success` | `+1.0` | object in target zone |
| `distance_shaping` | `-dist * 0.01` | dense; disabled by default for IL |

For imitation learning, only the sparse `success` reward is active. Dense shaping is
opt-in via `cfg.rewards.distance_shaping.weight = 0.01`.

### Termination Terms

| Term | Condition |
|------|-----------|
| `time_out` | `step_count >= 200` (200 steps = 6.67 s at 30 Hz) |
| `object_dropped` | object Z position < table height - 0.05 m |
| `success` | object center within target_zone_radius of target |

---

## Domain Randomization via EventTermCfg

DR is configured in `packages/lerobot-isaac-env/src/lerobot_isaac_env/randomization.py`.

```python
# Illustrative — not copy-paste code
@configclass
class EventCfg:
    randomize_object_pose = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",          # applied at every episode reset
        params={
            "position_range": ((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.0)),
            "velocity_range": ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            "asset_cfg": SceneEntityCfg("object"),
        }
    )

    randomize_lighting = EventTermCfg(
        func=mdp.randomize_light_color,
        mode="interval",       # applied periodically during episode
        interval_range_s=(5.0, 10.0),
        params={"intensity_range": (500, 2000)}
    )

    randomize_friction = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.2, 0.8),
            "asset_cfg": SceneEntityCfg("object"),
        }
    )
```

**DR is disabled at Stage 1** (fixed-position pick). The `replay_runner.py` selects
which events to apply via the `--randomize` flag:
```bash
--randomize object_pose lighting friction   # enables these 3 EventTermCfg entries
--randomize object_pose                     # only object pose, others disabled
```

Disabling is done by setting `EventTermCfg.enabled = False` at runtime:
```python
cfg.events.randomize_lighting.enabled = False
```

---

## USD Asset Wiring

The SO-101 articulation is defined in `so101_articulation.py`:

```python
SO101_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WORKSPACE}/packages/lerobot-isaac-env/assets/usd/so101.usd"
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 0.0,
            "joint2": -0.5,
            "joint3": 0.8,
            "joint4": 0.0,
            "joint5": 0.3,
            "gripper_joint": 0.0,
        }
    ),
    actuators={
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-5]"],
            stiffness=800.0,
            damping=40.0,
            effort_limit=400.0,
            velocity_limit=100.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint"],
            stiffness=400.0,
            damping=20.0,
        ),
    },
)
```

The USD file is NOT vendored in git. It is generated from the SO-ARM100 URDF:
```bash
python -m isaaclab.utils.urdf_converter \
  --input SO-ARM100.urdf \
  --output packages/lerobot-isaac-env/assets/usd/so101.usd \
  --merge-fixed-joints
```

See `packages/lerobot-isaac-env/assets/usd/README.md` for the full provenance.

---

## Physics Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Physics rate | 120 Hz | 4× the control rate (30 Hz) |
| Control rate | 30 Hz | Matches real SO-101 camera + servo rate |
| Gravity | 9.81 m/s² | Earth gravity |
| `num_envs` | 4–8 | RTX 3080 budget; reduce to 1 if OOM |
| GPU pipeline | enabled | PhysX GPU acceleration |
| `device` | `cuda:0` | GPU physics + tensor API |

---

## Gym Registration

Gym environments are registered in `packages/lerobot-isaac-env/src/lerobot_isaac_env/__init__.py`:

```python
gymnasium.register(
    id="Isaac-SO101-Pick-v0",
    entry_point="lerobot_isaac_env.tasks.pick:SO101PickEnv",
    kwargs={"cfg": SO101PickEnvCfg()},
)

gymnasium.register(
    id="Isaac-SO101-PickPlace-v0",
    entry_point="lerobot_isaac_env.tasks.pick_and_place:SO101PickPlaceEnv",
    kwargs={"cfg": SO101PickPlaceEnvCfg()},
)
```

Usage:
```python
import gymnasium as gym
import lerobot_isaac_env  # triggers registration

env = gym.make("Isaac-SO101-Pick-v0", headless=True)
obs, info = env.reset()
obs, reward, done, truncated, info = env.step(env.action_space.sample())
```

---

## RTX 3080 (10 GB) Constraints

| Config | VRAM Est. | Fits 3080? |
|--------|-----------|-----------|
| `num_envs=8`, no cameras | ~4 GB | Yes |
| `num_envs=8`, wrist cam 64x64 | ~7 GB | Yes |
| `num_envs=8`, wrist + overhead 480x480 | ~18 GB | No |
| `num_envs=4`, wrist cam 64x64 | ~4 GB | Yes |
| `num_envs=1`, full render | ~6 GB | Yes |

Recommendation: disable overhead camera during DR replay; enable only for policy evaluation.
Keep `num_envs <= 8` at all times.
