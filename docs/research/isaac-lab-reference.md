# Isaac Lab Reference

**Official docs:** https://isaac-sim.github.io/IsaacLab/
**GitHub:** https://github.com/isaac-sim/IsaacLab
**Paper:** https://arxiv.org/abs/2301.10896 (Isaac Gym predecessor; IsaacLab builds on this)
**Tutorial used in this workspace:** https://isaac-sim.github.io/IsaacLab/source/tutorials/03_envs/create_rl_env.html

**Related workspace docs:** [ARCHITECTURE.md](../../ARCHITECTURE.md) | [isaac-lab-integration.md](../internals/isaac-lab-integration.md)

---

## What is Isaac Lab

Isaac Lab is NVIDIA's open-source robot learning framework built on top of Isaac Sim
(Omniverse). It provides:
- GPU-accelerated physics (PhysX) with thousands of parallel environments
- A Manager-Based RL environment API
- USD-based robot asset loading
- Built-in domain randomization via an event manager system
- Gymnasium-compatible interface

Isaac Lab is the sim layer in this workspace. It replaces MuJoCo for SO-101 simulation.

---

## Key API Classes

### `ManagerBasedRLEnvCfg`

The base config class for all Isaac Lab RL environments.

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

@configclass
class SO101EnvCfg(ManagerBasedRLEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # Physics settings (inherited):
    # sim.dt = 1/120  (120 Hz physics)
    # decimation = 4   (30 Hz control)
```

Reference: https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.envs.html

### `ArticulationCfg`

Defines a robot from a USD asset file.

```python
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim.utils as sim_utils

SO101_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="<absolute_path>/so101.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False
        ),
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

Reference: https://isaac-sim.github.io/IsaacLab/source/tutorials/01_assets/run_articulation.html

### `EventTermCfg` (Domain Randomization)

Configures a domain randomization event.

```python
from isaaclab.managers import EventTermCfg, SceneEntityCfg
import isaaclab.envs.mdp as mdp

@configclass
class EventCfg:
    randomize_object_pose = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",   # or "interval" or "startup"
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "position_range": ((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.0)),
            "velocity_range": ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        }
    )
```

Event modes:
- `"reset"` — applied at each episode reset
- `"interval"` — applied at random intervals during episode
- `"startup"` — applied once at env creation

Reference: https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.managers.html#isaaclab.managers.EventManager

### Observation and Action Terms

```python
# Observation term (returns tensor per env):
from isaaclab.managers import ObservationTermCfg
import isaaclab.envs.mdp as mdp

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)

# Action term:
from isaaclab.managers import ActionTermCfg

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint[1-5]", "gripper_joint"],
        scale=1.0,
        use_default_offset=True,
    )
```

---

## USD Asset Management

### SO-101 URDF Source

URDF: https://github.com/TheRobotStudio/SO-ARM100/tree/main/URDF

The binary USD file is NOT vendored in git (it is ~50 MB). Generate it:
```bash
# Download URDF:
git clone https://github.com/TheRobotStudio/SO-ARM100 /tmp/SO-ARM100

# Convert to USD using Isaac Lab's converter:
python -m isaaclab.utils.urdf_converter \
  --input /tmp/SO-ARM100/URDF/SO-ARM100.urdf \
  --output packages/lerobot-isaac-env/assets/usd/so101.usd \
  --merge-fixed-joints \
  --make-instanceable

# Verify:
python -c "
import omni.isaac.core.utils.prims as prim_utils
prim_utils.define_prim('/World/Robot', 'Xform')
print('USD valid')
"
```

The full script is at `packages/lerobot-isaac-env/assets/usd/download_so101_urdf.sh`.

### USD in Version Control

USD files are in `.gitignore`. The gitignore entry:
```
packages/lerobot-isaac-env/assets/usd/*.usd
```

Rationale: USD files are binary blobs that bloat git history and change whenever
the URDF source changes. They are reproducibly generated from the URDF source.

---

## Headless Mode

Required for training. Isaac Lab supports headless via:

```python
# In SO101EnvCfg:
sim = SimulationCfg(headless=True)

# Or via CLI flag when calling isaaclab apps:
app = AppLauncher(headless=True)
simulation_app = app.app
```

When running via `lerobot-isaac-train`, headless is set to `True` by default.
Override with `--headless false` only when you need visual debugging.

Environment variable fallback: `DISPLAY=""` disables any accidental X11 connection attempts.

---

## Version Pinning

Pin in `pixi.toml`. Isaac Lab has pre-1.0 API churn — treat upgrades as a separate plan.

```toml
# pixi.toml (example — fill version after Phase 1 install):
[feature.isaaclab.pypi-dependencies]
# isaaclab = "==4.2.0"   # fill after Phase 1 install
```

Check current version: `python -c "import isaaclab; print(isaaclab.__version__)"`

---

## RTX 3080 (10 GB) Constraints

Isaac Lab recommends 16 GB VRAM. With 10 GB:
- Keep `num_envs <= 8` for physics-only (no cameras)
- Keep `num_envs <= 4` with 64x64 camera observations
- Keep `num_envs == 1` with full-resolution cameras
- Always enable AMP: `cfg.sim.use_gpu_pipeline = True`
- Disable overhead camera during DR replay (wrist camera only)

OOM recovery: the training adapter automatically halves `num_envs` on OOM and retries.

---

## Further Reading

- Manager-Based Env tutorial: https://isaac-sim.github.io/IsaacLab/source/tutorials/03_envs/create_rl_env.html
- Articulation tutorial: https://isaac-sim.github.io/IsaacLab/source/tutorials/01_assets/run_articulation.html
- Domain Randomization (events): https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.managers.html#isaaclab.managers.EventManager
- USD asset import: https://isaac-sim.github.io/IsaacLab/source/how-to/import_new_asset.html
- IsaacLab gymnasium wrapper: https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.envs.html#isaaclab.envs.ManagerBasedRLEnv
