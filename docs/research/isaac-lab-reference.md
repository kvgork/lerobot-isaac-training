# Isaac Lab Reference

> **Phase 0 placeholder.** Replace with full notes when Phase 1 implementation begins.
> This document captures the key API surface needed for `lerobot-isaac-env`.

**Official docs:** https://isaac-sim.github.io/IsaacLab/
**GitHub:** https://github.com/isaac-sim/IsaacLab

---

## Core API Surface

### ManagerBasedRLEnvCfg

The central config class for Manager-Based RL environments. Subclass this in `so101_env_cfg.py`.

```python
from isaaclab.envs import ManagerBasedRLEnvCfg

@configclass
class SO101EnvCfg(ManagerBasedRLEnvCfg):
    # Observation manager config
    observations: ObservationsCfg = ObservationsCfg()
    # Action manager config
    actions: ActionsCfg = ActionsCfg()
    # Reward manager config
    rewards: RewardsCfg = RewardsCfg()
    # Termination manager config
    terminations: TerminationsCfg = TerminationsCfg()
    # Event manager config (domain randomization)
    events: EventCfg = EventCfg()
```

### EventTermCfg (Domain Randomization)

Used to configure domain randomization events in Isaac Lab's event manager.

```python
from isaaclab.managers import EventTermCfg, SceneEntityCfg

@configclass
class EventCfg:
    randomize_object_pose = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={...}
    )
```

Event modes: `"reset"` (at episode start), `"interval"` (periodically), `"startup"` (once).

### ArticulationCfg (USD articulation)

Defines a robot as an Isaac Lab articulation from a USD file.

```python
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

SO101_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path="<path_to_so101.usd>"),
    actuators={
        "joint_pos": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=800.0,
            damping=40.0,
        )
    },
)
```

### Headless Mode

Always run without display in training. Isaac Lab supports headless via:
```python
sim_cfg = SimulationCfg(headless=True)
# Or via CLI: --headless
```

When using `pixi shell`, set: `DISPLAY=""` and ensure no GUI is attempted.

### USD Articulation for SO-101

The SO-101 URDF is available at: https://github.com/TheRobotStudio/SO-ARM100/tree/main/URDF

Convert to USD using Isaac Lab's built-in URDF importer:
```bash
python -m isaaclab.utils.urdf_converter \
  --input SO-ARM100.urdf \
  --output packages/lerobot-isaac-env/assets/usd/so101.usd \
  --merge-fixed-joints
```

See `packages/lerobot-isaac-env/assets/usd/download_so101_urdf.sh` for the full script.

---

## RTX 3080 (10 GB) Constraints

- Isaac Lab recommends 16 GB VRAM; 10 GB is documented as minimum for headless rollouts.
- Keep `num_envs` at 4–8 for training to avoid OOM.
- Disable image rendering in baseline configs (use joint state obs only initially).
- If OOM occurs, reduce `num_envs` to 1 and set `device="cuda:0"` with AMP enabled.

---

## Key Isaac Lab Versions

Pin in `pixi.toml` — treat version upgrades as a separate plan due to API churn.

```toml
[dependencies]
# isaaclab = "=<version>"  # fill in after Phase 1 install
```

---

## Further Reading

- Manager-Based RL Env guide: https://isaac-sim.github.io/IsaacLab/source/tutorials/03_envs/create_rl_env.html
- Articulation tutorial: https://isaac-sim.github.io/IsaacLab/source/tutorials/01_assets/run_articulation.html
- Domain Randomization (events): https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.managers.html#isaaclab.managers.EventManager
- USD asset import: https://isaac-sim.github.io/IsaacLab/source/how-to/import_new_asset.html
