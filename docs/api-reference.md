# API Reference — LeRobot Isaac Training Workspace

**Scope:** Public Python API surface of all 6 packages.
**For architecture:** [ARCHITECTURE.md](../ARCHITECTURE.md)
**For usage:** [USAGE.md](../USAGE.md)

This document covers only the public API: exported symbols intended for use across packages
or from external callers. Internal helpers are not documented here.

---

## `lerobot_isaac_meta`

**Package:** `packages/lerobot-isaac-meta/`
**Install:** `pip install -e packages/lerobot-isaac-meta`

### `workspace_paths`

```python
from lerobot_isaac_meta.workspace_paths import WorkspacePaths

class WorkspacePaths:
    """Resolves canonical paths within the workspace.

    Priority: LEROBOT_ISAAC_WORKSPACE env var > auto-detection from __file__.
    All paths are absolute Path objects.
    """

    @classmethod
    def root(cls) -> Path:
        """Return workspace root directory.

        Returns:
            Path: e.g. ~/workspaces/lerobot-isaac-training
        """

    @classmethod
    def datasets(cls) -> Path:
        """Return datasets/ directory (gitignored)."""

    @classmethod
    def outputs(cls) -> Path:
        """Return outputs/ directory (gitignored)."""

    @classmethod
    def configs(cls) -> Path:
        """Return packages/lerobot-isaac-configs/configs/ directory."""

    @classmethod
    def agent_state(cls) -> Path:
        """Return .agent-state/ directory."""
```

**Usage example:**
```python
from lerobot_isaac_meta.workspace_paths import WorkspacePaths

dataset_dir = WorkspacePaths.datasets() / "so101_pick_v1_filtered"
config_path = WorkspacePaths.configs() / "policy_smolvla.yaml"
```

### CLI Subcommands (`lerobot-isaac`)

| Subcommand | Description |
|-----------|-------------|
| `lerobot-isaac train [args]` | Alias for `lerobot-isaac-train` |
| `lerobot-isaac paths` | Print resolved workspace paths to stdout |
| `lerobot-isaac status` | Print build phase status and installed package versions |

---

## `lerobot_isaac_env`

**Package:** `packages/lerobot-isaac-env/`
**Install:** `pip install -e packages/lerobot-isaac-env`
**Heavy dep:** `isaaclab` (soft import; stubs available without it)

### `SO101EnvCfg`

```python
from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg, SO101PickEnvCfg, SO101PickPlaceEnvCfg

class SO101EnvCfg(ManagerBasedRLEnvCfg):
    """Base config for SO-101 Isaac Lab environments.

    All task-specific configs (SO101PickEnvCfg, SO101PickPlaceEnvCfg) inherit from this.
    Observation space mirrors LeRobotDataset v3.0 column names exactly.

    Attributes:
        num_envs (int): Number of parallel environments. Default 4.
        device (str): Torch device. Default "cuda:0".
        headless (bool): Disable rendering. Default True.
        observations (ObservationsCfg): joint_pos, joint_vel, wrist_cam, overhead_cam.
        actions (ActionsCfg): JointPositionAction (6-dim, radians).
        events (EventCfg): DR configuration (enable/disable per factor).
    """
```

### `make_env`

```python
from lerobot_isaac_env import make_env

def make_env(
    env_id: str,
    num_envs: int = 4,
    headless: bool = True,
    enable_dr: bool = False,
    device: str = "cuda:0",
) -> gymnasium.Env:
    """Create an Isaac Lab SO-101 environment.

    Args:
        env_id: One of "Isaac-SO101-Pick-v0", "Isaac-SO101-PickPlace-v0".
        num_envs: Parallel environments (keep <= 8 for RTX 3080).
        headless: True disables all rendering (required for training).
        enable_dr: If True, enables all EventTermCfg entries.
        device: CUDA device string.

    Returns:
        gymnasium.Env with obs/action spaces matching LeRobotDataset convention.

    Example:
        env = make_env("Isaac-SO101-Pick-v0", num_envs=4, headless=True)
        obs, info = env.reset()
        obs, rew, done, trunc, info = env.step(env.action_space.sample())
    """
```

### `build_articulation_cfg`

```python
from lerobot_isaac_env.so101_articulation import build_articulation_cfg

def build_articulation_cfg(usd_path: str | Path) -> ArticulationCfg:
    """Build the SO-101 ArticulationCfg from a USD file path.

    Args:
        usd_path: Absolute path to so101.usd.

    Returns:
        ArticulationCfg with joint specs matching SO-101 DYNAMIXEL servos.
    """
```

### Task Configs

```python
from lerobot_isaac_env.tasks.pick import SO101PickEnvCfg
from lerobot_isaac_env.tasks.pick_and_place import SO101PickPlaceEnvCfg

# SO101PickEnvCfg — Stage 1 (fixed-position pick)
# SO101PickPlaceEnvCfg — Stages 2-4 (variable position + obstacles)
# Both are ManagerBasedRLEnvCfg subclasses ready for make_env()
```

---

## `lerobot_isaac_adapters`

**Package:** `packages/lerobot-isaac-adapters/`
**Install:** `pip install -e packages/lerobot-isaac-adapters`
**CLI entrypoint:** `lerobot-isaac-train`

### `train.main`

```python
from lerobot_isaac_adapters.train import main

def main(args: list[str] | None = None) -> int:
    """Dispatch training to the appropriate target module.

    Parses CLI args (or accepts list for programmatic use).
    Always emits a metric line on stdout at each eval step.
    Supports --dry_run to print dispatched command without executing.

    Args:
        args: CLI argument list. If None, reads sys.argv[1:].

    Returns:
        Exit code (0 = success).

    Metric emission contract:
        Each target calls metric_extractor.emit(name, value) which prints:
        "<name>=<float>" to stdout.
    """
```

### `metric_extractor.emit`

```python
from lerobot_isaac_adapters.metric_extractor import MetricEmitter

class MetricEmitter:
    """Emits metrics in the format expected by autoresearch-ml-executor-worker."""

    def emit(self, name: str, value: float, step: int | None = None) -> None:
        """Print metric in autoresearch-compatible format.

        Args:
            name: Metric name (e.g. "pc_success", "recon_loss").
            value: Float value.
            step: Optional training step counter.

        Prints:
            "<name>=<value>" to stdout.
            Optionally logs to W&B if WANDB_API_KEY is set.

        Example:
            emitter = MetricEmitter()
            emitter.emit("pc_success", 0.73)   # prints: pc_success=0.73
        """
```

### `record_episodes`

```python
from lerobot_isaac_adapters.isaac_data_recorder import record_episodes

def record_episodes(
    env: gymnasium.Env,
    policy: Callable | None,
    dataset_path: str | Path,
    num_episodes: int,
    schema_features: dict | None = None,
) -> Path:
    """Record policy rollouts in Isaac Lab env to LeRobotDataset Parquet.

    If policy is None, records random actions (for testing).
    Validates schema against schema_features before writing.

    Args:
        env: Isaac Lab gymnasium env (SO-101).
        policy: Callable(obs) -> action, or None for random.
        dataset_path: Output dataset directory.
        num_episodes: Episodes to record.
        schema_features: Expected features dict; raises ValueError if mismatch.

    Returns:
        Path to written dataset.
    """
```

### Target Module Interface

All three target modules (`policy_lerobot`, `wm_dreamerv3`, `wm_leworldmodel`) implement:

```python
def train(cfg: DictConfig, dry_run: bool = False) -> dict:
    """Run training for this target.

    Args:
        cfg: Hydra DictConfig from config YAML.
        dry_run: If True, print command and return without executing.

    Returns:
        {
            "metric_name": str,    # e.g. "pc_success"
            "direction": str,      # "maximize" or "minimize"
            "value": float | None  # final metric value; None if dry_run
        }
    """
```

---

## `lerobot_isaac_autoresearch`

**Package:** `packages/lerobot-isaac-autoresearch/`
**Install:** `pip install -e packages/lerobot-isaac-autoresearch`

### `train_wrapper.main`

```python
from lerobot_isaac_autoresearch.train_wrapper import main

def main(args: list[str] | None = None) -> int:
    """Stable shim — forwards all args to lerobot-isaac-train.

    This is the script_path registered in all program.md files.
    Its sole purpose is to provide a stable entry point so program.md
    does not need to change if adapters package moves.

    Args:
        args: CLI argument list. If None, reads sys.argv[1:].

    Returns:
        Exit code from lerobot-isaac-train.
    """
```

### `program.md` Schema Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `metric.name` | str | Yes | Metric key to extract from stdout |
| `metric.direction` | str | Yes | `"maximize"` or `"minimize"` |
| `metric.regex` | str | Yes | Regex to parse metric from stdout |
| `budget.seconds_per_experiment` | int | Yes | Wall-clock budget per run |
| `budget.max_experiments` | int | Yes | Max number of experiments |
| `budget.plateau_limit` | int | Yes | Consecutive non-improving runs before stop |
| `baseline.script_path` | str | Yes | Path to train_wrapper.py |
| `baseline.args` | dict | Yes | Default training args |
| `hyperparameters` | dict | Yes | HP search space |
| `constraints.allow_architecture_change` | bool | No | Default: false for world models |

---

## `lerobot_isaac_synthetic`

**Package:** `packages/lerobot-isaac-synthetic/`
**Install:** `pip install -e packages/lerobot-isaac-synthetic`

### `replay_with_randomization`

```python
from lerobot_isaac_synthetic.isaac_dr.replay_runner import replay_with_randomization

def replay_with_randomization(
    source_dataset_path: str | Path,
    output_path: str | Path,
    num_augmentations: int = 5,
    randomize: list[str] = ("object_pose", "lighting", "friction"),
    headless: bool = True,
    num_envs: int = 4,
    dry_run: bool = False,
) -> Path:
    """Replay real episodes in Isaac Lab with domain randomization.

    For each episode in source_dataset, generates num_augmentations DR variants.

    Args:
        source_dataset_path: Real LeRobotDataset Parquet directory.
        output_path: Output DR Parquet dataset directory.
        num_augmentations: DR variants per real episode.
        randomize: DR factors to apply.
        headless: True required for training.
        num_envs: Parallel envs (keep <= 8 for RTX 3080).
        dry_run: Print intent without executing.

    Returns:
        Path to written DR dataset.

    Raises:
        NotImplementedError: If Isaac Lab is not installed (stubs active).
    """
```

### `write_episodes_to_lerobot_dataset`

```python
from lerobot_isaac_synthetic.isaac_dr.parquet_writer import write_episodes_to_lerobot_dataset

def write_episodes_to_lerobot_dataset(
    episodes: list[dict],
    output_path: str | Path,
    source: str = "sim_dr",
    reference_features: dict | None = None,
) -> Path:
    """Write rollout episodes to LeRobotDataset Parquet format.

    Args:
        episodes: List of dicts with keys: obs, actions, rewards, dones.
        output_path: Output dataset directory.
        source: Source tag for all episodes.
        reference_features: If provided, validates schema before writing.

    Returns:
        Path to written dataset.
    """
```

### `merge_datasets`

```python
from lerobot_isaac_synthetic.merge_utilities import merge_datasets

def merge_datasets(
    real_path: str | Path,
    output_path: str | Path,
    dr_path: str | Path | None = None,
    mimicgen_path: str | Path | None = None,
    real_weight: float = 1.0,
    dr_weight: float = 0.5,
    mimicgen_weight: float = 0.3,
) -> Path:
    """Merge real + DR + MimicGen datasets into a unified LeRobotDataset.

    Updates meta/info.json, meta/stats.json, meta/episodes.parquet.
    Idempotent: running twice with same inputs produces same output.

    Args:
        real_path: Real (filtered) Parquet dataset.
        output_path: Output merged dataset.
        dr_path: DR Parquet dataset (optional).
        mimicgen_path: MimicGen Parquet dataset (optional).
        real_weight: Sampling weight for real episodes.
        dr_weight: Sampling weight for DR episodes.
        mimicgen_weight: Sampling weight for MimicGen episodes.

    Returns:
        Path to merged dataset.

    Raises:
        ValueError: If source schemas are incompatible.
    """
```

---

## `lerobot_isaac_configs`

**Package:** `packages/lerobot-isaac-configs/`
**Install:** `pip install -e packages/lerobot-isaac-configs`

### `get_configs_dir`

```python
from lerobot_isaac_configs import get_configs_dir

def get_configs_dir() -> Path:
    """Return the absolute path to the configs/ directory.

    Works whether the package is installed editable (monorepo) or
    as a standalone pip package (post-spinout).

    Returns:
        Path to directory containing *.yaml config files.
    """
```

### `load_config`

```python
from lerobot_isaac_configs.loader import load_config

def load_config(name: str) -> DictConfig:
    """Load a named YAML config as a Hydra DictConfig.

    Args:
        name: Config file stem (without .yaml).
              One of: policy_smolvla, policy_act, policy_diffusion,
                       wm_dreamerv3, wm_leworldmodel, isaac_so101_pickplace.

    Returns:
        DictConfig with config values.

    Raises:
        FileNotFoundError: If config name is not found.

    Example:
        cfg = load_config("policy_smolvla")
        print(cfg.batch_size)   # 32
    """
```
