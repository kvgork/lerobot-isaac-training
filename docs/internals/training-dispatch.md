# Training Dispatch — Internals

**Cross-references:** [ARCHITECTURE.md](../../ARCHITECTURE.md) | [data-pipeline.md](./data-pipeline.md) | [autoresearch-integration.md](./autoresearch-integration.md)

---

## Overview

The training entrypoint `lerobot-isaac-train` is implemented by
`packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/train.py`.
It is a **thin dispatcher**: it reads `--target_arch`, resolves the config, and delegates to
one of three target modules. All targets share the same argument surface and the same
metric emission contract.

---

## Dispatch Flow

```
lerobot-isaac-train --target_arch <arch> --config <yaml> [options]
         |
         | train.py: parse args, load config, validate
         v
  +------+--------+----------------+-----------+
  |      |                         |           |
  v      v                         v           v
smolvla  act  diffusion      dreamerv3    le_world_model
  |      |       |               |              |
  v      v       v               v              v
targets/ targets/ targets/   targets/       targets/
policy_  policy_  policy_    wm_            wm_
lerobot  lerobot  lerobot    dreamerv3      leworldmodel
.py      .py      .py        .py            .py
```

All three `targets/*.py` modules expose the same interface:

```python
def train(cfg: DictConfig, dry_run: bool = False) -> dict:
    """
    Returns: {"metric_name": str, "direction": str, "value": float or None}
    Emits:   "<metric_name>=<value>" on stdout at each eval step.
    """
```

---

## Target Modules

### `targets/policy_lerobot.py`

**Target architectures:** `smolvla`, `act`, `diffusion`

Dispatches to LeRobot's training script as a subprocess:
```python
subprocess.run([
    sys.executable, "-m", "lerobot.scripts.train",
    "--config-path", cfg.lerobot_config_path,
    f"dataset_repo_id={cfg.dataset_path}",
    f"training.output_dir={cfg.output_dir}",
    f"training.num_steps={cfg.max_steps}",
])
```

Metric extraction: reads stdout from the subprocess and re-emits the last
`pc_success=<float>` line via `metric_extractor.emit()`.

**Dry-run mode:** prints the resolved subprocess command without executing.

### `targets/wm_dreamerv3.py`

**Target architecture:** `dreamerv3`

1. Accepts either Parquet or HDF5 path
2. If Parquet: auto-converts via `lerobot_world_model_bridge` skill (64x64 preset)
3. Dispatches to sheeprl or dreamer-v3-pytorch as subprocess
4. Emits `recon_loss=<float>` at each eval step

```python
subprocess.run([
    sys.executable, "-m", "sheeprl", "dreamer_v3",
    f"data.data_dir={hdf5_path}",
    "env.observation_size=[64,64]",
    f"algo.total_steps={cfg.max_steps}",
])
```

**Dry-run mode:** prints "would call dreamer training with these kwargs: {...}".

### `targets/wm_leworldmodel.py`

**Target architecture:** `le_world_model`

1. Auto-converts Parquet to HDF5 (96x96) if needed
2. Dispatches to HF LeWorldModel training script
3. Emits `pred_loss=<float>`

**Dry-run mode:** prints "would call leworldmodel training with these kwargs: {...}".

---

## Subprocess Arguments: Full Matrix

| Argument | Policy (LeRobot) | DreamerV3 (sheeprl) | LeWorldModel |
|----------|-----------------|---------------------|--------------|
| `--config` | hydra yaml path | env + algo yaml | yaml |
| `--dataset_path` | Parquet dir | HDF5 file | HDF5 file |
| `--output_dir` | checkpoint dir | checkpoint dir | checkpoint dir |
| `--max_steps` | `training.num_steps` | `algo.total_steps` | `training.steps` |
| `--seed` | `seed` | `seed` | `seed` |
| `--batch_size` | `training.batch_size` | `algo.batch_size` | `training.batch_size` |

Each target translates the unified `cfg` object into the backend's specific argument format.

---

## Metric Extraction

All targets use `metric_extractor.py` to emit metrics:

```python
from lerobot_isaac_adapters.metric_extractor import MetricEmitter

emitter = MetricEmitter()
emitter.emit("pc_success", 0.73)    # prints: pc_success=0.73
emitter.emit("recon_loss", 0.0317)  # prints: recon_loss=0.0317
emitter.emit("pred_loss", 0.0421)   # prints: pred_loss=0.0421
```

The output format is exactly `<name>=<value>` with no extra whitespace.
This matches the default regex in `autoresearch-ml-executor-worker`:
`(\w+)[=:\s]+([0-9.eE+-]+)`

The emitter also optionally logs to W&B if `WANDB_API_KEY` is set:
```python
# If wandb initialized:
wandb.log({metric_name: value, "step": step})
```

If W&B is not configured, stdout-only fallback is always active.

---

## OOM Retry Logic

The adapter includes an OOM recovery ladder for Isaac Lab environments and world models:

```
Attempt 1: cfg.num_envs=<config value>
   |-- OOM? -> reduce num_envs by half
Attempt 2: cfg.num_envs //= 2
   |-- OOM? -> reduce batch_size by half
Attempt 3: cfg.batch_size //= 2
   |-- OOM? -> enable gradient checkpointing
Attempt 4: cfg.gradient_checkpointing=True
   |-- OOM? -> FAIL with clear error message
```

The ladder is implemented in `train.py` wrapping each target call. Each retry is logged to
stdout and to `.agent-state/<session>/oom_events.jsonl` so the autoresearch loop can learn
to avoid OOM configurations.

---

## Config Resolution Order

When `train.py` resolves a config value, it follows this precedence:

```
CLI flag (highest)
  > YAML config override section
  > YAML config base section
  > Package default (lowest)
```

Example: if `configs/policy_smolvla.yaml` sets `batch_size: 32` but the CLI passes
`--batch_size 16`, the value `16` is used.

---

## `train_wrapper.py` (Autoresearch Shim)

`packages/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py`
is a stable shim between `autoresearch-ml-executor-worker` and `lerobot-isaac-train`.

Its sole purpose: `autoresearch-ml-executor-worker` requires a stable `script_path` in
`program.md`. The shim forwards all CLI args unchanged to `lerobot-isaac-train`, so if
the adapters package moves, only the shim needs updating — not all program.md files.

```python
# train_wrapper.py (illustrative):
import subprocess, sys

result = subprocess.run(
    ["lerobot-isaac-train"] + sys.argv[1:],
    capture_output=False  # pass through stdout so executor reads metrics
)
sys.exit(result.returncode)
```
