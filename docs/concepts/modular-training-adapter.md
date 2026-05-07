# Modular Training Adapter — Design Concept

**Cross-references:** [ARCHITECTURE.md](../../ARCHITECTURE.md) | [training-dispatch.md](../internals/training-dispatch.md) | [api-reference.md](../api-reference.md)

---

## Why One Entrypoint

Before this design, switching between training backends required changing:
- The training command
- The config format
- The metric extraction regex
- The autoresearch program.md's `script_path`

This meant each backend needed its own documentation path, its own pixi environment
activation step, and its own debug procedure.

The unified `lerobot-isaac-train` command solves this by:
1. Accepting a `--target_arch` flag that selects the backend
2. Mapping all backends to the same config structure
3. Emitting metrics in the same `<name>=<float>` format
4. Supporting `--dry_run` identically across all backends

The result: the autoresearch loop, the evaluation agent, and the CI smoke tests
all work with zero changes regardless of which backend is being trained.

---

## How `--target_arch` Works

The dispatch is a simple lookup:

```
--target_arch smolvla    -> targets/policy_lerobot.py  (train_fn=smolvla_train)
--target_arch act        -> targets/policy_lerobot.py  (train_fn=act_train)
--target_arch diffusion  -> targets/policy_lerobot.py  (train_fn=diffusion_train)
--target_arch dreamerv3  -> targets/wm_dreamerv3.py
--target_arch le_world_model -> targets/wm_leworldmodel.py
```

`policy_lerobot.py` handles all three LeRobot architectures because they share the same
LeRobot CLI (`lerobot.scripts.train`) and differ only in the config file passed.

Each target module:
- Is a separate file with no cross-target imports
- Catches its own import errors (soft imports for heavy deps)
- Handles its own dry-run printing
- Returns a `{"metric_name": ..., "direction": ...}` dict for the caller

---

## How to Add a New Architecture

To add a new training backend (e.g. `tdmpc2`):

1. Create `packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/wm_tdmpc2.py`:
   ```python
   def train(cfg, dry_run=False):
       try:
           import tdmpc2  # soft import
       except ImportError:
           raise ImportError("Install tdmpc2: pip install tdmpc2")
       if dry_run:
           print(f"DRY RUN: would call tdmpc2 training with {cfg}")
           return {"metric_name": "eval_reward", "direction": "maximize", "value": None}
       # ... run training ...
       return {"metric_name": "eval_reward", "direction": "maximize", "value": final_metric}
   ```

2. Add dispatch entry in `train.py`:
   ```python
   TARGETS = {
       ...
       "tdmpc2": "targets.wm_tdmpc2",
   }
   ```

3. Add a YAML config in `packages/lerobot-isaac-configs/configs/wm_tdmpc2.yaml`

4. Add a feature entry in the root `pixi.toml` (if heavy deps):
   ```toml
   [feature.tdmpc2.dependencies]
   tdmpc2 = ">=0.1"
   ```

5. Create `packages/lerobot-isaac-autoresearch/programs/tdmpc2.md` with metric/direction/mutations

That is all. The autoresearch loop, evaluation agent, and `lerobot-isaac` CLI all
automatically work with the new backend.

---

## Metric Emission Contract

This is the non-negotiable contract that every target module must satisfy:

```
Each eval step MUST emit exactly ONE line to stdout:
  <metric_name>=<float>

Where:
  - <metric_name> matches the name in program.md
  - <float> is a decimal number (no scientific notation preferred, but OK)
  - No leading/trailing whitespace
  - No extra text on the same line

Valid:   pc_success=0.73
         recon_loss=0.0317
         pred_loss=4.2e-2
Invalid: pc_success: 0.73       (colon)
         Final score = 0.73     (text prefix with spaces)
         0.73                   (no name)
```

Use `MetricEmitter.emit()` to guarantee correct format. Do not print metrics manually.

---

## Soft Import Pattern

Every target module uses soft imports so the default `pixi install` (dev environment)
does not fail even if the heavy deps are absent:

```python
# In wm_dreamerv3.py:
def train(cfg, dry_run=False):
    try:
        import sheeprl
    except ImportError:
        raise ImportError(
            "sheeprl not installed. "
            "Run: pixi install -e train-dreamer  or  pip install sheeprl[dreamer-v3]"
        )
    ...
```

The `except ImportError` at the top of the file (module level) is forbidden — it would
silently swallow the error. The soft import must be inside the `train()` function body
so the error is raised with a clear message when training is actually attempted.
