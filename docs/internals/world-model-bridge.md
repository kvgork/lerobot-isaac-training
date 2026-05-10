# World Model Bridge — Internals

**Cross-references:** [ARCHITECTURE.md](../../ARCHITECTURE.md) | [data-pipeline.md](./data-pipeline.md) | [training-dispatch.md](./training-dispatch.md)
**Skill source:** `${CLAUDE_CODE_ROOT}/skills/lerobot_world_model_bridge/`
**Skill docs:** `${CLAUDE_CODE_ROOT}/skills/lerobot_world_model_bridge/SKILL.md`
**Vault note:** `${VAULT_ROOT}/05-Wiki/entities/LeWorldModel.md`

---

## Overview

The `lerobot_world_model_bridge` skill (in the `claude_code` repo) handles all
Parquet-to-HDF5 conversions for world-model training. This document explains how
the bridge is wired into this workspace and the schema details for each target.

The workspace does NOT duplicate the bridge skill. It invokes it via:
```python
# Agent invocation:
# Task(lerobot-worldmodel-bridge, {input_parquet: ..., output_hdf5: ..., target: ...})

# Or direct skill invocation (Python):
from skills.lerobot_world_model_bridge.operations import convert_dataset
convert_dataset(input_parquet="datasets/so101_pick", output_hdf5="outputs/hdf5/...", target="dreamerv3")
```

---

## DreamerV3 HDF5 Schema

Target preset: `dreamerv3` with `image_size=64`

```
hdf5_file.hdf5
  /episode_000001/
      obs/
          image:   (T, 64, 64, 3)     uint8    -- wrist camera (primary)
          state:   (T, 12)            float32  -- joint_pos (6) + joint_vel (6)
      actions:     (T, 6)             float32  -- joint position targets (radians)
      rewards:     (T,)               float32  -- sparse {0.0, 1.0}
      dones:       (T,)               bool     -- episode termination
  /episode_000002/
      ...
```

Image resizing: bicubic downscale from raw resolution (480px) to 64px.
The wrist camera is used as the primary image observation (not overhead) because
it provides the most task-relevant information for pick-and-place.

Only the wrist camera is included in the DreamerV3 HDF5. Overhead camera
is excluded to save VRAM. This is a design choice; to include both, modify
the skill invocation with `cameras=["wrist", "overhead"]`.

---

## LeWorldModel HDF5 Schema

Target preset: `le_world_model` with `image_size=96`

```
hdf5_file.hdf5
  /episode_000001/
      obs/
          image:   (T, 96, 96, 3)     uint8    -- wrist camera
          state:   (T, 12)            float32  -- joint_pos + joint_vel
      actions:     (T, 6)             float32  -- joint position targets
      rewards:     (T,)               float32
      dones:       (T,)               bool
  ...
```

**IMPORTANT — Schema Warning:** The HF LeWorldModel HDF5 schema is partially undocumented.
The `(96,96)` preset in the skill is based on inspection of `quentinll/lewm-pusht` on
HuggingFace Hub. If schema drift occurs between the skill's output and what LeWM expects,
run the schema discovery script:
```python
import h5py
from huggingface_hub import hf_hub_download
path = hf_hub_download("quentinll/lewm-pusht", "data.hdf5")
with h5py.File(path) as f:
    def print_tree(name, obj):
        print(name, getattr(obj, 'shape', ''), getattr(obj, 'dtype', ''))
    f.visititems(print_tree)
```
Then compare with the skill's output and adjust the `le_world_model` preset if needed.

---

## Bridge Invocation Patterns

### Direct Agent Invocation (Recommended)

```
Task(lerobot-worldmodel-bridge, {
    input_parquet: "datasets/so101_pick_v1_filtered",
    output_hdf5: "outputs/hdf5/so101_pick_dreamerv3.hdf5",
    target: "dreamerv3",
    image_size: 64
})
```

The agent handles:
- Progress reporting
- Schema validation
- Error recovery (partial conversion resume)

### Automatic Conversion in Training Targets

`wm_dreamerv3.py` and `wm_leworldmodel.py` auto-detect Parquet input and convert:
```python
# In wm_dreamerv3.py (illustrative):
if dataset_path.endswith(".parquet") or Path(dataset_path).is_dir():
    hdf5_path = auto_convert(dataset_path, target="dreamerv3", image_size=64)
else:
    hdf5_path = dataset_path  # already HDF5
```

Pre-converting to HDF5 separately (via the agent) is faster if you run multiple
training experiments on the same dataset.

---

## Key Differences: DreamerV3 vs LeWorldModel

| Aspect | DreamerV3 | LeWorldModel |
|--------|-----------|-------------|
| Image size | 64x64 (RTX 3080 budget) | 96x96 (requires AMP + checkpointing) |
| Architecture | RSSM (GRU + CNN encoder) | JEPA-style latent predictor |
| Loss | ELBO: recon + KL + reward | next-embedding prediction |
| Metric | `recon_loss` (minimize) | `pred_loss` (minimize) |
| HDF5 schema | Well-documented; sheeprl-standard | Partially undocumented; inspect pusht |
| VRAM at batch=16 | ~8 GB (fits 3080) | ~12 GB (marginal; use batch=8) |
| Training time (RTX 3080) | ~2 h per run | ~1.5 h per run |

---

## Relationship to `lerobot-worldmodel-bridge` Agent

The agent (`${CLAUDE_CODE_ROOT}/agents/lerobot-worldmodel-bridge.md`) is a
higher-level wrapper that:
1. Invokes the skill's `convert_dataset()` function
2. Validates the output schema
3. Reports progress back to the calling session
4. Can be called from the training orchestrator as a pre-processing step

The skill's Python API is the low-level implementation; the agent provides the
Claude-invocable interface. In this workspace, always prefer the agent invocation
over calling the skill directly.
