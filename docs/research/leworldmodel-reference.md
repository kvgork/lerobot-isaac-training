# LeWorldModel Reference

**HuggingFace Hub:** https://huggingface.co/lerobot
**Reference dataset:** https://huggingface.co/datasets/quentinll/lewm-pusht
**LeRobot GitHub:** https://github.com/huggingface/lerobot

**Skill reference:** `/home/koen/tools/claude_code/skills/lerobot_world_model_bridge/`
**Vault note:** `/home/koen/Documents/Vaults/Local/05-Wiki/entities/LeWorldModel.md`
**Related workspace docs:** [world-model-bridge.md](../internals/world-model-bridge.md) | [ARCHITECTURE.md](../../ARCHITECTURE.md)

---

## What is HF LeWorldModel

LeWorldModel (Alibert et al. 2025) is Hugging Face's robot manipulation world model,
part of the LeRobot ecosystem. It learns to predict future observations from action sequences
using a JEPA-style (Joint Embedding Predictive Architecture) latent space.

Key properties:
- Learns from RGB + state observations (same format as LeRobot imitation learning)
- Predicts next observation embedding (not pixel-level reconstruction)
- Enables model-based planning: roll out hypothetical actions in latent space
- Designed for robot manipulation tasks; tested on PushT, BaxterStack, etc.

In this workspace, LeWorldModel is the second world-model target (alongside DreamerV3).
It is an alternative for users who prefer the HuggingFace ecosystem over sheeprl.

---

## Architecture Overview

LeWorldModel uses a JEPA architecture:

```
Encoder:     CNN / ViT -> observation embedding z_t  (shape: latent_dim)
Predictor:   MLP / Transformer -> predicted z_{t+1} from (z_t, a_t)
Target:      Exponential Moving Average of encoder (momentum encoder)

Training loss:
  pred_loss = MSE(predictor(z_t, a_t), stop_gradient(target_encoder(o_{t+1})))
```

Key difference from DreamerV3:
- DreamerV3 reconstructs observations (recon_loss)
- LeWorldModel predicts next-embedding (pred_loss)
- LeWorldModel does NOT reconstruct pixels — it only operates in latent space
- This makes training faster and avoids blurry reconstruction artifacts

---

## Integration with `lerobot_world_model_bridge`

The bridge skill handles Parquet → HDF5 conversion for LeWorldModel:

```
Task(lerobot-worldmodel-bridge, {
    input_parquet: "datasets/so101_pick_v1_filtered",
    output_hdf5: "outputs/hdf5/so101_lewm.hdf5",
    target: "le_world_model",
    image_size: 96
})
```

This produces an HDF5 file with the `(96,96)` image preset.

**IMPORTANT — Schema Warning:** The HF LeWorldModel HDF5 schema is partially undocumented.
The `(96,96)` preset is based on inspection of `quentinll/lewm-pusht`. Before running training,
verify the schema matches:

```python
import h5py
from huggingface_hub import hf_hub_download

# Inspect reference dataset:
path = hf_hub_download("quentinll/lewm-pusht", "data.hdf5")
with h5py.File(path) as f:
    def show(name, obj):
        print(name, getattr(obj, 'shape', ''), getattr(obj, 'dtype', ''))
    f.visititems(show)
```

Compare with the skill's output:
```python
with h5py.File("outputs/hdf5/so101_lewm.hdf5") as f:
    f.visititems(show)
```

If schemas differ, update the `le_world_model` preset in the bridge skill:
`/home/koen/tools/claude_code/skills/lerobot_world_model_bridge/`

---

## HDF5 Schema (Current Best-Effort)

```
so101_lewm.hdf5
  /episode_000001/
      obs/
          image:   (T, 96, 96, 3)   uint8    -- wrist camera
          state:   (T, 12)          float32  -- joint_pos (6) + joint_vel (6)
      actions:     (T, 6)           float32  -- joint targets (radians)
      rewards:     (T,)             float32
      dones:       (T,)             bool
```

Note: Unlike DreamerV3, LeWorldModel may require an additional `next_obs` key in the HDF5
for computing the target embedding. The bridge skill handles this automatically.

---

## RTX 3080 (10 GB) Constraints

From vault note `World-Models-(Robot-Manipulation).md`:

| Config | VRAM est. | Fits 3080? |
|--------|-----------|-----------|
| `image_size=96`, `batch=16` | ~12 GB | No |
| `image_size=96`, `batch=8`, AMP | ~7 GB | Yes |
| `image_size=96`, `batch=8`, AMP + grad ckpt | ~5 GB | Yes (comfortable) |
| `image_size=64`, `batch=16`, AMP | ~5 GB | Yes |

**Recommended config for RTX 3080:**
```yaml
# wm_leworldmodel.yaml
image_size: 96        # standard for LeWM
batch_size: 8         # halved from default
amp: true
gradient_checkpointing: true   # required at 10 GB
device: cuda:0
```

Training time: ~1.5 h for 96x96 with 16-step prediction windows on RTX 3080.

---

## Key Config Knobs

```yaml
# packages/lerobot-isaac-configs/configs/wm_leworldmodel.yaml
leworldmodel:
  image_size: 96
  batch_size: 8
  prediction_horizon: 16    # number of future steps to predict
  latent_dim: 256
  gaussian_reg: 0.01        # regularization on latent
  learning_rate: 1e-4
  momentum: 0.996           # EMA momentum for target encoder
  amp: true
  gradient_checkpointing: true
  device: cuda:0
  total_steps: 300000
  eval_every: 10000
  metric_regex: "pred_loss=([0-9.eE+-]+)"
```

---

## Autoresearch Integration

`packages/lerobot-isaac-autoresearch/programs/leworldmodel.md`:
- Metric: `pred_loss`, direction: `minimize`
- Baseline: calls `train_wrapper.py --target_arch le_world_model`
- Mutation operators: `learning_rate`, `batch_size`, `latent_dim`, `gaussian_reg`, `momentum`
- Budget: 5400 s per experiment (LeWM is faster than DreamerV3 at these settings)
- Plateau limit: 2

---

## Known Issues

1. **Undocumented HDF5 schema**: Primary risk. Mitigate by inspecting `quentinll/lewm-pusht`
   before first training run. See schema discovery script above.

2. **LeWM vs LeRobot version alignment**: LeWorldModel is part of the LeRobot library.
   Pin lerobot version in `pixi.toml`; do not use latest-on-main.

3. **Gradient checkpointing overhead**: Enabling it reduces VRAM at the cost of ~30% slower
   forward pass. Acceptable for RTX 3080.
