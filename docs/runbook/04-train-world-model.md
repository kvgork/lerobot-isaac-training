# Runbook 04: Train a World Model (DreamerV3 / LeWorldModel)

**Prerequisites:** Dataset collected (Runbook 02), Phase 2 impl (for full training)
**[Phase 2 impl required for full training — dry-run works now]**
**Expected outcome:** World model checkpoint in `outputs/`; `recon_loss` or `pred_loss` emitted

---

## Path Selection

| `--target_arch` | Model | Image size | RTX 3080 fit | Metric |
|-----------------|-------|-----------|--------------|--------|
| `dreamerv3` | DreamerV3 (RSSM) | 64×64 | Yes (8 GB) | `recon_loss` (minimize) |
| `le_world_model` | HF LeWorldModel | 96×96 | Marginal (12 GB) | `pred_loss` (minimize) |

---

## DreamerV3 Path

### Step 1: Convert Parquet to HDF5 (64×64)

```bash
# Via lerobot-worldmodel-bridge agent:
Task(lerobot-worldmodel-bridge, {
  input_parquet: "datasets/so101_pick_v1_filtered",
  output_hdf5: "outputs/hdf5/so101_pick_dreamerv3.hdf5",
  target: "dreamerv3",
  image_size: 64
})
```

Expected HDF5 layout:
```
episode_000001/
    obs/image: (T, 64, 64, 3) uint8
    obs/state: (T, 12) float32
    actions: (T, 6) float32
    rewards: (T,) float32
    dones: (T,) bool
```

### Step 2: Dry Run

```bash
lerobot-isaac-train \
  --target_arch dreamerv3 \
  --config packages/lerobot-isaac-configs/configs/wm_dreamerv3.yaml \
  --dry_run
```

### Step 3: Full Training

**[Phase 2 impl required — installs sheeprl]**

```bash
lerobot-isaac-train \
  --target_arch dreamerv3 \
  --config packages/lerobot-isaac-configs/configs/wm_dreamerv3.yaml \
  --dataset_path outputs/hdf5/so101_pick_dreamerv3.hdf5 \
  --output_dir outputs/dreamerv3_run1
```

Key config options in `wm_dreamerv3.yaml`:
```yaml
dreamerv3:
  image_size: 64        # keep 64 for RTX 3080
  batch_size: 16
  seq_len: 64
  amp: true             # reduces VRAM ~30%
  kl_coeff: 1.0
  num_steps: 200000
  eval_freq: 10000
```

### Step 4: Monitor

```bash
grep "recon_loss" outputs/dreamerv3_run1/train.log
# Expected format: recon_loss=0.0317
```

---

## LeWorldModel Path

### Step 1: Investigate HDF5 Schema First

**[Required before Step 2 — schema is undocumented]**

```bash
# Download a reference LeWM dataset:
python -c "
from huggingface_hub import snapshot_download
snapshot_download('quentinll/lewm-pusht', local_dir='/tmp/lewm-pusht')
"

# Inspect schema:
python -c "
import h5py
f = h5py.File('/tmp/lewm-pusht/pusht_episode_000001.hdf5')
def print_keys(name, obj): print(name, type(obj))
f.visititems(print_keys)
"
```

Compare with `lerobot_world_model_bridge` skill's `le_world_model` preset output.

### Step 2: Convert Parquet to HDF5 (96×96)

```bash
Task(lerobot-worldmodel-bridge, {
  input_parquet: "datasets/so101_pick_v1_filtered",
  output_hdf5: "outputs/hdf5/so101_pick_lewm.hdf5",
  target: "le_world_model",
  image_size: 96
})
```

### Step 3: Full Training

**[Phase 2 impl required]**

```bash
lerobot-isaac-train \
  --target_arch le_world_model \
  --config packages/lerobot-isaac-configs/configs/wm_leworldmodel.yaml \
  --dataset_path outputs/hdf5/so101_pick_lewm.hdf5 \
  --output_dir outputs/lewm_run1
```

Key config options:
```yaml
le_world_model:
  image_size: 96
  batch_size: 8         # reduce if OOM
  gradient_checkpointing: true
  amp: true
  eval_freq: 5000
```

---

## Autoresearch for World Models

```bash
# DreamerV3 autoresearch:
/autoresearch \
  packages/lerobot-isaac-autoresearch/programs/dreamerv3.md \
  --type ml_model

# LeWorldModel autoresearch:
/autoresearch \
  packages/lerobot-isaac-autoresearch/programs/leworldmodel.md \
  --type ml_model
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `NotImplementedError` | Phase 2 not implemented; use `--dry_run` |
| CUDA OOM (DreamerV3) | Use `image_size: 64`, `batch_size: 8`, `amp: true` |
| CUDA OOM (LeWM) | Use `batch_size: 4`, `gradient_checkpointing: true` |
| HDF5 schema mismatch | Follow schema investigation steps above |
| `sheeprl not found` | `pip install sheeprl[dreamer-v3]` |

---

## Reference Docs

- `docs/research/dreamerv3-reference.md` — DreamerV3 architecture + impl notes
- `docs/research/leworldmodel-reference.md` — HDF5 schema warning + skill cross-ref
- Skill: `/home/koen/tools/claude_code/skills/lerobot_world_model_bridge/SKILL.md`
