# DreamerV3 Reference

> **Phase 0 placeholder.** Replace with full notes when Phase 2 world-model implementation begins.
> This document covers what `lerobot-isaac-adapters/targets/wm_dreamerv3.py` needs to wire.

---

## What is DreamerV3

DreamerV3 (Hafner et al. 2023) is a model-based RL algorithm that learns a world model (RSSM) from observations and trains a policy purely in the latent space of the world model. Key properties:
- Works across discrete and continuous action spaces
- Handles sparse rewards via return normalization
- Learns from image observations (64×64 RGB typical; 96×96 also supported)

**Paper:** https://arxiv.org/abs/2301.04104

---

## Architecture: RSSM

The Recurrent State Space Model (RSSM) has two components:
1. **Sequence model** (GRU) — predicts next latent from action + prior
2. **Representation model** (CNN encoder) — infers latent from observation

Combined: `z_t = Representation(o_t, h_t)` where `h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})`

The world model trains via ELBO: reconstruction loss + KL divergence + reward prediction.

**Key metric emitted:** `recon_loss=<float>` — lower is better. The `autoresearch` loop minimizes this.

---

## Implementations

Two maintained implementations (both supported in this workspace via `wm_dreamerv3.py`):

### sheeprl (preferred)
- Repo: https://github.com/Eclectic-Sheep/sheeprl
- Well-maintained, modern, supports DreamerV3 with Fabric (Lightning) backend
- Install: `pip install sheeprl[dreamer-v3]`
- Train command:
  ```bash
  python -m sheeprl dreamer_v3 \
    --config dreamer_v3_default \
    data.data_dir=<hdf5_path> \
    env.observation_size=[64,64]
  ```

### dreamer-v3-pytorch (fallback)
- Repo: https://github.com/nm-wu/dreamer-v3-pytorch
- Standalone PyTorch implementation, lighter dependencies
- Install: `pip install dreamer-v3-pytorch`
- Train command varies by version — check their README

The adapter `wm_dreamerv3.py` abstracts the import so swapping implementations is a one-file change.

---

## HDF5 Input Format

DreamerV3 expects episodic HDF5 with keys per episode:
```
episode_000001/
    obs/
        image: (T, H, W, 3)  uint8
        state: (T, D)         float32
    actions: (T, A)           float32
    rewards: (T,)             float32
    dones:   (T,)             bool
```

For SO-101: `H=W=64`, `D=12` (joint_pos 6 + joint_vel 6), `A=6` (joint positions).

**Conversion:** Use `lerobot_world_model_bridge` skill — do NOT write custom converters.
```
Task(lerobot-worldmodel-bridge, {target: "dreamerv3", image_size: 64, ...})
```

---

## RTX 3080 Fit

From vault note `World-Models-(Robot-Manipulation).md`:
- 64×64 images, `batch_size=16`, `seq_len=64` → ~8 GB VRAM (fits RTX 3080)
- 96×96 images → requires ~12 GB (does NOT fit — use 64×64 for DreamerV3)
- Enable `torch.cuda.amp` (automatic mixed precision) to reduce footprint by ~30%

---

## Key Config Knobs (for `configs/wm_dreamerv3.yaml`)

```yaml
dreamerv3:
  image_size: 64          # keep at 64 for RTX 3080
  batch_size: 16
  seq_len: 64
  num_envs: 1             # during data collection phase
  amp: true               # automatic mixed precision
  device: cuda:0
  metric_regex: "recon_loss=([0-9.]+)"
```

---

## Autoresearch Integration

`programs/dreamerv3.md` program.md defines:
- Metric: `recon_loss`, direction: minimize
- Baseline command: calls `train_wrapper.py --target_arch dreamerv3`
- Mutation operators: learning_rate, batch_size, seq_len, kl_coeff, cnn_depth

The `autoresearch-ml-executor-worker` reads last stdout line matching `recon_loss=<float>`.
Ensure `wm_dreamerv3.py` emits this format on every eval step.
