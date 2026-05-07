# DreamerV3 Reference

**Paper:** https://arxiv.org/abs/2301.04104 (Hafner et al. 2023)
**sheeprl (preferred impl):** https://github.com/Eclectic-Sheep/sheeprl
**dreamer-v3-pytorch (fallback):** https://github.com/nm-wu/dreamer-v3-pytorch
**Official JAX impl (reference only):** https://github.com/danijar/dreamerv3

**Related workspace docs:** [world-model-bridge.md](../internals/world-model-bridge.md) | [ARCHITECTURE.md](../../ARCHITECTURE.md)

---

## What is DreamerV3

DreamerV3 (Hafner et al. 2023) is a model-based RL algorithm that trains entirely in the
latent space of a learned world model. Key properties:

- Works across discrete and continuous action spaces without tuning
- Handles sparse rewards via return normalization (no reward shaping needed)
- Learns from raw image observations (64×64 RGB standard; 96×96 also supported)
- Can train policies without any real environment interaction after world model training
- Achieves SOTA on Atari, DMControl, Minecraft, and robot manipulation

In this workspace, DreamerV3 is used as a **world model** target — we train the world
model on SO-101 teleoperation data to understand the robot's dynamics, not to train
a DreamerV3 policy (that would require RL rollouts).

---

## RSSM Architecture (Theory Summary)

The Recurrent State Space Model (RSSM) is DreamerV3's core:

```
Sequence model (GRU):
  h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
  (deterministic hidden state)

Representation model (CNN encoder):
  z_t = q(h_t, o_t)
  (stochastic latent from observation)

Prediction model (decoder):
  o_t_hat = p(h_t, z_t)
  (reconstruct observation from latent)
```

Training objective (ELBO):
```
L = E[ recon(o_t, o_t_hat) + KL(q || p) + reward_pred(r_t, h_t, z_t) ]
```

The `recon_loss` metric is the reconstruction component of this ELBO.
Lower `recon_loss` means the RSSM can better reconstruct observations from its latent,
indicating a higher-quality world model.

---

## Implementations in This Workspace

### sheeprl (preferred)

- Repo: https://github.com/Eclectic-Sheep/sheeprl
- Well-maintained, modern PyTorch + Lightning Fabric backend
- DreamerV3 variant: `dreamer_v3` algorithm
- Install: `pixi install -e train-dreamer` (includes sheeprl[dreamer-v3])
- Train command:
  ```bash
  python -m sheeprl dreamer_v3 \
    --config dreamer_v3_default \
    data.data_dir=<hdf5_path> \
    env.observation_size=[64,64] \
    algo.total_steps=500000
  ```

### dreamer-v3-pytorch (fallback)

- Repo: https://github.com/nm-wu/dreamer-v3-pytorch
- Standalone PyTorch implementation, lighter deps than sheeprl
- Use if sheeprl is unavailable or has incompatibility
- Install: `pip install dreamer-v3-pytorch`

The adapter `wm_dreamerv3.py` abstracts the import behind a config flag:
```yaml
# wm_dreamerv3.yaml
implementation: sheeprl   # or: dreamer-v3-pytorch
```

---

## HDF5 Input Format

DreamerV3 expects episodic HDF5 produced by `lerobot_world_model_bridge`:

```
dreamerv3_data.hdf5
  /episode_000001/
      obs/
          image:   (T, 64, 64, 3)   uint8    -- wrist camera
          state:   (T, 12)          float32  -- joint_pos (6) + joint_vel (6)
      actions:     (T, 6)           float32  -- joint targets (radians)
      rewards:     (T,)             float32  -- sparse {0.0, 1.0}
      dones:       (T,)             bool
  /episode_000002/
      ...
```

**Convert from Parquet:**
```
Task(lerobot-worldmodel-bridge, {
    input_parquet: "datasets/so101_pick_v1_filtered",
    output_hdf5: "outputs/hdf5/so101_dreamerv3.hdf5",
    target: "dreamerv3",
    image_size: 64
})
```

Do NOT write custom converters. The skill handles image resize, dtype casting,
and schema validation.

---

## RTX 3080 (10 GB) Configuration

From vault note `World-Models-(Robot-Manipulation).md`:

| Config | VRAM est. | Fits 3080? |
|--------|-----------|-----------|
| `image_size=64`, `batch=16`, `seq_len=64` | ~8 GB | Yes |
| `image_size=64`, `batch=32`, `seq_len=64` | ~14 GB | No |
| `image_size=96`, `batch=16`, `seq_len=64` | ~12 GB | No |
| `image_size=64`, `batch=16`, AMP enabled | ~5 GB | Yes (comfortable) |

**Recommended config for RTX 3080:**
```yaml
# wm_dreamerv3.yaml
image_size: 64
batch_size: 16
seq_len: 64
amp: true            # automatic mixed precision
gradient_checkpointing: false  # not needed with batch=16
device: cuda:0
```

---

## Key Config Knobs

```yaml
# packages/lerobot-isaac-configs/configs/wm_dreamerv3.yaml
dreamerv3:
  image_size: 64
  batch_size: 16
  seq_len: 64
  num_envs: 1             # for data collection (not training)
  amp: true
  device: cuda:0
  metric_regex: "recon_loss=([0-9.eE+-]+)"
  # RSSM architecture:
  rssm_dim: 512           # latent dimension
  rssm_depth: 4           # GRU depth
  encoder_depth: 32       # CNN channel depth multiplier
  # Training:
  learning_rate: 1e-4
  kl_scale: 1.0           # KL loss weight
  reward_scale: 1.0
  total_steps: 500000
  eval_every: 10000
```

---

## Autoresearch Integration

`packages/lerobot-isaac-autoresearch/programs/dreamerv3.md` defines the search problem:
- Metric: `recon_loss`, direction: `minimize`
- Baseline: calls `train_wrapper.py --target_arch dreamerv3`
- Mutation operators: `learning_rate`, `batch_size`, `seq_len`, `kl_scale`, `rssm_dim`
- Budget: 7200 s per experiment (DreamerV3 is slow at seq_len=64)
- Plateau limit: 2 (tighter than policy because training is expensive)

The `autoresearch-ml-executor-worker` reads the last stdout line matching `recon_loss=<float>`.
`wm_dreamerv3.py` MUST emit this format on every eval step via `MetricEmitter`.

---

## Known Issues

1. **sheeprl version compatibility**: sheeprl's DreamerV3 API has changed across versions.
   Pin the version in `pixi.toml` after a working version is found.

2. **H5py vs HDF5 version**: Some HDF5 versions have incompatible file formats. Use h5py
   >= 3.10 for best compatibility.

3. **Sequence model warmup**: DreamerV3's GRU needs a warmup period (~50k steps) before
   `recon_loss` stabilizes. Do not judge model quality by early steps.
