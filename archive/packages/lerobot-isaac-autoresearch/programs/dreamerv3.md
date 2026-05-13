# DreamerV3 World Model Autoresearch Program

<!-- Autoresearch config for DreamerV3 world model training.
     Run with:
       cd ~/tools/claude_code
       /autoresearch ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/dreamerv3.md --type ml_model
-->

## Research Goal
Minimize reconstruction loss for DreamerV3 world model trained on SO-101
Isaac Lab rollouts. Target: recon_loss < 0.03 (sufficient for imagination-based
policy rollouts). If available, also track pc_success_imagined as a proxy for
imagination fidelity.

## Training Script
path: src/lerobot_isaac_autoresearch/train_wrapper.py
entry_args: "--target_arch dreamerv3 --dataset {dataset} --output_dir {out} --steps {steps}"
# Adapter reads HDF5 produced by lerobot_world_model_bridge (64x64 preset).
# If a Parquet path is supplied, the adapter auto-converts via the bridge skill.

## Metric
name: recon_loss
direction: minimize
source: stdout
# wm_dreamerv3.py emits: recon_loss=0.0317
# If imagination-based success is available, also emitted as: pc_success_imagined=0.XX
regex: 'recon_loss[=:\s]+([0-9.eE+-]+)'

## Budget
seconds_per_experiment: 7200    # 2 h per run (RTX 3080, DreamerV3 short rollouts)
max_experiments: 5
plateau_limit: 2                # world-model training is expensive; tighter plateau

## Constraints
allow_architecture_change: false    # DreamerV3 arch is paper-fixed (RSSM + decoder)
allow_optimizer_change: true
allow_data_pipeline_change: true    # sequence length, image augmentation OK
# VRAM ceiling: 10 GB. image_size=64, batch_size <= 16, seq_len <= 64.
# OOM recovery: train_wrapper.py halves batch_size once and retries automatically.

## Experiment Tracker
wandb: false    # set true after `wandb login`
mlflow: false

---

## Operators Priority

1. tune_hyperparams
2. change_scheduler
3. add_regularization

---

## Hyperparameter Search Space

```yaml
# RSSM dimensions
rssm_deter: [256, 512, 1024]
rssm_stoch: [32, 64]
rssm_classes: [32, 64]

# Training
batch_size: [8, 16]
batch_length: [32, 64]          # sequence length (steps per trajectory sample)
lr: [1e-4, 3e-4]
weight_decay: [0.0, 1e-4]
scheduler: [linear_warmup_cosine, cosine]

# Losses
kl_scale: [0.1, 0.3, 1.0]      # KL divergence weight
recon_scale: [1.0]              # reconstruction loss weight (fixed)
reward_scale: [0.5, 1.0]

# Image
image_size: [64]                # fixed — lerobot_world_model_bridge 64x64 preset
use_amp: true                   # required for 10 GB

# Evaluation
eval_every: 5000
steps: [50000, 100000]
```

---

## Notes

- Dataset format: HDF5 at `~/workspaces/lerobot-isaac-training/datasets/<task>_dreamerv3/`
  - Produced by `lerobot_world_model_bridge` skill with `preset=dreamerv3` (64x64 images)
  - Source: `${CLAUDE_CODE_ROOT}/skills/lerobot_world_model_bridge/`
- DreamerV3 implementation: `sheeprl` (default) or `nm-wu/dreamer-v3-pytorch`
  - Adapter selects via `configs/wm_dreamerv3.yaml` — set `impl: sheeprl` or `impl: nmwu`
- Checkpoint outputs: `~/workspaces/lerobot-isaac-training/outputs/dreamerv3/<job_name>/`
- `recon_loss` is the pixel reconstruction MSE (lower is better image prediction)
- `pc_success_imagined` (if emitted) estimates task success in DreamerV3's imagination
- Full adapter arg reference: `python -m lerobot_isaac_adapters.train --help`
