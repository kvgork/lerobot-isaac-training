# HF LeWorldModel Autoresearch Program

<!-- Autoresearch config for HF LeWorldModel training.
     Run with:
       cd ~/tools/claude_code
       /autoresearch ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/leworldmodel.md --type ml_model
-->

## Research Goal
Minimize next-embedding prediction loss for HF LeWorldModel trained on SO-101
teleop and Isaac Lab rollouts (96x96 images). Target: pred_loss < 0.05
(sufficient for planning and data augmentation via model rollouts).

## Training Script
path: src/lerobot_isaac_autoresearch/train_wrapper.py
entry_args: "--target_arch le_world_model --dataset {dataset} --output_dir {out} --steps {steps}"
# Adapter reads HDF5 produced by lerobot_world_model_bridge (96x96 preset).
# LeWM uses next-embedding (JEPA-style) prediction; no pixel reconstruction loss.

## Metric
name: pred_loss
direction: minimize
source: stdout
# wm_leworldmodel.py emits: pred_loss=0.0412
regex: 'pred_loss[=:\s]+([0-9.eE+-]+)'

## Budget
seconds_per_experiment: 5400    # 1.5 h per run (RTX 3080, 96x96, 16-step windows)
max_experiments: 5
plateau_limit: 2                # LeWM training is expensive; tighter plateau

## Constraints
allow_architecture_change: false    # LeWM arch is paper-fixed (JEPA encoder + predictor)
allow_optimizer_change: true
allow_data_pipeline_change: true    # window length, augmentation, normalisation OK
# VRAM ceiling: 10 GB. image_size=96, batch_size <= 8. Use gradient checkpointing.
# OOM recovery: train_wrapper.py halves batch_size once and retries automatically.

## Experiment Tracker
wandb: false    # set true after `wandb login`
mlflow: false

---

## Operators Priority

1. tune_hyperparams
2. change_scheduler
3. modify_data_pipeline

---

## Hyperparameter Search Space

```yaml
# Architecture (width only — no structural change)
latent_dim: [128, 256, 512]
num_heads: [4, 8]
depth: [4, 6]                   # transformer depth — keep <= 6 for 10 GB

# Training
batch_size: [4, 8]
window_length: [8, 16]          # steps per prediction window
lr: [1e-4, 3e-4, 5e-4]
weight_decay: [0.0, 1e-4]
scheduler: [cosine, linear_warmup_cosine]
warmup_steps: [500, 1000]
use_amp: true                   # required for 10 GB

# Regularisation
gaussian_reg: [0.0, 0.01, 0.1]  # Gaussian smoothness regulariser (LeWM paper)
dropout: [0.0, 0.1]

# Image
image_size: [96]                # fixed — lerobot_world_model_bridge 96x96 preset

# Evaluation
eval_every: 5000
steps: [50000, 100000]
```

---

## Notes

- Dataset format: HDF5 at `~/workspaces/lerobot-isaac-training/datasets/<task>_lewm/`
  - Produced by `lerobot_world_model_bridge` skill with `preset=le_world_model` (96x96 images)
  - Source: `${CLAUDE_CODE_ROOT}/skills/lerobot_world_model_bridge/`
  - WARNING (from vault): LeWM HDF5 schema is undocumented; use skill's schema_discovery
    helper to inspect `quentinll/lewm-pusht` if mismatch occurs
- Checkpoint outputs: `~/workspaces/lerobot-isaac-training/outputs/lewm/<job_name>/`
- `pred_loss` is the JEPA-style next-embedding cosine prediction loss (lower is better)
- Full adapter arg reference: `python -m lerobot_isaac_adapters.train --help`
