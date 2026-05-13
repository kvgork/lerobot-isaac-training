# LeRobot Policy Autoresearch Program

<!-- Autoresearch config for LeRobot policy training (SmolVLA / ACT / Diffusion).
     Run with:
       cd ~/tools/claude_code
       /autoresearch ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/lerobot-policy.md --type ml_model
-->

## Research Goal
Maximize pc_success for SO-101 manipulation tasks using LeRobot policy training
on an RTX 3080 (10 GB VRAM). Target: pc_success >= 0.85 (curriculum advance threshold).
Start with SmolVLA (pretrained on SO-101 data); fall back to ACT or Diffusion.

## Training Script
path: src/lerobot_isaac_autoresearch/train_wrapper.py
entry_args: "--target_arch smolvla --dataset {dataset} --output_dir {out} --steps {steps}"
# Templated values filled by autoresearch-ml-proposer-worker each iteration.
# Fallbacks:
#   entry_args: "--target_arch act --dataset {dataset} --output_dir {out} --steps {steps}"
#   entry_args: "--target_arch diffusion --dataset {dataset} --output_dir {out} --steps {steps}"

## Metric
name: pc_success
direction: maximize
source: stdout
# train_wrapper.py guarantees a final stdout line: pc_success=0.XXXX
# Original LeRobot format: {'pc_success': 14.0, ...} — wrapper normalises to [0,1].
regex: 'pc_success[=:\s]+([0-9.]+)'

## Budget
seconds_per_experiment: 7200    # 2 h per run (RTX 3080 — SmolVLA fine-tune)
max_experiments: 10
plateau_limit: 3                # stop after 3 consecutive non-improvements

## Constraints
allow_architecture_change: false    # SmolVLA/ACT/Diffusion arch is fixed; no new layers
allow_optimizer_change: true
allow_data_pipeline_change: true    # augmentation, normalization, source mixing OK
# VRAM ceiling: 10 GB. Proposer must not raise batch_size above 16.
# OOM recovery: train_wrapper.py halves batch_size once and retries automatically.

## Experiment Tracker
wandb: false    # set true after `wandb login` in pixi env
mlflow: false

---

## Operators Priority

1. tune_hyperparams
2. change_scheduler
3. modify_data_pipeline
4. add_regularization
5. change_optimizer

---

## Hyperparameter Search Space

### SmolVLA (primary — pretrained on SO-101)
```yaml
batch_size: [4, 8, 16]
lr: [1e-4, 3e-4, 5e-4]
weight_decay: [1e-4, 1e-3]
scheduler: [cosine, linear_warmup_cosine]
warmup_steps: [500, 1000]
use_amp: true
steps: [20000, 40000]
```

### ACT (secondary — faster inference)
```yaml
batch_size: [2, 4, 8]
lr: [1e-4, 3e-4, 1e-3]
chunk_size: [50, 100]
kl_weight: [5, 10, 20]
scheduler: [cosine, linear_warmup_cosine]
use_amp: true
steps: [30000, 60000]
```

### Diffusion Policy (fallback — more VRAM-friendly)
```yaml
batch_size: [4, 8, 16]
lr: [1e-4, 3e-4, 1e-3]
weight_decay: [1e-4, 1e-3]
num_diffusion_iters: [50, 100]
scheduler: [cosine, plateau]
steps: [20000, 40000]
```

---

## pc_success Thresholds

| Value | Meaning | Action |
|-------|---------|--------|
| < 0.40 | Baseline / random | Continue training |
| 0.40–0.60 | Learning | Autoresearch explores |
| 0.60–0.85 | Good | Trigger sim augmentation |
| >= 0.85 | Target | Advance curriculum stage |

---

## Notes

- Dataset path: `~/workspaces/lerobot-isaac-training/datasets/<task>/`
- Checkpoint outputs: `~/workspaces/lerobot-isaac-training/outputs/train/<job_name>/`
- `pc_success` is the fraction of eval episodes that completed the task (0.0–1.0)
- Source mixing (real + DR + mimicgen) is controlled via `--dataset` arg; see
  `lerobot-isaac-synthetic` package for dataset merge utilities
- Full adapter arg reference: `python -m lerobot_isaac_adapters.train --help`
