# LeRobot SmolVLA Fine-Tune — Autoresearch Program

<!-- Domain-aware program for SmolVLA fine-tuning. SmolVLA is pretrained on
     SO-101 data so the search space is narrow and the learning rate sweet
     spot is significantly lower than diffusion's.
-->

## Domain
domain: lerobot_isaac
domain_knowledge: programs/_domain_knowledge.md
stack: lerobot 0.5+ (smolvla extra installed), RTX 3080 10 GB, SO-101

## Research Goal
Fine-tune SmolVLA on `kvgork/so101-pickplace1` and maximize `pc_success`.
Target ≥0.85 for curriculum advance.

## Training Script
path: src/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py
entry_args: "--target_arch smolvla --dataset datasets/kvgork/so101-pickplace1 --output_dir {out} --steps {steps} --batch_size {batch_size} --lr {lr} --seed {seed}"
env: train-policy
python: .pixi/envs/train-policy/bin/python

## Metric
name: pc_success
direction: maximize
source: stdout
regex: 'pc_success[=:\s]+([0-9.eE+\-]+)'

## Budget
seconds_per_experiment: 3600       # 1 h per run (SmolVLA forward is heavier)
max_experiments: 8
plateau_limit: 3
oom_recovery: halve_batch_size_once

## Constraints
allow_architecture_change: false
allow_optimizer_change: true
allow_data_pipeline_change: true   # augmentation on
allow_remainder_args: true
vram_ceiling_gb: 10
batch_size_max: 8                  # SmolVLA image inputs are large
batch_size_default: 4

## Operators Priority

1. `tune_hyperparams` — `optimizer.lr` first (1e-5, 3e-5, 5e-5),
   then `warmup_steps`, then `weight_decay`.
2. `change_scheduler` — linear_warmup_cosine almost always helps for
   pretrained transformers.
3. `add_regularization` — weight_decay 1e-4 (AdamW), grad clip 1.0.
4. `modify_data_pipeline` — RandomCrop / ColorJitter image augmentations.

## Hyperparameter Search Space

```yaml
batch_size: [2, 4, 8]
lr: [1e-5, 3e-5, 5e-5, 1e-4]       # narrow — SmolVLA hates lr>1e-4
weight_decay: [1e-4, 1e-3]
warmup_steps: [200, 500, 1000]
steps: [5000, 20000, 40000]
seed: [42]
scheduler: [linear_warmup_cosine, cosine_annealing]
optimizer: [adamw]                  # Adam alone tends to overfit head
# Remainder:
policy.chunk_size: [50, 100]
```

## Mutation Hints

- **Baseline:** `lr=3e-5 batch_size=4 warmup_steps=500 steps=10000`.
- **Pretrained collapse:** if pc_success drops below 0.4 after step 1000,
  abort — likely catastrophic forgetting from lr too high.
- **Plateau ≥0.7:** try data augmentation (modify_data_pipeline) before
  raising lr.

## Stopping Rules

Same as diffusion (3-plateau, max_experiments=8).
