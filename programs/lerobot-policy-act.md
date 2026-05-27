# LeRobot ACT — Autoresearch Program

<!-- ACT (Action Chunking Transformer). Fastest inference of the three
     policy archs. Trains from scratch; needs more steps than SmolVLA.
-->

## Domain
domain: lerobot_isaac
domain_knowledge: programs/_domain_knowledge.md
stack: lerobot 0.5+, RTX 3080 10 GB, SO-101

## Research Goal
Maximize `pc_success` for ACT on `kvgork/so101-pickplace1`. Goal ≥0.75.

## Training Script
path: src/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py
entry_args: "--target_arch act --dataset datasets/kvgork/so101-pickplace1 --output_dir {out} --steps {steps} --batch_size {batch_size} --lr {lr} --seed {seed}"
env: train-policy
python: .pixi/envs/train-policy/bin/python

## Metric
name: pc_success
direction: maximize
source: stdout
regex: 'pc_success[=:\s]+([0-9.eE+\-]+)'

## Budget
seconds_per_experiment: 2700       # 45 min per run
max_experiments: 8
plateau_limit: 3

## Constraints
allow_architecture_change: false
allow_optimizer_change: true
allow_data_pipeline_change: true
allow_remainder_args: true
vram_ceiling_gb: 10
batch_size_max: 8
batch_size_default: 4

## Operators Priority

1. `tune_hyperparams` — `optimizer.lr` (1e-4 → 5e-4 range),
   `policy.chunk_size` (50, 100), `policy.kl_weight` (5, 10, 20).
2. `change_scheduler` — linear_warmup_cosine.
3. `add_regularization` — grad clip 1.0.

## Hyperparameter Search Space

```yaml
batch_size: [2, 4, 8]
lr: [1e-4, 3e-4, 5e-4, 1e-3]
chunk_size: [50, 100]
kl_weight: [5, 10, 20]
warmup_steps: [500, 1000]
steps: [30000, 60000, 100000]
seed: [42, 1337]
scheduler: [linear_warmup_cosine]
optimizer: [adamw]
```

## Mutation Hints

- **Baseline:** `lr=3e-4 batch_size=4 chunk_size=50 kl_weight=10 steps=30000`.
- **High kl_weight (≥20):** if pc_success regresses, the KL term is
  overpowering — reduce.
- **chunk_size=100:** doubles attention cost; only try after lr is dialed in.

## Stopping Rules

Standard (3-plateau, max_experiments=8).
