# LeWorldModel (in-process) — Autoresearch Program

<!-- In-process minimal LeWM-style trainer (lerobot_isaac_adapters._lewm_minimal).
     Tiny model (790K params) — runs fast, great for HP-search smoke.
-->

## Domain
domain: lerobot_isaac
domain_knowledge: programs/_domain_knowledge.md
stack: in-process torch trainer (no upstream CLI needed), RTX 3080

## Research Goal
Minimize `pred_loss` for the LeWM-style next-embedding predictor on the
bridged SO-101 96×96 HDF5.

## Training Script
path: src/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py
entry_args: "--target_arch le_world_model --dataset outputs/pipeline-validation-so101/bridge/so101_lewm_full.hdf5 --output_dir {out} --steps {steps} --batch_size {batch_size} --lr {lr} --seed {seed}"
env: train-lewm
python: .pixi/envs/train-lewm/bin/python

## Metric
name: pred_loss
direction: minimize
source: stdout
regex: 'pred_loss[=:\s]+([0-9.eE+\-]+)'

## Budget
seconds_per_experiment: 600        # 10 min — model is tiny, fast to converge
max_experiments: 8
plateau_limit: 3

## Constraints
allow_architecture_change: true    # Embedding size, encoder channels — editable in _lewm_minimal.py
allow_optimizer_change: false      # Hardcoded Adam in the trainer
allow_data_pipeline_change: true   # Window size, stride at bridge step
allow_remainder_args: false
vram_ceiling_gb: 10
batch_size_max: 16
batch_size_default: 8

## Operators Priority

1. `tune_hyperparams` — `lr` (1e-5 → 1e-3), `steps` (300, 1000, 3000),
   `batch_size`.
2. `change_architecture` — bump `embed_dim` 128 → 256 in `_lewm_minimal.py`
   (requires source edit; only do this in `refine` mode after lr is tuned).
3. `modify_data_pipeline` — re-bridge with different window_size (8, 16, 32).

## Hyperparameter Search Space

```yaml
batch_size: [4, 8, 16]
lr: [1e-5, 5e-5, 1e-4, 3e-4, 1e-3]
steps: [300, 1000, 3000, 10000]
seed: [42, 1337]
# Architectural (require code edit, refine mode only):
embed_dim: [128, 256]
encoder_channels: ["32/64/128/256", "48/96/192/384"]
```

## Mutation Hints

- **Baseline:** `lr=1e-4 batch_size=8 steps=1000`. Should reach
  `pred_loss<0.01` within seconds on the SO-101 96×96 HDF5.
- **lr too high (≥3e-4):** pred_loss explodes after step ~50; halve.
- **Window doubling:** improves long-horizon prediction but the bridge
  has to re-emit. Only worth doing after lr / batch is dialed in.

## Stopping Rules

Standard (3-plateau, max_experiments=8). Tight budget — 8×10 min ≈ 1.5 h.
