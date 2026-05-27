# DreamerV3 World Model — Autoresearch Program

<!-- sheeprl-backed DreamerV3 training against the SO-101 HDF5 produced by
     lerobot_world_model_bridge. Custom HDF5 replay env via the
     lerobot-isaac-adapters sheeprl_plugin.
-->

## Domain
domain: lerobot_isaac
domain_knowledge: programs/_domain_knowledge.md
stack: sheeprl 0.5.8.dev (git master), Isaac Sim 6.0 (not used directly), RTX 3080

## Research Goal
Minimize `recon_loss` for DreamerV3 on the bridged SO-101 HDF5
(`outputs/.../bridge/so101_dreamerv3_full.hdf5`, 64×64 images, window 16).

## Training Script
path: src/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py
entry_args: "--target_arch dreamerv3 --dataset outputs/pipeline-validation-so101/bridge/so101_dreamerv3_full.hdf5 --output_dir {out} --steps {steps} --batch_size {batch_size} --lr {lr} --seed {seed}"
env: train-dreamer
python: .pixi/envs/train-dreamer/bin/python
remainder: "-- env.capture_video=False fabric.accelerator=gpu fabric.devices=1"

## Metric
name: recon_loss
direction: minimize
source: stdout
regex: 'recon_loss[=:\s]+([0-9.eE+\-]+)'

## Budget
seconds_per_experiment: 3600       # 1 h — dreamer rollout + train is slow
max_experiments: 8
plateau_limit: 3

## Constraints
allow_architecture_change: false   # Hidden in algo.world_model.*
allow_optimizer_change: false      # sheeprl uses its own
allow_data_pipeline_change: true   # different image_size / window
allow_remainder_args: true
vram_ceiling_gb: 10
batch_size_max: 16
batch_size_default: 8

## Operators Priority

1. `tune_hyperparams` — `algo.world_model.optimizer.lr` (1e-5, 1e-4, 3e-4),
   then `algo.per_rank_batch_size`, then `algo.per_rank_sequence_length`.
2. `change_scheduler` — sheeprl doesn't expose easy scheduler mutation; skip.
3. `modify_data_pipeline` — re-bridge with image_size 48 (lighter) or
   window 32 (longer context); requires regenerating the HDF5 first.

## Hyperparameter Search Space (sheeprl Hydra paths)

```yaml
algo.per_rank_batch_size: [4, 8, 16]
algo.world_model.optimizer.lr: [1e-5, 3e-5, 1e-4, 3e-4]
algo.per_rank_sequence_length: [32, 64]
algo.replay_ratio: [1, 2, 4]
algo.world_model.discrete_size: [16, 32, 64]
algo.world_model.stochastic_size: [16, 32, 64]
algo.gamma: [0.99, 0.997]
algo.total_steps: [100000, 500000, 1000000]
seed: [42, 1337]
```

## Mutation Hints

- **Baseline:** `algo.world_model.optimizer.lr=1e-4 algo.per_rank_batch_size=8 algo.total_steps=200000`.
- **Sequence length doubling:** doubles VRAM. Only try if batch_size is ≤4.
- **replay_ratio bump:** more updates per env step — helps when env is fast
  (our HDF5 replay env is instant). Try 2 or 4.
- **discrete_size + stochastic_size doubling:** capacity bump; only after
  lr is tuned.

## Stopping Rules

Standard (3-plateau, max_experiments=8). Special-case: if `recon_loss`
grows by 2× in two consecutive trials, the proposer's last mutation is
diverging — revert to previous best before next mutation.
