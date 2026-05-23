# DreamerV3 World Model on Isaac Lab — Autoresearch Program

<!-- Sister program to wm-dreamerv3.md but targets the live Isaac Lab
     pick-place env (env=isaac_so101) instead of the HDF5 replay env.
     Actor head learns task control because actions affect physics and a
     real reward signal is emitted by lerobot-isaac-env's RewardManager.
-->

## Domain
domain: lerobot_isaac
domain_knowledge: programs/_domain_knowledge.md
stack: sheeprl 0.5.8.dev + Isaac Lab v2.3.2 (via lerobot-isaac-env) + RTX 3080

## Research Goal
Train a DreamerV3 actor that drives the SO-101 toward pick-place success
in the live Isaac Lab env, then test it on the real robot. Replaces the
prior `wm-dreamerv3.md` sweep which used the HDF5 replay env and produced
a non-task-directed actor (replay env ignores actions + emits 0 reward).

## Training Script
path: src/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py
entry_args: "--target_arch dreamerv3 --output_dir {out} --steps {steps} --batch_size {batch_size} --lr {lr} --seed {seed}"
env: train-dreamer
python: .pixi/envs/train-dreamer/bin/python
remainder: "-- --env isaac_so101 env.num_envs={num_envs} env.image_size={image_size} env.max_episode_steps={max_episode_steps} env.headless=True fabric.accelerator=gpu fabric.devices=1"

## Metric
name: episode_return
direction: maximize
source: stdout
regex: 'episode_return[=:\s]+([0-9.eE+\-]+)'
# Secondary (TB-scraped post-trial):
#   reward_mean — sheeprl's `Rewards/rew_avg` tag
#   pc_success  — derived from successful-episode count if env emits
#                 `is_success` info flag

## Budget
seconds_per_experiment: 3600       # 1 h — Isaac Lab step is slower than HDF5 replay
max_experiments: 8
plateau_limit: 3

## Constraints
allow_architecture_change: false
allow_optimizer_change: false      # sheeprl pins its own
allow_data_pipeline_change: true   # may tune env knobs (num_envs / image_size)
allow_remainder_args: true
vram_ceiling_gb: 10
batch_size_max: 16
batch_size_default: 8

## Operators Priority

1. `tune_hyperparams` — `algo.world_model.optimizer.lr` (1e-5, 1e-4, 3e-4),
   then `algo.replay_ratio` (1, 2, 4), then `env.num_envs` (1, 4, 8 —
   parallel envs accelerate experience collection).
2. `modify_data_pipeline` — `env.image_size` (32, 64, 96 — bigger = more
   VRAM, slower step but better visuals for the actor).
3. `change_scheduler` — sheeprl doesn't expose easy scheduler mutation; skip.

## Hyperparameter Search Space

```yaml
algo.per_rank_batch_size:           [4, 8, 16]
algo.world_model.optimizer.lr:      [1e-5, 3e-5, 1e-4, 3e-4]
algo.replay_ratio:                  [1, 2, 4]
algo.world_model.discrete_size:     [16, 32, 64]
algo.world_model.stochastic_size:   [16, 32, 64]
algo.total_steps:                   [50000, 100000, 200000]
env.num_envs:                       [1, 4, 8]
env.image_size:                     [32, 64]
env.max_episode_steps:              [300, 600]
seed:                               [42, 1337]
```

## Mutation Hints

- **Baseline:** `algo.world_model.optimizer.lr=1e-4
  algo.per_rank_batch_size=8 algo.replay_ratio=2 algo.total_steps=100000
  env.num_envs=4 env.image_size=64 env.max_episode_steps=600`. Reuses
  the HP winner from the prior replay-env sweep (`wm-bash-20260522-211616`,
  trial 11: r=2, D=64, S=64) since dynamics learning likely transfers
  even though the actor signal didn't.
- **num_envs > 1 first.** Parallel envs are the cheapest single knob —
  same wall-clock, N× experience.
- **Replay-ratio bump:** more updates per env step — useful when Isaac
  Lab step is slow (≥0.05 s/tick). Try 2 then 4.
- **image_size=32 sanity run** if step rate < 15 Hz on RTX 3080.

## Stopping Rules

Standard (3-plateau, max_experiments=8). Hard-stop early if `reward_mean`
goes NaN — `lerobot_isaac_env.rewards.success_reward` divides by joint
distance; check for collapse before resuming.

## Cross-references

- Wrapper env: `src/lerobot-isaac-adapters/.../sheeprl_plugin/isaac_env.py`
- Hydra cfg:   `src/lerobot-isaac-adapters/.../sheeprl_plugin/configs/env/isaac_so101.yaml`
- Sibling rewards: `src/lerobot-isaac-env/.../rewards.py`
  (`success_reward`, `action_l2_penalty`, `joint_vel_penalty`)
- Companion sweep (replay env, ARCHIVED): `programs/wm-dreamerv3.md`
- Plan: `plans/2026-05-23-wm-isaac-env-plan.md` Phase C4.
