# DreamerV3 Isaac Lab — HP Autoresearch Program

<!-- Sister to wm-dreamerv3-isaac.md but encodes the actual HP sweep over
     the four knobs identified in plans/2026-05-23-wm-isaac-autoresearch-plan.md:
     actor entropy, replay_ratio, actor min_std, WM optimizer lr.
-->

## Domain
domain: lerobot_isaac
domain_knowledge: programs/_domain_knowledge.md
stack: sheeprl 0.5.8.dev + Isaac Lab v2.3.2 + RTX 3080

## Research Goal
Find an HP combination that escapes the v7/v8 actor-collapse failure
mode (Grads/actor → 0 by step ~10k, Rewards/rew_avg flat at random
baseline ~-62). Sweep 4 axes; ratchet best by final `Rewards/rew_avg`
scraped from TensorBoard. Verify winners don't show premature
`Grads/actor` collapse.

## Training Script
path: scripts/_run_autoresearch_wm_isaac.sh
entry_args: "(driven by bash sweep — see Trial Pool below)"
env: sim
python: .pixi/envs/sim/bin/python
remainder: "(per-trial Hydra overrides built from the pool)"

## Metric
name: rew_avg
direction: maximize
source: tensorboard
tag: "Rewards/rew_avg"
extractor: scripts/_scrape_tb_to_history.py --metric Rewards/rew_avg

## Budget
seconds_per_experiment: 10800       # 3 h hard ceiling per trial
max_experiments: 10
plateau_limit: 4                    # 4 consecutive non-improvers → stop sweep

## Constraints
allow_architecture_change: false
allow_optimizer_change: false       # sheeprl pins, only sweep its lr scalar
allow_data_pipeline_change: false   # env is fixed at isaac_so101
allow_remainder_args: true
vram_ceiling_gb: 10
batch_size_max: 16
batch_size_default: 16

## Trial Pool (10 configs)

Format per row: `ENT_COEF|REPLAY_RATIO|MIN_STD|WM_LR|STEPS|LABEL`

```
3e-4|0.5|0.1|1e-4|60000|baseline
1e-2|0.5|0.1|1e-4|60000|high-entropy
3e-3|0.5|0.3|1e-4|60000|mid-ent-min_std
1e-2|1.0|0.3|1e-4|60000|v8-config
3e-4|0.25|0.1|1e-4|60000|low-replay
3e-4|2.0|0.1|1e-4|30000|high-replay
1e-2|0.5|0.5|1e-4|60000|high-ent-high-min_std
1e-2|0.5|0.3|3e-5|60000|low-wm-lr
1e-2|0.5|0.3|3e-4|60000|high-wm-lr
1e-2|0.5|0.1|1e-4|60000|ablate-min_std
```

## Frozen (NOT swept)

```yaml
algo.per_rank_batch_size:           16
env.num_envs:                       1
env.image_size:                     64
env.max_episode_steps:              300
fabric.precision:                   bf16-mixed
algo.world_model.discrete_size:     32
algo.world_model.stochastic_size:   32
reward.weight:                      10.0
reward.distance_scale:              0.4
reward.ee_body_name:                gripper_link
```

## Stopping Rules

Standard plateau (4 consecutive non-improvers vs current best, or
max_experiments reached). Hard-stop early if no trial achieves
`rew_avg > -58` (10 points off random baseline) by trial 5 → indicates
the sweep regime is wrong → escalate to v2 sweep with wider knobs.

## Forensic Criteria (logged, NOT ratchet)

For each trial also scrape:

- `Grads/actor` final value → ≥ 0.05 means actor not saturated
- `State/post_entropy` final → ≥ 5.0 means WM not over-confident
- `Loss/observation_loss` final → ≤ 0.5 means WM dynamics learning
- `Loss/policy_loss` final → magnitude check (negative = entropy
  dominates reward; not necessarily bad)

The "winner" is the trial with highest `rew_avg` AND
`Grads/actor ≥ 0.05` (rules out v7/v8-class fake winners).

## Cross-References

- Plan: `plans/2026-05-23-wm-isaac-autoresearch-plan.md`
- Single-trial runner: `scripts/_run_wm_isaac_overnight.sh`
- TB scrape: `scripts/_scrape_tb_to_history.py`
- Sister program (single-trial): `programs/wm-dreamerv3-isaac.md`
