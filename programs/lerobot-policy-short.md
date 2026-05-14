# LeRobot Policy Autoresearch (short budget) — SO-101

<!-- Short-budget variant of programs/lerobot-policy.md tuned for a ~30 min
     end-to-end smoke that actually populates .agent-state/<sess>/autoresearch/.

     Run with:
       cd ~/tools/claude_code
       /autoresearch ~/workspaces/lerobot-isaac-training/programs/lerobot-policy-short.md --type ml_model
-->

## Research Goal
Maximize `pc_success` (proxy: minimize train loss when no eval rollouts run)
for the SO-101 diffusion policy on the kvgork/so101-pickplace1 dataset.
Tight per-iter budget so the loop completes in ~30 min total.

## Training Script
path: archive/packages/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py
entry_args: "--target_arch diffusion --dataset datasets/kvgork/so101-pickplace1 --output_dir {out} --steps {steps}"

## Metric
name: pc_success
direction: maximize
source: stdout
regex: 'pc_success[=:\s]+([0-9.eE+\-]+)'

## Budget
seconds_per_experiment: 480     # 8 min per iter (3 iters fit in ~30 min)
max_experiments: 3
plateau_limit: 2

## Constraints
allow_architecture_change: false
allow_optimizer_change: true
allow_data_pipeline_change: false
# RTX 3080 ceiling: batch_size <= 16, OOM recovery halves once.

## Experiment Tracker
wandb: false
mlflow: false

---

## Operators Priority

1. tune_hyperparams
2. change_optimizer

---

## Hyperparameter Search Space

```yaml
batch_size: [4, 8, 16]
lr: [1e-5, 5e-5, 1e-4, 3e-4]
steps: [400, 800, 1200]
seed: [42, 1337]
```
