# LeRobot Diffusion Policy — Autoresearch Program

<!-- Domain-aware autoresearch program for diffusion policy training on SO-101.
     Run:
       cd ~/tools/claude_code
       /autoresearch ~/workspaces/lerobot-isaac-training/programs/lerobot-policy-diffusion.md --type ml_model
     Or via wrapper:
       bash scripts/run_autoresearch.sh --program diffusion
-->

## Domain
domain: lerobot_isaac
domain_knowledge: programs/_domain_knowledge.md
stack: lerobot 0.5+, RTX 3080 10 GB, SO-101 6-DOF arm

## Research Goal
Maximize `pc_success` (open-loop action-MSE proxy) for the diffusion policy on
the `kvgork/so101-pickplace1` dataset. Plateau-stop and report best config.

## Training Script
path: archive/packages/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py
entry_args: "--target_arch diffusion --dataset datasets/kvgork/so101-pickplace1 --output_dir {out} --steps {steps} --batch_size {batch_size} --lr {lr} --seed {seed}"
env: train-policy
python: .pixi/envs/train-policy/bin/python

## Metric
name: pc_success
direction: maximize
source: stdout
regex: 'pc_success[=:\s]+([0-9.eE+\-]+)'
note: |
  pc_success is the open-loop action-MSE proxy from
  `scripts/_open_loop_eval.py`: pc_success = 1 / (1 + mse). The
  `lerobot-train` subprocess does NOT emit this directly — train_wrapper
  appends an eval step on the saved checkpoint at exit.

## Budget
seconds_per_experiment: 1800     # 30 min per run on RTX 3080
max_experiments: 10
plateau_limit: 3
oom_recovery: halve_batch_size_once

## Constraints
allow_architecture_change: false   # Diffusion-only; arch_swap handled by separate program
allow_optimizer_change: true
allow_data_pipeline_change: true
allow_remainder_args: true
vram_ceiling_gb: 10
batch_size_max: 16
batch_size_default: 8

## Experiment Tracker
wandb: false
mlflow: false
state_dir: .agent-state/{session_id}/autoresearch/lerobot-policy-diffusion

---

## Operators Priority (refine order)

1. `tune_hyperparams` — `optimizer.lr` first (5e-5, 1e-4, 3e-4 grid),
   then `batch_size` (4, 8, 16), then `weight_decay` (1e-4, 1e-3).
2. `change_scheduler` — add cosine_annealing if absent.
3. `modify_data_pipeline` — increase synthetic DR variance once a real-only
   baseline plateaus.
4. `add_regularization` — gradient clip 1.0, dropout 0.1 in diffusion head.
5. `change_optimizer` — Adam → AdamW (most realistic improvement direction
   for diffusion).

## Hyperparameter Search Space

```yaml
batch_size: [4, 8, 16]               # 8 default; 16 only if VRAM headroom
lr: [3e-5, 5e-5, 1e-4, 3e-4]         # diffusion sweet spot ~1e-4
weight_decay: [0, 1e-4, 1e-3]
steps: [10000, 30000, 80000]
seed: [42, 1337, 7]
scheduler: [none, cosine_annealing]
optimizer: [adam, adamw]
# Forwarded via remainder args:
policy.n_action_steps: [4, 8, 16]    # diffusion chunk length
policy.num_train_timesteps: [50, 100, 200]  # diffusion denoising steps
```

## Mutation Hints

- **First experiment:** baseline at `lr=1e-4 batch_size=8 steps=20000` to
  anchor the metric.
- **If pc_success ≈ baseline after 3 trials:** swap optimizer to AdamW with
  weight_decay 1e-4 (decoupled regularization).
- **If pc_success degrades 2 trials in a row:** halve `lr` (likely diverging).
- **If two OOMs in a row:** halve `batch_size` (mandatory; see domain
  knowledge §1).

## Stopping Rules

- Stop on plateau (3 consecutive non-improvements within ±2% of best).
- Stop on `max_experiments` reached.
- Stop on 2 unrecoverable crashes in a row (not OOM — those self-recover).

## Report Layout

After completion, produce a Markdown report at
`.agent-state/{session_id}/autoresearch/lerobot-policy-diffusion/report.md`:

- Best `pc_success`, with full config snapshot.
- Per-trial trace: trial, metric, config delta, status, duration.
- Top 3 operator wins (which mutations actually moved the needle).
- Recommended next program (e.g. `lerobot-policy-smolvla.md` if diffusion
  saturated).
