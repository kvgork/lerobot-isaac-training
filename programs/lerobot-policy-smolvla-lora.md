# LeRobot SmolVLA + LoRA Fine-Tune — Autoresearch Program

<!-- Companion program to lerobot-policy-smolvla.md. Studies rank →
     pc_success effect with PEFT LoRA adapters on the SmolVLA backbone.
     See plans/2026-05-19-lora-autoresearch-plan.md §Findings for the
     rationale behind every range below.
-->

## Domain
domain: lerobot_isaac
domain_knowledge: programs/_domain_knowledge.md
stack: lerobot 0.5+ (smolvla extra) + peft>=0.10, RTX 3080 10 GB, SO-101

## Research Goal
Fine-tune SmolVLA with PEFT LoRA on `kvgork/so101-pickplace1`. Study the
rank → pc_success effect: identify the rank sweet spot for VLA fine-tunes
(expected r=64-128 per LoRA-SP, arXiv:2603.07404, which ran the exact
SmolVLA ablation: r=8 → 0%, r=128 → 40-93% per task). HuggingFace's own
SmolVLA PEFT example defaults to r=64. Target pc_success ≥
baseline-non-LoRA at <8% trainable params.

## Training Script
path: src/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py
entry_args: "--target_arch smolvla --dataset datasets/kvgork/so101-pickplace1 --output_dir {out} --steps {steps} --batch_size {batch_size} --use_lora --lora_rank {lora_rank} --lora_alpha {lora_alpha} --lora_dropout {lora_dropout} --lora_target_modules {lora_target_modules}"
env: train-policy
python: .pixi/envs/train-policy/bin/python

## Metric
name: pc_success
direction: maximize
source: stdout
regex: 'pc_success[=:\s]+([0-9.eE+\-]+)'

## Budget
seconds_per_experiment: 3600
max_experiments: 12      # 4 ranks × ~3 mutation alternatives
plateau_limit: 3
oom_recovery: halve_batch_size_once

## Constraints
allow_architecture_change: false
allow_optimizer_change: true
allow_data_pipeline_change: false
allow_remainder_args: true
allow_lora_mutation: true        # NEW — gates the tune_lora operator
allow_rank_extension: true       # NEW — proposer may probe r=256/r=512 if curve still rising at r=128
rank_extension_cap: 512          # hard upper bound (>=r=512 approaches full-FT param count)
rank_rising_threshold: 0.05      # pc_success delta vs prev rank step that counts as "still rising"
vram_ceiling_gb: 10
batch_size_max: 8                # r=128 attn_qkvo adds ~320 MB opt state; r=512 ~1.3 GB
batch_size_default: 6

## Operators Priority

1. `tune_lora` — vary rank first (16 → 32 → 64 → 128), then alpha
   (r vs 2r), then dropout, then target_modules preset.
2. `tune_hyperparams` — optimizer.lr 1e-5 → 3e-5 → 5e-5; warmup_steps
   200/500/1000.
3. `change_scheduler` — linear_warmup_cosine (default for transformers).
4. `add_regularization` — weight_decay 1e-4 (AdamW only).

## Hyperparameter Search Space

```yaml
lora_rank:          [16, 32, 64, 128]    # VLA ladder — see plan §2.6
lora_alpha_factor:  [1, 2]                # multiplied with rank
lora_dropout:       [0.0, 0.05, 0.1]
lora_target_modules: [attn_qv, attn_qkvo]
batch_size:         [4, 6, 8]
lr:                 [1e-5, 3e-5, 5e-5]
warmup_steps:       [200, 500]
steps:              [5000, 10000, 20000]
seed:               [42]
scheduler:          [linear_warmup_cosine]
optimizer:          [adamw]
```

## Mutation Hints

- **Baseline:** `lora_rank=64 lora_alpha=64 lora_dropout=0.05 lora_target_modules=attn_qv lr=3e-5 batch_size=6 warmup_steps=500 steps=10000`. (HF SmolVLA PEFT example default.)
- **Rank sweep order:** start at r=64 (baseline = HF default), then probe
  r=32 (cheaper), then r=128 (saturation point per LoRA-SP), then r=16
  (lower bound). Stop probing higher r if pc_success(r=128) ≤
  pc_success(r=64).
- **Alpha tied to rank:** always set `alpha = k * r` for k ∈ {1, 2}.
  Hu et al. 2021 §3.4 motivates this. At r=128, k=1 keeps effective
  scale at 1.0 (recommended starting point; k=2 doubles update
  magnitude).
- **`pc_success` regression vs r=64 baseline by ≥0.10:** revert and try
  alpha = 2*r at the same rank before continuing.
- **Plateau at any rank:** switch to `tune_hyperparams` (lr) before
  bumping rank.

## Stopping Rules

Standard: 3 consecutive plateaus or max_experiments=12. Override: stop
early if r=64 and r=128 both hit pc_success ≥ 0.7 — the goal is the
curve shape, not absolute SOTA.

**Auto-extension stop:** when `allow_rank_extension=true`, the proposer
may propose r=256 (and then r=512) past the default `{16,32,64,128}`
ladder. Extension trigger: pc_success(r=128) − pc_success(r=64) ≥
`rank_rising_threshold` (default 0.05). Extension halts once curve
plateaus OR `rank_extension_cap` is reached.

## Cross-References

- Plan: `plans/2026-05-19-lora-autoresearch-plan.md` (§Findings is the
  rank-range citation table).
- Companion (non-LoRA baseline): `lerobot-policy-smolvla.md`.
- Domain pack §13: `_domain_knowledge.md`.
