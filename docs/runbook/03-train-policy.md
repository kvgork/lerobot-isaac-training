# Runbook 03: Train a LeRobot Policy

**Prerequisites:** Dataset collected and filtered (Runbook 02), LeRobot installed
**[Phase 2 impl required for full training — stubs work for dry-run]**
**Expected outcome:** Policy checkpoint in `outputs/`; `pc_success` metric emitted

---

## Policy Architecture Options

| `--target_arch` | Algorithm | Use case |
|-----------------|-----------|----------|
| `smolvla` | SmolVLA (vision-language-action) | Best general manipulation |
| `act` | Action Chunking Transformer | Fast inference, table-top tasks |
| `diffusion` | Diffusion Policy | Complex trajectory distributions |
| `vla_jepa` | VLA-JEPA (lerobot 0.6.0 world-model policy) | Sample-efficient BC; WM auxiliary at train, dropped at inference. **RTX-3080 fit.** |
| `fastwam` | FastWAM (lerobot 0.6.0 world-model policy) | Video-gen WM expert. **~5B — needs >>10 GB VRAM.** |
| `lingbot_va` | LingBot-VA (lerobot 0.6.0 world-model policy) | Autoregressive video+action WM (train **and** inference). **~5B + ~20 GB frozen — big HW only.** |

### World-model policies (lerobot 0.6.0)

`vla_jepa` / `fastwam` / `lingbot_va` are lerobot 0.6.0 policies that use a world
model *during training*. They dispatch through the same `lerobot-train` path as the
plain policies and emit `pc_success`, so every step below works unchanged — only
`--target_arch` differs. Fine-tune from a pretrained checkpoint by passing
`--policy.path=` after `--`; the adapter then omits its auto `--policy.type`
(passing both is a lerobot/draccus conflict):

```bash
# From scratch (RTX 3080 — keep batch small):
lerobot-isaac-train --target_arch vla_jepa --dataset datasets/so101-pickplace-new \
  --batch_size 4 --steps 20000 --output_dir outputs/vla_jepa_run1

# Fine-tune from the pretrained VLA-JEPA checkpoint:
lerobot-isaac-train --target_arch vla_jepa --dataset datasets/so101-pickplace-new \
  --output_dir outputs/vla_jepa_ft -- --policy.path=lerobot/VLA-JEPA-Pretrain
```

`fastwam` / `lingbot_va` are registered but need >>10 GB VRAM; install their extras
first (`LEROBOT_EXTRAS=training,smolvla,feetech,vla_jepa,fastwam bash scripts/install_train_deps.sh`).

---

## Step 1: Choose Config

All configs live in `packages/lerobot-isaac-configs/configs/`:
```bash
ls packages/lerobot-isaac-configs/configs/
# policy_smolvla.yaml  policy_act.yaml  policy_diffusion.yaml
# wm_dreamerv3.yaml  wm_leworldmodel.yaml  isaac_so101_pickplace.yaml
```

Edit the relevant YAML to set `dataset_path` and `output_dir`:
```yaml
# packages/lerobot-isaac-configs/configs/policy_smolvla.yaml
dataset_path: datasets/so101_pick_v1_filtered
output_dir: outputs/smolvla_run1
batch_size: 32
num_steps: 100000
eval_freq: 5000
```

---

## Step 2: Dry Run (works now with scaffolding)

```bash
lerobot-isaac-train \
  --target_arch smolvla \
  --config packages/lerobot-isaac-configs/configs/policy_smolvla.yaml \
  --dry_run
```

Expected: prints dispatched command without executing. Exit 0.

---

## Step 3: Full Training Run

**[Phase 2 impl required]**

```bash
lerobot-isaac-train \
  --target_arch smolvla \
  --config packages/lerobot-isaac-configs/configs/policy_smolvla.yaml \
  --dataset_path datasets/so101_pick_v1_filtered \
  --output_dir outputs/smolvla_run1
```

This calls `targets/policy_lerobot.py` which internally invokes `lerobot-train`
(lerobot 0.5+) with `--dataset.repo_id` / `--dataset.root` / `--batch_size` /
`--steps` / `--optimizer.lr` / `--policy.push_to_hub=false`.

### Step 3b: Train on successful demonstrations only (`--successes_only`)

If your dataset was recorded with `robot-data-recorder` **including failure
episodes** (operator pressed `f`), BC training should skip the failures —
imitating a failed trajectory teaches the wrong actions. Reward/done are not
parquet features, so success is read from the recorder's
`meta/episode_labels.json` sidecar and forwarded to `lerobot-train` as
`--dataset.episodes`:

```bash
lerobot-isaac-train \
  --target_arch smolvla \
  --dataset datasets/so101_pick_v1 \
  --successes_only \
  --output_dir outputs/smolvla_run1
```

- Policy archs only; requires a **single local** `--dataset` (the sidecar lives
  on disk next to the parquet).
- No-op with a printed warning if the dataset is unlabelled, has zero
  successes, or is an HF repo / multi-local set.
- Composes with the SAL/TED quality filter (`lerobot_dataset_quality`): success
  filtering drops *failed* demos, quality filtering drops *low-smoothness* demos.
- Failures are still valuable for **world-model** training — they stay in the
  HDF5 output and broaden state-space coverage. Only the BC/parquet path filters them.

---

## Step 4: Monitor Training

If W&B is configured:
```bash
wandb login
# Training logs to wandb automatically when WANDB_API_KEY is set
```

Or check stdout:
```bash
lerobot-isaac-train ... 2>&1 | tee outputs/smolvla_run1/train.log
grep "pc_success" outputs/smolvla_run1/train.log
```

---

## Step 5: Evaluate Policy

```bash
# The evaluation agent reads pc_success from W&B or stdout:
Task(lerobot-evaluation-agent, {
  checkpoint_path: "outputs/smolvla_run1/checkpoints/last",
  dataset_path: "datasets/so101_pick_v1_filtered",
  eval_episodes: 20,
  metric: "pc_success"
})
```

The agent returns: `ADVANCE` (pc_success > threshold), `CONTINUE` (more training needed), or `COLLECT_MORE` (dataset too small).

---

## Step 6: Advance Curriculum (if ADVANCE)

```bash
Task(lerobot-curriculum-agent, {
  workspace_root: "~/workspaces/lerobot-isaac-training",
  current_stage: 1,
  eval_metric: "pc_success",
  eval_value: 0.85,
  advance_threshold: 0.80
})
```

---

## Step 7: Run Autoresearch (optional)

For automated hyperparameter search:
```bash
/autoresearch \
  packages/lerobot-isaac-autoresearch/programs/lerobot-policy.md \
  --type ml_model
```

---

## Metric Contract

Every eval step MUST emit exactly: `pc_success=<float>` on stdout.
The `metric_extractor.py` module handles this — do not emit this format manually.

```python
from lerobot_isaac_adapters.metric_extractor import MetricEmitter
emitter = MetricEmitter()
emitter.emit("pc_success", 0.73)  # prints: pc_success=0.73
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `NotImplementedError` from target | Phase 2 not yet implemented — use `--dry_run` |
| CUDA OOM | Reduce `batch_size` in config; check `num_envs` |
| `lerobot not found` | `pip install lerobot` |
| Policy never converges | Check dataset quality; increase `num_steps` |
