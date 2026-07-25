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
`--target_arch` differs. Pass extra `--policy.*` flags after `--`; when you load a
pretrained checkpoint via `--policy.path=`, the adapter omits its auto
`--policy.type` (passing both is a lerobot/draccus conflict).

#### RTX-3080 fine-tune recipe (GPU-verified 2026-07-11)

> [!tip] Auto-applied since the adapter update
> The adapter now **auto-injects this recipe** for `vla_jepa` fine-tuned from a
> `--policy.path`. You can just run:
> ```bash
> PYTORCH_ALLOC_CONF=expandable_segments:True \
> lerobot-isaac-train --target_arch vla_jepa --dataset datasets/local/so101-pickplace-new \
>   --successes_only --batch_size 2 --steps 20000 --output_dir outputs/vla_jepa_real_so101 \
>   -- --policy.path=lerobot/VLA-JEPA-Pretrain --save_freq=5000
> ```
> `policy_lerobot` appends `--policy.freeze_qwen=true` + `--policy.reinit_modules=[...]`
> (unless you set them yourself) and, when `--policy.path` is a **local** checkpoint
> dir, materialises a camera-count-patched copy under `<output_dir>/_wm_policy_patched`.
> Override any flag by passing it explicitly; disable the whole thing with
> `LEROBOT_ISAAC_WM_AUTORECIPE=0`. Camera adaptation only runs for a local dir — for
> the HF repo id, pre-materialise the 1-camera copy (step 3 below) so `--policy.path`
> points at a local dir. The rest of this section documents what the recipe does.

The `lerobot/VLA-JEPA-Pretrain` checkpoint is **7-dim action / 8-dim state /
2-camera** (`exterior_1_left`, `exterior_2_left`). Fine-tuning on SO-101
(6-action / 12-state / 1 overhead cam) on a 10 GB card needs **three** things —
a naked `--policy.path=lerobot/VLA-JEPA-Pretrain` fails (state-dict size
mismatch → camera KeyError → OOM):

1. **`--policy.freeze_qwen=true` (required for fit).** A full fp32 fine-tune of the
   2B Qwen3-VL backbone OOMs — weights alone are ~7.7 GB and the Adam states would
   need ~16 GB. Freezing the VLM backbone (train only the action expert + JEPA WM)
   fits at **~9 GB / batch 2**. It is also the correct small-data choice (50 demos
   must not retrain a 2B VLM). `torch_dtype` is already `bfloat16` by default.
2. **`--policy.reinit_modules=[...]`** to randomly re-init the embodiment-specific
   heads whose shapes differ from the pretrain robot (else `load_state_dict` raises
   on the size mismatch): `action_encoder`, `action_decoder`, `state_encoder`.
3. **A local 1-camera config.** The model hard-requires every camera it declares
   (`_prepare_model_inputs` indexes each key), so `--rename_map` alone can't fix a
   2→1 camera-count mismatch. Copy the pretrained dir and patch `config.json`'s
   `input_features` down to your single camera key (`observation.images.overhead`),
   then point `--policy.path` at the copy. See `scripts/` or the combined plan
   `plans/2026-07-11-combined-today-plan.md` for the patch snippet.

```bash
REINIT='["model.action_model.action_encoder","model.action_model.action_decoder","model.action_model.state_encoder"]'
PYTORCH_ALLOC_CONF=expandable_segments:True \
lerobot-isaac-train --target_arch vla_jepa --dataset datasets/local/so101-pickplace-new \
  --successes_only --batch_size 2 --steps 20000 --output_dir outputs/vla_jepa_real_so101 \
  -- --policy.path=outputs/vla_jepa_pretrain_so101 \
     --policy.reinit_modules="$REINIT" --policy.freeze_qwen=true --save_freq=5000
```

Verified throughput: **4.3 step/s (~9 GB VRAM, 97 % util — compute-bound, not
decode-bound); 20k steps ≈ 76 min.** Drop to `--batch_size 1` only if a heavier
background GPU load steals the ~1 GB headroom.

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
