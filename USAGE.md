# Usage Runbook — LeRobot + Isaac Lab Training Workspace

This document indexes all runbooks in `docs/runbook/` with exact commands per task.
Steps that require Phase X implementation beyond current scaffolding are marked **[Phase X impl required]**.

---

## Task Index

| Task | Runbook | Phase Req |
|------|---------|-----------|
| Bootstrap workspace | `docs/runbook/01-bootstrap.md` | Phase 0 scaffolding ✅ |
| Collect real teleop data | `docs/runbook/02-collect-data.md` | Phase 0 ✅ (agent wiring) |
| Train a LeRobot policy | `docs/runbook/03-train-policy.md` | Phase 2 impl |
| Train DreamerV3 / LeWorldModel | `docs/runbook/04-train-world-model.md` | Phase 2 impl |
| Generate DR synthetic data | `docs/runbook/05-augment-with-dr.md` | Phase 4 impl |
| Generate MimicGen synthetic data | `docs/runbook/06-augment-with-mimicgen.md` | Deferred |

---

## 1. Collect Real Teleop Data

**Agent:** `lerobot-data-collection-agent`
**Skill:** `lerobot_dataset_quality` (SAL + TED filtering)

```bash
# Invoke the data-collection agent:
# Task(lerobot-data-collection-agent, {
#   dataset_path: "~/workspaces/lerobot-isaac-training/datasets/so101_pick",
#   robot: "so101",
#   quality_filter: true
# })

# Or check dataset quality directly:
# (skill auto-invoked when pattern "filter dataset" or "dataset quality" is used)
```

See `docs/runbook/02-collect-data.md` for step-by-step.

---

## 2. Filter Dataset Quality

**Skill:** `lerobot_dataset_quality`

The skill runs SAL (Scene Anomaly Localization) and TED (Trajectory Edit Distance) filtering:
```bash
# Auto-invoked by lerobot-data-collection-agent, or manually:
# Task via skill: lerobot_dataset_quality.filter(
#   dataset_path="datasets/so101_pick",
#   sal_threshold=0.3,
#   ted_threshold=0.5
# )
```

---

## 3. Train a LeRobot Policy

**[Phase 2 impl required for full training]**

```bash
# Dry-run (scaffolding, works now):
lerobot-isaac-train \
  --target_arch smolvla \
  --config packages/lerobot-isaac-configs/configs/policy_smolvla.yaml \
  --dataset_path datasets/so101_pick \
  --dry_run

# Full training (requires Phase 2 impl + LeRobot installed):
lerobot-isaac-train \
  --target_arch smolvla \
  --config packages/lerobot-isaac-configs/configs/policy_smolvla.yaml \
  --dataset_path datasets/so101_pick \
  --output_dir outputs/smolvla_run1

# Other policy targets:
lerobot-isaac-train --target_arch act ...
lerobot-isaac-train --target_arch diffusion ...
```

See `docs/runbook/03-train-policy.md` for full cycle.

---

## 4. Train DreamerV3 World Model

**[Phase 2 impl required]**

```bash
# Step 1: Convert Parquet to HDF5 (64x64 for RTX 3080):
# Task(lerobot-worldmodel-bridge, {
#   input_parquet: "datasets/so101_pick",
#   output_hdf5: "outputs/hdf5/so101_pick_64.hdf5",
#   target: "dreamerv3",
#   image_size: 64
# })

# Step 2: Train (requires sheeprl or dreamer-v3-pytorch installed):
lerobot-isaac-train \
  --target_arch dreamerv3 \
  --config packages/lerobot-isaac-configs/configs/wm_dreamerv3.yaml \
  --dataset_path outputs/hdf5/so101_pick_64.hdf5 \
  --output_dir outputs/dreamerv3_run1

# Metric emitted: recon_loss=<float>
```

---

## 5. Train LeWorldModel

**[Phase 2 impl required]**

```bash
# Step 1: Convert Parquet to HDF5 (96x96):
# Task(lerobot-worldmodel-bridge, {
#   input_parquet: "datasets/so101_pick",
#   output_hdf5: "outputs/hdf5/so101_pick_96.hdf5",
#   target: "le_world_model",
#   image_size: 96
# })

# Step 2: Train:
lerobot-isaac-train \
  --target_arch le_world_model \
  --config packages/lerobot-isaac-configs/configs/wm_leworldmodel.yaml \
  --dataset_path outputs/hdf5/so101_pick_96.hdf5 \
  --output_dir outputs/lewm_run1

# Metric emitted: pred_loss=<float>
# Note: LeWM HDF5 schema is undocumented — use lerobot_world_model_bridge skill's
#       96x96 preset. See skills/lerobot_world_model_bridge/SKILL.md for schema notes.
```

See `docs/runbook/04-train-world-model.md` for both paths.

---

## 6. Run Autoresearch Loop

**[Phase 3 impl required for full search; program.md files exist now]**

```bash
# From the claude_code repo context (agents installed via ~/.claude/agents/):
/autoresearch \
  ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/lerobot-policy.md \
  --type ml_model

# For DreamerV3:
/autoresearch \
  ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/dreamerv3.md \
  --type ml_model

# For LeWorldModel:
/autoresearch \
  ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/leworldmodel.md \
  --type ml_model

# Or invoke the orchestrator directly:
# Task(autoresearch-loop-orchestrator, {
#   program_path: "packages/lerobot-isaac-autoresearch/programs/lerobot-policy.md",
#   workspace_root: "$PWD"
# })
```

---

## 7. Generate DR Synthetic Data

**[Phase 4 impl required]**

```bash
# DR replay (replays real episodes with domain randomization):
python -m lerobot_isaac_synthetic.isaac_dr.replay_runner \
  --source_dataset_path datasets/so101_pick \
  --output_path datasets/so101_pick_dr \
  --num_augmentations 5 \
  --randomize object_pose lighting friction

# Dry run (works now with scaffolding):
python -m lerobot_isaac_synthetic.isaac_dr.replay_runner \
  --source_dataset_path datasets/so101_pick \
  --output_path /tmp/test_dr \
  --dry_run

# Merge real + DR datasets:
# merge_utilities.merge_datasets(
#   real_path="datasets/so101_pick",
#   dr_path="datasets/so101_pick_dr",
#   output_path="datasets/so101_merged",
#   real_weight=1.0,
#   dr_weight=0.5
# )
```

See `docs/runbook/05-augment-with-dr.md`.

---

## 8. Generate MimicGen Synthetic Data

**[Deferred — Phase 4b stub only]**

```bash
# Enable flag (required):
export LEROBOT_MIMICGEN_ENABLED=1

# Invoke via agent (preferred):
# Task(lerobot-sim-augmentation-agent, {
#   source_dataset: "datasets/so101_pick",
#   output_path: "datasets/so101_mimicgen",
#   num_demonstrations: 100
# })

# Or via bridge_invocation stub (currently raises NotImplementedError):
# python -m lerobot_isaac_synthetic.mimicgen.bridge_invocation ...
```

See `docs/runbook/06-augment-with-mimicgen.md`.

---

## 9. Advance Curriculum Stage

**[Phase 2+ impl required]**

```bash
# Invoke curriculum agent after evaluation:
# Task(lerobot-curriculum-agent, {
#   workspace_root: "$PWD",
#   current_stage: 1,
#   eval_metric: "pc_success",
#   eval_value: 0.85,
#   advance_threshold: 0.80
# })

# Curriculum stages (6-stage manipulation ladder):
# Stage 1: Fixed-position pick
# Stage 2: Pick with variable object position
# Stage 3: Pick-and-place
# Stage 4: Pick-and-place with obstacles
# Stage 5: Insertion
# Stage 6: (Future) multi-step manipulation
```

---

## 10. Spinout Package to Standalone Repo

```bash
# See ARCHITECTURE.md spinout procedure section.
# Quick example for lerobot-isaac-env:
git subtree split -P packages/lerobot-isaac-env -b spinout-env
cd /tmp
git clone ~/workspaces/lerobot-isaac-training spinout-env-repo
cd spinout-env-repo
git checkout spinout-env
# Adjust pyproject.toml paths and push to new remote.
```

---

## Common Pitfalls

- **RTX 3080 OOM:** Keep `--num_envs 4-8` for Isaac Lab. Use `--image_size 64` for DreamerV3.
- **USD path:** SO-101 USD must be downloaded before Phase 1 runs. See `packages/lerobot-isaac-env/assets/usd/README.md`.
- **Isaac Lab headless:** Always pass `headless=True` in configs unless you have a display.
- **Parquet→HDF5:** Use `lerobot_world_model_bridge` skill — do NOT write custom HDF5 converters.
- **Autoresearch metric format:** Each eval step must emit exactly `<name>=<float>` on stdout.

For full orientation: `CLAUDE.md` | `ARCHITECTURE.md` | `docs/runbook/`
