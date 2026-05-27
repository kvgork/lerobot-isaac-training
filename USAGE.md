# Usage Runbook — LeRobot + Isaac Lab Training Workspace

**Workspace:** `~/workspaces/lerobot-isaac-training/`
**For architecture context:** `ARCHITECTURE.md`
**For workspace orientation:** `CLAUDE.md`
**Plan reference:** `${CLAUDE_CODE_ROOT}/plans/2026-05-06-lerobot-isaac-workspace-plan.md`

> **Architecture note (2026-05-13):** Post-spinout. Only `lerobot-isaac-meta` lives
> in `packages/`. The 7 siblings are public GitHub repos at `github.com/kvgork/<name>`.
> See `docs/runbook/00-install.md` for the install path.

---

## Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| OS | Ubuntu 22.04 LTS | Isaac Lab tested on 22.04; 20.04 may work |
| GPU | NVIDIA RTX 3080 10 GB | 16 GB recommended; 10 GB requires `--num_envs 4-8` and AMP |
| CUDA | 12.x | Isaac Sim requires CUDA 12; LeRobot works with 11.8+ |
| Python | 3.10+ | 3.11 recommended; pixi manages the interpreter |
| Disk | 50 GB free | Isaac Sim ~15 GB, USD assets ~2 GB, datasets + checkpoints variable |
| RAM | 32 GB | 16 GB minimum; 32 GB for comfortable parallel env usage |
| Display | optional | Isaac Lab headless mode (`headless=True`) does not require X11 |
| pixi | latest | `curl -fsSL https://pixi.sh/install.sh \| bash` |

---

## First-Time Setup

Run these steps in order. Each step is idempotent — safe to re-run.

```bash
# 1. Enter workspace
cd ~/workspaces/lerobot-isaac-training

# 2. Install pixi if absent
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc   # or restart shell

# 3. Install base environment (dev tooling + 6 sibling packages)
pixi install

# 4. Activate workspace shell
pixi shell

# 5. Install packages in editable mode (also done by pixi, but explicit for IDEs)
uv sync
# Verify:
python -c "import lerobot_isaac_meta; print('meta OK')"
python -c "import lerobot_isaac_adapters; print('adapters OK')"
python -c "import lerobot_isaac_synthetic; print('synthetic OK')"

# 6. Set workspace environment variable (add to ~/.bashrc for persistence)
export LEROBOT_ISAAC_WORKSPACE=~/workspaces/lerobot-isaac-training

# 7. Install Isaac Lab (system-level; GPU + ~15 GB disk required)
#    Skip if you only need policy training (no simulation)
pixi run install-isaac-lab
# Verify: python -c "import isaaclab; print('Isaac Lab OK')"

# 8. Download SO-101 USD asset (requires Isaac Lab installed in step 7)
pixi run download-usd
# Or: bash packages/lerobot-isaac-env/assets/usd/download_so101_urdf.sh
# Expected: packages/lerobot-isaac-env/assets/usd/so101.usd

# 9. Smoke test
lerobot-isaac-train \
  --target_arch smolvla \
  --config packages/lerobot-isaac-configs/configs/policy_smolvla.yaml \
  --dry_run
# Expected: "DRY RUN" message, exit 0

# 10. Run workspace tests
pixi run test
# Or: pytest packages/*/tests/
# Expected: all tests pass (no Isaac Lab or LeRobot required)
```

---

## Daily Workflows

### Workflow A: Collect Real Teleop Data

**Prerequisites:** SO-101 connected via USB; LeRobot installed (`pixi install -e train-policy`)

**Step 1 — Start teleoperation recording:**
```bash
pixi shell -e train-policy

# Invoke the data-collection agent (auto-filters via SAL + TED):
# Task(lerobot-data-collection-agent, {
#   dataset_path: "datasets/so101_pick_v1",
#   robot: "so101",
#   quality_filter: true,
#   min_episodes: 50
# })

# Or use LeRobot directly:
python -m lerobot.scripts.record \
  --robot-path lerobot/configs/robot/so101.yaml \
  --fps 30 \
  --repo-id local/so101_pick_v1 \
  --root datasets/ \
  --num-episodes 50
```

**Step 2 — Verify dataset:**
```bash
python -c "
import pandas as pd
df = pd.read_parquet('datasets/so101_pick_v1/data/chunk-000/episode_000001.parquet')
print('Columns:', df.columns.tolist())
print('Episodes recorded successfully')
"
```

**Expected output:** Dataset in `datasets/so101_pick_v1/` with Parquet files and `meta/info.json`.

**Troubleshooting:**
- `robot not found`: check USB connection; run `lerobot find-robot --robot so101`
- `no episodes saved`: check FPS (30 Hz required); verify camera feed with `lerobot show-robot`

---

### Workflow B: Filter and Quality-Check Data

**Prerequisites:** Raw dataset in `datasets/`; `lerobot_dataset_quality` skill

**Step 1 — Run quality filter:**
```bash
# The skill runs SAL (Scene Anomaly Localization) and TED (Trajectory Edit Distance):
# Task via skill invocation:
# lerobot_dataset_quality.filter(
#   dataset_path="datasets/so101_pick_v1",
#   output_path="datasets/so101_pick_v1_filtered",
#   sal_threshold=0.3,
#   ted_threshold=0.5
# )

# Or via data-collection agent which wraps both:
# Task(lerobot-data-collection-agent, {
#   dataset_path: "datasets/so101_pick_v1",
#   quality_filter: true
# })
```

**Step 2 — Verify filter results:**
```bash
python -c "
import json, pathlib
info = json.loads(pathlib.Path('datasets/so101_pick_v1_filtered/meta/info.json').read_text())
print('Episodes after filter:', info['total_episodes'])
print('Source:', info.get('source', 'real'))
"
```

**Expected output:** Filtered dataset with fewer episodes, all tagged `source="real"`.

**Skill reference:** `${CLAUDE_CODE_ROOT}/skills/lerobot_dataset_quality/`

---

### Workflow C: Train SmolVLA / ACT / Diffusion Policy

**Prerequisites:** Filtered dataset in `datasets/`; LeRobot installed
**Phase status:** Phase 2 impl required for full training; `--dry_run` works now

**Step 1 — Choose target architecture:**
```bash
pixi shell -e train-policy
# Architectures: smolvla (best general), act (fast inference), diffusion (complex trajectories)
```

**Step 2 — Edit config:**
```bash
# Edit packages/lerobot-isaac-configs/configs/policy_smolvla.yaml:
# dataset_path: datasets/so101_pick_v1_filtered
# output_dir: outputs/smolvla_run1
# batch_size: 32
# num_steps: 100000
# eval_freq: 5000
```

**Step 3 — Dry run (scaffolding — works now):**
```bash
lerobot-isaac-train \
  --target_arch smolvla \
  --config packages/lerobot-isaac-configs/configs/policy_smolvla.yaml \
  --dry_run
```

**Step 4 — Full training:**
```bash
lerobot-isaac-train \
  --target_arch smolvla \
  --config packages/lerobot-isaac-configs/configs/policy_smolvla.yaml \
  --dataset_path datasets/so101_pick_v1_filtered \
  --output_dir outputs/smolvla_run1
# Emits: pc_success=<float> on each eval step
```

**Step 5 — Evaluate:**
```bash
# Task(lerobot-evaluation-agent, {
#   checkpoint_path: "outputs/smolvla_run1/checkpoints/last",
#   dataset_path: "datasets/so101_pick_v1_filtered",
#   eval_episodes: 20,
#   metric: "pc_success"
# })
```

**Expected output:** Checkpoint in `outputs/smolvla_run1/`; `pc_success=<float>` on stdout.
**See also:** `docs/runbook/03-train-policy.md`

---

### Workflow D: Train DreamerV3 World Model

**Prerequisites:** Filtered dataset; sheeprl installed; `pixi install -e train-dreamer`
**Phase status:** Phase 2 impl required; dry-run works

**Step 1 — Convert Parquet to HDF5 (64x64 for RTX 3080):**
```bash
# Task(lerobot-worldmodel-bridge, {
#   input_parquet: "datasets/so101_pick_v1_filtered",
#   output_hdf5: "outputs/hdf5/so101_pick_64.hdf5",
#   target: "dreamerv3",
#   image_size: 64
# })
# Skill: ${CLAUDE_CODE_ROOT}/skills/lerobot_world_model_bridge/
```

**Step 2 — Train:**
```bash
pixi shell -e train-dreamer
lerobot-isaac-train \
  --target_arch dreamerv3 \
  --config packages/lerobot-isaac-configs/configs/wm_dreamerv3.yaml \
  --dataset_path outputs/hdf5/so101_pick_64.hdf5 \
  --output_dir outputs/dreamerv3_run1
# Emits: recon_loss=<float> on each eval step
```

**Step 3 — Monitor:**
```bash
grep "recon_loss" outputs/dreamerv3_run1/train.log
```

**RTX 3080 note:** Use `image_size: 64`, `batch_size: 16`, `amp: true`. Do NOT use 96x96 for DreamerV3.
**Expected output:** RSSM checkpoint; `recon_loss` decreasing over steps.
**See also:** `docs/runbook/04-train-world-model.md`, `docs/research/dreamerv3-reference.md`

---

### Workflow E: Train HF LeWorldModel

**Prerequisites:** Filtered dataset; LeWorldModel installed; `pixi install -e train-lewm`
**Phase status:** Phase 2 impl required

**Step 1 — Convert Parquet to HDF5 (96x96):**
```bash
# Task(lerobot-worldmodel-bridge, {
#   input_parquet: "datasets/so101_pick_v1_filtered",
#   output_hdf5: "outputs/hdf5/so101_pick_96.hdf5",
#   target: "le_world_model",
#   image_size: 96
# })
```

**Step 2 — Train:**
```bash
pixi shell -e train-lewm
lerobot-isaac-train \
  --target_arch le_world_model \
  --config packages/lerobot-isaac-configs/configs/wm_leworldmodel.yaml \
  --dataset_path outputs/hdf5/so101_pick_96.hdf5 \
  --output_dir outputs/lewm_run1
# Emits: pred_loss=<float>
```

**RTX 3080 note:** LeWM at 96x96 requires gradient checkpointing + AMP + `batch_size=8`.
**Expected output:** LeWM checkpoint; `pred_loss` decreasing.
**See also:** `docs/runbook/04-train-world-model.md`, `docs/research/leworldmodel-reference.md`

---

### Workflow F: Run Autoresearch HP Search

**Prerequisites:** Training adapter functional (Phase 2 impl); program.md configured
**Phase status:** program.md files exist; Phase 3 impl required for full end-to-end run

**Step 1 — Choose program:**
```bash
ls packages/lerobot-isaac-autoresearch/programs/
# lerobot-policy.md  dreamerv3.md  leworldmodel.md
```

**Step 2 — Invoke autoresearch loop:**
```bash
# For LeRobot policy (maximizes pc_success):
/autoresearch \
  ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/lerobot-policy.md \
  --type ml_model

# For DreamerV3 (minimizes recon_loss):
/autoresearch \
  ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/dreamerv3.md \
  --type ml_model

# For LeWorldModel (minimizes pred_loss):
/autoresearch \
  ~/workspaces/lerobot-isaac-training/packages/lerobot-isaac-autoresearch/programs/leworldmodel.md \
  --type ml_model
```

**Step 3 — Monitor results:**
```bash
# Results land in .agent-state/<session>/
ls .agent-state/
```

**Program structure:** Each `program.md` specifies: metric, direction, baseline command, mutation operators, time budget.
**Agent reference:** `${CLAUDE_CODE_ROOT}/agents/orchestrators/autoresearch-loop-orchestrator.md`
**See also:** `docs/runbook/` (no dedicated runbook yet; see programs/*.md for inline docs)

---

### Workflow G: Generate DR Synthetic Data

**Prerequisites:** Isaac Lab installed; filtered dataset; `pixi install -e sim`
**Phase status:** Phase 4 impl required; dry-run works now

**Step 1 — Dry run:**
```bash
pixi shell -e sim
python -m lerobot_isaac_synthetic.isaac_dr.replay_runner \
  --source_dataset_path datasets/so101_pick_v1_filtered \
  --output_path /tmp/test_dr_output \
  --dry_run
# Expected: "DRY RUN — would replay N episodes"
```

**Step 2 — Full DR replay:**
```bash
python -m lerobot_isaac_synthetic.isaac_dr.replay_runner \
  --source_dataset_path datasets/so101_pick_v1_filtered \
  --output_path datasets/so101_pick_dr_v1 \
  --num_augmentations 5 \
  --randomize object_pose lighting friction \
  --headless true \
  --num_envs 4
# Each real episode -> 5 DR variants; 5x dataset expansion
```

**Step 3 — Merge real + DR:**
```bash
python -c "
from lerobot_isaac_synthetic.merge_utilities import merge_datasets
merge_datasets(
    real_path='datasets/so101_pick_v1_filtered',
    dr_path='datasets/so101_pick_dr_v1',
    output_path='datasets/so101_merged_v1',
    real_weight=1.0,
    dr_weight=0.5
)
"
```

**Step 4 — Verify merged schema:**
```bash
python -c "
import pandas as pd
df = pd.read_parquet('datasets/so101_merged_v1/data/chunk-000/episode_000001.parquet')
assert 'action' in df.columns
print('source values:', df['source'].unique())  # ['real', 'sim_dr']
print('Schema OK')
"
```

**Expected output:** Merged dataset with `source` column containing `"real"` and `"sim_dr"` values.
**See also:** `docs/runbook/05-augment-with-dr.md`

---

### Workflow H: Generate MimicGen Synthetic Data (Deferred)

**Status: DEFERRED** — `bridge_invocation.py` raises `NotImplementedError` by default.

**To enable (Phase 4b):**
```bash
pip install mimicgen robosuite
export LEROBOT_MIMICGEN_ENABLED=1
# Then implement bridge_invocation.py or use the agent:
# Task(lerobot-sim-augmentation-agent, {
#   source_dataset: "datasets/so101_pick_v1_filtered",
#   output_path: "datasets/so101_mimicgen",
#   num_demonstrations: 100
# })
```

**Skill reference:** `${CLAUDE_CODE_ROOT}/skills/lerobot_mimicgen_bridge/`
**Agent reference:** `${CLAUDE_CODE_ROOT}/agents/workers/lerobot-sim-augmentation-agent.md`
**See also:** `docs/runbook/06-augment-with-mimicgen.md`, `docs/research/mimicgen-reference.md`

---

### Workflow I: Merge Real + Sim Datasets

**Prerequisites:** At least two source datasets (real + DR or real + MimicGen)

```bash
python -c "
from lerobot_isaac_synthetic.merge_utilities import merge_datasets

# Merge real + DR:
merge_datasets(
    real_path='datasets/so101_pick_v1_filtered',
    dr_path='datasets/so101_pick_dr_v1',
    output_path='datasets/so101_merged_v1',
    real_weight=1.0,
    dr_weight=0.5
)

# Add MimicGen to existing merged (when available):
# merge_datasets(
#     real_path='datasets/so101_merged_v1',
#     mimicgen_path='datasets/so101_mimicgen',
#     output_path='datasets/so101_full_v1',
#     real_weight=1.0,
#     mimicgen_weight=0.3
# )
"
```

The merged dataset updates `meta/info.json`, `meta/stats.json`, and `meta/episodes.parquet`.
Each episode retains its `source` tag for per-source weighting in training configs.

---

### Workflow J: Curriculum Advance

**Prerequisites:** Policy trained and evaluated (`pc_success` available); Phase 2+ impl

```bash
# After evaluation returns pc_success >= 0.80:
# Task(lerobot-curriculum-agent, {
#   workspace_root: "~/workspaces/lerobot-isaac-training",
#   current_stage: 1,
#   eval_metric: "pc_success",
#   eval_value: 0.85,
#   advance_threshold: 0.80
# })
```

**Stage ladder (6 stages):**
| Stage | Task | DR Range | Advance Threshold |
|-------|------|----------|-------------------|
| 1 | Fixed-position pick | 0 cm | 0.80 |
| 2 | Pick with variable object position | ±3 cm | 0.80 |
| 3 | Pick-and-place | ±5 cm | 0.75 |
| 4 | Pick-and-place with obstacles | ±5 cm | 0.70 |
| 5 | Insertion | ±2 cm | 0.65 |
| 6 | (Future) multi-step manipulation | TBD | TBD |

**Agent reference:** `${CLAUDE_CODE_ROOT}/agents/orchestrators/lerobot-curriculum-agent.md`

---

### Workflow K: View Metrics Dashboard

**Prerequisites:** `pixi install -e dashboard`; `LEROBOT_ISAAC_WORKSPACE` set
**See also:** `docs/runbook/07-dashboard.md` for the full runbook with tab guide and troubleshooting.

#### Live UI

```bash
pixi run -e dashboard dashboard
# Opens http://localhost:8501
```

The sidebar offers session selector, refresh interval, watch-files toggle, mode radio
(Live / Compare 2-way / Compare N-way), save-snapshot button, and export-report button.
All 8 tabs show "No data yet" when `outputs/` is empty — this is normal on a fresh workspace.

#### Static HTML report

```bash
# Inline plotly.js (~5 MB self-contained, works offline)
pixi run -e dashboard report --workspace=$PWD

# CDN plotly.js (~50 KB, requires internet for offline viewing)
pixi run -e dashboard report --workspace=$PWD --cdn
```

Output: `outputs/reports/<run_id>/report.html`
Side effect: auto-saves a snapshot to `outputs/snapshots/<run_id>/` (disable with `--no-snapshot`).

#### Save a snapshot

```bash
pixi run -e dashboard snapshot --workspace=$PWD --label=baseline

# List existing snapshots
pixi run -e dashboard snapshot --workspace=$PWD list
```

Output: `outputs/snapshots/<timestamp>-<label>/meta.json` + `loaders/`

#### Compare 2-way

```bash
# By snapshot label / ID
pixi run -e dashboard compare --workspace=$PWD --snapshots baseline after-dr

# With CDN plotly
pixi run -e dashboard compare --workspace=$PWD --snapshots baseline after-dr --cdn
```

Output: `outputs/reports/compare-baseline-vs-after-dr/report.html`
Layout: each tab split into two columns; delta KPI strip above (pc_success, train_loss).

#### Compare N-way

```bash
pixi run -e dashboard compare --workspace=$PWD \
  --snapshots baseline exp-lr1e3 exp-lr5e4 exp-dr5x \
  --mode nway
```

Output: time-series traces from all snapshots overlaid; snapshot label as legend.

### Workflow L: Batch Train Multiple Backends and Auto-Compare

**Prerequisites:** any pixi env that has the target backends installed (`full` covers all)
**See also:** `docs/runbook/08-batch-train-and-compare.md` for the full schema and runbook.

Run multiple `target_arch`s sequentially on the same dataset and render a single
N-way / 2-way HTML compare report — the canonical "train SmolVLA and LeWorldModel
on the same data, then compare" workflow.

```bash
# 1. Drop / edit a batch YAML (see batch_example.yaml in lerobot-isaac-configs)
$EDITOR packages/lerobot-isaac-configs/src/lerobot_isaac_configs/configs/batches/example.yaml

# 2. Verify dispatch (no checkpoints written, no snapshots taken)
pixi run -e default lerobot-isaac-batch \
    --config packages/lerobot-isaac-configs/src/lerobot_isaac_configs/configs/batches/example.yaml \
    --workspace . \
    --dry_run

# 3. Run for real
pixi run -e full lerobot-isaac-batch --config <your-batch.yaml> --workspace .

# 3b. Pixi shortcut for the example
pixi run train-and-compare
```

Output: `outputs/reports/compare-<batch_id>/report.html` plus per-run snapshots
under `outputs/snapshots/<batch_id>-<run_id>/`.

Failure handling: `on_failure: continue` (default) skips failed runs from compare;
`on_failure: abort` halts the batch on first failure.

---

## CLI Reference

### `lerobot-isaac-train` Flags

Entrypoint: `packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/train.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--target_arch` | str | required | One of: `smolvla`, `act`, `diffusion`, `dreamerv3`, `le_world_model` |
| `--config` | path | required | Path to YAML config (in `packages/lerobot-isaac-configs/configs/`) |
| `--dataset_path` | path | from config | Override dataset path from config |
| `--output_dir` | path | from config | Override output directory |
| `--max_steps` | int | from config | Override max training steps |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--isaac_env_id` | str | None | Isaac Lab env ID (for `isaac_data_recorder`) |
| `--dry_run` | flag | False | Print dispatched command without executing |
| `--num_envs` | int | from config | Number of parallel Isaac Lab environments |

### `lerobot-isaac` Subcommand Table

Entrypoint: `packages/lerobot-isaac-meta/src/lerobot_isaac_meta/cli.py`

| Subcommand | Description |
|-----------|-------------|
| `lerobot-isaac train` | Alias for `lerobot-isaac-train` |
| `lerobot-isaac paths` | Print resolved workspace paths |
| `lerobot-isaac status` | Show build status and installed packages |

### `replay_runner` Flags

Entrypoint: `packages/lerobot-isaac-synthetic/src/lerobot_isaac_synthetic/isaac_dr/replay_runner.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--source_dataset_path` | path | required | Input Parquet dataset |
| `--output_path` | path | required | Output DR Parquet dataset |
| `--num_augmentations` | int | 5 | DR variants per real episode |
| `--randomize` | list | all | DR factors: `object_pose`, `lighting`, `friction`, `camera_fov` |
| `--headless` | bool | true | Run without display |
| `--num_envs` | int | 4 | Parallel environments (keep ≤8 for RTX 3080) |
| `--dry_run` | flag | False | Print what would run without executing |

### Dashboard CLI Commands

| Command | Description |
|---------|-------------|
| `lerobot-isaac-dashboard` | Start live Streamlit app (default port 8501) |
| `lerobot-isaac-report --workspace=PATH [--cdn] [--no-snapshot] [--with-csv]` | Export static HTML report |
| `lerobot-isaac-snapshot --workspace=PATH [--label=LABEL] [save\|list]` | Save or list snapshots |
| `lerobot-isaac-compare --workspace=PATH --snapshots A B [C ...] [--mode 2way\|nway] [--cdn]` | Export compare report |

---

## Pixi Reference

### `pixi run <task>` Index

| Task | Description |
|------|-------------|
| `pixi run test` | Run all workspace tests (`pytest packages/*/tests/`) |
| `pixi run lint` | Run ruff linter |
| `pixi run fmt` | Run ruff formatter |
| `pixi run install-isaac-lab` | Install Isaac Lab (system-level; GPU required) |
| `pixi run download-usd` | Download SO-101 USD from TheRobotStudio/SO-ARM100 |

### Environment Selection

```bash
# Default (dev tooling only):
pixi install && pixi shell

# LeRobot policy training:
pixi install -e train-policy && pixi shell -e train-policy

# DreamerV3 world model:
pixi install -e train-dreamer && pixi shell -e train-dreamer

# LeWorldModel:
pixi install -e train-lewm && pixi shell -e train-lewm

# Isaac Lab simulation:
pixi install -e sim && pixi shell -e sim

# Metrics dashboard:
pixi install -e dashboard && pixi shell -e dashboard

# All targets:
pixi install -e full && pixi shell -e full
```

---

## Spinout Workflow

To extract a package to its own standalone repository:

```bash
# 1. Create spinout branch
git subtree split -P packages/lerobot-isaac-env -b spinout-env

# 2. Clone into a new directory
cd /tmp
git clone ~/workspaces/lerobot-isaac-training lerobot-isaac-env-standalone
cd lerobot-isaac-env-standalone
git checkout spinout-env

# 3. Activate the dormant pixi.toml (remove "dormant" comment)
#    Update pyproject.toml: change sibling path deps to PyPI deps
#    Change: lerobot-isaac-configs = {path = "../lerobot-isaac-configs"}
#    To:     lerobot-isaac-configs = ">=0.1.0"

# 4. Push to new remote
git remote set-url origin git@github.com:yourorg/lerobot-isaac-env.git
git push -u origin spinout-env:main
```

For cleaner history, use `git filter-repo` — see `ARCHITECTURE.md §Spinout Mechanics`.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `import isaaclab` fails | Isaac Lab not installed | Run `pixi run install-isaac-lab`; stubs work without it |
| `lerobot-isaac-train: command not found` | Packages not installed | Run `uv sync` or `pip install -e packages/lerobot-isaac-meta` |
| `so101.usd not found` | USD not downloaded | Run `pixi run download-usd`; see `packages/lerobot-isaac-env/assets/usd/README.md` |
| CUDA OOM during Isaac sim | Too many parallel envs | Reduce `--num_envs` to 1–4; set `headless: true`; enable AMP |
| CUDA OOM during DreamerV3 | Image size too large | Use `image_size: 64`; set `batch_size: 16`; enable `amp: true` |
| CUDA OOM during LeWM | LeWM at 96x96 too large | Enable gradient checkpointing + AMP; set `batch_size: 8` |
| `NotImplementedError` in targets | Phase 2 not yet implemented | Use `--dry_run` to test dispatch without training |
| Parquet schema mismatch | Isaac Lab obs names differ from LeRobot | Check `observation.state`, `action` columns; use `isaac_data_recorder.py` |
| `pixi install` fails | Network issues or missing CUDA | Try `pixi install --verbose`; check CUDA version with `nvidia-smi` |
| Autoresearch loop hangs | Training script not emitting metric | Ensure `metric_extractor.emit("<name>", value)` is called |
| `lerobot not found` | LeRobot not installed in environment | Use `pixi shell -e train-policy` |
| Dashboard tabs all blank | plotly not installed or wrong env | `pixi install -e dashboard`; verify `pixi run -e dashboard python -c "import plotly"` |
| Dashboard snapshot reload fails | Snapshot from newer dashboard version | Upgrade dashboard or re-save snapshot from current version |
