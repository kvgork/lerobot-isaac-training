# Runbook 05: Augment with Isaac Lab Domain Randomization

**Prerequisites:** Dataset collected (Runbook 02), Isaac Lab installed (Phase 1 impl)
**[Phase 4 impl required for actual replay — dry-run works now]**
**Expected outcome:** DR-augmented Parquet dataset in `datasets/`; merged dataset ready for training

---

## Overview

Isaac Lab DR replay replays real SO-101 episodes through the simulator with randomized:
- **Object pose** — initial object position/orientation varied
- **Lighting** — scene illumination varied
- **Friction** — gripper and table friction varied

Output: synthetic `LeRobotDataset` Parquet episodes tagged `source="sim_dr"`.

---

## Step 1: Verify Isaac Lab Environment

```bash
python -c "import lerobot_isaac_env; print('env OK')"
python -c "from lerobot_isaac_synthetic.isaac_dr import replay_runner; print('DR OK')"
```

---

## Step 2: Dry Run

```bash
python -m lerobot_isaac_synthetic.isaac_dr.replay_runner \
  --source_dataset_path datasets/so101_pick_v1_filtered \
  --output_path /tmp/test_dr_output \
  --dry_run
```

Expected: prints "DRY RUN — would replay N episodes", exits 0.

---

## Step 3: Full DR Replay

**[Phase 4 impl required + Isaac Lab installed]**

```bash
python -m lerobot_isaac_synthetic.isaac_dr.replay_runner \
  --source_dataset_path datasets/so101_pick_v1_filtered \
  --output_path datasets/so101_pick_dr_v1 \
  --num_augmentations 5 \
  --randomize object_pose lighting friction \
  --headless true \
  --num_envs 4
```

Parameters:
- `--num_augmentations`: how many DR variants per real episode (5 = 5× dataset expansion)
- `--randomize`: which DR factors to apply (space-separated)
- `--headless true`: required unless display available
- `--num_envs 4`: parallel environments; keep ≤8 for RTX 3080

---

## Step 4: Verify DR Dataset Schema

```bash
python -c "
import pandas as pd
df = pd.read_parquet('datasets/so101_pick_dr_v1/data/chunk-000/episode_000001.parquet')
print('Columns:', df.columns.tolist())
# Must include: observation.state (12,), action (6,), source='sim_dr'
assert 'action' in df.columns
print('Schema OK')
"
```

The dataset schema MUST match real SO-101 episodes exactly (same column names, same shapes). `parquet_writer.py` enforces this.

---

## Step 5: Merge Real + DR Datasets

```python
from lerobot_isaac_synthetic.merge_utilities import merge_datasets

merge_datasets(
    real_path="datasets/so101_pick_v1_filtered",
    dr_path="datasets/so101_pick_dr_v1",
    output_path="datasets/so101_merged_v1",
    real_weight=1.0,
    dr_weight=0.5
)
```

Or via CLI:
```bash
python -c "
from lerobot_isaac_synthetic.merge_utilities import merge_datasets
merge_datasets(
    real_path='datasets/so101_pick_v1_filtered',
    dr_path='datasets/so101_pick_dr_v1',
    output_path='datasets/so101_merged_v1'
)
"
```

---

## Step 6: Use Merged Dataset for Training

```bash
lerobot-isaac-train \
  --target_arch smolvla \
  --config packages/lerobot-isaac-configs/configs/policy_smolvla.yaml \
  --dataset_path datasets/so101_merged_v1 \
  --output_dir outputs/smolvla_dr_run1
```

---

## Domain Randomization Config Reference

The DR parameters are defined in `packages/lerobot-isaac-env/src/lerobot_isaac_env/randomization.py`:

```python
# Object pose DR
randomize_object_pose: EventTermCfg(
    params={
        "position_range": [[-0.1, 0.1], [-0.1, 0.1], [0.0, 0.0]],
        "rotation_range": [[-15.0, 15.0]]  # degrees
    }
)

# Lighting DR
randomize_lighting: EventTermCfg(...)

# Friction DR
randomize_friction: EventTermCfg(
    params={"friction_range": [0.3, 1.2]}
)
```

To add new DR factors, add `EventTermCfg` entries to `randomization.py` and include the factor name in `--randomize`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `NotImplementedError` in replay_runner | Phase 4 not impl; use `--dry_run` |
| CUDA OOM during replay | Reduce `--num_envs` to 1; set `--headless true` |
| Schema mismatch after merge | Check `observation.state` shape in both datasets |
| Isaac Lab env not loading | Verify USD path; see Runbook 01 Step 6 |
| DR episodes look unrealistic | Tighten randomization ranges in `randomization.py` |
