# Runbook 02: Collect Real Teleop Data

**Prerequisites:** Bootstrap complete (Runbook 01), SO-101 arm connected, LeRobot installed
**Agent:** `lerobot-data-collection-agent`
**Skill:** `lerobot_dataset_quality` (SAL + TED filtering)
**Expected outcome:** Filtered LeRobot Parquet dataset in `datasets/`

---

## Step 1: Create Dataset Directory

```bash
mkdir -p ~/workspaces/lerobot-isaac-training/datasets
cd ~/workspaces/lerobot-isaac-training
```

---

## Step 2: Record Teleop Demonstrations

Use LeRobot's built-in teleoperation recording:

```bash
# Record with SO-101 via LeRobot:
python -m lerobot.scripts.control_robot \
  --robot-path lerobot/configs/robot/so101.yaml \
  --fps 30 \
  --root datasets/ \
  --repo-id local/so101_pick_v1 \
  record \
  --warmup-time-s 5 \
  --episode-time-s 30 \
  --reset-time-s 5 \
  --num-episodes 50
```

**[Requires LeRobot installed + SO-101 connected]**

Target: 50–100 demonstrations per task. Quality matters more than quantity.

---

## Step 3: Invoke the Data Collection Agent

The `lerobot-data-collection-agent` runs SAL + TED filtering automatically:

```
Task(lerobot-data-collection-agent, {
  dataset_path: "datasets/so101_pick_v1",
  robot: "so101",
  quality_filter: true,
  sal_threshold: 0.3,
  ted_threshold: 0.5,
  output_path: "datasets/so101_pick_v1_filtered"
})
```

The agent:
1. Loads the raw Parquet dataset
2. Runs SAL (Scene Anomaly Localization) to detect corrupted/anomalous episodes
3. Runs TED (Trajectory Edit Distance) to remove near-duplicate episodes
4. Outputs filtered dataset to `output_path`
5. Reports rejection statistics

---

## Step 4: Verify Dataset Quality

```bash
# Check dataset structure:
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('datasets/so101_pick_v1_filtered')
print(f'Episodes: {ds.num_episodes}')
print(f'Frames: {len(ds)}')
print(f'Features: {list(ds.features.keys())}')
"
```

Expected features for SO-101:
- `observation.state`: shape `(12,)` — joint_pos 6 + joint_vel 6
- `action`: shape `(6,)` — target joint positions
- `observation.images.top`: JPEG binary (if camera attached)

---

## Step 5: Tag Dataset Source

Ensure real episodes are tagged `source="real"` in the Parquet metadata. This is needed for weighted mixing in `merge_utilities.py`:

```bash
# Check source tag:
python -c "
import pandas as pd
df = pd.read_parquet('datasets/so101_pick_v1_filtered/data/chunk-000/episode_000001.parquet')
print(df.columns.tolist())
"
```

---

## Step 6: Register in Workspace Config

Add dataset path to `packages/lerobot-isaac-configs/configs/policy_smolvla.yaml`:
```yaml
dataset_path: datasets/so101_pick_v1_filtered
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| SO-101 not detected | Check USB connection; verify robot config path |
| Too many rejected episodes | Lower `sal_threshold` to 0.2; check lighting consistency |
| `lerobot not found` | Install: `pip install lerobot` or via pixi |
| Dataset schema mismatch | Check `observation.state` shape — must be `(12,)` for SO-101 |

---

## Next Step

With a filtered dataset, proceed to training:
- Policy training: `docs/runbook/03-train-policy.md`
- World model training: `docs/runbook/04-train-world-model.md`
- DR augmentation: `docs/runbook/05-augment-with-dr.md`
