# Data Pipeline — Internals

**Cross-references:** [ARCHITECTURE.md](../../ARCHITECTURE.md) | [training-dispatch.md](./training-dispatch.md) | [synthetic-data.md](./synthetic-data.md)

---

## Overview

All data in this workspace flows through a single canonical format: the LeRobot Parquet dataset
(`LeRobotDataset` v3.0 schema). Every ingestion path (real teleop, Isaac Lab DR replay, MimicGen)
produces this same format so training is format-agnostic and datasets can be merged freely.

---

## LeRobotDataset Schema (v3.0)

### Directory Layout

```
datasets/<name>/
  data/
    chunk-000/
      episode_000001.parquet
      episode_000002.parquet
      ...
    chunk-001/
      ...
  meta/
    info.json          # total_episodes, total_frames, fps, features dict
    stats.json         # per-column mean, std, min, max
    episodes.parquet   # episode-level metadata: episode_index, length, source
    tasks.jsonl        # task description per episode (optional)
  videos/              # optional: MP4 for each camera per episode
```

### Per-Episode Parquet Columns

| Column | Shape | Dtype | Source |
|--------|-------|-------|--------|
| `observation.state` | `(T, 12)` | float32 | joint_pos (6) + joint_vel (6) |
| `observation.images.wrist` | `(T, H, W, 3)` | uint8 | wrist camera RGB |
| `observation.images.overhead` | `(T, H, W, 3)` | uint8 | overhead camera RGB |
| `action` | `(T, 6)` | float32 | joint position targets (radians) |
| `next.reward` | `(T,)` | float32 | sparse success reward (0.0 or 1.0) |
| `next.done` | `(T,)` | bool | episode termination flag |
| `episode_index` | scalar | int64 | unique index in dataset |
| `frame_index` | `(T,)` | int64 | frame counter within episode |
| `timestamp` | `(T,)` | float64 | seconds from episode start |
| `source` | scalar | str | `"real"` / `"sim_dr"` / `"mimicgen"` |
| `task_index` | scalar | int64 | index into tasks.jsonl (optional) |

**Convention:** `H=W=480` for real data; `H=W=64` for DreamerV3 HDF5 conversion; `H=W=96` for LeWM.
The Parquet stores real resolution; world-model targets resize at conversion time.

---

## Ingestion Paths

### Path 1 — Real Teleoperation

```
SO-101 hardware (30 Hz)
  -> LeRobot record script (python -m lerobot.scripts.record)
  -> LeRobotDataset.create() + add_frame() per timestep
  -> datasets/<name>/data/chunk-000/episode_XXXXXX.parquet
```

Column mapping from LeRobot's SO-101 config:
- `observation.state` = concatenated `[joint_pos, joint_vel]` from DYNAMIXEL read
- `action` = commanded joint position targets (raw radians, NOT normalized)
- Cameras: wrist camera on joint 6; overhead camera on tripod

### Path 2 — Isaac Lab DR Replay

```
Existing Parquet episode
  -> replay_runner.py reads each frame as action sequence
  -> Isaac Lab env reset with EventTermCfg DR applied
  -> replay actions step-by-step (30 Hz)
  -> parquet_writer.py writes obs/actions to new episode
  -> source="sim_dr" tag added
```

The parquet_writer uses `LeRobotDataset.create()` with the **identical** `features` dict as the
real dataset. Schema enforcement is strict — a mismatch raises `ValueError` before any frames
are written.

### Path 3 — MimicGen (Deferred)

```
Real Parquet
  -> lerobot_mimicgen_bridge skill  (Parquet -> MimicGen HDF5)
  -> MimicGen + robosuite/MuJoCo (internal; not Isaac Lab)
  -> lerobot_mimicgen_bridge skill  (MimicGen HDF5 -> Parquet)
  -> source="mimicgen" tag
```

MimicGen HDF5 uses end-effector-space actions; the bridge converts to joint-space for LeRobot.
Status: deferred. See [`synthetic-data.md`](./synthetic-data.md).

---

## Episode Tagging

The `source` field in `meta/episodes.parquet` records the origin of each episode. It is set by:

1. `parquet_writer.py` — sets `source="sim_dr"` for DR episodes
2. `bridge_invocation.py` — sets `source="mimicgen"` for MimicGen episodes
3. `LeRobot record script` — leaves unset; `merge_utilities.py` backfills `"real"` for episodes without a source tag

The `source` column is preserved through `merge_utilities.merge_datasets()`.

---

## Format Conversions

### Parquet to DreamerV3 HDF5

Use `lerobot_world_model_bridge` skill:
```python
# Task(lerobot-worldmodel-bridge, {target: "dreamerv3", image_size: 64, ...})
```

Output HDF5 schema:
```
episode_000001/
    obs/
        image: (T, 64, 64, 3)   uint8
        state: (T, 12)          float32
    actions: (T, 6)             float32
    rewards: (T,)               float32
    dones:   (T,)               bool
```

Image resize: bicubic downscale from 480px to 64px applied during conversion.
DO NOT write custom converters; always use the skill.
Skill path: `${CLAUDE_CODE_ROOT}/skills/lerobot_world_model_bridge/`

### Parquet to LeWorldModel HDF5

Same skill, different preset:
```python
# Task(lerobot-worldmodel-bridge, {target: "le_world_model", image_size: 96, ...})
```

Output HDF5 schema: `(T, 96, 96, 3)` image. The schema is partially undocumented; inspect
`quentinll/lewm-pusht` on HuggingFace Hub for reference. See [`world-model-bridge.md`](./world-model-bridge.md).

---

## Dataset Merge Logic

`merge_utilities.merge_datasets()` performs:

1. Read `meta/info.json` from each source dataset
2. Reassign `episode_index` sequentially (starting from 0)
3. Update `meta/episodes.parquet` with new indices and source tags
4. Concatenate Parquet files into `chunk-000/` (splitting into new chunks at 1000 episodes)
5. Recompute `meta/stats.json` (weighted mean/std using `real_weight` and `dr_weight`)
6. Write updated `meta/info.json` with `total_episodes` and `sources` list

Deduplication: episodes with identical `(source, original_episode_index)` are not duplicated
if merge is run multiple times. The key is stored in `meta/merge_manifest.json`.

---

## Schema Invariant Enforcement

The `isaac_data_recorder.py` module includes a pre-flight check before writing any episodes:

```python
# Pseudo-code (illustrative):
expected_features = load_features_from_config(cfg)
actual_features = {
    "observation.state": {"shape": (12,), "dtype": "float32"},
    "action": {"shape": (6,), "dtype": "float32"},
    ...
}
if actual_features != expected_features:
    raise ValueError(f"Schema mismatch: {diff}")
```

This ensures sim episodes can be merged with real episodes without column conflicts.
