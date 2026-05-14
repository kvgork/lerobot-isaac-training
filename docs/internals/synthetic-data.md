# Synthetic Data — Internals

**Cross-references:** [pipeline-overview.md](../pipeline-overview.md) | [ARCHITECTURE.md](../../ARCHITECTURE.md) | [data-pipeline.md](./data-pipeline.md) | [isaac-lab-integration.md](./isaac-lab-integration.md)
**Package:** `lerobot-isaac-synthetic` (bare repo: `~/workspaces/spinouts/lerobot-isaac-synthetic/`; editable source: `src/lerobot-isaac-synthetic/`; installed via `git+file://`).

> **Real end-to-end synthetic data generation works** as of 2026-05-14 on the
> Isaac Sim 6.0 + Isaac Lab 0.54 + lerobot 0.5 stack. See
> [pipeline-overview.md §5](../pipeline-overview.md#5-where-each-recent-bugfix-lives)
> for the bugfix trail. Single-command reproduction:
> `bash scripts/run_full_pipeline.sh --skip-policy --skip-worldmodel --skip-eval`.

---

## Overview

Two complementary paths expand the training corpus beyond real teleoperation:

1. **Isaac Lab Domain Randomization (Priority Path)** — replay real episodes in the
   simulator with randomized scene parameters. Fast, always produces valid episodes.
2. **MimicGen Bridge (Deferred Path)** — use MimicGen to generate novel trajectory
   configurations from a small real-demo seed. More diverse but requires more setup.

Both paths produce Parquet files in the same `LeRobotDataset` schema as real data,
tagged with a `source` column value of `"sim_dr"` or `"mimicgen"`.

---

## Path 1: Isaac Lab DR Replay Loop

### Architecture

```
replay_runner.py
    |
    | for each real episode:
    |   for each augmentation (1..N):
    |
    +-- 1. load_episode(real_parquet_path, episode_idx)
    |       returns: sequence of action frames [(action_0, ..., action_T)]
    |
    +-- 2. env.reset(DR_seed=seed_i)
    |       EventManager applies DR config:
    |         - object pose randomization
    |         - lighting randomization
    |         - friction randomization
    |
    +-- 3. for action in actions:
    |         obs, reward, done, _ = env.step(action)
    |         record_frame(obs, action, reward, done)
    |
    +-- 4. parquet_writer.write_episode(frames, source="sim_dr")
    |
    v
datasets/<name>_dr/  (source="sim_dr" in every episode)
```

### `replay_runner.py` Key Logic

The runner reads the real episode's action sequence (NOT its observations) and replays
it open-loop in the sim. This means the robot's trajectory in sim may diverge from real
if the DR changes the scene significantly — that divergence is the desired augmentation.

**Important:** The runner does NOT use the real episode's observations. It only replays
actions. This ensures that the sim observations (which have different visual appearance
due to DR) are recorded, not the real ones.

### `parquet_writer.py` Key Logic

The writer uses `LeRobotDataset.create()` with the same `features` dict as the real dataset.
It calls `add_frame()` for each timestep and `save_episode()` at the end.

Schema enforcement occurs at `create()` time: if the real dataset's features dict does not
match the writer's expected features, `ValueError` is raised before any data is written.

---

## Path 2: MimicGen Bridge (Deferred)

### Status

`bridge_invocation.py` raises `NotImplementedError` by default. Enable via:
```bash
export LEROBOT_MIMICGEN_ENABLED=1
pip install mimicgen robosuite
```

### Architecture (When Enabled)

```
bridge_invocation.py
    |
    +-- 1. lerobot_mimicgen_bridge.to_mimicgen(real_parquet_path)
    |       converts Parquet -> MimicGen HDF5
    |       (end-effector-space action conversion)
    |
    +-- 2. MimicGen.generate(
    |         source_hdf5=mimicgen_input.hdf5,
    |         num_demos=N,
    |         randomization_config=...
    |       )
    |       runs internally in robosuite/MuJoCo (NOT Isaac Lab)
    |
    +-- 3. lerobot_mimicgen_bridge.from_mimicgen(mimicgen_output.hdf5)
    |       converts MimicGen HDF5 -> Parquet
    |       source="mimicgen" tag applied
    |
    v
datasets/<name>_mimicgen/
```

**Key gap:** Joint-to-end-effector-space conversion for SO-101 has NOT been calibrated.
The `lerobot_mimicgen_bridge` skill supports the conversion but requires a robot-specific
kinematic model. This is a Phase 4b task. See `docs/research/mimicgen-reference.md`.

---

## Dataset Merge Logic

`merge_utilities.merge_datasets()` merges two or more Parquet datasets into one.

### Merge Steps

```python
# Illustrative (not production code):
def merge_datasets(real_path, dr_path, output_path, real_weight=1.0, dr_weight=0.5):
    # 1. Read source info
    real_info = read_info(real_path)
    dr_info = read_info(dr_path)

    # 2. Validate schema compatibility
    assert real_info["features"] == dr_info["features"], "Schema mismatch!"

    # 3. Reassign episode indices (0-based, sequential)
    real_episodes = list_episodes(real_path)   # e.g. 0..49
    dr_episodes = list_episodes(dr_path)        # e.g. 0..249
    # merged: 0..49 (real) + 50..299 (dr)

    # 4. Copy Parquet files
    copy_with_new_index(real_episodes, output_path, start_index=0)
    copy_with_new_index(dr_episodes, output_path, start_index=len(real_episodes))

    # 5. Update meta/episodes.parquet with source tags
    write_episodes_parquet(output_path, real_episodes + dr_episodes)

    # 6. Recompute meta/stats.json (weighted)
    stats = compute_weighted_stats(real_episodes, dr_episodes, real_weight, dr_weight)
    write_stats(output_path, stats)

    # 7. Write meta/info.json
    write_info(output_path, total_episodes=len(real_episodes)+len(dr_episodes))

    # 8. Write merge manifest (for idempotency)
    write_manifest(output_path, sources=[real_path, dr_path], weights=[real_weight, dr_weight])
```

### Deduplication

The merge is idempotent: if run twice with the same inputs, the output is identical.
The `meta/merge_manifest.json` records `(source_path, episode_count, hash)` per source;
if a source is already present in the output with the same hash, it is skipped.

### Source Weighting in Training

Training configs can weight sources differently using the `source` column:
```yaml
# In policy_smolvla.yaml:
dataset_mixing:
  real: 1.0
  sim_dr: 0.5
  mimicgen: 0.3
```

The LeRobot training script reads this section and applies weighted sampling per source.
Episodes tagged `source="real"` are sampled 2x more frequently than `source="sim_dr"`
when `real=1.0, dr=0.5`.
