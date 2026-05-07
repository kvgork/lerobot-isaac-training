# MimicGen Reference

**Paper:** https://arxiv.org/abs/2310.17596 (Mandlekar et al. 2023)
**GitHub:** https://github.com/NVlabs/mimicgen
**Robosuite (required dep):** https://github.com/ARISE-Initiative/robosuite

**Skill reference:** `/home/koen/tools/claude_code/skills/lerobot_mimicgen_bridge/`
**Agent reference:** `/home/koen/tools/claude_code/agents/workers/lerobot-sim-augmentation-agent.md`
**Vault note:** `/home/koen/Documents/Vaults/Local/05-Wiki/entities/MimicGen.md`
**Related workspace docs:** [synthetic-data.md](../internals/synthetic-data.md) | [ARCHITECTURE.md](../../ARCHITECTURE.md)

---

## What is MimicGen

MimicGen (Mandlekar et al. 2023) is a data augmentation system for robot manipulation.
Given a small set of human demonstrations (~10–50), it generates large-scale synthetic
datasets (1000s of episodes) by:

1. **Segmenting** demonstrations into sub-tasks (e.g. grasp phase, transport phase)
2. **Placing** robot and objects in novel configurations
3. **Re-timing** and interpolating the demonstration trajectories to fit new configurations
4. **Simulating** the result in MuJoCo / robosuite to verify physical plausibility

Output: a large set of valid demonstrations across diverse initial conditions.

---

## Integration Architecture

MimicGen runs in its native MuJoCo/robosuite environment. It is NOT an Isaac Lab tool.
The integration with this workspace is purely at the data level:

```
Real SO-101 Parquet (datasets/)
         |
         | lerobot_mimicgen_bridge skill
         | (Parquet -> MimicGen HDF5 with end-effector-space conversion)
         v
MimicGen HDF5 (tmp/)
         |
         | MimicGen + robosuite (runs internally in MuJoCo)
         | agent: lerobot-sim-augmentation-agent
         v
Augmented MimicGen HDF5
         |
         | lerobot_mimicgen_bridge skill
         | (MimicGen HDF5 -> Parquet with joint-space conversion)
         v
MimicGen Parquet (datasets/so101_mimicgen/)
  source = "mimicgen"
         |
         | merge_utilities.merge_datasets()
         v
Merged Parquet (datasets/so101_merged/)
```

MimicGen and Isaac Lab are parallel, independent tools. Isaac Lab is the primary sim
environment for this workspace; MimicGen is an augmentation tool that produces additional
training data via a completely separate code path.

---

## Current Status: DEFERRED

This path is deferred. `bridge_invocation.py` raises `NotImplementedError` by default.

**Reason for deferral:**
1. Joint-to-end-effector-space conversion for SO-101 requires calibrated kinematics
2. The `lerobot_mimicgen_bridge` skill requires robot-specific schema alignment for SO-101
3. DR replay (Isaac Lab path) is faster to set up and likely sufficient for early stages

**When to enable:**
- When DR replay alone is insufficient (e.g. policy fails to generalize to novel initial configs)
- When you have ~10 high-quality real demonstrations to seed MimicGen
- When SO-101 kinematic model is calibrated for end-effector-space actions

---

## How to Enable (Phase 4b)

```bash
# Step 1: Install dependencies
pip install mimicgen robosuite

# Step 2: Enable the path
export LEROBOT_MIMICGEN_ENABLED=1

# Step 3: Calibrate SO-101 kinematic model
# (fill in kinematic model in lerobot_mimicgen_bridge skill config)

# Step 4: Use the agent
# Task(lerobot-sim-augmentation-agent, {
#   source_dataset: "datasets/so101_pick_v1_filtered",
#   output_path: "datasets/so101_mimicgen",
#   num_demonstrations: 100,
#   task: "pick_and_place"
# })
```

The `lerobot-sim-augmentation-agent` orchestrates the full pipeline. Do not invoke
`bridge_invocation.py` directly — use the agent for error handling and progress tracking.

---

## Known Integration Gap (from Vault)

The vault note `MimicGen.md` flags a known gap:
- LeRobot's Parquet schema uses **joint-space actions** (radians)
- MimicGen's HDF5 schema uses **end-effector-space actions** (x,y,z + quaternion)
- The `lerobot_mimicgen_bridge` skill handles the conversion but requires:
  - A robot-specific kinematic model (forward/inverse kinematics)
  - Calibration of the SO-101's kinematic chain

This gap has NOT been resolved for SO-101. It is the primary blocker for Phase 4b.

---

## MimicGen vs Isaac Lab DR: When to Use Which

| Criterion | Isaac Lab DR | MimicGen |
|-----------|-------------|---------|
| Setup effort | Low (env already exists) | High (kinematic calibration needed) |
| Diversity type | Visual + physics variation | Spatial configuration variation |
| Trajectory validity | High (replays real actions) | Variable (retiming can fail) |
| Demo requirement | Any dataset | 10–50 high-quality demos |
| Isaac Lab required | Yes | No (uses MuJoCo) |
| Episode count | Real count × `num_augmentations` | Up to 1000× real count |
| Current status | Scaffolded (Phase 4a) | Deferred (Phase 4b stub) |

Use Isaac Lab DR for early stages (visual generalization).
Use MimicGen when you need spatial diversity that DR cannot provide.

---

## Source Tagging

All MimicGen-generated episodes in the merged dataset are tagged `source="mimicgen"`.
Training configs can weight sources differently:
```yaml
dataset_mixing:
  real: 1.0
  sim_dr: 0.5
  mimicgen: 0.3
```

The `real` source is weighted highest because real demos are the ground truth.
`sim_dr` is intermediate (Isaac Lab physics is close to real).
`mimicgen` is lowest (MuJoCo physics and retimed trajectories are less faithful).
