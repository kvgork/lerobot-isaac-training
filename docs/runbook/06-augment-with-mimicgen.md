# Runbook 06: Augment with MimicGen (Deferred Path)

**Prerequisites:** Dataset collected (Runbook 02), MimicGen + robosuite installed
**[DEFERRED — Phase 4b stub only. `bridge_invocation.py` raises NotImplementedError by default.]**
**Expected outcome (when implemented):** MimicGen-augmented Parquet in `datasets/`

---

## Status

This runbook documents the intended workflow for when Phase 4b is implemented.
Current state: `packages/lerobot-isaac-synthetic/src/lerobot_isaac_synthetic/mimicgen/bridge_invocation.py` is a stub that raises `NotImplementedError` unless `LEROBOT_MIMICGEN_ENABLED=1`.

MimicGen is the **secondary** augmentation path. Prefer DR replay (Runbook 05) first.

---

## Prerequisites for Enabling MimicGen

```bash
# 1. Install MimicGen and robosuite:
pip install mimicgen robosuite

# 2. Enable the bridge:
export LEROBOT_MIMICGEN_ENABLED=1

# 3. Implement bridge_invocation.py:
# Open: packages/lerobot-isaac-synthetic/src/lerobot_isaac_synthetic/mimicgen/bridge_invocation.py
# Follow the docstring in run_mimicgen() to wire the real call
```

---

## Architecture

```
Real SO-101 Parquet episodes
        │
        ▼ lerobot_mimicgen_bridge skill
        │   Parquet → MimicGen HDF5 (robodemo format)
        │   Skill: /home/koen/tools/claude_code/skills/lerobot_mimicgen_bridge/
        │
        ▼ MimicGen augmentation (runs in MuJoCo/robosuite)
        │   Input: N real demos → Output: M synthetic demos (M >> N)
        │
        ▼ lerobot_mimicgen_bridge skill
        │   MimicGen HDF5 → LeRobot Parquet
        │
        ▼ merge_utilities.merge_datasets() with source="mimicgen"
```

Note: MimicGen uses MuJoCo internally — this is NOT a conflict with our Isaac Lab stack. MimicGen is an isolated tool producing Parquet output.

---

## Step 1: Convert Real Episodes to MimicGen Format

```bash
# Via lerobot_mimicgen_bridge skill:
# (auto-invoked by lerobot-sim-augmentation-agent)
#
# Or manually (after implementing):
# python -m lerobot_mimicgen_bridge.convert \
#   --input datasets/so101_pick_v1_filtered \
#   --output /tmp/so101_mimicgen_input.hdf5 \
#   --robot so101
```

**Known gap:** SO-101 schema alignment with MimicGen HDF5 has not been validated. Consult vault note `MimicGen.md` and `lerobot_mimicgen_bridge` SKILL.md before implementing.

---

## Step 2: Run MimicGen Augmentation

**Preferred path — use the agent:**

```bash
Task(lerobot-sim-augmentation-agent, {
  source_dataset: "datasets/so101_pick_v1_filtered",
  output_path: "datasets/so101_mimicgen_v1",
  num_demonstrations: 200,
  robot: "so101",
  task: "pick"
})
```

The agent orchestrates the full pipeline including conversion, MimicGen execution, and back-conversion.

**Or via bridge_invocation stub (after Phase 4b impl):**

```bash
export LEROBOT_MIMICGEN_ENABLED=1
python -m lerobot_isaac_synthetic.mimicgen.bridge_invocation \
  --source_dataset_path datasets/so101_pick_v1_filtered \
  --output_path datasets/so101_mimicgen_v1 \
  --num_demonstrations 200
```

---

## Step 3: Merge with Real and DR Datasets

```python
from lerobot_isaac_synthetic.merge_utilities import merge_datasets

# Merge all three sources:
merge_datasets(
    real_path="datasets/so101_pick_v1_filtered",
    dr_path="datasets/so101_pick_dr_v1",      # from Runbook 05
    mimicgen_path="datasets/so101_mimicgen_v1",
    output_path="datasets/so101_full_v1",
    real_weight=1.0,
    dr_weight=0.5,
    mimicgen_weight=0.3
)
```

Source tagging: `real_weight` applies to `source="real"`, `dr_weight` to `source="sim_dr"`, `mimicgen_weight` to `source="mimicgen"`.

---

## Step 4: Train with Full Dataset

```bash
lerobot-isaac-train \
  --target_arch smolvla \
  --config packages/lerobot-isaac-configs/configs/policy_smolvla.yaml \
  --dataset_path datasets/so101_full_v1 \
  --output_dir outputs/smolvla_full_run1
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `NotImplementedError` | Set `LEROBOT_MIMICGEN_ENABLED=1` AND implement bridge_invocation.py |
| MimicGen schema mismatch | Consult `lerobot_mimicgen_bridge` SKILL.md + vault MimicGen.md |
| robosuite not found | `pip install robosuite` |
| MimicGen generation fails | Check robot model compatibility with robosuite |
| SO-101 not in robosuite | May need custom robot model — see MimicGen docs |

---

## Reference Docs

- `docs/research/mimicgen-reference.md` — architecture + known gaps
- Skill: `/home/koen/tools/claude_code/skills/lerobot_mimicgen_bridge/SKILL.md`
- Agent: `/home/koen/tools/claude_code/agents/workers/lerobot-sim-augmentation-agent.md`
- Vault: `/home/koen/Documents/Vaults/Local/05-Wiki/entities/MimicGen.md`
- MimicGen paper: https://arxiv.org/abs/2310.17596
