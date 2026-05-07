# MimicGen Reference

> **Phase 0 placeholder.** Replace with full notes when Phase 4b MimicGen implementation begins.
> Cross-reference to existing skill and vault documentation.

---

## Primary References

**Skill path:** `/home/koen/tools/claude_code/skills/lerobot_mimicgen_bridge/`
**Skill docs:** `/home/koen/tools/claude_code/skills/lerobot_mimicgen_bridge/SKILL.md`
**Agent:** `/home/koen/tools/claude_code/agents/workers/lerobot-sim-augmentation-agent.md`
**Vault note:** `/home/koen/Documents/Vaults/Local/05-Wiki/entities/MimicGen.md`

The `lerobot_mimicgen_bridge` skill handles Parquet ↔ MimicGen HDF5 conversion.
The `lerobot-sim-augmentation-agent` orchestrates the full MimicGen pipeline.

---

## What is MimicGen

MimicGen (Mandlekar et al. 2023) is a data augmentation system for robot manipulation. Given a small set of human demonstrations, it generates large-scale synthetic datasets by:
1. Placing robot and objects in novel configurations
2. Re-timing and interpolating the demonstration trajectories
3. Simulating the result in MuJoCo / robosuite

**Paper:** https://arxiv.org/abs/2310.17596
**GitHub:** https://github.com/NVlabs/mimicgen

---

## Integration Architecture

```
Real SO-101 Parquet demos
        │
        ▼ lerobot_mimicgen_bridge skill (Parquet → MimicGen HDF5)
        │
        ▼ MimicGen augmentation (runs in MuJoCo/robosuite internally)
        │
        ▼ lerobot_mimicgen_bridge skill (MimicGen HDF5 → Parquet)
        │
        ▼ merge_utilities.merge_datasets() with source="mimicgen"
        │
        ▼ merged Parquet → training
```

Note: MimicGen uses MuJoCo internally. This is NOT a replacement of our Isaac Lab stack — MimicGen is an isolated tool that produces Parquet output consumed normally by the training pipeline.

---

## Deferred Status

This path is **deferred** in the current build. `bridge_invocation.py` raises `NotImplementedError` by default.

To enable:
1. Install MimicGen + robosuite: `pip install mimicgen robosuite`
2. Set: `export LEROBOT_MIMICGEN_ENABLED=1`
3. Implement `packages/lerobot-isaac-synthetic/src/lerobot_isaac_synthetic/mimicgen/bridge_invocation.py`
4. Or use the agent: `Task(lerobot-sim-augmentation-agent, {...})`

---

## Known Integration Gap (from Vault)

The vault note `MimicGen.md` flags a known gap: LeRobot's Parquet schema does not have a 1:1 mapping to MimicGen's HDF5 schema for end-effector-space demos. The `lerobot_mimicgen_bridge` skill handles this conversion but requires manual schema alignment for new robot types. For SO-101, this alignment has NOT been done yet — treat this as a Phase 4b task.

---

## Source Tagging

All MimicGen-generated episodes in the merged dataset are tagged `source="mimicgen"` by `merge_utilities.py`. The policy training configs can weight sources differently:
```yaml
dataset_mixing:
  real: 1.0
  sim_dr: 0.5
  mimicgen: 0.3
```
