# LeWorldModel Reference

> **Phase 0 placeholder.** Replace with full notes when Phase 2 LeWorldModel implementation begins.
> Cross-reference to existing skill documentation.

---

## Primary Reference

The `lerobot_world_model_bridge` skill in the `claude_code` repo is the authoritative source for LeWorldModel integration in this workspace:

**Skill path:** `/home/koen/tools/claude_code/skills/lerobot_world_model_bridge/`
**Skill docs:** `/home/koen/tools/claude_code/skills/lerobot_world_model_bridge/SKILL.md`

That skill already handles:
- Parquet → HDF5 conversion for LeWorldModel (`(96,96)` image preset)
- HDF5 schema documentation and schema-discovery script
- `wm_leworldmodel.py` stub wiring guidance

---

## What is HF LeWorldModel

LeWorldModel (Alibert et al. 2025) is Hugging Face's robot manipulation world model, part of the LeRobot ecosystem. It learns to predict future observations and rewards from action sequences, enabling model-based planning without a physical simulator.

- GitHub/HuggingFace: search `lerobot-world-model` on HuggingFace Hub
- Reference dataset to inspect schema: `quentinll/lewm-pusht`
- Vault note: `/home/koen/Documents/Vaults/Local/05-Wiki/entities/LeWorldModel.md`

---

## HDF5 Schema Warning

**The HF LeWorldModel HDF5 schema is undocumented.** The vault note flags this as a known risk.

Recommended approach (from Phase 2 plan):
1. Download `quentinll/lewm-pusht` from HuggingFace Hub
2. Inspect the HDF5 structure: `python -c "import h5py; f=h5py.File('pusht.hdf5'); print(list(f.keys()))"`
3. Diff with the output of `lerobot_world_model_bridge` skill's `le_world_model` preset
4. Adjust schema if needed before running training

The `lerobot_world_model_bridge` skill's `(96,96)` preset is the current best-effort schema.

---

## RTX 3080 Fit

- LeWorldModel at `(96,96)`: requires ~12 GB VRAM — does NOT fit RTX 3080 in standard mode
- Mitigation: enable gradient checkpointing + AMP + `batch_size=8`
- See vault note `World-Models-(Robot-Manipulation).md` for the full RTX-3080 fit table

---

## Key Metric

**`pred_loss`** — prediction loss on held-out transitions. Lower is better.
The `autoresearch` loop minimizes this (see `programs/leworldmodel.md`).

---

## Adapter Wiring

`packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/wm_leworldmodel.py` is the stub to implement. It should:
1. Accept `--config configs/wm_leworldmodel.yaml` and `--dataset_path <hdf5>`
2. Call the LeWorldModel training script as subprocess
3. Emit `pred_loss=<float>` on stdout at each eval step

Consult `lerobot-worldmodel-bridge` agent for conversion guidance:
```
Task(lerobot-worldmodel-bridge, {
  input_parquet: "datasets/so101_pick",
  output_hdf5: "outputs/hdf5/so101_pick_96.hdf5",
  target: "le_world_model",
  image_size: 96
})
```
