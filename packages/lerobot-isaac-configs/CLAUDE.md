# lerobot-isaac-configs — Package Orientation

**Role:** Leaf package. Provides YAML config files and a minimal loader.
**Phase:** 0 (skeleton) — configs are stubs; real values added in Phase 2.
**Status:** Skeleton complete.

## Purpose

Central location for all training configuration files. Being a leaf package
(no internal deps) means any package can safely depend on it.

## Public API

- `lerobot_isaac_configs.load_config(name: str) -> dict`
  Loads `configs/{name}.yaml` relative to the package root.
  Raises `FileNotFoundError` if the config does not exist.

## Dependencies

External only: `pyyaml`
Internal siblings: none

## Key Files

- `src/lerobot_isaac_configs/loader.py` — config resolver and YAML loader
- `configs/` — YAML config files (stubs in Phase 0)

## Spinout Procedure

Can be extracted to standalone repo for use across multiple robot projects:
```bash
git subtree split -P packages/lerobot-isaac-configs -b spinout-configs
```

## Adding a New Config

1. Add `configs/{name}.yaml`
2. Ensure the file is a valid YAML dict at the top level
3. Access via `load_config("{name}")`
