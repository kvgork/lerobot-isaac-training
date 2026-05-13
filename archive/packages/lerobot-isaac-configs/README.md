# lerobot-isaac-configs

Shared YAML configuration files for all `lerobot-isaac-*` packages.

This is a **leaf package** — it has no internal cross-package dependencies.
Any sibling package may depend on it; it depends on nothing internal.

---

## Purpose

Central location for all training configuration files. Config files are stored in
`src/lerobot_isaac_configs/configs/` so they are accessible via `importlib.resources`
regardless of install method (editable, wheel, or zip). The `load_config()` function
provides a simple name-based lookup with env-var override for CI/containers.

---

## Status

**Phase 0 — skeleton complete.** All config files are stubs with placeholder values.
Real hyperparameter values are populated in Phase 2 after hardware testing.

| Config | Status |
|--------|--------|
| `policy_smolvla.yaml` | Stub — Phase 2 |
| `policy_act.yaml` | Stub — Phase 2 |
| `policy_diffusion.yaml` | Stub — Phase 2 |
| `wm_dreamerv3.yaml` | Stub — Phase 2 |
| `wm_leworldmodel.yaml` | Stub — Phase 2 |
| `isaac_so101_pickplace.yaml` | Stub — Phase 1/2 |
| `loader.py` | Implemented |

---

## Installation

### Monorepo mode (pixi)

```bash
pixi install   # from workspace root
```

### Standalone mode

```bash
cd packages/lerobot-isaac-configs
pixi install
```

### Direct pip install

```bash
pip install -e packages/lerobot-isaac-configs/
```

---

## Quick Example

```python
from lerobot_isaac_configs import load_config

# Load a config by name (without .yaml extension)
cfg = load_config("wm_dreamerv3")
print(cfg)           # dict with YAML content

cfg = load_config("policy_smolvla")
print(cfg.get("batch_size", "not set"))
```

```python
# List all available configs
from lerobot_isaac_configs.loader import list_configs

names = list_configs()
print(names)
# ['isaac_so101_pickplace', 'policy_act', 'policy_diffusion', 'policy_smolvla',
#  'wm_dreamerv3', 'wm_leworldmodel']
```

---

## Public API

- **`load_config(name: str) -> dict`** — load a YAML config by name (without `.yaml`).
  Raises `FileNotFoundError` if the config does not exist.
  Raises `ValueError` if the YAML is not a top-level mapping.
- **`list_configs() -> list[str]`** — return names of all available configs.
- **`get_configs_dir() -> Path`** — return the path to the `configs/` directory.

---

## Config Files

| Config file | Training target | Notes |
|-------------|-----------------|-------|
| `policy_smolvla.yaml` | SmolVLA fine-tuning | Phase 2 |
| `policy_act.yaml` | ACT from scratch | Phase 2 |
| `policy_diffusion.yaml` | Diffusion Policy | Phase 2 |
| `wm_dreamerv3.yaml` | DreamerV3 world model | Phase 2 |
| `wm_leworldmodel.yaml` | HF LeWorldModel | Phase 2 |
| `isaac_so101_pickplace.yaml` | Isaac Lab env per-stage settings | Phase 1/2 |

All are stubs in Phase 0. Real values are added in Phase 2.

---

## Dependencies

### Python (pyproject.toml)

```
pyyaml>=6.0
```

### Sibling package dependencies

None. This is a leaf package with no internal dependencies.

### Heavy/external dependencies

None. Only `pyyaml` required.

---

## Configuration

### Environment variable: `LEROBOT_ISAAC_CONFIGS_DIR`

Override the configs directory (useful in Docker/CI or for custom config sets):

```bash
export LEROBOT_ISAAC_CONFIGS_DIR=/opt/custom_configs
python -c "from lerobot_isaac_configs import load_config; print(load_config('wm_dreamerv3'))"
```

If set to a non-existent path, `get_configs_dir()` raises `FileNotFoundError`.

**Resolution order:**
1. `LEROBOT_ISAAC_CONFIGS_DIR` env var (if set and valid)
2. `importlib.resources.files("lerobot_isaac_configs") / "configs"` — works for
   both editable installs and wheel installs

---

## Adding a New Config

1. Add `src/lerobot_isaac_configs/configs/{name}.yaml`
2. Ensure the file is a valid YAML mapping at the top level (not a list or scalar)
3. Access via `load_config("{name}")`
4. Add a test in `tests/test_loader.py` verifying the config loads and has expected keys

---

## Running Tests

```bash
cd packages/lerobot-isaac-configs
pytest tests/ -v
```

All tests pass without any external dependencies.

---

## Spinout

Can be extracted as a standalone repo for use across multiple robot projects:

```bash
git subtree split -P packages/lerobot-isaac-configs -b spinout-configs
git checkout spinout-configs
git remote add origin git@github.com:user/lerobot-isaac-configs.git
git push -u origin main
```

See also: `../../docs/ARCHITECTURE.md` — spinout section.
