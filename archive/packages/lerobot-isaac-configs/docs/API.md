# lerobot-isaac-configs — Public API Reference

---

## Module: `lerobot_isaac_configs`

Top-level package. Exports `load_config` directly.

```python
import lerobot_isaac_configs
print(lerobot_isaac_configs.__version__)  # "0.1.0"

from lerobot_isaac_configs import load_config
cfg = load_config("wm_dreamerv3")
```

**Exports:** `load_config`, `__version__`

---

## Module: `lerobot_isaac_configs.loader`

Config resolver and YAML loader.

---

### `get_configs_dir() -> Path`

Return the path to the `configs/` directory.

**Resolution order:**

1. `LEROBOT_ISAAC_CONFIGS_DIR` environment variable (if set and valid directory).
2. `importlib.resources.files("lerobot_isaac_configs") / "configs"` — works for both
   editable installs (`pip install -e`) and wheel installs.

**Returns:** `Path` — absolute path to the configs directory.

**Raises:**
- `FileNotFoundError` — if `LEROBOT_ISAAC_CONFIGS_DIR` is set but the path does not
  exist or is not a directory.
- `FileNotFoundError` — if the package-internal `configs/` directory cannot be found
  (likely indicates a broken install).

**Example:**
```python
from lerobot_isaac_configs.loader import get_configs_dir

configs_dir = get_configs_dir()
print(configs_dir)  # /path/to/lerobot_isaac_configs/configs
print(list(configs_dir.glob("*.yaml")))
```

---

### `load_config(name: str) -> dict[str, Any]`

Load a YAML config by name.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Config name without `.yaml` extension (e.g. `"wm_dreamerv3"`). |

**Returns:** `dict[str, Any]` — parsed YAML content.

**Raises:**
- `FileNotFoundError` — if `configs/{name}.yaml` does not exist. Error message lists
  available configs.
- `ValueError` — if the YAML content is not a mapping (dict) at the top level.

**Example:**
```python
from lerobot_isaac_configs import load_config

cfg = load_config("wm_dreamerv3")
print(cfg.get("image_size"))   # 64  (Phase 0 stub value)

# Attempting to load a non-existent config:
try:
    load_config("nonexistent")
except FileNotFoundError as e:
    print(e)  # Config 'nonexistent' not found... Available configs: [...]
```

---

### `list_configs() -> list[str]`

Return the names of all available configs (without `.yaml` extension).

**Returns:** `list[str]` — sorted list of config names.

**Example:**
```python
from lerobot_isaac_configs.loader import list_configs

print(list_configs())
# ['isaac_so101_pickplace', 'policy_act', 'policy_diffusion', 'policy_smolvla',
#  'wm_dreamerv3', 'wm_leworldmodel']
```

---

## Available Config Files

All configs are Phase 0 stubs. Real values populated in Phase 2.

| Name | File | Training target |
|------|------|-----------------|
| `policy_smolvla` | `configs/policy_smolvla.yaml` | SmolVLA fine-tuning on SO-101 |
| `policy_act` | `configs/policy_act.yaml` | ACT from scratch |
| `policy_diffusion` | `configs/policy_diffusion.yaml` | Diffusion Policy |
| `wm_dreamerv3` | `configs/wm_dreamerv3.yaml` | DreamerV3 world model |
| `wm_leworldmodel` | `configs/wm_leworldmodel.yaml` | HF LeWorldModel |
| `isaac_so101_pickplace` | `configs/isaac_so101_pickplace.yaml` | Isaac Lab env per-stage settings |

---

## Environment Variable

`LEROBOT_ISAAC_CONFIGS_DIR` — override the configs directory:

```bash
export LEROBOT_ISAAC_CONFIGS_DIR=/opt/my_configs
```

---

## Cross-Package References

- `load_config()` is called by `../../lerobot-isaac-adapters/docs/API.md` — `train.main()`
  when `--config` is omitted
- The `CONFIGS_DIR` constant in `../../lerobot-isaac-meta/docs/API.md` points to the
  same directory resolved here
