# lerobot-isaac-configs — Usage Examples

All examples require only `pyyaml` (installed with the package). No external ML deps.

---

## Example 1 — Import and verify

```python
import lerobot_isaac_configs
print(lerobot_isaac_configs.__version__)  # 0.1.0
print("import ok")
```

Expected output:
```
0.1.0
import ok
```

---

## Example 2 — Load a config by name

```python
from lerobot_isaac_configs import load_config

cfg = load_config("wm_dreamerv3")
print(type(cfg))  # <class 'dict'>
print(cfg)        # {'image_size': 64, ...} (stub values)
```

---

## Example 3 — List all available configs

```python
from lerobot_isaac_configs.loader import list_configs

names = list_configs()
print(names)
```

Expected output:
```
['isaac_so101_pickplace', 'policy_act', 'policy_diffusion', 'policy_smolvla',
 'wm_dreamerv3', 'wm_leworldmodel']
```

---

## Example 4 — Handle missing config gracefully

```python
from lerobot_isaac_configs import load_config

try:
    cfg = load_config("nonexistent_config")
except FileNotFoundError as e:
    print(f"Not found: {e}")
```

Expected output:
```
Not found: Config 'nonexistent_config' not found at .../configs/nonexistent_config.yaml.
Available configs: ['isaac_so101_pickplace', 'policy_act', ...]
```

---

## Example 5 — Override configs directory with env var

```python
import os
import tempfile
import yaml
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    # Write a custom config
    (tmp_path / "my_custom.yaml").write_text(
        yaml.dump({"batch_size": 16, "lr": 1e-3})
    )

    os.environ["LEROBOT_ISAAC_CONFIGS_DIR"] = str(tmp_path)

    import importlib
    import lerobot_isaac_configs.loader as loader_mod
    importlib.reload(loader_mod)  # force re-resolution

    # load_config uses get_configs_dir() deferred — no reload needed
    from lerobot_isaac_configs.loader import load_config as lc
    cfg = lc("my_custom")
    print(cfg)  # {'batch_size': 16, 'lr': 0.001}

    del os.environ["LEROBOT_ISAAC_CONFIGS_DIR"]
```

---

## Example 6 — Use configs in a training script

Shows how `lerobot-isaac-adapters` uses this package (illustrative).

```python
from lerobot_isaac_configs import load_config

def get_training_config(target_arch: str) -> dict:
    config_map = {
        "smolvla": "policy_smolvla",
        "act": "policy_act",
        "diffusion": "policy_diffusion",
        "dreamerv3": "wm_dreamerv3",
        "le_world_model": "wm_leworldmodel",
    }
    name = config_map.get(target_arch)
    if name is None:
        raise ValueError(f"Unknown arch: {target_arch}")
    return load_config(name)

cfg = get_training_config("smolvla")
print(f"Using config keys: {list(cfg.keys())}")
```

---

## Example 7 — Get configs directory path

```python
from lerobot_isaac_configs.loader import get_configs_dir

configs_dir = get_configs_dir()
print(configs_dir)
# /path/to/site-packages/lerobot_isaac_configs/configs

# List all YAML files
for f in sorted(configs_dir.glob("*.yaml")):
    print(f.name)
```

Expected output:
```
isaac_so101_pickplace.yaml
policy_act.yaml
policy_diffusion.yaml
policy_smolvla.yaml
wm_dreamerv3.yaml
wm_leworldmodel.yaml
```
