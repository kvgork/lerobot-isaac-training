# lerobot-isaac-configs

Shared YAML configuration files for all `lerobot-isaac-*` packages.

This is a **leaf package** — it has no internal cross-package dependencies.
Any sibling package may depend on it; it depends on nothing internal.

## Contents

`configs/` directory holds one YAML file per training target:

| Config file | Target | Phase |
|-------------|--------|-------|
| `policy_smolvla.yaml` | SmolVLA fine-tuning | 2 |
| `policy_act.yaml` | ACT from scratch | 2 |
| `policy_diffusion.yaml` | Diffusion Policy | 2 |
| `wm_dreamerv3.yaml` | DreamerV3 world model | 2 |
| `wm_leworldmodel.yaml` | HF LeWorldModel | 2 |
| `isaac_so101_pickplace.yaml` | Isaac Lab env per-stage settings | 1/2 |

All configs are stubs in Phase 0 — real values added in Phase 2.

## Public API

```python
from lerobot_isaac_configs import load_config

cfg = load_config("wm_dreamerv3")  # loads configs/wm_dreamerv3.yaml
print(cfg["image_size"])  # 64
```

## Cross-Package Coupling

This package is a dependency of `lerobot-isaac-meta` and any package that needs configs.
It does NOT import from any sibling package.
