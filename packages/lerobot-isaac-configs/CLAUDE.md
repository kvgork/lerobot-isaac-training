# lerobot-isaac-configs — Package Orientation

**Role:** Leaf package. Provides YAML config files and a minimal loader.
**Phase:** 0 (skeleton) — configs are stubs; real values added in Phase 2.
**Status:** Loader implemented and tested. All config files are stubs.

---

## What This Package Does

Central location for all training configuration YAML files used by the workspace.
The loader uses `importlib.resources` to resolve the `configs/` directory, ensuring
it works correctly under both editable installs (`pip install -e`) and wheel installs.
Supports env-var override (`LEROBOT_ISAAC_CONFIGS_DIR`) for CI and container usage.

---

## Public API

- `lerobot_isaac_configs.load_config(name: str) -> dict` — load `configs/{name}.yaml`
- `lerobot_isaac_configs.loader.list_configs() -> list[str]` — all available config names
- `lerobot_isaac_configs.loader.get_configs_dir() -> Path` — the configs directory path

Resolution order for `get_configs_dir()`:
1. `LEROBOT_ISAAC_CONFIGS_DIR` env var (if set and valid directory)
2. `importlib.resources.files("lerobot_isaac_configs") / "configs"`

---

## Internal Structure

| File | Role |
|------|------|
| `src/lerobot_isaac_configs/__init__.py` | Exports `load_config`, `__version__` |
| `src/lerobot_isaac_configs/loader.py` | `load_config()`, `list_configs()`, `get_configs_dir()` |
| `src/lerobot_isaac_configs/configs/` | YAML config files (stubs in Phase 0) |
| `tests/test_loader.py` | Tests for loader, env-var override, error cases |

---

## Coupling (plan §11.6)

- **No imports from any sibling package.** This is a true leaf.
- All sibling packages that need configs depend on THIS package.
- Dependency graph: `meta → configs`, `adapters → configs`, `autoresearch → adapters → configs`
- Safe for any sibling to depend on; no circular dep risk.

---

## Heavy Dependencies

None. Only `pyyaml` required.

---

## How to Extend

### Add a new config file

1. Add `src/lerobot_isaac_configs/configs/{name}.yaml` with a top-level mapping.
2. Verify: `python -c "from lerobot_isaac_configs import load_config; print(load_config('{name}'))"`
3. Add a test case in `tests/test_loader.py`.

### Use a custom configs directory

```bash
export LEROBOT_ISAAC_CONFIGS_DIR=/path/to/my_configs
```

Or at test time:
```python
import os
os.environ["LEROBOT_ISAAC_CONFIGS_DIR"] = str(tmp_path)
```

---

## Testing Notes

Tests in `tests/test_loader.py`:
- `test_load_all_configs` — loads every available config, checks it returns a dict
- `test_load_nonexistent_raises` — verifies `FileNotFoundError` on unknown config
- `test_list_configs` — verifies list is non-empty and contains known names
- `test_env_var_override` — `LEROBOT_ISAAC_CONFIGS_DIR` redirect works
- `test_env_var_nonexistent_raises` — missing dir raises `FileNotFoundError`
- `test_non_mapping_yaml_raises` — YAML list at top level raises `ValueError`

All pass without external deps (only pyyaml).

---

## Spinout Note

```bash
git subtree split -P packages/lerobot-isaac-configs -b spinout-configs
```

After spinout, this package becomes a standalone PyPI package. Other workspace packages
update their `pyproject.toml` to reference the PyPI version.
See `../../docs/ARCHITECTURE.md` (spinout section).

---

## Source-of-Truth Pointers

- Build plan: `${CLAUDE_CODE_ROOT}/plans/2026-05-06-lerobot-isaac-workspace-plan.md` — Phase 0
- Workspace ARCHITECTURE.md: `../../docs/ARCHITECTURE.md`
