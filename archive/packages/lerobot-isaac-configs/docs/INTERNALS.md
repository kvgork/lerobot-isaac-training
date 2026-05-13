# lerobot-isaac-configs — Internals

---

## File Structure Walk-through

```
packages/lerobot-isaac-configs/
├── pyproject.toml           — dep: pyyaml>=6.0
├── pixi.toml
├── README.md / CLAUDE.md / docs/
├── src/
│   └── lerobot_isaac_configs/
│       ├── __init__.py              — exports load_config, __version__
│       ├── loader.py                — get_configs_dir(), load_config(), list_configs()
│       └── configs/                 — YAML config files
│           ├── policy_smolvla.yaml
│           ├── policy_act.yaml
│           ├── policy_diffusion.yaml
│           ├── wm_dreamerv3.yaml
│           ├── wm_leworldmodel.yaml
│           └── isaac_so101_pickplace.yaml
└── tests/
    └── test_loader.py               — loader tests including env-var override
```

---

## Key Design: `importlib.resources` Resolution

The `configs/` directory lives **inside** the package tree at
`src/lerobot_isaac_configs/configs/`. This is the preferred location for data files
in modern Python packaging:

- For editable installs: `importlib.resources.files("lerobot_isaac_configs") / "configs"`
  resolves to the source tree directly.
- For wheel installs: the same call resolves to the installed package tree.
- No `MANIFEST.in` or `package_data` needed because `hatchling` includes all files
  within the declared package directory automatically.

The `get_configs_dir()` function calls `str(configs_traversable)` to materialise the
`Traversable` to a real filesystem path. This works for both sdist and editable installs
but may not work for zip-based installs (which are rare in practice for this use case).

---

## `load_config()` Implementation

```python
def load_config(name: str) -> dict:
    configs_dir = get_configs_dir()
    config_path = configs_dir / f"{name}.yaml"
    if not config_path.exists():
        available = sorted(p.stem for p in configs_dir.glob("*.yaml"))
        raise FileNotFoundError(
            f"Config {name!r} not found at {config_path}. "
            f"Available configs: {available}"
        )
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(...)
    return data
```

Key design choices:
- `get_configs_dir()` is called on every invocation (not cached). This allows
  `LEROBOT_ISAAC_CONFIGS_DIR` changes to take effect without module reload.
- `yaml.safe_load` (not `yaml.load`) — prevents arbitrary code execution.
- Top-level mapping validation prevents accidental YAML lists.
- Error message includes available configs to aid discoverability.

---

## Phase 0 Config Stubs

All YAML files in Phase 0 are minimal stubs with placeholder values. Example
`wm_dreamerv3.yaml` (typical stub structure):

```yaml
# DreamerV3 world model config — Phase 0 stub
# Real values populated in Phase 2 after hardware testing
image_size: 64
batch_size: 16
lr: 3e-4
seed: 42
```

Phase 2 will replace these with validated hyperparameters from autoresearch runs.

---

## Soft-Import Strategy

No soft imports. This package only uses:
- Python stdlib (`importlib.resources`, `os`, `pathlib`)
- `pyyaml` (declared as hard dep)

---

## Test Architecture

`tests/test_loader.py` has 6+ tests:

- `test_load_all_configs` — iterates `list_configs()`, loads each, checks `isinstance(cfg, dict)`
- `test_load_nonexistent_raises` — `load_config("_missing_")` raises `FileNotFoundError`
- `test_list_configs` — result is a non-empty list containing `"wm_dreamerv3"`
- `test_get_configs_dir_returns_directory` — result is an existing directory
- `test_env_var_override` — sets `LEROBOT_ISAAC_CONFIGS_DIR` to `tmp_path`, creates
  a `.yaml` file there, loads it via `load_config()`
- `test_env_var_nonexistent_raises` — missing path raises `FileNotFoundError`
- `test_non_mapping_yaml_raises` — YAML with `[1, 2, 3]` at top level raises `ValueError`

Tests use `tmp_path` pytest fixture and `importlib.reload()` where needed.

---

## Known Limitations

1. **Config stubs** — all 6 YAML files are placeholders. Using them in a real training
   run will produce default/suboptimal hyperparameters.

2. **No schema validation** — `load_config()` returns a raw dict without checking that
   required keys are present. Adding a schema validator (e.g. `jsonschema`) is planned
   for Phase 2.

3. **No merging** — configs cannot inherit from or override each other. If multiple
   training stages need slight config variants, they must be separate files or the caller
   must merge the dicts manually.

4. **`importlib.resources` path materialisation** — the `str(configs_traversable)` call
   works for regular filesystem installs but would fail for zip-embedded packages. This
   is acceptable for the workspace use case.

---

## Future Work

| Item | Plan |
|------|------|
| Populate stub configs | Phase 2 — after autoresearch identifies best hyperparameters |
| Schema validation | Add `jsonschema` or `pydantic` validator to `load_config()` |
| Config inheritance | Allow `_base: other_config` key for DRY config management |
| Config versioning | Add `_version:` key and migration support |
