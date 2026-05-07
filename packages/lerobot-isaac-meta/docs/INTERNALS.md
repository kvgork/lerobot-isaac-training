# lerobot-isaac-meta — Internals

How the package works internally.

---

## File Structure Walk-through

```
packages/lerobot-isaac-meta/
├── pyproject.toml           — package metadata; lists all 5 siblings as deps
├── pixi.toml                — pixi environment (Python 3.12, pytest)
├── README.md                — user-facing documentation
├── CLAUDE.md                — AI agent orientation
├── docs/
│   ├── API.md               — full public API reference
│   ├── EXAMPLES.md          — usage examples
│   └── INTERNALS.md         — this file
├── src/
│   └── lerobot_isaac_meta/
│       ├── __init__.py      — exports cli, workspace_paths, __version__
│       ├── cli.py           — argparse entrypoint + subcommand registry
│       └── workspace_paths.py — canonical path constants + ensure_dirs()
└── tests/
    └── test_paths.py        — 7 tests for workspace_paths
```

---

## Key Data Structures

### `cli._SUBCOMMANDS`

```python
_SUBCOMMANDS: dict[str, tuple[callable, str]] = {
    "train":            (_cmd_train,            "train a policy or world model (Phase 2+)"),
    "record":           (_cmd_record,           "record SO-101 teleop data (Phase 2+)"),
    "dr-replay":        (_cmd_dr_replay,        "replay with Isaac Lab domain randomization (Phase 4+)"),
    "mimicgen-augment": (_cmd_mimicgen_augment, "augment dataset via MimicGen (Phase 4b, deferred)"),
}
```

This dict is the sole registry for subcommands. `build_parser()` iterates it to
register `subparsers.add_parser(name, help=help_text)` entries. Adding a new
subcommand requires only adding one entry here plus a `_cmd_*` handler.

### `workspace_paths` module-level constants

All five path constants (`WORKSPACE_ROOT`, `DATASETS_DIR`, `OUTPUTS_DIR`,
`CONFIGS_DIR`, `AGENT_STATE_DIR`) are resolved at module import time by calling
`_resolve_workspace_root()`. They are plain `pathlib.Path` objects.

`_resolve_workspace_root()` checks `os.environ.get("LEROBOT_ISAAC_WORKSPACE")`
first; on miss it walks 4 levels up from `workspace_paths.py`:

```
workspace_paths.py               — level 0
  src/lerobot_isaac_meta/        — level 0 (same file)
    .parent  (lerobot_isaac_meta) — level 1
    .parent  (src)               — level 2
    .parent  (lerobot-isaac-meta)— level 3
    .parent  (packages)          — level 4
    .parent  (workspace root)    — level 5 ... wait
```

Actually the path is:
`packages/lerobot-isaac-meta/src/lerobot_isaac_meta/workspace_paths.py`

`Path(__file__).parents[4]` = workspace root.

---

## Soft-Import Strategy

This package has **no soft imports**. It imports only:
- Python stdlib (`os`, `pathlib`, `argparse`, `sys`)
- Sibling packages are NOT imported at module load (cli stubs only print text)

The sibling packages themselves handle soft-imports of Isaac Lab, lerobot, etc.

---

## Test Architecture

`tests/test_paths.py` has 7 tests, all using pytest fixtures (`tmp_path`):

1. **`test_workspace_root_is_directory`** — basic sanity: `WORKSPACE_ROOT.is_dir()`
2. **`test_workspace_root_contains_pyproject`** — checks workspace marker exists
3. **`test_path_constants_are_absolute`** — all 5 constants must be absolute
4. **`test_path_constants_are_under_workspace_root`** — `DATASETS_DIR`, `OUTPUTS_DIR`,
   `AGENT_STATE_DIR` must start with `WORKSPACE_ROOT`
5. **`test_env_var_override`** — sets `LEROBOT_ISAAC_WORKSPACE`, reloads module,
   checks `WORKSPACE_ROOT` matches the override
6. **`test_env_var_nonexistent_raises`** — non-existent path raises `FileNotFoundError`
7. **`test_ensure_dirs_creates_directories`** — verifies `datasets/`, `outputs/`,
   `.agent-state/` are created by `ensure_dirs()`

Tests 5–7 use `importlib.reload()` to force re-evaluation of module-level constants.
The `finally` blocks restore the original env var and reload the module again to
avoid test pollution.

---

## Known Limitations

1. **Subcommands are stubs** — all 4 subcommands print guidance text and return 0.
   Real delegation to sibling packages is added during Phases 2 and 4.

2. **`CONFIGS_DIR` is not created by `ensure_dirs()`** — it points inside the
   installed `lerobot-isaac-configs` package, which is managed by pip/pixi, not
   by the runtime. If the configs package is not installed, `CONFIGS_DIR` will not
   exist.

3. **Module reload for env-var testing** — the module-level path resolution means
   env var changes after import require `importlib.reload()`. This is intentional
   (fail-fast at import) but means runtime `LEROBOT_ISAAC_WORKSPACE` changes don't
   take effect without reload.

---

## Future Un-stubbing Plan

| Phase | Change |
|-------|--------|
| Phase 2 | `_cmd_train` — import and call `lerobot_isaac_adapters.train.main` |
| Phase 2 | `_cmd_record` — invoke `lerobot-data-collection-agent` |
| Phase 4 | `_cmd_dr_replay` — call `lerobot_isaac_synthetic.isaac_dr.replay_runner.main` |
| Phase 4b | `_cmd_mimicgen_augment` — enable when MimicGen path is activated |

Each un-stubbing is a 2-5 line change inside the relevant `_cmd_*` function in `cli.py`.
