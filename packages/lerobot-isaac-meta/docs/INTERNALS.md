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
    ├── conftest.py            — sibling-pkg sys.path injection (monorepo only)
    ├── test_paths.py          — workspace_paths resolution tests
    ├── test_cli_record.py     — record subcommand wiring
    ├── test_cli_quality_filter.py — quality-filter subcommand
    ├── test_batch_config.py   — batch CLI config parsing
    └── test_batch_runner.py   — batch CLI runner
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
    "quality-filter":   (_cmd_quality_filter,   "filter low-quality episodes via SAL+TED"),
}
```

This dict is the sole registry for subcommands. `build_parser()` iterates it to
register `subparsers.add_parser(name, help=help_text)` entries. Adding a new
subcommand requires only adding one entry here plus a `_cmd_*` handler.

### `workspace_paths` module-level constants

All five path constants (`WORKSPACE_ROOT`, `DATASETS_DIR`, `OUTPUTS_DIR`,
`CONFIGS_DIR`, `AGENT_STATE_DIR`) are resolved at module import time by calling
`_resolve_workspace_root()`. They are `pathlib.Path` objects **or** ``None``
when the package is installed standalone outside any workspace.

`_resolve_workspace_root()` resolution order:

1. ``LEROBOT_ISAAC_WORKSPACE_ROOT`` env var (preferred).
2. ``LEROBOT_ISAAC_WORKSPACE`` env var (legacy alias).
3. Walk upward from CWD looking for a directory whose ``pixi.toml`` contains
   ``[workspace]`` or ``[tool.pixi.workspace]``, or whose ``pyproject.toml``
   contains ``[tool.uv.workspace]`` / ``[tool.pixi.workspace]``.
4. Walk upward from ``Path(__file__).parent`` with the same marker test.
5. ``None`` — no workspace; ``ensure_dirs()`` becomes a no-op and
   ``require_workspace_root()`` raises ``RuntimeError`` with guidance.

This marker-driven walk-up replaces the older ``Path(__file__).parents[4]``
hard-coded depth. The old approach silently produced wrong paths in
site-packages installs, virtualenvs, and standalone (post-spinout) layouts.

---

## Soft-Import Strategy

This package has **no soft imports**. It imports only:
- Python stdlib (`os`, `pathlib`, `argparse`, `sys`)
- Sibling packages are NOT imported at module load (cli stubs only print text)

The sibling packages themselves handle soft-imports of Isaac Lab, lerobot, etc.

---

## Test Architecture

Tests are split across multiple files. They use two complementary gating
strategies depending on the dependency at hand:

- **``importlib.util.find_spec`` skip** — for tests that need a sibling
  Python package importable (e.g. ``robot_data_recorder``,
  ``lerobot_isaac_adapters``). Portable across monorepo and standalone trees.
- **``@pytest.mark.requires_workspace_root``** — reserved for tests whose
  semantics are tied to the monorepo workspace layout itself (cross-package
  coupling, sibling source-tree probing). Auto-skipped via ``conftest.py``
  when ``_in_monorepo()`` returns false.

``tests/test_paths.py`` exercises ``workspace_paths``:

1. ``test_workspace_root_is_directory_or_none`` — sanity for either branch.
2. ``test_path_constants_are_absolute_when_set`` — absoluteness invariant.
3. ``test_path_constants_are_under_workspace_root`` — child-of-root invariant.
4. ``test_workspace_root_contains_marker_when_set`` — verifies discovery hit
   a real workspace.
5. ``test_env_var_override_primary`` / ``test_env_var_override_legacy``
   — both env-var names work.
6. ``test_env_var_nonexistent_raises`` — non-existent path raises ``FileNotFoundError``.
7. ``test_walk_up_discovery_finds_pixi_workspace_marker`` — discovery walks up.
8. ``test_walk_up_discovery_returns_none_in_unmarked_tree`` — fall-through.
9. ``test_require_workspace_root_raises_when_none`` — explicit guard.
10. ``test_ensure_dirs_creates_directories`` — idempotent dir creation.
11. ``test_ensure_dirs_is_noop_when_standalone`` — graceful standalone behavior.

Tests that use env-var overrides reload the module with ``importlib.reload``
to force re-evaluation of module-level constants. The ``finally`` blocks
restore the original env state and reload again to avoid test pollution.

---

## Known Limitations

1. **Subcommand stubs** — ``train``, ``dr-replay``, ``mimicgen-augment``
   print guidance text and return 0. Real delegation lands in Phases 2 / 4.
   ``record`` and ``quality-filter`` are fully wired.

2. **``CONFIGS_DIR`` is not created by ``ensure_dirs()``** — it points inside
   the installed ``lerobot-isaac-configs`` package, which is managed by
   pip/pixi, not by the runtime. If the configs package is not installed,
   ``CONFIGS_DIR`` will not exist.

3. **Module reload for env-var testing** — module-level path resolution means
   env-var changes after import require ``importlib.reload()``. This is
   intentional (fail-fast at import) but means runtime
   ``LEROBOT_ISAAC_WORKSPACE_ROOT`` changes don't take effect without reload.

4. **Standalone discovery edge case** — if a user runs from a deeply nested
   ``cwd`` *outside* the monorepo while the ``__file__`` for the installed
   package is *inside* it (e.g. editable install), discovery may pick the
   monorepo. This is correct behavior — set
   ``LEROBOT_ISAAC_WORKSPACE_ROOT`` to override.

---

## Future Un-stubbing Plan

| Phase | Change |
|-------|--------|
| Phase 2 | `_cmd_train` — import and call `lerobot_isaac_adapters.train.main` |
| Phase 4 | `_cmd_dr_replay` — call `lerobot_isaac_synthetic.isaac_dr.replay_runner.main` |
| Phase 4b | `_cmd_mimicgen_augment` — enable when MimicGen path is activated |

Each un-stubbing is a 2-5 line change inside the relevant `_cmd_*` function in `cli.py`.
