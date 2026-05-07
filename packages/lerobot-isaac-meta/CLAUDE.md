# lerobot-isaac-meta — Package Orientation

**Role:** Umbrella package. Depends on all siblings. Exposes top-level CLI and workspace paths.
**Phase:** 0 (skeleton) — CLI stubs only; subcommands wired in Phases 2–4.
**Status:** Skeleton complete. No Isaac Lab or lerobot required at import time.

---

## What This Package Does

`lerobot-isaac-meta` provides two things:
1. The `lerobot-isaac` CLI entrypoint with subcommands (`train`, `record`, `dr-replay`,
   `mimicgen-augment`) that delegate to the appropriate sibling packages.
2. The `workspace_paths` module: a canonical resolver for workspace-level directory paths
   that supports env-var override (`LEROBOT_ISAAC_WORKSPACE`) for CI/container usage.

All CLI subcommands are stubs until their target phases are implemented.

---

## Public API Surface

- `lerobot_isaac_meta.cli.main(argv=None) -> int` — registered `lerobot-isaac` script
- `lerobot_isaac_meta.cli.build_parser() -> argparse.ArgumentParser` — full parser
- `lerobot_isaac_meta.workspace_paths.WORKSPACE_ROOT` — absolute Path to workspace root
- `lerobot_isaac_meta.workspace_paths.DATASETS_DIR` — `WORKSPACE_ROOT/datasets`
- `lerobot_isaac_meta.workspace_paths.OUTPUTS_DIR` — `WORKSPACE_ROOT/outputs`
- `lerobot_isaac_meta.workspace_paths.CONFIGS_DIR` — path inside lerobot-isaac-configs
- `lerobot_isaac_meta.workspace_paths.AGENT_STATE_DIR` — `WORKSPACE_ROOT/.agent-state`
- `lerobot_isaac_meta.workspace_paths.ensure_dirs() -> None` — idempotent dir creation

---

## Internal Structure

| File | Role |
|------|------|
| `src/lerobot_isaac_meta/__init__.py` | Exports `cli`, `workspace_paths`, `__version__` |
| `src/lerobot_isaac_meta/cli.py` | argparse CLI with subcommand registry (`_SUBCOMMANDS` dict) |
| `src/lerobot_isaac_meta/workspace_paths.py` | Path resolver; resolves from env var or `__file__` |
| `tests/test_paths.py` | 6 tests: resolution, absoluteness, env-var override, ensure_dirs |

The `_SUBCOMMANDS` registry in `cli.py` is a `dict[str, (handler, help_text)]`. Adding
a new subcommand means adding one entry to this dict plus a `_cmd_*` handler function.

---

## Coupling (plan §11.6)

- **Depends on:** all 5 sibling packages (`env`, `adapters`, `autoresearch`, `synthetic`, `configs`)
- **No sibling imports meta.** This is a strict one-way dependency.
- **workspace_paths** only uses Python stdlib (`os`, `pathlib`) — no sibling imports.
- **cli.py** does NOT import sibling packages at module load; subcommand handlers
  are stubs that only print messages. Real calls are added during Phases 2–4.

---

## Heavy Dependencies

None. All heavy deps (Isaac Lab, lerobot, sheeprl, transformers) are contained in
sibling packages and are soft-imported there. Meta itself has zero heavyweight deps.

---

## How to Extend

### Add a new CLI subcommand

1. Add a `_cmd_<name>(args: argparse.Namespace) -> int` function in `cli.py`.
2. Register it in `_SUBCOMMANDS`:
   ```python
   "my-cmd": (_cmd_my_cmd, "one-line description"),
   ```
3. Add a test in `tests/test_paths.py` (or a new test file) verifying the parser
   accepts the subcommand name.

### Wire a stub to a real implementation

Replace the `print(...)` + `return 0` body of the relevant `_cmd_*` function with
the actual import and call. Example for `train`:
```python
def _cmd_train(args):
    from lerobot_isaac_adapters.train import main as train_main
    return train_main(args)
```

---

## Testing Notes

All tests in `tests/test_paths.py` run **without** any external deps.

- `test_workspace_root_is_directory` — verifies `__file__`-based resolution
- `test_workspace_root_contains_pyproject` — workspace marker check
- `test_path_constants_are_absolute` — all 5 constants must be absolute paths
- `test_path_constants_are_under_workspace_root` — child-of-root invariant
- `test_env_var_override` — `LEROBOT_ISAAC_WORKSPACE` overrides resolution
- `test_env_var_nonexistent_raises` — missing path raises `FileNotFoundError`
- `test_ensure_dirs_creates_directories` — idempotent dir creation

---

## Spinout Note

This package is the workspace orchestrator and is not typically spun out.
Individual sibling packages are the ones extracted as standalone repos.
See `../../docs/ARCHITECTURE.md` (spinout section) for the exact commands.

---

## Source-of-Truth Pointers

- Build plan: `/home/koen/tools/claude_code/plans/2026-05-06-lerobot-isaac-workspace-plan.md` — Phase 0
- Workspace ARCHITECTURE.md: `../../docs/ARCHITECTURE.md`
- Autoresearch orchestrator: `~/.claude/agents/orchestrators/autoresearch-loop-orchestrator.md`
- LeRobot specialist: `~/.claude/agents/lerobot-specialist.md`
