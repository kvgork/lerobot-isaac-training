# lerobot-isaac-meta — Package Orientation

**Role:** Umbrella package. Depends on all siblings. Exposes top-level CLI and workspace paths.
**Phase:** 0 (skeleton) — CLI stubs only; subcommands wired in Phases 2–4.
**Status:** Skeleton complete.

## Purpose

- Provide `lerobot-isaac` CLI entrypoint
- Provide canonical workspace path resolver (`workspace_paths.py`)
- Aggregate all sibling packages as a single installable unit

## Public API

- `lerobot_isaac_meta.cli:main` — registered as `lerobot-isaac` script
- `lerobot_isaac_meta.workspace_paths.WORKSPACE_ROOT` — resolves workspace root
- `lerobot_isaac_meta.workspace_paths.{DATASETS_DIR, OUTPUTS_DIR, CONFIGS_DIR, AGENT_STATE_DIR}`

## Dependencies

External: none (delegates to siblings)
Siblings: all 5 (`env`, `adapters`, `autoresearch`, `synthetic`, `configs`)

## Spinout Procedure

This package stays as the workspace orchestrator — it is NOT typically spun out.
The individual sibling packages are the ones that become standalone repos.

```bash
# Example: extract env package
git subtree split -P packages/lerobot-isaac-env -b spinout-env
```

## Key Files

- `src/lerobot_isaac_meta/cli.py` — argparse CLI with subcommand registry
- `src/lerobot_isaac_meta/workspace_paths.py` — resolves paths from env var or __file__
- `tests/test_paths.py` — verifies path resolution
