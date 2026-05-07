# lerobot-isaac-meta

Umbrella package for the LeRobot + Isaac Lab training workspace.

This package combines all sibling packages and exposes the top-level `lerobot-isaac` CLI.
It is the entry point for the workspace — install this and all dependencies come with it.

---

## Purpose

`lerobot-isaac-meta` serves as the integration layer for the entire SO-101 training
workspace. It provides two things:

1. **`lerobot-isaac` CLI** — a top-level argparse-based command with subcommands
   (`train`, `record`, `dr-replay`, `mimicgen-augment`) that delegate to the
   appropriate sibling packages.
2. **`workspace_paths` module** — a canonical path resolver that provides absolute
   `Path` constants (`WORKSPACE_ROOT`, `DATASETS_DIR`, etc.) and supports environment
   variable override for CI and containerised deployments.

All subcommands are currently stubs that print "not yet wired" messages. Real
implementations are wired in during Phases 2 and 4.

---

## Status

**Phase 0 — scaffolding complete.** CLI stubs, path resolver, and tests are
implemented. No external runtime dependencies beyond the sibling packages.

| Component | Status |
|-----------|--------|
| `cli.py` | Skeleton — stubs only |
| `workspace_paths.py` | Implemented — fully functional |
| `train` subcommand | Stub — wired in Phase 2 |
| `record` subcommand | Stub — wired in Phase 2 |
| `dr-replay` subcommand | Stub — wired in Phase 4 |
| `mimicgen-augment` subcommand | Stub — deferred Phase 4b |

---

## Installation

### Monorepo mode (pixi)

```bash
# From workspace root:
pixi install
pixi run python -c "import lerobot_isaac_meta; print('ok')"
```

### Standalone mode (standalone pixi.toml)

```bash
cd packages/lerobot-isaac-meta
pixi install
```

### Direct pip install

```bash
# Editable install (recommended for development):
pip install -e packages/lerobot-isaac-meta/

# Or install all workspace packages at once:
pip install -e packages/lerobot-isaac-env/ \
            -e packages/lerobot-isaac-adapters/ \
            -e packages/lerobot-isaac-autoresearch/ \
            -e packages/lerobot-isaac-synthetic/ \
            -e packages/lerobot-isaac-configs/ \
            -e packages/lerobot-isaac-meta/
```

---

## Quick Example

```python
# This example requires no external deps — only the workspace packages.
from lerobot_isaac_meta.workspace_paths import WORKSPACE_ROOT, DATASETS_DIR, OUTPUTS_DIR

print(WORKSPACE_ROOT)   # /home/user/workspaces/lerobot-isaac-training
print(DATASETS_DIR)     # /home/user/workspaces/lerobot-isaac-training/datasets
print(OUTPUTS_DIR)      # /home/user/workspaces/lerobot-isaac-training/outputs

# Ensure workspace directories exist before a training run
from lerobot_isaac_meta.workspace_paths import ensure_dirs
ensure_dirs()           # creates datasets/, outputs/, .agent-state/ if absent

# Use a resolved path
dataset_path = DATASETS_DIR / "so101_pick_real_v1"
print(dataset_path.exists())  # True if data is present
```

```bash
# CLI smoke test — no dependencies required
lerobot-isaac --version
# lerobot-isaac 0.1.0 (Phase 0 scaffold)

lerobot-isaac --help
lerobot-isaac train --help
```

---

## Public API

- **`lerobot_isaac_meta.cli.main(argv=None) -> int`** — CLI entrypoint registered
  as the `lerobot-isaac` console script.
- **`lerobot_isaac_meta.cli.build_parser() -> argparse.ArgumentParser`** — returns
  the parser with all subcommands registered; useful for testing.
- **`lerobot_isaac_meta.workspace_paths.WORKSPACE_ROOT`** — `Path` to the workspace
  root; resolved from `LEROBOT_ISAAC_WORKSPACE` env var or `__file__`.
- **`lerobot_isaac_meta.workspace_paths.DATASETS_DIR`** — `WORKSPACE_ROOT/datasets`.
- **`lerobot_isaac_meta.workspace_paths.OUTPUTS_DIR`** — `WORKSPACE_ROOT/outputs`.
- **`lerobot_isaac_meta.workspace_paths.CONFIGS_DIR`** — path to configs inside
  `lerobot-isaac-configs` package.
- **`lerobot_isaac_meta.workspace_paths.AGENT_STATE_DIR`** — `WORKSPACE_ROOT/.agent-state`.
- **`lerobot_isaac_meta.workspace_paths.ensure_dirs() -> None`** — idempotent
  directory creation for gitignored runtime dirs.

---

## Dependencies

### Python dependencies (pyproject.toml)

All sibling packages are listed as dependencies; no third-party external packages
are required by this package itself.

```
lerobot-isaac-env
lerobot-isaac-adapters
lerobot-isaac-autoresearch
lerobot-isaac-synthetic
lerobot-isaac-configs
```

### Sibling package dependency graph

```
meta ──► env
     ──► adapters ──► configs
     ──► autoresearch ──► adapters
     ──► synthetic ──► (env, configs at runtime)
     ──► configs
```

Rule (plan §11.6): siblings do NOT import from `lerobot-isaac-meta`. The
import direction is always `meta → everything else`.

### Heavy/external dependencies

None directly. Heavy deps (Isaac Lab, lerobot, sheeprl) are soft-imported by the
sibling packages at call time and are not required for the meta package itself.

---

## Configuration

### Environment variable: `LEROBOT_ISAAC_WORKSPACE`

Override the workspace root directory (useful in Docker/CI):

```bash
export LEROBOT_ISAAC_WORKSPACE=/opt/workspace
python -c "from lerobot_isaac_meta.workspace_paths import WORKSPACE_ROOT; print(WORKSPACE_ROOT)"
# /opt/workspace
```

If this variable is set to a path that does not exist, `workspace_paths` raises
`FileNotFoundError` at import time.

### Subcommand arguments

Each subcommand accepts its own arguments (to be wired in later phases). Run
`lerobot-isaac <subcommand> --help` for the current stub output.

---

## Running Tests

```bash
# From the package directory:
cd packages/lerobot-isaac-meta
pytest tests/ -v

# From the workspace root:
pytest packages/lerobot-isaac-meta/tests/ -v
```

All tests pass without any external dependencies. The path tests use `tmp_path`
fixtures to verify env-var override and `ensure_dirs` behaviour.

---

## CLI Reference

```bash
lerobot-isaac --help
lerobot-isaac --version

# Subcommands (stubs until their phases are complete):
lerobot-isaac train --help          # Phase 2 — delegates to lerobot-isaac-adapters
lerobot-isaac record --help         # Phase 2 — invokes lerobot-data-collection-agent
lerobot-isaac dr-replay --help      # Phase 4 — delegates to lerobot-isaac-synthetic
lerobot-isaac mimicgen-augment --help  # Phase 4b (deferred) — MimicGen bridge
```

---

## Spinout

This package is the workspace orchestrator and is **not typically extracted** as a
standalone repo. Individual sibling packages are the ones that become standalone repos.

If extraction is needed:

```bash
# From workspace root:
git subtree split -P packages/lerobot-isaac-meta -b spinout-meta
git checkout spinout-meta
# Remove workspace deps from pyproject.toml; pin sibling-pkg versions on PyPI
git remote add origin git@github.com:user/lerobot-isaac-meta.git
git push -u origin main
```

See also: `../../docs/ARCHITECTURE.md` — spinout section.
