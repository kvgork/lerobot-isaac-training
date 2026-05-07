# lerobot-isaac-meta

Umbrella package for the LeRobot + Isaac Lab training workspace.

This package combines all sibling packages and exposes the top-level `lerobot-isaac` CLI.
It is the entry point for the workspace — install this and all dependencies come with it.

## Public API Surface

- `lerobot_isaac_meta.cli.main` — `lerobot-isaac` CLI entrypoint
- `lerobot_isaac_meta.workspace_paths` — canonical path resolver (WORKSPACE_ROOT, etc.)

## Dependencies

All sibling packages:
- `lerobot-isaac-env` (Phase 1)
- `lerobot-isaac-adapters` (Phase 2)
- `lerobot-isaac-autoresearch` (Phase 3)
- `lerobot-isaac-synthetic` (Phase 4)
- `lerobot-isaac-configs` (Phase 0 skeleton)

## Install

```bash
# From workspace root (uv workspace):
uv sync

# Standalone (after spinout):
pip install -e packages/lerobot-isaac-meta/
```

## CLI

```bash
lerobot-isaac --help
lerobot-isaac train --help      # Phase 2+
lerobot-isaac record --help     # Phase 2+
lerobot-isaac dr-replay --help  # Phase 4+
lerobot-isaac mimicgen-augment --help  # Phase 4b+
```

## Cross-Package Coupling

This package depends on all siblings. Siblings do NOT depend on this package.
Import direction: `meta → env, adapters, autoresearch, synthetic, configs`
