# ADR-0002: Pixi Workspace with Per-Package Dormant pixi.toml

**Status:** Accepted
**Date:** 2026-05-06
**Deciders:** Project team

---

## Context

This monorepo contains 6 Python packages with heterogeneous dependency profiles:

- Some packages only need pure-Python deps (pytest, pyyaml)
- Some need LeRobot + torch
- Some need DreamerV3 / sheeprl
- One needs HF LeWorldModel
- One needs Isaac Lab (GPU + full Omniverse runtime)

A naive single `requirements.txt` would pull in all deps for all packages, making the
default environment multi-GB and slow. We also want each package to be independently
publishable and spinnable out to a standalone repo without carrying monorepo-specific glue.

---

## Decision

Use **pixi** as the workspace dependency manager with the following structure:

```
pixi.toml             # workspace root — active, manages all environments
packages/
  lerobot-isaac-configs/
    pixi.toml         # per-package — dormant in monorepo, active post-spinout
    pyproject.toml
  ...
```

The root `pixi.toml` defines multiple environments via features:

| Environment | Features | Purpose |
|-------------|----------|---------|
| `default` | `dev` | Unit tests, lint, format |
| `train-policy` | `dev` + `lerobot` | LeRobot policy training |
| `train-dreamer` | `dev` + `lerobot` + `dreamerv3` | DreamerV3 |
| `train-lewm` | `dev` + `lerobot` + `leworldmodel` | HF LeWorldModel |
| `sim` | `dev` + `lerobot` + `isaaclab` | Isaac Lab (post-install) |
| `full` | all | All targets |

Each per-package `pixi.toml` is a **dormant** standalone config. It is ignored by the
workspace resolver but activates automatically when the package is spun out via
`git subtree split`.

---

## Rationale

### Why pixi over conda/pip/poetry

- **Conda + pip hybrid:** pixi manages both conda packages (e.g., `cudatoolkit`) and
  PyPI packages in a single lockfile. This is critical for Isaac Lab (CUDA) and
  torch (GPU build).
- **Lockfile reproducibility:** `pixi.lock` pins every transitive dep — conda and PyPI —
  ensuring reproducible environments across machines.
- **Feature-based environments:** pixi's feature system allows orthogonal opt-in deps
  without N×M environment explosion. Adding a new arch is one `[feature.newarch]` block.
- **Speed:** pixi resolves and installs in seconds vs. minutes for conda alone.

### Why dormant per-package pixi.toml

The per-package `pixi.toml` serves a dual purpose:

1. **Documentation:** it describes the package's self-contained deps for developers
   reading a single package in isolation.
2. **Spinout readiness:** when `git subtree split` extracts the package, the per-package
   `pixi.toml` becomes the active config immediately — no manual setup required.

If we only had the root `pixi.toml`, a spun-out package would have no dep spec.

### Why not pyproject.toml extras for environments

`pyproject.toml` extras can express optional deps but cannot manage conda packages,
cannot produce lockfiles, and require a separate venv per environment. pixi's feature
system is a strict superset.

---

## Consequences

**Positive:**
- Single command (`pixi install`) for the common case (tests + lint)
- Reproducible lockfile for all GPU deps
- Seamless spinout: each package carries its own standalone dep spec
- New architectures added in one TOML block

**Negative:**
- pixi is a relatively new tool; some team members may not know it
- Per-package `pixi.toml` can drift from root `pixi.toml` if not maintained
- `pixi.lock` is large and binary-like; diffs are hard to review

**Mitigation:**
- `CONTRIBUTING.md` explains the pixi workflow
- CI uses `prefix-dev/setup-pixi` action for consistency
- Per-package `pixi.toml` is kept intentionally minimal (no version pinning);
  pinning lives in the root lockfile

---

## Alternatives Considered

| Alternative | Why rejected |
|-------------|-------------|
| Poetry | No conda support; GPU deps (cudatoolkit) not manageable |
| Conda + pip manually | No lockfile; no feature/environment abstraction; slow |
| uv workspaces | No conda support; GPU dep management limited |
| Nix | Correct approach but team expertise missing; steep learning curve |
