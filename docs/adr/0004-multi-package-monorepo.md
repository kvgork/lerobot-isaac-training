# ADR-0004: Multi-Package Monorepo (6 Packages)

**Status:** Accepted
**Date:** 2026-05-06
**Deciders:** Project team

---

## Context

The LeRobot + Isaac Lab training system spans several distinct functional areas:

- Environment simulation (Isaac Lab MDP, physics, rendering)
- Training adapters (dispatch to LeRobot / DreamerV3 / LeWorldModel backends)
- Configuration management (YAML configs, schema validation)
- Synthetic data generation (DR replay, Parquet writer)
- Autoresearch integration (program.md, metric history, plateau detection)
- Cross-cutting utilities and shared types

Early prototyping used a flat `src/` layout with a single `pyproject.toml`. This became
unwieldy as the heavy-dep matrix grew: any change to the Isaac Lab env layer required
testing all training code, even when the two are orthogonal.

---

## Decision

Structure the workspace as a **multi-package monorepo** with 6 packages under `packages/`:

| Package | Responsibility | Heavy deps |
|---------|---------------|------------|
| `lerobot-isaac-meta` | Shared types, protocols, utilities | None |
| `lerobot-isaac-configs` | YAML config schemas, validation, loaders | None (pydantic) |
| `lerobot-isaac-env` | Isaac Lab MDP environment + USD wiring | `isaaclab` |
| `lerobot-isaac-adapters` | Modular train.py, metric extractors, backends | `lerobot`, `dreamerv3`, `lewm` |
| `lerobot-isaac-synthetic` | DR replay loop, Parquet writer, MimicGen bridge | `isaaclab` |
| `lerobot-isaac-autoresearch` | Autoresearch ML loop integration | `lerobot` |

Coupling flows **upward only**: `meta` ← `configs` ← (`env`, `adapters`) ← `synthetic`/`autoresearch`.
No package imports a package above it in this hierarchy. Circular imports are a lint error.

---

## Rationale

### Why 6 packages instead of a monolith

**Dependency isolation:** `lerobot-isaac-configs` has no heavy deps and can be installed
in a CI job in seconds. A monolith would require Isaac Lab in every job.

**Parallel development:** the env layer and the adapter layer can be developed and tested
independently. A broken Isaac Lab install does not block work on training adapters.

**Faster CI:** per-package matrix jobs run in parallel (see ADR-0001 CI design). A monolith
would serialize everything.

**Cognitive scope:** each package has a clear, bounded responsibility. Reviewers can
understand a PR touching only `lerobot-isaac-configs` without loading the entire system.

### Coupling rules

The one-way dependency graph is enforced by:

1. `pyproject.toml` `[project.dependencies]` — only lists packages strictly below in the
   hierarchy. Cross-level imports are not declared as deps.
2. Ruff `I` rules — import ordering violations surface if a package tries to import a
   peer at the same level (soft signal, not hard block).
3. Code review convention — PRs that add a cross-level dep require explicit justification.

`lerobot-isaac-meta` has no deps on other workspace packages and serves as the shared
vocabulary (protocols, dataclasses, type aliases).

### Spinout strategy

Each package is designed to be spun out to a standalone repo via:

```bash
git subtree split --prefix=packages/<pkg> -b spinout/<pkg>
git push <new-remote> spinout/<pkg>:main
```

For this to work:
- Each package has its own `pyproject.toml`, `pixi.toml`, `README.md`, and `tests/`
- The per-package `pixi.toml` is a dormant standalone dep spec (see ADR-0002)
- No package contains hardcoded absolute paths to sibling packages

The smoke test (`scripts/spinout_smoke_test.sh`) verifies spinout readiness for any
target package.

---

## Consequences

**Positive:**
- Fine-grained dependency control
- Parallel CI jobs (6× speedup vs serial)
- Clean spinout path to standalone repos
- Independent versioning and publishing possible

**Negative:**
- 6 `pyproject.toml` files to maintain
- Cross-package refactoring requires touching multiple packages
- Developers must understand the coupling hierarchy

**Mitigation:**
- `lerobot-isaac-meta` as shared vocabulary reduces cross-package coupling
- Root `pixi.toml` workspace config manages all 6 packages in one `pixi install`
- ADR-0003 soft-import discipline keeps the default env lightweight for all packages

---

## Alternatives Considered

| Alternative | Why rejected |
|-------------|-------------|
| Single monolith package | Heavy-dep explosion; slow CI; cognitive overload |
| 2-package split (env + rest) | Still mixes training adapters with autoresearch; insufficient isolation |
| Separate git repos from day 1 | Too early to commit to boundaries; monorepo allows easy refactoring |
| src-layout single package with extras | Extras don't manage heavy GPU deps (conda); no spinout path |
