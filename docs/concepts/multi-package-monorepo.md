# Multi-Package Monorepo — Design Concept

**Cross-references:** [ARCHITECTURE.md](../../ARCHITECTURE.md) | [pixi-workspace.md](./pixi-workspace.md)

---

## Rationale

This workspace uses a multi-package monorepo (six packages under `packages/`) rather than
a single flat package for two reasons:

### 1. Independent Spinout

Each subsystem can become a standalone pip-installable repository later.
`lerobot-isaac-env` could be published as a general SO-101 Isaac Lab wrapper,
independent of any training tooling. `lerobot-isaac-synthetic` could be a general
domain-randomization toolkit. With a monorepo, these extractions are clean `git subtree split`
operations with no refactoring needed inside the package.

### 2. Heavy Dep Isolation

`lerobot-isaac-env` depends on Isaac Lab (~15 GB). `lerobot-isaac-adapters` depends on
LeRobot, sheeprl, or transformers depending on `target_arch`. By separating packages,
the dependency groups are isolated. A machine doing config edits only needs
`lerobot-isaac-configs` installed. A machine doing autoresearch management only needs
`lerobot-isaac-autoresearch`. No single machine needs everything at once.

---

## Coupling Rules

The coupling rules (from `ARCHITECTURE.md §Cross-Package Coupling`) exist to make spinout clean.
If `lerobot-isaac-adapters` imported `lerobot-isaac-synthetic` directly, extracting
`adapters` to its own repo would require also extracting `synthetic`. The rules prevent this.

The rule hierarchy:
```
lerobot-isaac-configs     <- leaf, no deps, everyone imports it
lerobot-isaac-env         <- only isaaclab, numpy, torch
lerobot-isaac-synthetic   <- env (soft) + configs
lerobot-isaac-adapters    <- configs + env (soft via recorder)
lerobot-isaac-autoresearch <- configs only (calls adapters as subprocess)
lerobot-isaac-meta        <- all of the above (umbrella)
```

---

## Spinout Strategy

### When to Spinout

Spinout a package when:
- It has reached a stable API that other projects want to consume independently
- Its issue tracker would be cluttered by unrelated issues from other packages
- It has different release cadence needs (e.g. `env` changes with Isaac Lab releases;
  `configs` changes with every experiment)

### Spinout Procedure (Quick Reference)

```bash
# Choose package to extract:
PKG=lerobot-isaac-env

# Create branch with only that package's history:
git subtree split -P packages/$PKG -b spinout-$PKG

# Clone into standalone repo:
git clone . /tmp/$PKG-standalone
cd /tmp/$PKG-standalone
git checkout spinout-$PKG

# Activate dormant pixi.toml:
# Remove "# DORMANT: activate after spinout" comment from pixi.toml

# Update pyproject.toml — change sibling path deps to PyPI deps:
# [project.dependencies]
# lerobot-isaac-configs = ">=0.1.0"   # was: {path = "../lerobot-isaac-configs"}

# Push to new remote:
git remote set-url origin git@github.com:yourorg/$PKG.git
git push -u origin spinout-$PKG:main
```

Full procedure: see `ARCHITECTURE.md §Spinout Mechanics`.

---

## uv Workspace vs Pip Workspace

This monorepo uses a `uv workspace` (declared in the root `pyproject.toml`):

```toml
[tool.uv.workspace]
members = ["packages/*"]
```

Advantages of uv workspace over plain pip:
- Lock file covers all packages simultaneously
- Cross-package editable installs managed automatically
- `uv sync` installs all packages and their deps in one command
- Faster than pip for large dependency graphs

The workspace is also compatible with pixi (pixi calls uv for Python dep resolution).

---

## Version Policy

All six packages share the same version number in the monorepo.
Version is set in the root `pyproject.toml` and referenced in each `packages/*/pyproject.toml`:
```toml
# packages/lerobot-isaac-env/pyproject.toml
[project]
version = "0.1.0"  # kept in sync with root manually
```

After spinout, each package adopts independent versioning.
