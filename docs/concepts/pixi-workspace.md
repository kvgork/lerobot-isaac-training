# Pixi Workspace — Design Concept

**Cross-references:** [CLAUDE.md](../../CLAUDE.md) | [multi-package-monorepo.md](./multi-package-monorepo.md)

---

## Why Pixi

This workspace uses `pixi` as the environment manager rather than conda, virtualenv, or plain pip.

| Concern | pixi approach |
|---------|--------------|
| Isaac Lab deps (CUDA, USD libs) | conda packages via pixi's conda channel |
| Python deps (LeRobot, sheeprl) | PyPI packages via pixi's pypi section |
| Lock file | `pixi.lock` covers both conda + PyPI in one file |
| Activation | `pixi shell` or `pixi run <task>` — no `conda activate` |
| CI compatibility | `pixi install --frozen` reproduces exact lock |
| Multiple envs | Defined in `pixi.toml`; activated with `-e <name>` |

The critical advantage: Isaac Lab requires specific CUDA library versions that are best
managed via conda. `pip install isaaclab` often fails due to CUDA ABI mismatches.
pixi handles the conda + pip split cleanly in one `pixi.toml`.

---

## Features vs Environments

`pixi.toml` uses a two-level system: **features** are sets of packages; **environments** are
named combinations of features.

```toml
# Simplified pixi.toml structure:

[project]
name = "lerobot-isaac-workspace"
channels = ["conda-forge", "nvidia", "pytorch"]
platforms = ["linux-64"]

# Features: independent dep groups
[feature.dev.dependencies]
python = "3.11.*"
pytest = ">=7.0"
ruff = ">=0.4"

[feature.lerobot.pypi-dependencies]
lerobot = {git = "https://github.com/huggingface/lerobot"}

[feature.dreamerv3.pypi-dependencies]
sheeprl = {extras = ["dreamer-v3"]}

[feature.leworldmodel.pypi-dependencies]
transformers = ">=4.40"

[feature.isaaclab]
# Isaac Sim is installed by a post-install script, not via pixi directly
# This feature adds the Python path glue
[feature.isaaclab.tasks]
install-isaac-lab = "bash scripts/install_isaac_lab.sh"

# Environments: named combinations of features
[environments]
default = {features = ["dev"], solve-group = "default"}
train-policy = {features = ["dev", "lerobot"]}
train-dreamer = {features = ["dev", "lerobot", "dreamerv3"]}
train-lewm = {features = ["dev", "lerobot", "leworldmodel"]}
sim = {features = ["dev", "lerobot", "isaaclab"]}
full = {features = ["dev", "lerobot", "dreamerv3", "leworldmodel", "isaaclab"]}
```

---

## Dormant Per-Package `pixi.toml`

Each `packages/*/pixi.toml` is **dormant** in monorepo mode. It contains a comment:
```toml
# DORMANT: This pixi.toml is inactive in the monorepo.
# Activate after spinout: remove this comment and run pixi install.
```

The root `pixi.toml` is the active config. When a package is spun out to its own repo,
its `pixi.toml` is activated (comment removed) and becomes the primary env config.

This avoids two sources of truth for the same deps while monorepo is active,
while ensuring each package is ready for standalone use after spinout.

---

## Common Commands

```bash
# Install default environment (dev tooling + 6 packages):
pixi install

# Activate default shell:
pixi shell

# Install and activate specific environment:
pixi install -e train-policy
pixi shell -e train-policy

# Run a task without activating shell:
pixi run test
pixi run -e sim test

# List available tasks:
pixi task list

# Update lock file (after changing pixi.toml):
pixi update

# Install with exact lock (for CI):
pixi install --frozen
```

---

## Adding a New Environment

1. Add feature to `pixi.toml`:
   ```toml
   [feature.myfeature.pypi-dependencies]
   my-package = ">=1.0"
   ```

2. Add environment:
   ```toml
   [environments]
   train-myfeature = {features = ["dev", "lerobot", "myfeature"]}
   ```

3. Update `pixi.lock`:
   ```bash
   pixi update
   ```

4. Document in `CLAUDE.md §Pixi Workspace` environment table.

5. Update `ARCHITECTURE.md §Pixi Workspace Layout` table.

---

## Relationship to `uv sync`

pixi manages conda + PyPI deps and the Python interpreter.
`uv sync` manages the Python workspace packages (`packages/*` editable installs).
They are complementary:

```bash
# Full setup:
pixi install       # installs all conda + PyPI deps
pixi shell         # activates environment
uv sync            # installs all 6 workspace packages in editable mode
```

Running `pixi run test` automatically calls `uv sync` as a prerequisite via the task
definition in `pixi.toml`:
```toml
[tasks]
test = {cmd = "pytest packages/*/tests/", depends-on = ["sync"]}
sync = "uv sync"
```
