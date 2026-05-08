# LeRobot + Isaac Lab Training Workspace

This workspace is a modular monorepo for autonomous SO-101 robot manipulation training. It integrates Isaac Lab (physics simulation), LeRobot (imitation learning policies: SmolVLA / ACT / Diffusion Policy), and two world-model targets (DreamerV3 and HF LeWorldModel) under a single unified training entrypoint. A Streamlit metrics dashboard (`lerobot-isaac-dashboard`) provides live and static visibility into all pipeline artefacts with snapshot save/load and run comparison. The design separates concerns into eight pip-installable packages so each subsystem can later be extracted to its own repository.

## Features

**Simulation and policies:** The `lerobot-isaac-env` package wraps the SO-101 arm in an Isaac Lab `ManagerBasedRLEnv` with full domain randomization. The `lerobot-isaac-adapters` package provides a single `lerobot-isaac-train` command with a `--target_arch` selector that dispatches to LeRobot policy training, DreamerV3, or LeWorldModel without changing any other argument. Configs are centralized in `lerobot-isaac-configs` and consumed by all packages.

**Autoresearch and synthetic data:** The `lerobot-isaac-autoresearch` package contains `program.md` files consumed by the `autoresearch-loop-orchestrator` agent for automated hyperparameter search. The `lerobot-isaac-synthetic` package provides Isaac Lab domain-randomization replay (priority path) and a MimicGen bridge stub (deferred path) for expanding the training corpus beyond real teleoperation. All agents and skills referenced here live in the `claude_code` repo at `/home/koen/tools/claude_code/` and are NOT duplicated in this workspace.

**Metrics dashboard:** The `lerobot-isaac-dashboard` package provides a Streamlit + Plotly live dashboard and static HTML reports over all pipeline artefacts (datasets, checkpoints, eval results, autoresearch history, curriculum stage). Supports snapshot save/load and 2-way / N-way run comparison with zero external services.

---

## Table of Contents

- [Quickstart](#quickstart)
- [Package Map](#package-map)
- [Build Status](#build-status)
- [Documentation](#documentation)
- [Key Docs](#key-docs)

---

## Quickstart

1. **Clone and enter workspace:**
   ```bash
   cd ~/workspaces/lerobot-isaac-training
   ```

2. **Install pixi environment (default — dev tooling + all sibling packages):**
   ```bash
   pixi install
   pixi shell
   ```
   To use a specific environment (e.g. with LeRobot policies included):
   ```bash
   pixi install -e train-policy
   pixi shell -e train-policy
   ```
   Available environments: `default`, `train-policy`, `train-dreamer`, `train-lewm`, `sim`, `dashboard`, `full`.
   See `CLAUDE.md §Pixi Workspace` for the full environment table.

   Alternative (uv-only, no conda deps):
   ```bash
   uv sync
   # Or individually: pip install -e packages/lerobot-isaac-meta[all]
   ```

3. **Install Isaac Lab** (system-level, GPU required — separate from pixi install):
   ```bash
   pixi run install-isaac-lab
   # Or manually: bash scripts/install_isaac_lab.sh
   # Verify: python -c "import isaaclab"
   ```

4. **Set workspace env var:**
   ```bash
   export LEROBOT_ISAAC_WORKSPACE=~/workspaces/lerobot-isaac-training
   ```

5. **Download SO-101 USD asset:**
   ```bash
   pixi run download-usd
   # Or manually: bash packages/lerobot-isaac-env/assets/usd/download_so101_urdf.sh
   # See packages/lerobot-isaac-env/assets/usd/README.md for details
   ```

6. **Smoke test — dry run policy training:**
   ```bash
   lerobot-isaac-train --target_arch smolvla \
     --config packages/lerobot-isaac-configs/configs/policy_smolvla.yaml \
     --dry_run
   ```

7. **Smoke test — dry run world model:**
   ```bash
   lerobot-isaac-train --target_arch dreamerv3 \
     --config packages/lerobot-isaac-configs/configs/wm_dreamerv3.yaml \
     --dry_run
   ```

8. **Run workspace tests:**
   ```bash
   pixi run test
   # Or: pytest packages/*/tests/
   ```

9. **Open the metrics dashboard:**
   ```bash
   pixi run -e dashboard dashboard
   # Opens http://localhost:8501
   ```

10. **Open workspace CLAUDE.md for full orientation:**
    ```bash
    cat CLAUDE.md
    ```

---

## Package Map

| Package | Path | Description |
|---------|------|-------------|
| `lerobot-isaac-meta` | `packages/lerobot-isaac-meta/` | Umbrella CLI (`lerobot-isaac`) + workspace path resolver; depends on all siblings |
| `lerobot-isaac-env` | `packages/lerobot-isaac-env/` | Isaac Lab `ManagerBasedRLEnv` for SO-101: obs, actions, rewards, DR, tasks |
| `lerobot-isaac-adapters` | `packages/lerobot-isaac-adapters/` | Unified `lerobot-isaac-train` entrypoint; dispatches by `--target_arch` |
| `lerobot-isaac-autoresearch` | `packages/lerobot-isaac-autoresearch/` | `program.md` configs for autoresearch loop; `train_wrapper.py` shim |
| `lerobot-isaac-synthetic` | `packages/lerobot-isaac-synthetic/` | Isaac Lab DR replay + MimicGen bridge stub + dataset merge utilities |
| `lerobot-isaac-configs` | `packages/lerobot-isaac-configs/` | Shared YAML configs per `target_arch`; leaf package (no internal deps) |
| `lerobot-isaac-recorder` | `packages/lerobot-isaac-recorder/` | D435 camera + SO-101 teleop dual-write recorder (Parquet + LeWM HDF5) |
| `lerobot-isaac-dashboard` | `packages/lerobot-isaac-dashboard/` | Streamlit + Plotly metrics dashboard; live UI + static HTML + snapshots + compare |

---

## Build Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Workspace bootstrap (skeleton, 6 packages, CLAUDE.md, configs, .gitignore) | Done |
| Phase 1 | Isaac Lab SO-101 env stubs (`lerobot-isaac-env`) | Done (scaffolding) |
| Phase 2 | Modular training adapter (`lerobot-isaac-adapters`) | Done (scaffolding) |
| Phase 3 | Autoresearch loop wiring (`lerobot-isaac-autoresearch`) | Done (scaffolding) |
| Phase 4 | Synthetic data generation (`lerobot-isaac-synthetic`) | Done (scaffolding) |
| Phase 5 | Documentation polish (this file + ARCHITECTURE + USAGE + runbooks) | Done |
| Phase A | lerobot-isaac-dashboard (live UI + static report + snapshots + compare) | Done (281 tests) |
| Phase 1 impl | Wire real Isaac Lab imports, full obs/action/reward impl | Future work |
| Phase 2 impl | Wire real LeRobot/DreamerV3/LeWM backends in adapters | Future work |
| Phase 3 impl | Run autoresearch end-to-end with metrics | Future work |
| Phase 4 impl | Implement DR replay + enable MimicGen path | Future work |

---

## Documentation

Full documentation is organized into four areas:

### Top-Level References

| Doc | Description |
|-----|-------------|
| `CLAUDE.md` | Session orientation: agents, skills, vault links, navigation guide |
| `ARCHITECTURE.md` | System diagrams, state machine, coupling rules, spinout mechanics, glossary |
| `USAGE.md` | All 11 workflows with exact commands; CLI reference; common errors |

### Runbooks (`docs/runbook/`)

Step-by-step task guides:

| Runbook | Task |
|---------|------|
| `01-bootstrap.md` | First-time setup: pixi, Isaac Lab, USD, smoke tests |
| `02-collect-data.md` | Collect and quality-filter SO-101 teleop data |
| `03-train-policy.md` | Train SmolVLA / ACT / Diffusion policy end-to-end |
| `04-train-world-model.md` | Train DreamerV3 or LeWorldModel |
| `05-augment-with-dr.md` | Generate DR synthetic data via Isaac Lab replay |
| `06-augment-with-mimicgen.md` | MimicGen augmentation (deferred path) |
| `07-dashboard.md` | Live + static metrics dashboard: start, tabs, snapshots, compare, troubleshoot |

### Deep-Dives (`docs/internals/`)

Internal implementation details:

| Doc | Description |
|-----|-------------|
| `data-pipeline.md` | Full data lifecycle: LeRobotDataset schema, conversions, merge logic |
| `training-dispatch.md` | How `train.py` dispatches, subprocess args, OOM retry, metric contract |
| `autoresearch-integration.md` | program.md schema, mutation operators, metric history, plateau detection |
| `isaac-lab-integration.md` | MDP terms, EventTermCfg DR, USD wiring, physics config, RTX 3080 limits |
| `world-model-bridge.md` | DreamerV3 vs LeWM HDF5 schemas, bridge patterns, schema warnings |
| `synthetic-data.md` | DR replay loop, parquet writer, merge dedup, MimicGen deferred path |

### Concepts (`docs/concepts/`)

Design rationale and patterns:

| Doc | Description |
|-----|-------------|
| `modular-training-adapter.md` | Why one entrypoint; how to add a new training backend |
| `soft-import-discipline.md` | Why heavy deps are lazy; pattern; testing without heavy deps |
| `multi-package-monorepo.md` | Rationale for 6 packages; coupling rules; spinout strategy |
| `pixi-workspace.md` | Why pixi; features vs environments; dormant per-package config |

### External References (`docs/research/`)

Library-specific reference notes:

| Doc | Description |
|-----|-------------|
| `isaac-lab-reference.md` | Key API classes, USD setup, headless mode, version pinning |
| `dreamerv3-reference.md` | RSSM theory, sheeprl vs dreamer-v3-pytorch, HDF5 schema, config |
| `leworldmodel-reference.md` | JEPA architecture, HDF5 schema warning, RTX 3080 config |
| `mimicgen-reference.md` | Pipeline, deferred status, integration gap, when to enable |

### API Reference

`docs/api-reference.md` — Public Python API for all packages with signatures and examples.

`packages/lerobot-isaac-dashboard/docs/API.md` — Full dashboard public API (loaders, tabs, report, snapshots, compare).

---

## Key Docs

- `CLAUDE.md` — workspace orientation for any Claude session opened here
- `ARCHITECTURE.md` — system diagram, data flow, cross-package coupling rules
- `USAGE.md` — runbook index with exact commands per task
- `docs/runbook/` — step-by-step runbooks (bootstrap → collect → train → augment → dashboard)
- `docs/research/` — reference notes for Isaac Lab, DreamerV3, LeWorldModel, MimicGen
- `packages/lerobot-isaac-dashboard/docs/` — dashboard API, examples, and internals

**Full build plan:** `/home/koen/tools/claude_code/plans/2026-05-06-lerobot-isaac-workspace-plan.md`
