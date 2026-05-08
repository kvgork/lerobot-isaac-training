# LeRobot + Isaac Lab Training Workspace — Orientation

**Workspace:** `~/workspaces/lerobot-isaac-training/`
**Purpose:** Isaac Lab + LeRobot + LeWorldModel monorepo for SO-101 manipulation training
**Status as of 2026-05-06:** Phases 0–5 complete (scaffolding). Training impl is future work.
**Build plan:** `/home/koen/tools/claude_code/plans/2026-05-06-lerobot-isaac-workspace-plan.md`

---

## Hard Scope Lock (2026-05-06)

This workspace currently contains ONLY scaffolding. No training runs, no synthetic data
generation. Each phase below adds real implementation.

---

## Repo–Workspace Contract

This workspace uses agents and skills from the `claude_code` repo — they are NOT duplicated here.

| What | Source of truth | Installed copy |
|------|----------------|----------------|
| Agents | `/home/koen/tools/claude_code/agents/` | `~/.claude/agents/` |
| Skills | `/home/koen/tools/claude_code/skills/` | (used in-repo) |
| Configs | `packages/lerobot-isaac-configs/configs/` | (this workspace) |
| Datasets | `datasets/` | (this workspace, gitignored) |
| Outputs | `outputs/` | (this workspace, gitignored) |
| Agent state | `.agent-state/` | (this workspace, gitignored) |

To update installed agents after editing source: `cd /home/koen/tools/claude_code && ./install.sh`

---

## Package Map (8 packages under `packages/`)

| Package | Dir | Phase | Status |
|---------|-----|-------|--------|
| `lerobot-isaac-meta` | `packages/lerobot-isaac-meta/` | 0 | Scaffolded |
| `lerobot-isaac-env` | `packages/lerobot-isaac-env/` | 1 | Un-stubbed (real Isaac Lab API; soft-import) |
| `lerobot-isaac-adapters` | `packages/lerobot-isaac-adapters/` | 2 | Un-stubbed (subprocess dispatchers) |
| `lerobot-isaac-autoresearch` | `packages/lerobot-isaac-autoresearch/` | 3 | Un-stubbed |
| `lerobot-isaac-synthetic` | `packages/lerobot-isaac-synthetic/` | 4 | Un-stubbed (DR replay; MimicGen deferred) |
| `lerobot-isaac-configs` | `packages/lerobot-isaac-configs/` | 0/A | 6 YAML configs populated |
| `lerobot-isaac-recorder` | `packages/lerobot-isaac-recorder/` | §14 | D435 + SO-101 dual-write (Parquet + LeWM HDF5) |
| `lerobot-isaac-dashboard` | `packages/lerobot-isaac-dashboard/` | §dashboard | Live + static metrics dashboard with snapshot save/load + 2-way and N-way compare |

---

## Where to Find Things

- **Docs:** `docs/research/` (external lib refs), `docs/runbook/` (how-tos)
- **YAML configs:** `packages/lerobot-isaac-configs/configs/`
- **Datasets (gitignored):** `datasets/`
- **Checkpoints/outputs (gitignored):** `outputs/`
- **Agent orchestration state (gitignored):** `.agent-state/{sessionId}/events.jsonl`
- **Package CLAUDE.md files:** each `packages/*/CLAUDE.md`
- **Top-level docs:** `README.md` | `ARCHITECTURE.md` | `USAGE.md`

---

## How to Continue Work

### Invoke training orchestrator (Phase 2+)
```
Task(lerobot-training-orchestrator, {workspace_root: "$PWD"})
```
The orchestrator expects:
- `datasets/` for input Parquet files
- `outputs/` for checkpoints
- `packages/lerobot-isaac-configs/configs/` for per-target YAML

### Train a world model (Phase 2+)
Step 1 — Convert Parquet to HDF5:
```
Task(lerobot-worldmodel-bridge, {input_parquet: "datasets/...", output_hdf5: "outputs/..."})
```
Step 2 — Train:
```bash
python -m lerobot_isaac_adapters.train \
  --target_arch dreamerv3 \
  --config packages/lerobot-isaac-configs/configs/wm_dreamerv3.yaml
```

### Run autoresearch (Phase 3+)
```
/autoresearch packages/lerobot-isaac-autoresearch/programs/lerobot-policy.md --type ml_model
```

---

## Pixi Workspace

The root `pixi.toml` is the **active pixi workspace** for the entire monorepo.
Each `packages/<pkg>/pixi.toml` is **dormant** in monorepo mode — it only activates
after a package is spun out to a standalone repo via `git subtree split`.

### Available environments

| Environment | Features included | Use case |
|-------------|-------------------|----------|
| `default` | dev | Unit tests, linting, format |
| `train-policy` | dev + lerobot | Train LeRobot policies (ACT / SmolVLA / Diffusion) |
| `train-dreamer` | dev + lerobot + dreamerv3 | Train DreamerV3 world model |
| `train-lewm` | dev + lerobot + leworldmodel | Train HF LeWorldModel |
| `sim` | dev + lerobot + isaaclab | Isaac Lab simulation (post-install) |
| `dashboard` | dev + dashboard | Live + static metrics dashboard |
| `full` | all features | All targets simultaneously |

### Common commands

```bash
# Install all conda + pip deps for the default environment
pixi install

# Install for a specific environment
pixi install -e train-policy

# Run tests in the default environment
pixi run test

# Run tests inside the sim environment
pixi run -e sim test

# Lint / format
pixi run lint
pixi run fmt

# Install Isaac Lab (system-level; requires GPU)
pixi run install-isaac-lab

# Download SO-101 USD asset
pixi run download-usd

# Start metrics dashboard
pixi run -e dashboard dashboard
```

Note: `pixi install` does NOT run `pixi run install-isaac-lab`.
Isaac Lab requires a separate manual step (GPU + disk space).

---

## Build Status Checklist

- [x] Phase 0 — Workspace Bootstrap (skeleton, 6 package stubs, top-level files)
- [x] Phase 1 — Isaac Lab SO-101 Environment (`lerobot-isaac-env` scaffolded)
- [x] Phase 2 — Modular Training Adapter (`lerobot-isaac-adapters` scaffolded)
- [x] Phase 3 — Autoresearch ML-Loop Integration (`lerobot-isaac-autoresearch` scaffolded)
- [x] Phase 4 — Synthetic Data Generation (`lerobot-isaac-synthetic` scaffolded)
- [x] Phase 5 — Documentation finalization (README, ARCHITECTURE, USAGE, runbooks, research docs)
- [x] Phase A — lerobot-isaac-dashboard package (live UI + static report + snapshot/compare)
- [x] Phase 1 impl — Isaac Lab MDP wiring (soft-import; cfg construction green; camera obs deferred)
- [x] Phase 2 impl — LeRobot / DreamerV3 / LeWM backends wired (subprocess + metric extraction; dry-run smoke green)
- [x] Phase 3 impl — Autoresearch e2e dry-run green (`train_wrapper → train → metric` chain enforced by test)
- [x] Phase 4a impl — Isaac DR replay + parquet writer + merge utilities wired (dry-run green)
- [ ] Phase 4b impl — MimicGen bridge path (deferred per plan; gated by `LEROBOT_MIMICGEN_ENABLED=1`)
- [ ] Real-data smoke — repeat dry-run smoke against actual SO-101 teleop dataset once collected
- [ ] Camera observation wiring — `wrist_camera_rgb` / `overhead_camera_rgb` need `CameraCfg` in scene (Isaac Lab tutorial 04)
- [ ] Insertion task — `tasks/insertion.py` Stage 5 stub (`NotImplementedError`)

---

## Reused Agents (10) — Source Paths

| Agent | Source path |
|-------|-------------|
| `lerobot-training-orchestrator` | `/home/koen/tools/claude_code/agents/orchestrators/lerobot-training-orchestrator.md` |
| `lerobot-data-collection-agent` | `/home/koen/tools/claude_code/agents/workers/lerobot-data-collection-agent.md` |
| `lerobot-evaluation-agent` | `/home/koen/tools/claude_code/agents/workers/lerobot-evaluation-agent.md` |
| `lerobot-sim-augmentation-agent` | `/home/koen/tools/claude_code/agents/workers/lerobot-sim-augmentation-agent.md` |
| `lerobot-curriculum-agent` | `/home/koen/tools/claude_code/agents/orchestrators/lerobot-curriculum-agent.md` |
| `lerobot-worldmodel-bridge` | `/home/koen/tools/claude_code/agents/lerobot-worldmodel-bridge.md` |
| `lerobot-specialist` | `/home/koen/tools/claude_code/agents/lerobot-specialist.md` |
| `autoresearch-loop-orchestrator` | `/home/koen/tools/claude_code/agents/orchestrators/autoresearch-loop-orchestrator.md` |
| `autoresearch-ml-executor-worker` | `/home/koen/tools/claude_code/agents/workers/autoresearch-ml-executor-worker.md` |
| `autoresearch-ml-proposer-worker` | `/home/koen/tools/claude_code/agents/workers/autoresearch-ml-proposer-worker.md` |

## Reused Skills (4) — Source Paths

| Skill | Source path |
|-------|-------------|
| `lerobot_world_model_bridge` | `/home/koen/tools/claude_code/skills/lerobot_world_model_bridge/` |
| `lerobot_mimicgen_bridge` | `/home/koen/tools/claude_code/skills/lerobot_mimicgen_bridge/` |
| `lerobot_dataset_quality` | `/home/koen/tools/claude_code/skills/lerobot_dataset_quality/` |
| `autoresearch` | `/home/koen/tools/claude_code/skills/autoresearch/` |

---

## Agent Routing (ask these agents for help)

| Question type | Agent | Installed path |
|---------------|-------|----------------|
| LeRobot datasets, policies, training | `lerobot-specialist` | `~/.claude/agents/lerobot-specialist.md` |
| Isaac Lab env design, USD, obs/actions | (use `ros2-learning-mentor` for ROS2 context) | — |
| World model (DreamerV3 / LeWM) | `lerobot-worldmodel-bridge` | `~/.claude/agents/lerobot-worldmodel-bridge.md` |
| Data collection quality | `lerobot-data-collection-agent` | `~/.claude/agents/workers/lerobot-data-collection-agent.md` |
| Curriculum / stage progression | `lerobot-curriculum-agent` | `~/.claude/agents/orchestrators/lerobot-curriculum-agent.md` |
| Autoresearch loop | `autoresearch-loop-orchestrator` | `~/.claude/agents/orchestrators/autoresearch-loop-orchestrator.md` |
| Evaluation / policy advancement | `lerobot-evaluation-agent` | `~/.claude/agents/workers/lerobot-evaluation-agent.md` |
| Metrics dashboard / pipeline visibility / snapshot compare | (no agent — see `docs/runbook/07-dashboard.md`) | — |
| Python patterns, code quality | `python-best-practices` | `~/.claude/agents/python-best-practices.md` |
| Debugging | `debugging-detective` | `~/.claude/agents/debugging-detective.md` |

---

## Common Pitfalls

- **RTX 3080 OOM:** Keep `--num_envs` at 4–8 for Isaac Lab; use `--image_size 64` for DreamerV3.
  The `lerobot-program.md` template already encodes an OOM recovery ladder.
- **USD path resolution:** SO-101 USD must be downloaded before Phase 1 runs.
  See `packages/lerobot-isaac-env/assets/usd/README.md` for the download/conversion script.
- **Isaac Lab headless mode:** Always pass `headless=True` unless you have a display.
- **Parquet → HDF5 direction:** `lerobot_world_model_bridge` skill handles this.
  Do NOT write custom HDF5 converters — use the skill.
- **LeWM HDF5 schema:** Undocumented. Use `(96,96)` preset in `lerobot_world_model_bridge`.
  See `skills/lerobot_world_model_bridge/SKILL.md` for schema notes.
- **Spinout to standalone repo:** Use `git subtree split` per Section 11.7 of the build plan.

---

## Vault Links (Second Brain context)

- SO-101 hardware / sim notes: `/home/koen/Documents/Vaults/Local/05-Wiki/entities/SO-101.md`
- LeWorldModel schema: `/home/koen/Documents/Vaults/Local/05-Wiki/entities/LeWorldModel.md`
- MimicGen integration: `/home/koen/Documents/Vaults/Local/05-Wiki/entities/MimicGen.md`
- World-Models RTX-3080 fit table: `/home/koen/Documents/Vaults/Local/05-Wiki/concepts/World-Models-(Robot-Manipulation).md`
- Autonomous ML Training Loop: `/home/koen/Documents/Vaults/Local/05-Wiki/concepts/Autonomous-ML-Training-Loop.md`

---

## Related Documentation

- `README.md` — project quickstart and package map
- `ARCHITECTURE.md` — system diagram, data flow, coupling rules
- `USAGE.md` — runbook index with exact commands
- `docs/runbook/` — step-by-step runbooks per task
- `docs/research/` — Isaac Lab, DreamerV3, LeWorldModel, MimicGen reference notes

---

## Documentation Map

All documentation files in this workspace with one-line descriptions:

| File | Description |
|------|-------------|
| `README.md` | Project overview, quickstart, package map, build status |
| `ARCHITECTURE.md` | Full system architecture: diagrams, state machine, coupling rules, glossary |
| `USAGE.md` | Comprehensive runbook: all 10 workflows, CLI reference, common errors |
| `CLAUDE.md` (this file) | Session orientation: agents, skills, pitfalls, vault links |
| `docs/api-reference.md` | Public Python API for all 6 packages: signatures + examples |
| `docs/runbook/01-bootstrap.md` | First-time setup: pixi, Isaac Lab, USD, smoke tests |
| `docs/runbook/02-collect-data.md` | Collect and quality-filter SO-101 teleop data |
| `docs/runbook/03-train-policy.md` | Train SmolVLA / ACT / Diffusion policy end-to-end |
| `docs/runbook/04-train-world-model.md` | Train DreamerV3 or LeWorldModel |
| `docs/runbook/05-augment-with-dr.md` | Generate DR synthetic data via Isaac Lab replay |
| `docs/runbook/06-augment-with-mimicgen.md` | MimicGen augmentation (deferred path) |
| `docs/runbook/07-dashboard.md` | Live + static metrics dashboard: start, tabs, snapshots, compare, troubleshoot |
| `docs/research/isaac-lab-reference.md` | Isaac Lab API, USD setup, RTX 3080 constraints |
| `docs/research/dreamerv3-reference.md` | DreamerV3 theory, sheeprl, HDF5 schema, config knobs |
| `docs/research/leworldmodel-reference.md` | LeWorldModel architecture, HDF5 schema warning, config |
| `docs/research/mimicgen-reference.md` | MimicGen pipeline, deferred status, when to enable |
| `docs/internals/data-pipeline.md` | Full data lifecycle: schema, conversions, tagging, merge |
| `docs/internals/training-dispatch.md` | How train.py dispatches, subprocess args, OOM retry |
| `docs/internals/autoresearch-integration.md` | program.md schema, operators, metric history, plateau |
| `docs/internals/isaac-lab-integration.md` | MDP terms, DR config, USD wiring, physics params |
| `docs/internals/world-model-bridge.md` | DreamerV3 vs LeWM HDF5 schemas, bridge patterns |
| `docs/internals/synthetic-data.md` | DR replay loop, parquet writer, merge logic, dedup |
| `docs/concepts/modular-training-adapter.md` | Why one entrypoint, how to add a new arch |
| `docs/concepts/soft-import-discipline.md` | Why heavy deps are lazy, pattern, testing strategy |
| `docs/concepts/multi-package-monorepo.md` | Rationale, coupling rules, spinout strategy |
| `docs/concepts/pixi-workspace.md` | Why pixi, features vs environments, dormant config |

---

## How to Navigate ("I want to X, read Y")

| I want to... | Read this |
|-------------|----------|
| Get started for the first time | `docs/runbook/01-bootstrap.md` |
| Collect real SO-101 data | `docs/runbook/02-collect-data.md` |
| Train a policy | `docs/runbook/03-train-policy.md` + `docs/research/` for the chosen arch |
| Train a world model | `docs/runbook/04-train-world-model.md` + `docs/research/dreamerv3-reference.md` |
| Generate synthetic data | `docs/runbook/05-augment-with-dr.md` |
| Run autoresearch HP search | `USAGE.md §Workflow F` + `docs/internals/autoresearch-integration.md` |
| View metrics / compare runs | `docs/runbook/07-dashboard.md` |
| Understand how the system fits together | `ARCHITECTURE.md` |
| Understand the data format | `docs/internals/data-pipeline.md` |
| Understand how training dispatch works | `docs/internals/training-dispatch.md` |
| Add a new training backend | `docs/concepts/modular-training-adapter.md` |
| Spin out a package to its own repo | `ARCHITECTURE.md §Spinout Mechanics` + `docs/concepts/multi-package-monorepo.md` |
| Look up a function signature | `docs/api-reference.md` |
| Understand a term in the codebase | `ARCHITECTURE.md §Glossary` |

---

## Production Hygiene (added 2026-05-07)

### CI / GitHub Actions

Workflows live in `.github/workflows/`:

| File | Purpose |
|------|---------|
| `ci.yml` | Per-package matrix (6 pkg × 2 Python versions) + workspace integration job |
| `lint.yml` | PR-only ruff check + ruff format check + TOML validation |

**Matrix shape:** 12 parallel per-package jobs (6 packages × Python 3.10 + 3.11) run on
every push/PR to `main`. A 13th workspace-level job runs after all 12 pass, installing
via `pixi install` and running `pytest packages/*/tests/ -m 'not integration'`.

**Triggers:** push to `main`, PRs to `main`.

Heavy-dep tests (`requires_isaaclab`, `requires_lerobot`, `requires_dreamerv3`) are
skipped in CI by marker exclusion. They require a GPU runner and manual invocation.

### Pre-commit Hooks

`.pre-commit-config.yaml` is at the workspace root. Install once with:

```bash
pre-commit install                          # runs on every git commit
pre-commit install --hook-type pre-push     # additionally runs on git push
```

Hooks run in order:
1. `pre-commit-hooks` — whitespace, EOF, YAML, TOML, large-file, line-ending checks
2. `ruff` — lint with auto-fix
3. `ruff-format` — formatting
4. `pytest-check` (pre-push stage) — runs tests for any test files newer than last commit

Do NOT run `pre-commit install` in CI (hooks run natively via `lint.yml`).

### Spinout Smoke Test

```bash
# Default target: lerobot-isaac-configs (smallest package)
bash scripts/spinout_smoke_test.sh

# Other packages
bash scripts/spinout_smoke_test.sh lerobot-isaac-meta
bash scripts/spinout_smoke_test.sh lerobot-isaac-adapters
bash scripts/spinout_smoke_test.sh lerobot-isaac-dashboard
```

The script uses `git subtree split` to extract the package to a temp dir, checks that
`pyproject.toml`, `pixi.toml`, `README.md`, and `src/<pkg>` all exist, then runs
`pytest tests/ -q`. Do NOT run during an uncommitted-changes session — subtree split
requires a clean tree.

### ADR Index

Architecture Decision Records live in `docs/adr/`. See `docs/adr/README.md` for the
full index. Current ADRs:

| ADR | One-line summary |
|-----|-----------------|
| 0001 | Isaac Lab chosen over MuJoCo for GPU parallelism + native DR + USD |
| 0002 | Pixi workspace with dormant per-package pixi.toml for spinout readiness |
| 0003 | Heavy deps lazy-imported inside functions; packages importable with no GPU deps |
| 0004 | 6-package monorepo with one-way coupling and independent spinout path |
| 0005 | Single train.py --target_arch + MetricExtractor gives autoresearch a stable interface |
| 0006 | Streamlit + Plotly + jinja2 for dashboard; dual-render Tab.render; local-files-only; Parquet + JSON snapshots |

When making a significant architectural decision, add an ADR following the template in
`docs/adr/README.md`.
