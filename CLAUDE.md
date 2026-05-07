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

## Package Map (6 packages under `packages/`)

| Package | Dir | Phase | Status |
|---------|-----|-------|--------|
| `lerobot-isaac-meta` | `packages/lerobot-isaac-meta/` | 0 | Scaffolded |
| `lerobot-isaac-env` | `packages/lerobot-isaac-env/` | 1 | Scaffolded (stubs) |
| `lerobot-isaac-adapters` | `packages/lerobot-isaac-adapters/` | 2 | Scaffolded |
| `lerobot-isaac-autoresearch` | `packages/lerobot-isaac-autoresearch/` | 3 | Scaffolded |
| `lerobot-isaac-synthetic` | `packages/lerobot-isaac-synthetic/` | 4 | Scaffolded (stubs) |
| `lerobot-isaac-configs` | `packages/lerobot-isaac-configs/` | 0 | Scaffolded |

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
- [ ] Phase 1 impl — Wire real Isaac Lab imports and full MDP implementation
- [ ] Phase 2 impl — Wire real LeRobot/DreamerV3/LeWM backends
- [ ] Phase 3 impl — Run autoresearch end-to-end with real metrics
- [ ] Phase 4 impl — Implement DR replay; enable MimicGen path

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
