# Architecture — LeRobot + Isaac Lab Training Workspace

**Status as of 2026-05-06:** Phases 0–5 complete (scaffolding). Training/eval wiring is future work.

---

## System Diagram (ASCII)

```
┌──────────────────────────────────────────────────────────────────────┐
│              LeRobot + Isaac Lab Training Workspace                   │
│                ~/workspaces/lerobot-isaac-training/                  │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    lerobot-isaac-meta                         │    │
│  │  lerobot-isaac CLI  •  workspace_paths.py                     │    │
│  │  Depends on: env | adapters | autoresearch | synthetic |      │    │
│  │              configs                                          │    │
│  └──────────────────────┬───────────────────────────────────────┘    │
│                          │                                             │
│           ┌──────────────┼──────────────────────────┐                 │
│           │              │                           │                 │
│  ┌────────▼───────┐  ┌───▼─────────────────┐  ┌────▼──────────────┐ │
│  │lerobot-isaac   │  │lerobot-isaac-        │  │lerobot-isaac-     │ │
│  │-env            │  │adapters              │  │autoresearch       │ │
│  │                │  │                      │  │                   │ │
│  │ SO101EnvCfg    │  │ train.py             │  │ programs/*.md     │ │
│  │ ManagerBased   │  │ ──target_arch──▶     │  │ train_wrapper.py  │ │
│  │ RLEnv (Isaac)  │  │  policy_lerobot.py   │  │                   │ │
│  │ DR event mgr   │  │  wm_dreamerv3.py     │  └──────┬────────────┘ │
│  │ tasks/pick*    │  │  wm_leworldmodel.py  │         │               │
│  └────────┬───────┘  │ metric_extractor.py  │         │               │
│           │          │ isaac_data_recorder  │         │               │
│           │          └──────────────────────┘         │               │
│           │                                            │               │
│  ┌────────▼──────────────────────────────────────┐    │               │
│  │lerobot-isaac-synthetic                         │    │               │
│  │                                                │    │               │
│  │  isaac_dr/replay_runner.py (priority path)     │    │               │
│  │  isaac_dr/parquet_writer.py                    │    │               │
│  │  mimicgen/bridge_invocation.py (deferred)      │    │               │
│  │  merge_utilities.py                            │    │               │
│  └────────────────────────────────────────────────┘    │               │
│                                                          │               │
│  ┌───────────────────────────────────────────────────────▼────────┐   │
│  │lerobot-isaac-configs  (leaf — no internal deps)                 │   │
│  │  configs/policy_smolvla.yaml  wm_dreamerv3.yaml  ...           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## External Agent + Skill Layer (repo — NOT duplicated here)

```
/home/koen/tools/claude_code/
├── agents/orchestrators/
│   ├── lerobot-training-orchestrator.md   ← top-level loop driver
│   ├── lerobot-curriculum-agent.md        ← 6-stage manipulation ladder
│   └── autoresearch-loop-orchestrator.md  ← Karpathy autoresearch
├── agents/workers/
│   ├── lerobot-data-collection-agent.md   ← teleop quality filter
│   ├── lerobot-evaluation-agent.md        ← pc_success → ADVANCE / CONTINUE
│   ├── lerobot-sim-augmentation-agent.md  ← MimicGen pipeline
│   ├── autoresearch-ml-executor-worker.md ← runs train, parses metric
│   └── autoresearch-ml-proposer-worker.md ← 6 mutation operators
├── agents/
│   ├── lerobot-worldmodel-bridge.md       ← Parquet→HDF5/npz/WebDataset
│   └── lerobot-specialist.md             ← Q&A backstop
└── skills/
    ├── lerobot_world_model_bridge/        ← Parquet↔HDF5 conversion
    ├── lerobot_mimicgen_bridge/           ← Parquet↔MimicGen HDF5
    ├── lerobot_dataset_quality/           ← SAL+TED filtering
    └── autoresearch/                      ← operator defs + metric history
```

Installed to `~/.claude/agents/` via `cd /home/koen/tools/claude_code && ./install.sh`.

---

## Data Flow

```
SO-101 teleoperation
        │
        ▼ LeRobot Parquet (datasets/)
        │
        ├──► lerobot_dataset_quality skill ──► filtered Parquet
        │         (SAL + TED filtering)
        │
        ├──► lerobot-isaac-synthetic/isaac_dr
        │         replay_runner.py  ──► DR-augmented Parquet  ─┐
        │         (domain randomized)                           │
        │                                                        ├──► merged Parquet
        ├──► lerobot-isaac-synthetic/mimicgen (deferred)        │    (merge_utilities.py)
        │         bridge_invocation.py ──► MimicGen Parquet ────┘
        │
        ▼ merged/filtered Parquet
        │
        ├──► policy path (--target_arch smolvla/act/diffusion)
        │         lerobot train → checkpoint → eval → pc_success
        │
        └──► world model path
              │
              ├── lerobot_world_model_bridge skill
              │     Parquet → HDF5 (64×64 for DreamerV3, 96×96 for LeWM)
              │
              ├── --target_arch dreamerv3
              │     sheeprl / dreamer-v3-pytorch → recon_loss
              │
              └── --target_arch le_world_model
                    HF LeWorldModel → pred_loss

Metric emitted on stdout: <name>=<float>
Parsed by autoresearch-ml-executor-worker regex: (\w+)[=:\s]+([0-9.eE+-]+)

Checkpoints → outputs/   (gitignored)
```

---

## Cross-Package Coupling Rules (Section 11.6)

| Rule | Details |
|------|---------|
| `lerobot-isaac-env` has NO sibling imports | Pure Isaac Lab + torch. Safe to extract. |
| `lerobot-isaac-configs` is a leaf | No imports from any sibling. All siblings may import it. |
| `lerobot-isaac-adapters` does NOT import `lerobot-isaac-env` directly | Accesses env via `isaac_data_recorder.py` which soft-imports isaaclab. |
| `lerobot-isaac-autoresearch` only calls `adapters` as subprocess | No Python import coupling. |
| `lerobot-isaac-synthetic` imports `lerobot-isaac-env` softly | All Isaac Lab imports deferred to function bodies. |
| All Isaac Lab / LeRobot / sheeprl / transformers imports are soft | `try/except ImportError` at module level. Tests run without heavy deps. |

---

## Autoresearch Integration Flow

```
User runs:
  /autoresearch packages/lerobot-isaac-autoresearch/programs/lerobot-policy.md

  autoresearch-loop-orchestrator
      │ reads program.md (metric, direction, baseline, mutations)
      │
      ├──► autoresearch-ml-proposer-worker
      │         generates mutations on hyperparams
      │
      └──► autoresearch-ml-executor-worker
                calls train_wrapper.py (script_path from program.md)
                  └─► lerobot_isaac_adapters.train
                        dispatches to policy or world-model target
                        emits: pc_success=0.73  (or recon_loss=0.031)
                executor parses last metric line
                logs to autoresearch skill metric history
```

Three `program.md` files:
- `programs/lerobot-policy.md` — metric: `pc_success`, maximize
- `programs/dreamerv3.md` — metric: `recon_loss`, minimize
- `programs/leworldmodel.md` — metric: `pred_loss`, minimize

---

## World-Model Dispatch (`--target_arch` selector)

```
lerobot-isaac-train --target_arch=<value>
        │
        ├── smolvla / act / diffusion
        │       → targets/policy_lerobot.py
        │         subprocess: lerobot.scripts.train
        │         metric: pc_success
        │
        ├── dreamerv3
        │       → targets/wm_dreamerv3.py
        │         subprocess: sheeprl dreamer_v3 (or dreamer-v3-pytorch)
        │         metric: recon_loss
        │         image_size: 64×64 (RTX 3080 budget)
        │
        └── le_world_model
                → targets/wm_leworldmodel.py
                  subprocess: leworldmodel train
                  metric: pred_loss
                  image_size: 96×96 (requires HF LeWorldModel)
```

---

## Repo–Workspace Contract

| Location | Role |
|----------|------|
| `/home/koen/tools/claude_code/agents/` | Source of truth for agents (edit here) |
| `/home/koen/tools/claude_code/skills/` | Source of truth for skills (edit here) |
| `~/.claude/agents/` | Installed copies (what Claude invokes) |
| `~/workspaces/lerobot-isaac-training/packages/` | All Python implementation |
| `~/workspaces/lerobot-isaac-training/datasets/` | LeRobot Parquet datasets (gitignored) |
| `~/workspaces/lerobot-isaac-training/outputs/` | Training checkpoints (gitignored) |
| `~/workspaces/lerobot-isaac-training/.agent-state/` | Orchestrator state (gitignored) |

To deploy agent edits: `cd /home/koen/tools/claude_code && ./install.sh`

---

## Synthetic Data Strategy

| Path | Priority | Status |
|------|----------|--------|
| Isaac Lab domain randomization replay | **Primary** | Scaffolded (Phase 4); impl deferred |
| MimicGen bridge (MuJoCo-internal) | Deferred | Stub only; enable via `LEROBOT_MIMICGEN_ENABLED=1` |

MimicGen still runs MuJoCo internally — we do not replace that. Its output is converted back to LeRobot Parquet via `lerobot_mimicgen_bridge` skill and merged into the training corpus via `merge_utilities.py`.

---

## Spinout Procedure (Section 11.7)

Any package under `packages/` can be extracted to a standalone repo:

```bash
# Example: extract env package
git subtree split -P packages/lerobot-isaac-env -b spinout-env
git push git@github.com:yourorg/lerobot-isaac-env.git spinout-env:main

# Or with git filter-repo (cleaner history):
git clone . /tmp/lerobot-isaac-env-repo
cd /tmp/lerobot-isaac-env-repo
git filter-repo --path packages/lerobot-isaac-env/ --path-rename packages/lerobot-isaac-env/:
```

Update the extracted repo's `pyproject.toml` to remove `packages/` prefix from package paths. The sibling package deps (`lerobot-isaac-configs`) become PyPI deps in the standalone repo.
