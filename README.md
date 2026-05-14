# lerobot-isaac-training

Modular training stack for the **SO-101 robot arm** combining Isaac Lab simulation,
LeRobot imitation-learning policies, and two world-model backends (DreamerV3 and
HF LeWorldModel) behind a single training entrypoint, with a metrics dashboard,
domain-randomization synthetic-data pipeline, and an autoresearch hyperparameter loop.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pixi](https://img.shields.io/badge/managed%20by-pixi-yellow)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](.pre-commit-config.yaml)

> **Status — research / scaffolding.** Phases 0 – 5 (workspace bootstrap, package
> wiring, dry-run smoke tests, documentation) are complete. Real training requires
> a GPU, Isaac Lab, and an SO-101 dataset. See [Build Status](#build-status).

---

## What's inside

**Thin-meta-repo architecture** (post-spinout, 2026-05-13): only
`lerobot-isaac-meta` lives in `packages/` as a live workspace member. The other
7 packages are spun out to local bare repos at `~/workspaces/spinouts/<name>.git`
and installed via `git+file://` URLs. History-only copies remain in
`archive/packages/<name>/`. See `docs/runbook/00-install.md`.

| Package | Source of truth | Role |
|---------|-----------------|------|
| [`lerobot-isaac-meta`](packages/lerobot-isaac-meta/) | live workspace | Umbrella CLI + workspace path resolver |
| `lerobot-isaac-env` | `~/workspaces/spinouts/lerobot-isaac-env.git` | Isaac Lab `ManagerBasedRLEnv` for SO-101 (obs / actions / rewards / DR) |
| `lerobot-isaac-adapters` | `~/workspaces/spinouts/lerobot-isaac-adapters.git` | Single `lerobot-isaac-train` entrypoint; dispatches by `--target_arch` |
| `lerobot-isaac-autoresearch` | `~/workspaces/spinouts/lerobot-isaac-autoresearch.git` | `program.md` configs + wrapper for the autoresearch HP-search loop |
| `lerobot-isaac-synthetic` | `~/workspaces/spinouts/lerobot-isaac-synthetic.git` | DR-replay synthetic-data pipeline + MimicGen bridge stub |
| `lerobot-isaac-configs` | `~/workspaces/spinouts/lerobot-isaac-configs.git` | Shared YAML configs per `target_arch` |
| `robot-data-recorder` | `~/workspaces/spinouts/robot-data-recorder.git` | RealSense D435 + SO-101 teleop dual-write recorder (standalone — not a meta dep) |
| `lerobot-isaac-dashboard` | `~/workspaces/spinouts/lerobot-isaac-dashboard.git` | Streamlit + Plotly metrics dashboard with snapshot + N-way compare |

> **TODO:** swap `file://` URLs for `https://github.com/kvgork/<name>.git` once
> GitHub repos are published. See `docs/runbook/09-publish-to-github.md`.

Three training backends, one CLI:

```bash
lerobot-isaac-train --target_arch smolvla     ...   # imitation policy
lerobot-isaac-train --target_arch dreamerv3   ...   # world model (sheeprl)
lerobot-isaac-train --target_arch le_world_model ... # world model (HF LeWorldModel)
```

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/kvgork/lerobot-isaac-training.git
cd lerobot-isaac-training
```

### 2. Install dependencies via pixi

The 6 sibling spun-out packages must exist as bare repos at `~/workspaces/spinouts/<name>/`
(no `.git` suffix) before `pixi install` can resolve them. If you haven't yet created them,
see `docs/runbook/00-install.md`.

```bash
curl -fsSL https://pixi.sh/install.sh | bash    # one-time
pixi install                                    # default env: meta editable + 6 git+file:// installs
```

#### Editable dev mode (opt-in)

Want to edit a sibling package and see changes reflected without reinstalling?
Use the opt-in `editable` env, which installs the 6 siblings as editable path deps
from local clones under `src/`:

```bash
pixi run sync               # clone the 6 spinouts into src/<name>/  (idempotent)
pixi install -e editable    # resolve siblings as editable path deps
pixi run -e editable test   # tests run against the editable clones

# Later: pull updates into the local clones
pixi run sync-update        # git fetch && git pull --ff-only on each src/lerobot-isaac-*
```

Default `pixi install` is unaffected — the two envs live side-by-side. Recorder dev is
also opt-in: `pixi run sync-recorder` clones it, then install manually with
`pixi run -e default pip install -e src/robot-data-recorder`.

#### Standalone install (no monorepo, no pixi)

```bash
bash scripts/install.sh
# or directly:
pip install "git+file:///path/to/lerobot-isaac-meta@main[post-spinout]"
```

The default `pixi install` activates `scripts/setup_env.sh`, which exports
`LEROBOT_ISAAC_WORKSPACE`, `CLAUDE_CODE_ROOT`, and `LEROBOT_CLAUDE_CODE_ROOT`
automatically inside the pixi environment.

Other environments (see `pixi.toml`):

| env | purpose |
|------|---------|
| `default` | dev tooling, lint, unit tests (siblings via git+file://) |
| `editable` | sibling dev with edit-and-reload (siblings via path dep from `src/`) |
| `train-policy` | LeRobot policy training (SmolVLA / ACT / Diffusion) |
| `train-dreamer` | DreamerV3 world-model training |
| `train-lewm` | HF LeWorldModel training |
| `sim` | Isaac Lab simulation (post `install-isaac-lab`) |
| `record` | SO-101 + D435 teleoperation recorder |
| `dashboard` | metrics dashboard (Streamlit + Plotly) |
| `full` | all features simultaneously |

```bash
pixi shell -e train-policy
```

### 3. Smoke test (no GPU required)

```bash
pixi run test                          # unit tests across all packages
pixi run lint && pixi run fmt          # ruff
lerobot-isaac-train --target_arch smolvla \
    --config packages/lerobot-isaac-configs/configs/policy_smolvla.yaml \
    --dry_run                          # verifies dispatch chain end-to-end
```

### 4. Optional — install Isaac Lab and download the SO-101 USD asset (GPU required)

```bash
pixi run install-isaac-lab             # ≈ 30 min, requires NVIDIA GPU + driver ≥ 535
pixi run download-usd                  # converts SO-101 URDF → USD via Isaac Lab
```

### 5. Optional — start the metrics dashboard

```bash
pixi run -e dashboard dashboard        # http://localhost:8501
```

---

## How it fits together

```
                                ┌─────────────────────────┐
   ┌──────────────┐    Parquet  │  robot-data-recorder    │
   │   SO-101 +   │────────────▶│  (D435 + teleop)        │
   │   D435 cam   │             └────────────┬────────────┘
   └──────────────┘                          │
                                             ▼  datasets/
                ┌────────────────────────────────────────────────┐
                │           datasets/<task>/<repo_id>/           │
                └─────────────┬───────────────────┬──────────────┘
                              │                   │
                              ▼                   ▼
              ┌─────────────────────┐   ┌───────────────────────┐
              │ lerobot-isaac-      │   │ lerobot-isaac-        │
              │ synthetic           │   │ adapters              │
              │ (DR replay +        │──▶│ (train.py dispatch    │
              │  MimicGen bridge)   │   │  by --target_arch)    │
              └─────────────────────┘   └──────────┬────────────┘
                                                   │
                          ┌──────────────┬─────────┴─────────┬──────────────┐
                          ▼              ▼                   ▼              ▼
                    ┌─────────┐   ┌─────────────┐   ┌──────────────┐   ┌──────────┐
                    │ SmolVLA │   │ DreamerV3   │   │ LeWorldModel │   │ ACT /    │
                    │  / ACT  │   │  (sheeprl)  │   │   (HF)       │   │ Diffusion│
                    └────┬────┘   └──────┬──────┘   └──────┬───────┘   └─────┬────┘
                         │               │                 │                 │
                         └───────────────┴────────┬────────┴─────────────────┘
                                                  ▼
                                       outputs/checkpoints/
                                       outputs/eval/
                                                  │
                                                  ▼
                                ┌──────────────────────────────────┐
                                │ lerobot-isaac-dashboard          │
                                │ (live UI + static report +       │
                                │  snapshots + N-way compare)      │
                                └──────────────────────────────────┘
```

The autoresearch loop (`lerobot-isaac-autoresearch`) wraps `lerobot-isaac-train`
to run mutation-driven hyperparameter search over any of the four backends.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full diagram, coupling rules,
and spinout mechanics.

---

## Documentation

| Want to … | Read |
|-----------|------|
| **Get the full pipeline picture in one doc** | [**`docs/pipeline-overview.md`**](docs/pipeline-overview.md) |
| Run everything end-to-end with one command | [`scripts/run_full_pipeline.sh`](scripts/run_full_pipeline.sh) (or `pixi run pipeline`) |
| Get started for the first time | [`docs/runbook/01-bootstrap.md`](docs/runbook/01-bootstrap.md) |
| Collect SO-101 teleop data | [`docs/runbook/02-collect-data.md`](docs/runbook/02-collect-data.md) |
| Train a policy | [`docs/runbook/03-train-policy.md`](docs/runbook/03-train-policy.md) |
| Train a world model | [`docs/runbook/04-train-world-model.md`](docs/runbook/04-train-world-model.md) |
| Generate synthetic data | [`docs/runbook/05-augment-with-dr.md`](docs/runbook/05-augment-with-dr.md) |
| Run autoresearch | [`docs/internals/autoresearch-integration.md`](docs/internals/autoresearch-integration.md) + [`USAGE.md`](USAGE.md) §F |
| View metrics / compare runs | [`docs/runbook/07-dashboard.md`](docs/runbook/07-dashboard.md) |
| Batch train multiple archs + auto-compare | [`docs/runbook/08-batch-train-and-compare.md`](docs/runbook/08-batch-train-and-compare.md) |
| Understand the architecture | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Browse the public Python API | [`docs/api-reference.md`](docs/api-reference.md) |
| Contribute code | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

The full reference layout:

- `README.md` — this file (overview, quickstart, package map)
- `ARCHITECTURE.md` — diagrams, data flow, coupling rules, glossary
- `USAGE.md` — every workflow with exact commands; CLI reference; common errors
- `CONTRIBUTING.md` — pre-commit, code style, PR flow, testing
- `CHANGELOG.md` — release history
- `docs/runbook/` — step-by-step task guides
- `docs/internals/` — implementation deep-dives
- `docs/concepts/` — design rationale (modular adapter, soft imports, monorepo, pixi)
- `docs/research/` — library reference notes (Isaac Lab, DreamerV3, LeWorldModel, MimicGen)
- `docs/adr/` — Architecture Decision Records
- `packages/<pkg>/CLAUDE.md` — per-package orientation

---

## Build Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Workspace bootstrap (skeleton, packages, configs, .gitignore) | Done |
| 1 | Isaac Lab SO-101 env wired (soft-import; cfg construction green; camera obs deferred) | Done |
| 2 | Modular training adapter (subprocess + metric extraction; dry-run smoke green) | Done |
| 3 | Autoresearch e2e dry-run (`train_wrapper → train → metric` chain) | Done |
| 4a | Isaac DR replay + parquet writer + merge utilities | Done |
| 4b | MimicGen bridge path | Deferred (gated by `LEROBOT_MIMICGEN_ENABLED=1`) |
| 5 | Documentation polish | Done |
| A | Metrics dashboard (live UI + static report + snapshots + compare; 281 tests) | Done |
| — | Real-data smoke against actual SO-101 teleop dataset | Pending data |
| — | Camera observation wiring (`wrist_camera_rgb` / `overhead_camera_rgb`) | Pending CameraCfg in scene |
| — | Insertion task (`tasks/insertion.py` Stage 5 stub) | Pending |

---

## External dependencies

This workspace expects two adjacent repositories at runtime; both are auto-detected
or overridable via env var (set up by `scripts/setup_env.sh` / pixi activation):

| Variable | Default search order | What it points to |
|----------|---------------------|-------------------|
| `CLAUDE_CODE_ROOT` | `LEROBOT_CLAUDE_CODE_ROOT` → `~/tools/claude_code` → sibling dir | `claude_code` repo (agents + skills) |
| `LEROBOT_ISAAC_WORKSPACE` | this repo root (auto) | this workspace root |
| `VAULT_ROOT` | `~/Documents/Vaults/Local` if present | optional Obsidian vault for cross-references |

Workspace agents and skills (referenced from this repo's docs) live in the
[`claude_code`](https://github.com/anthropics/claude-code) repo and are **not**
duplicated here.

System-level prerequisites (handled by the install scripts in `scripts/`):

- **Isaac Sim** (NVIDIA Omniverse) — required for Isaac Lab
- **Isaac Lab** v2.1.0 — installed via `pixi run install-isaac-lab`
- **LeRobot** — installed via `pixi run install-lerobot` or `pip install lerobot[all]`
- **NVIDIA GPU** + driver ≥ 535 (RTX 3080 or better recommended)

See [`scripts/README.md`](scripts/README.md) for full execution order.

---

## License

MIT — see [`LICENSE`](LICENSE).

## Contributing

Pull requests welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).
