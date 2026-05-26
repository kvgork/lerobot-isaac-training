# LeRobot + Isaac Lab Training Workspace — Orientation

**Workspace:** `~/workspaces/lerobot-isaac-training/`
**Purpose:** Isaac Lab + LeRobot + LeWorldModel monorepo for SO-101 manipulation training
**Status as of 2026-05-06:** Phases 0–5 complete (scaffolding). Training impl is future work.
**Build plan:** `${CLAUDE_CODE_ROOT}/plans/2026-05-06-lerobot-isaac-workspace-plan.md`

---

## Hard Scope Lock (2026-05-06)

This workspace currently contains ONLY scaffolding. No training runs, no synthetic data
generation. Each phase below adds real implementation.

---

## Repo–Workspace Contract

This workspace uses agents and skills from the `claude_code` repo — they are NOT duplicated here.

| What | Source of truth | Installed copy |
|------|----------------|----------------|
| Agents | `${CLAUDE_CODE_ROOT}/agents/` | `~/.claude/agents/` |
| Skills | `${CLAUDE_CODE_ROOT}/skills/` | (used in-repo) |
| Configs | `packages/lerobot-isaac-configs/configs/` | (this workspace) |
| Datasets | `datasets/` | (this workspace, gitignored) |
| Outputs | `outputs/` | (this workspace, gitignored) |
| Agent state | `.agent-state/` | (this workspace, gitignored) |

To update installed agents after editing source: `cd ${CLAUDE_CODE_ROOT} && ./install.sh`

---

## Architecture: Thin-Meta-Repo (post-spinout, 2026-05-13)

Only `lerobot-isaac-meta` lives in `packages/` as a live workspace member.
The other 6 siblings have been spun out to local bare git repos at
`~/workspaces/spinouts/<name>/` (NO `.git` suffix — they are bare directories whose
`config` declares `bare = true`) and are installed via `git+file://` URLs in the
default pixi environment.

A separate opt-in `editable` pixi environment installs the 6 siblings as editable
path deps from local clones at `src/<name>/` — for hands-on development without
reinstalls. See "Editable dev mode" below.

`robot-data-recorder` is a **standalone hardware package** and is NOT a meta dep.
It is opt-in via `pixi run sync-recorder`.

**TODO:** swap `file://` URLs for `https://github.com/kvgork/<name>.git` once GitHub
repos exist (see `docs/runbook/09-publish-to-github.md`).

## Package Map (8 packages, 1 live + 6 spun-out + 1 standalone)

| Package | Location | Install path (default env) |
|---------|----------|----------------------------|
| `lerobot-isaac-meta` | `packages/lerobot-isaac-meta/` (live) | editable workspace member |
| `lerobot-isaac-env` | `~/workspaces/spinouts/lerobot-isaac-env/` | `git+file://...@main` |
| `lerobot-isaac-adapters` | `~/workspaces/spinouts/lerobot-isaac-adapters/` | `git+file://...@main` |
| `lerobot-isaac-autoresearch` | `~/workspaces/spinouts/lerobot-isaac-autoresearch/` | `git+file://...@main` |
| `lerobot-isaac-synthetic` | `~/workspaces/spinouts/lerobot-isaac-synthetic/` | `git+file://...@main` |
| `lerobot-isaac-configs` | `~/workspaces/spinouts/lerobot-isaac-configs/` | `git+file://...@main` |
| `lerobot-isaac-dashboard` | `~/workspaces/spinouts/lerobot-isaac-dashboard/` | `git+file://...@main` |
| `robot-data-recorder` | `~/workspaces/spinouts/robot_data_recorder/` | standalone — opt-in via `pixi run sync-recorder` |

In `editable` env, the 6 siblings install from `src/<name>/` (editable=true, path dep)
instead. See "Editable dev mode" below.

History-only copies of the spun-out packages remain in `archive/packages/<name>/`
(via `git mv`, full history preserved). Treat them as read-only — edit the bare
repos (default mode) or the `src/<name>/` clones (editable mode) for canonical changes.

---

## Where to Find Things

- **Docs:** `docs/research/` (external lib refs), `docs/runbook/` (how-tos)
- **YAML configs:** `packages/lerobot-isaac-configs/configs/` (in default env, served by `importlib.resources` from the git+file install; in editable env, served from `src/lerobot-isaac-configs/src/lerobot_isaac_configs/configs/`)
- **Editable sibling clones (gitignored):** `src/<name>/` — only present after `pixi run sync`
- **Datasets (gitignored):** `datasets/`
- **Checkpoints/outputs (gitignored):** `outputs/`
- **Agent orchestration state (gitignored):** `.agent-state/{sessionId}/events.jsonl`
- **Package CLAUDE.md files:** each `packages/*/CLAUDE.md` (and each `src/*/CLAUDE.md` once cloned)
- **Top-level docs:** `README.md` | `ARCHITECTURE.md` | `USAGE.md`

---

## How to Continue Work

### Default workflow (siblings installed from git+file://, no source-level editing)

```bash
pixi install               # default env: meta editable + 6 git+file:// installs
pixi run test              # 659 passing, 14 skipped
```

### Editable dev mode (opt-in, edit-and-reload on siblings)

Use this mode when you want to make changes to a sibling package (e.g.
`lerobot-isaac-adapters`) and have them reflected in the workspace immediately
without `pip install --force-reinstall`.

```bash
pixi run sync              # clones the 6 spinouts into src/<name>/ (idempotent)
pixi install -e editable   # resolves 6 siblings as editable path deps
pixi shell -e editable     # enter the env
# edit src/lerobot-isaac-<pkg>/src/... — changes reflect on next import
pixi run sync-update       # later: pull updates from the bare repos
```

Then commit + push **inside the relevant `src/<pkg>/`** directory — each is an
independent git checkout of its bare repo. The workspace `.gitignore` ignores
`src/<pkg>/` entirely.

| Mode | Env | Sibling source |
|------|-----|----------------|
| Default | `default` | git+file:// from `~/workspaces/spinouts/<name>` |
| Editable | `editable` | path deps from `src/<name>/` (editable=true) |

Switching modes does NOT require uninstalling — `.pixi/envs/default/` and
`.pixi/envs/editable/` live side-by-side.

### Recorder dev (opt-in, separate from meta deps)

```bash
pixi run sync-recorder                           # clones into src/robot-data-recorder
pixi run -e default pip install -e src/robot-data-recorder   # install into chosen env
```

### Invoke training orchestrator (Phase 2+)
```
Task(lerobot-training-orchestrator, {workspace_root: "$PWD"})
```
The orchestrator expects:
- `datasets/` for input Parquet files
- `outputs/` for checkpoints
- Configs served by `lerobot_isaac_configs.load_config(...)` (resolved via importlib.resources)

### Train a world model (Phase 2+)
Step 1 — Convert Parquet to HDF5:
```
Task(lerobot-worldmodel-bridge, {input_parquet: "datasets/...", output_hdf5: "outputs/..."})
```
Step 2 — Train:
```bash
python -m lerobot_isaac_adapters.train \
  --target_arch dreamerv3 \
  --config <(python -c "from lerobot_isaac_configs import get_configs_dir; print(get_configs_dir() / 'wm_dreamerv3.yaml')")
```

### Run autoresearch (Phase 3+)
```
/autoresearch <path-to-program.md> --type ml_model
```

---

## Pixi Workspace

The root `pixi.toml` is the **active pixi workspace** for the entire monorepo.
Each `packages/<pkg>/pixi.toml` is **dormant** in monorepo mode — it only activates
after a package is spun out to a standalone repo via `git subtree split`.

### Available environments

| Environment | Features included | Use case |
|-------------|-------------------|----------|
| `default` | dev + git-siblings | Unit tests, linting, format — siblings installed from git+file:// |
| `editable` | dev + editable-siblings | Sibling dev with edit-and-reload — siblings from `src/<name>/` |
| `train-policy` | dev + lerobot + git-siblings | Train LeRobot policies (ACT / SmolVLA / Diffusion) |
| `train-dreamer` | dev + lerobot + dreamerv3 + git-siblings | Train DreamerV3 world model |
| `train-lewm` | dev + lerobot + leworldmodel + git-siblings | Train HF LeWorldModel |
| `sim` | dev + lerobot + isaaclab + git-siblings | Isaac Lab simulation (post-install) |
| `dashboard` | dev + dashboard + git-siblings | Live + static metrics dashboard |
| `full` | all features + git-siblings | All targets simultaneously |

The `git-siblings` and `editable-siblings` features are mutually exclusive — they
provide the same 6 package names from different URL sources. Pixi cannot resolve
both in one env, so the `editable` env opts out of `git-siblings`.

### Common commands

```bash
# Install all conda + pip deps for the default environment
pixi install

# Install for a specific environment
pixi install -e train-policy

# Editable dev mode: clone-and-install workflow
pixi run sync                  # one-time: clone 6 spinouts into src/<name>/
pixi install -e editable       # resolve as editable path deps
pixi run sync-update           # later: git fetch && git pull --ff-only on each clone
pixi run sync-recorder         # opt-in clone of robot_data_recorder into src/

# Run tests in the default environment
pixi run test

# Run tests inside the editable environment
pixi run -e editable test

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
- [x] Phase B — Thin-meta-repo + opt-in editable-siblings workflow (`pixi run sync` + `pixi install -e editable`)
- [x] Phase 1 impl — Isaac Lab MDP wiring (soft-import; cfg construction green; camera obs deferred)
- [x] Phase 2 impl — LeRobot / DreamerV3 / LeWM backends wired (subprocess + metric extraction; dry-run smoke green)
- [x] Phase 3 impl — Autoresearch e2e dry-run green (`train_wrapper → train → metric` chain enforced by test)
- [x] Phase 4a impl — Isaac DR replay + parquet writer + merge utilities wired (dry-run green)
- [ ] Phase 4b impl — MimicGen bridge path (deferred per plan; gated by `LEROBOT_MIMICGEN_ENABLED=1`)
- [ ] Real-data smoke — repeat dry-run smoke against actual SO-101 teleop dataset once collected
- [x] Camera observation wiring — `d435_rgb` `CameraCfg` wired in `so101_env_cfg.py` matching real D435 wrist cam (DR100 Phase 1, commit `592b53d`). Runtime verification deferred until GPU/Isaac Lab available.
- [ ] Insertion task — `tasks/insertion.py` Stage 5 stub (`NotImplementedError`)

---

## Reused Agents (10) — Source Paths

| Agent | Source path |
|-------|-------------|
| `lerobot-training-orchestrator` | `${CLAUDE_CODE_ROOT}/agents/orchestrators/lerobot-training-orchestrator.md` |
| `lerobot-data-collection-agent` | `${CLAUDE_CODE_ROOT}/agents/workers/lerobot-data-collection-agent.md` |
| `lerobot-evaluation-agent` | `${CLAUDE_CODE_ROOT}/agents/workers/lerobot-evaluation-agent.md` |
| `lerobot-sim-augmentation-agent` | `${CLAUDE_CODE_ROOT}/agents/workers/lerobot-sim-augmentation-agent.md` |
| `lerobot-curriculum-agent` | `${CLAUDE_CODE_ROOT}/agents/orchestrators/lerobot-curriculum-agent.md` |
| `lerobot-worldmodel-bridge` | `${CLAUDE_CODE_ROOT}/agents/lerobot-worldmodel-bridge.md` |
| `lerobot-specialist` | `${CLAUDE_CODE_ROOT}/agents/lerobot-specialist.md` |
| `autoresearch-loop-orchestrator` | `${CLAUDE_CODE_ROOT}/agents/orchestrators/autoresearch-loop-orchestrator.md` |
| `autoresearch-ml-executor-worker` | `${CLAUDE_CODE_ROOT}/agents/workers/autoresearch-ml-executor-worker.md` |
| `autoresearch-ml-proposer-worker` | `${CLAUDE_CODE_ROOT}/agents/workers/autoresearch-ml-proposer-worker.md` |

## Reused Skills (4) — Source Paths

| Skill | Source path |
|-------|-------------|
| `lerobot_world_model_bridge` | `${CLAUDE_CODE_ROOT}/skills/lerobot_world_model_bridge/` |
| `lerobot_mimicgen_bridge` | `${CLAUDE_CODE_ROOT}/skills/lerobot_mimicgen_bridge/` |
| `lerobot_dataset_quality` | `${CLAUDE_CODE_ROOT}/skills/lerobot_dataset_quality/` |
| `autoresearch` | `${CLAUDE_CODE_ROOT}/skills/autoresearch/` |

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
- **`pixi install -e editable` requires `pixi run sync` first** — the editable env's
  path deps cannot canonicalize if `src/<pkg>/` does not yet exist on disk.
- **Bare-repo URLs have no `.git` suffix.** The on-disk bare repos are named
  `~/workspaces/spinouts/<name>/` (NOT `<name>.git`). Recorder is the exception:
  `~/workspaces/spinouts/robot_data_recorder/` (underscore, working tree).
- **No eager `from . import <runnable_module>` in package `__init__.py`:** if the
  submodule is also invokable as `python -m <pkg>.<mod>`, eager re-export triggers
  `RuntimeWarning: '<pkg>.<mod>' found in sys.modules`. Use a deferred local
  import in callers, or document a `from <pkg>.<mod> import main` form. Affects
  `lerobot_isaac_adapters.train`, `lerobot_isaac_autoresearch.train_wrapper`,
  `lerobot_isaac_synthetic.isaac_dr.replay_runner`.
- **Dry-run is the default acceptance bar pre-data:** every entrypoint must accept
  `--dry_run`, print the resolved subprocess command, and exit 0 — never actually
  reach the heavy backend.
- **Heavy training deps are NOT pip-installed by `pixi install`.** Run
  `bash scripts/install_train_deps.sh` (or `pixi run install-train-deps`) once after
  `pixi install` to put `lerobot` in `train-policy`/`train-lewm` and `sheeprl` (from
  git, `--ignore-requires-python` on Py3.12) in `train-dreamer`. Documented in
  `docs/runbook/00-install.md §Step 4`.
- **lerobot 0.5+ CLI broke older adapter flags.** Adapter `policy_lerobot.py` emits
  `--batch_size` / `--steps` / `--optimizer.lr` / `--config_path` /
  `--policy.push_to_hub=false` — NOT the legacy `--training.*` / `--config` shape.
  If you see `unknown argument --training.batch_size` from `lerobot-train`, the
  adapter is stale and must be reinstalled from the bare repo.
- **Local LeRobotDataset path:** pass the on-disk root as `--dataset`. The adapter
  splits it into `--dataset.repo_id=<parent>/<name>` + `--dataset.root=<path>`.
  Do NOT pre-flatten it into an HF cache layout.
- **LeWorldModel real training is BLOCKED (2026-05-13).** `lerobot 0.5.x` does not
  ship `lerobot.scripts.train_world_model`. The `le_world_model` adapter dry-run
  works but real dispatch fails. Use `--target_arch dreamerv3` for any actual
  WM training until upstream lands a CLI.
- **Bridge `dtype: image` (PNG bytes in parquet) is now supported** as of
  claude_code commit 4e6e21c — the older bridge required MP4 files under
  `videos/`. If you re-pin the bridge skill, ensure the
  `_load_episode_frames_from_parquet` helper is present (cv2-free PIL path).
- **SmolVLA throughput on RTX 3080 / so101-pickplace1.** Measured 2026-05-15
  via `scripts/_smoke_train.sh`.
  - **Without cache:** 1.45 step/s = 5.8 samples/s (data-bound — CPU PNG
    decode dominates). VRAM 4.4 GB. 20k steps ≈ 230 min.
  - **With `--cache_frames`:** **10.1 step/s = 40 samples/s** (7× win;
    compute-bound on `updt_s=0.095 s`). VRAM 3.8 GB train + ~7 GB RAM
    cache (uint8). Warmup cost: ~16 min for 7491 rows at 4 DataLoader
    workers. 20k steps ≈ 33 min train + 16 min warmup = 49 min total.
  - **Disk-cached warmup:** the post-warmup cache is pickled to
    `outputs/cache_storage/<sig>.pt` (6.94 GB) and reloaded in **6.2 s**
    on subsequent runs with the same dataset signature. Saves ~15.9 min
    per subprocess — critical for autoresearch sweeps that spawn N
    trials. Disable via `LEROBOT_ISAAC_CACHE_DISK_DIR=none`.
  - **batch_size>4 hurts:** doubling batch halves step-rate (samples/s
    unchanged) — bigger batches starve the data loader. Drop to 2 only
    on OOM.
  - Use the cached path for any real run; uncached only for diagnostics.
    See `plans/2026-05-15-dataloader-gpu-decode-plan.md` (approach A).
- **LeRobotDataset returns FLOAT32 normalized images, NOT uint8** (lerobot 0.5.1).
  Per-row shape is also `(T, 3, H, W)` not `(3, H, W)` — the `T` dim comes
  from `cfg.policy.observation_delta_indices` (defaults to `[0]` for SmolVLA,
  giving T=1 but still 4-D). Pad masks `<key>_is_pad` are also returned
  per row. Any in-RAM cache wrapper must:
  1. Detect 4-D image tensors (not just 3-D).
  2. uint8-compress + lazy-decast to float32/255 on read (saves 4× RAM,
     keeps quantization error ≤ 1/255).
  3. Parallelize the warmup decode via `torch.utils.data.DataLoader(
     num_workers=N)` — single-thread decode is ~3× slower than the
     steady-state lerobot dataloader (which already runs 4 workers).
  Implementation lives in
  `lerobot_isaac_adapters.data.cached_dataset.CachedDatasetWrapper` and
  ships with the `--cache_frames` flag on `lerobot_isaac_adapters.train`.
  Enabled cache size for so101-pickplace1 = 6.9 GB uint8 (vs 27.6 GB
  float32 — would blow any reasonable RAM cap).
- **Smoke-script I/O buffering** (`scripts/_smoke_train.sh`): subshell +
  shell redirect + `timeout SIGTERM` previously discarded the python
  subprocess stdout when the watchdog fired (0-byte `train.log`).
  Fix: drop the subshell, use `PYTHONUNBUFFERED=1 stdbuf -oL -eL python
  -u` + a direct `>> "$TRAIN_LOG" 2>&1` append-redirect. Confirmed via
  smoke E → smoke G chain. If you reuse the watchdog pattern in any
  new script, copy this shape, not the legacy subshell+pipe one.
- **SmolVLA needs `--policy.load_vlm_weights=true` explicitly.** Default
  `SmolVLAConfig.load_vlm_weights=False` leaves the SmolVLM2-500M backbone
  at random init — useless even though `freeze_vision_encoder=True` and
  `train_expert_only=True` are sensible defaults. The adapter does NOT add
  this flag automatically; pass it via the `--` remainder.
  - To resume from a checkpoint: `--policy.pretrained_path=<DIR>` (Path,
    not HF repo id — `--policy.path=...` does NOT exist in lerobot 0.5.1).
  - First launch downloads ~2 GB from
    `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. Prefetch via
    `bash scripts/_run_smolvla_tonight.sh --prefetch-weights` to keep the
    training watchdog budget for actual training.
  - VRAM: batch 4 fits on RTX 3080 10GB with the default
    expert-only-trained config. Drop to batch 2 if OOM.

---

## Vault Links (Second Brain context)

- SO-101 hardware / sim notes: `${VAULT_ROOT}/05-Wiki/entities/SO-101.md`
- LeWorldModel schema: `${VAULT_ROOT}/05-Wiki/entities/LeWorldModel.md`
- MimicGen integration: `${VAULT_ROOT}/05-Wiki/entities/MimicGen.md`
- World-Models RTX-3080 fit table: `${VAULT_ROOT}/05-Wiki/concepts/World-Models-(Robot-Manipulation).md`
- Autonomous ML Training Loop: `${VAULT_ROOT}/05-Wiki/concepts/Autonomous-ML-Training-Loop.md`

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
| `docs/pipeline-overview.md` | **End-to-end pipeline walkthrough**: data collection → autoresearch. Single source of truth for stage layout, contracts, and recent bugfix trail. |
| `docs/api-reference.md` | Public Python API for all 6 packages: signatures + examples |
| `docs/runbook/00-install.md` | Thin-meta-repo install: default + editable modes + recorder dev |
| `docs/runbook/01-bootstrap.md` | First-time setup: pixi, Isaac Lab, USD, smoke tests |
| `docs/runbook/02-collect-data.md` | Collect and quality-filter SO-101 teleop data |
| `docs/runbook/03-train-policy.md` | Train SmolVLA / ACT / Diffusion policy end-to-end |
| `docs/runbook/04-train-world-model.md` | Train DreamerV3 or LeWorldModel |
| `docs/runbook/05-augment-with-dr.md` | Generate DR synthetic data via Isaac Lab replay |
| `docs/runbook/06-augment-with-mimicgen.md` | MimicGen augmentation (deferred path) |
| `docs/runbook/07-dashboard.md` | Live + static metrics dashboard: start, tabs, snapshots, compare, troubleshoot |
| `docs/runbook/08-batch-train-and-compare.md` | Batch-train multiple `target_arch`s sequentially and auto-render N-way compare report |
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
| Get started for the first time | `docs/runbook/00-install.md` then `docs/runbook/01-bootstrap.md` |
| Edit a sibling package's source | `docs/runbook/00-install.md` §"Editable dev mode" |
| Collect real SO-101 data | `docs/runbook/02-collect-data.md` |
| Train a policy | `docs/runbook/03-train-policy.md` + `docs/research/` for the chosen arch |
| Train a world model | `docs/runbook/04-train-world-model.md` + `docs/research/dreamerv3-reference.md` |
| Generate synthetic data | `docs/runbook/05-augment-with-dr.md` |
| Run autoresearch HP search | `programs/README.md` → `bash scripts/run_autoresearch.sh --program <name>` (domain-aware) + `docs/internals/autoresearch-integration.md` |
| View metrics / compare runs | `docs/runbook/07-dashboard.md` |
| **See the full pipeline in one doc** | **`docs/pipeline-overview.md`** |
| Run everything end-to-end (one cmd) | `scripts/run_full_pipeline.sh` (or `pixi run pipeline`) |
| **Deploy trained policy on real SO-101** | **`docs/runbook/10-deploy-to-hardware.md`** + `lerobot-isaac-deploy` |
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
