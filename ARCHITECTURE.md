# Architecture — LeRobot + Isaac Lab Training Workspace

**Status as of 2026-05-06:** Phases 0–5 complete (scaffolding). Training/eval wiring is future work.
**Plan reference:** `/home/koen/tools/claude_code/plans/2026-05-06-lerobot-isaac-workspace-plan.md`

---

## System Overview

This workspace is an eight-package Python monorepo that connects the SO-101 robot arm (physical hardware) to three training backends (LeRobot imitation-learning policies, DreamerV3 world model, HF LeWorldModel) via a single unified training entrypoint. An Isaac Lab simulation environment provides domain-randomized synthetic episodes to augment the real teleoperation corpus. A standalone autoresearch loop (driven by the `autoresearch-loop-orchestrator` agent from the `claude_code` repo) performs automated hyperparameter search over any of the three backends. The `lerobot-isaac-dashboard` package provides a read-only metrics surface over all pipeline artefacts (local files only). All agents and skills referenced here live in `/home/koen/tools/claude_code/` and are NOT duplicated in this workspace.

---

## Component Diagram

```
+--------------------------------------------------------------------------+
|          LeRobot + Isaac Lab Training Workspace                          |
|          ~/workspaces/lerobot-isaac-training/                           |
|                                                                          |
|  +----------------------------------------------------------------------+|
|  |  lerobot-isaac-meta  (umbrella — depends on all siblings)            ||
|  |    lerobot-isaac  CLI  |  workspace_paths.py path resolver           ||
|  +---------------+--------+------+---------------------+---------+------+|
|                  |               |                     |         |       |
|    +-------------+----+  +-------+-----------+  +-----+----+    |       |
|    | lerobot-isaac-env|  |lerobot-isaac-     |  |lerobot-  |    |       |
|    |                  |  |adapters           |  |isaac-    |    |       |
|    |  SO101EnvCfg     |  |                   |  |autorese- |    |       |
|    |  ManagerBased-   |  |  train.py         |  |arch      |    |       |
|    |  RLEnv (Isaac)   |  |  --target_arch--> |  |          |    |       |
|    |                  |  |   policy_lerobot  |  |programs/ |    |       |
|    |  observations.py |  |   wm_dreamerv3    |  |  *.md    |    |       |
|    |  actions.py      |  |   wm_leworldmodel |  |          |    |       |
|    |  rewards.py      |  |  metric_extractor |  |train_    |    |       |
|    |  terminations.py |  |  isaac_data_      |  |wrapper.py|    |       |
|    |  randomization.py|  |  recorder.py      |  |          |    |       |
|    |  tasks/          |  +-------------------+  +----------+    |       |
|    |    pick.py       |                                          |       |
|    |    pick_place.py |  +-----------------------------------+   |       |
|    |    insertion.py  |  | lerobot-isaac-synthetic           |   |       |
|    |  assets/usd/     |  |                                   |   |       |
|    +------------------+  |  isaac_dr/replay_runner.py        |   |       |
|                          |  isaac_dr/parquet_writer.py        |   |       |
|                          |  mimicgen/bridge_invocation.py     |   |       |
|                          |  merge_utilities.py               |   |       |
|                          +-----------------------------------+   |       |
|                                                                   |       |
|  +---------------------------------------------------------------+----+  |
|  |  lerobot-isaac-configs  (leaf — no internal deps)                  |  |
|  |    configs/policy_smolvla.yaml   wm_dreamerv3.yaml   ...          |  |
|  +--------------------------------------------------------------------+  |
|                                                                          |
|  +--------------------------------------------------------------------+  |
|  |  lerobot-isaac-dashboard  (read-only leaf — no code coupling)     |  |
|  |    loaders/  — 9 loaders for local artefacts                      |  |
|  |    tabs/     — 8 Streamlit tab modules (live + static dual-render) |  |
|  |    snapshots.py  — save/load/list workspace snapshots             |  |
|  |    compare.py    — 2-way and N-way snapshot comparison            |  |
|  |    report.py     — static HTML exporter                           |  |
|  |                                                                    |  |
|  |  Reads from (arrows IN):                                          |  |
|  |    datasets/           <-- load_parquet_dataset, load_synthetic   |  |
|  |    outputs/            <-- load_eval_results, load_checkpoints,   |  |
|  |                            load_training_logs, load_curriculum     |  |
|  |    .agent-state/       <-- load_autoresearch, load_events         |  |
|  |                                                                    |  |
|  |  Writes to (arrows OUT — training artefacts never touched):       |  |
|  |    outputs/snapshots/  -- snapshot save                           |  |
|  |    outputs/reports/    -- static HTML export                      |  |
|  +--------------------------------------------------------------------+  |
|                                                                          |
+--------------------------------------------------------------------------+

External systems (NOT in this workspace):

  Isaac Lab   <----  lerobot-isaac-env  +  synthetic/isaac_dr
  LeRobot     <----  lerobot-isaac-adapters/targets/policy_lerobot.py
  DreamerV3   <----  lerobot-isaac-adapters/targets/wm_dreamerv3.py
  LeWorldModel<----  lerobot-isaac-adapters/targets/wm_leworldmodel.py
  W&B         <----  metric_extractor (optional; stdout fallback always active)
  MimicGen    <----  lerobot-isaac-synthetic/mimicgen/ (deferred)

Agent/skill layer (claude_code repo — NOT duplicated here):
  /home/koen/tools/claude_code/
    agents/orchestrators/
      lerobot-training-orchestrator.md
      lerobot-curriculum-agent.md
      autoresearch-loop-orchestrator.md
    agents/workers/
      lerobot-data-collection-agent.md
      lerobot-evaluation-agent.md
      lerobot-sim-augmentation-agent.md
      autoresearch-ml-executor-worker.md
      autoresearch-ml-proposer-worker.md
    agents/
      lerobot-worldmodel-bridge.md
      lerobot-specialist.md
    skills/
      lerobot_world_model_bridge/
      lerobot_mimicgen_bridge/
      lerobot_dataset_quality/
      autoresearch/
```

---

## Data Flow Diagrams

### (a) Real-Data Pipeline: SO-101 Teleop to LeRobotDataset

```
  SO-101 Robot (hardware)
         |
         | teleoperation (30 Hz, joint positions + wrist/overhead camera)
         v
  LeRobotDataset Parquet  (datasets/so101_<task>_raw/)
         |
         | lerobot_dataset_quality skill (SAL + TED filtering)
         | agent: lerobot-data-collection-agent
         v
  Filtered Parquet  (datasets/so101_<task>_v1_filtered/)
         |
         | source="real" tag in meta/info.json
         v
  [ready for training or augmentation]
```

### (b) Sim-DR Pipeline: Real Episodes to Augmented Dataset

```
  Filtered Parquet  (datasets/so101_<task>_v1_filtered/)
         |
         | lerobot-isaac-synthetic/isaac_dr/replay_runner.py
         | reads each episode, replays in Isaac Lab with EventTermCfg DR
         |   - object pose  +-10 cm (Stage 1: 0)
         |   - lighting variation
         |   - table + gripper friction  [0.3, 1.2]
         |   - camera FOV jitter
         v
  DR episodes  (source="sim_dr") written by parquet_writer.py
         |
         | lerobot-isaac-synthetic/merge_utilities.merge_datasets()
         | real_weight=1.0, dr_weight=0.5 (configurable)
         v
  Merged Parquet  (datasets/so101_<task>_merged/)
         |
         | meta/info.json updated: episodes from both sources
         | source column: "real" | "sim_dr"
         v
  [ready for training]
```

### (c) MimicGen Pipeline: Real Demos to Augmented Dataset (DEFERRED)

```
  Filtered Parquet  (datasets/so101_<task>_v1_filtered/)
         |
         | lerobot_mimicgen_bridge skill  (Parquet -> MimicGen HDF5)
         v
  MimicGen HDF5  (end-effector-space format)
         |
         | MimicGen augmentation (runs INTERNALLY in MuJoCo/robosuite)
         | agent: lerobot-sim-augmentation-agent
         v
  Augmented MimicGen HDF5  (N * real demos -> synthetic demos)
         |
         | lerobot_mimicgen_bridge skill  (MimicGen HDF5 -> Parquet)
         v
  MimicGen Parquet  (source="mimicgen")
         |
         | merge_utilities.merge_datasets() with dr + real
         v
  Merged Parquet  (source="real" | "sim_dr" | "mimicgen")

  NOTE: Enable with LEROBOT_MIMICGEN_ENABLED=1
        Requires: pip install mimicgen robosuite
        Current status: bridge_invocation.py raises NotImplementedError
```

### (d) Training Pipeline: Dataset to Checkpoint

```
  Filtered/Merged Parquet  (or HDF5 for world models)
         |
         | lerobot-isaac-train  (packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/train.py)
         | --target_arch {smolvla|act|diffusion|dreamerv3|le_world_model}
         | --config packages/lerobot-isaac-configs/configs/<config>.yaml
         |
         +--[smolvla/act/diffusion]--> targets/policy_lerobot.py
         |      subprocess: python -m lerobot.scripts.train
         |      metric emitted: pc_success=<float>
         |
         +--[dreamerv3]-------------> targets/wm_dreamerv3.py
         |      auto-converts Parquet->HDF5 via lerobot_world_model_bridge
         |      subprocess: sheeprl dreamer_v3  (or dreamer-v3-pytorch)
         |      image_size: 64x64  (RTX 3080 budget)
         |      metric emitted: recon_loss=<float>
         |
         +--[le_world_model]--------> targets/wm_leworldmodel.py
                auto-converts Parquet->HDF5 via lerobot_world_model_bridge
                subprocess: leworldmodel train
                image_size: 96x96  (requires gradient checkpointing + AMP)
                metric emitted: pred_loss=<float>
         |
         v
  outputs/<run_name>/checkpoints/  (gitignored)
         |
         | metric_extractor.emit("<metric>=<float>") on every eval step
         v
  stdout line: <metric>=<float>
```

### (e) Autoresearch Loop: Autonomous Hyperparameter Search

```
  User writes / edits:
  packages/lerobot-isaac-autoresearch/programs/<target>.md
  (defines: metric, direction, baseline command, mutation operators, budget)
         |
         | /autoresearch <program_path> --type ml_model
         v
  autoresearch-loop-orchestrator  (reads program.md)
         |
         +---> autoresearch-ml-proposer-worker
         |       applies 1 of 6 mutation operators to current best config
         |       returns: new hyperparameter dict
         |
         +---> autoresearch-ml-executor-worker
                 calls: packages/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py
                   which calls: lerobot-isaac-train --target_arch ... <mutated args>
                 parses stdout: last line matching <metric>=<float>
                 logs result to: skills/autoresearch/ metric history
         |
         | loop until plateau_limit exceeded or budget exhausted
         v
  Best hyperparameter config  +  history in .agent-state/<session>/

  Three programs (separate metric/direction):
    lerobot-policy.md  -> pc_success  (maximize)
    dreamerv3.md       -> recon_loss  (minimize)
    leworldmodel.md    -> pred_loss   (minimize)
```

### (f) Dashboard Pipeline: Artefacts to Metrics Surface

```
  datasets/  +  outputs/  +  .agent-state/
       |               |               |
       |               |               |
       v               v               v
  lerobot-isaac-dashboard  (read-only; no GPU deps)
       |
       | loaders/   — reads local files, returns LoaderResult (never raises)
       | tabs/      — renders Plotly figures (dual-render: live + static)
       |
       +--[live]--> Streamlit UI  http://localhost:8501
       |              pixi run -e dashboard dashboard
       |
       +--[static]-> outputs/reports/<run_id>/report.html
       |              pixi run -e dashboard report
       |
       +--[snapshot]-> outputs/snapshots/<id>/
       |                 pixi run -e dashboard snapshot --label=baseline
       |
       +--[compare]--> outputs/reports/compare-<A>-vs-<B>/report.html
                         pixi run -e dashboard compare --snapshots A B
                         pixi run -e dashboard compare --snapshots A B C --mode nway

  IMPORTANT: dashboard is read-only w.r.t. training artefacts.
  It only writes to outputs/snapshots/ and outputs/reports/.
```

---

## State Machine

The `lerobot-training-orchestrator` drives a 6-stage curriculum. Below is the full state machine.

### ASCII State Diagram

```
        [START]
            |
            v
      +----------+
      |  COLLECT  |  <--- lerobot-data-collection-agent
      +----------+        quality filter: SAL + TED
            |
            | dataset size >= min_episodes
            v
  +------------------+
  |  TRAIN_BASELINE  |  <--- lerobot-isaac-train --target_arch smolvla
  +------------------+       (or act / diffusion per config)
            |
            | training complete (max_steps or time budget reached)
            v
      +-----------+
      |  EVAL_GATE |  <--- lerobot-evaluation-agent
      +-----------+        reads pc_success from W&B or stdout
            |
            +-- pc_success < 0.5 ---------> COLLECT  (collect more data)
            |
            +-- 0.5 <= pc_success < 0.80 -> AUGMENT_SIM
            |
            +-- pc_success >= 0.80 -------> ADVANCE_CURRICULUM
            |
            v
    +-------------+
    | AUGMENT_SIM |  <--- lerobot-isaac-synthetic (DR replay)
    +-------------+       expands dataset with sim_dr episodes
            |
            | DR dataset generated and merged
            v
  +------------------+
  |   TRAIN_FULL     |  <--- lerobot-isaac-train on merged dataset
  +------------------+
            |
            | training complete
            v
      +-----------+
      |  EVAL_GATE |  (second pass)
      +-----------+
            |
            +-- pc_success < 0.80 ---------> TRAIN_FULL  (repeat)
            |
            +-- pc_success >= 0.80 -------> ADVANCE_CURRICULUM
            |
            v
  +--------------------+
  | ADVANCE_CURRICULUM |  <--- lerobot-curriculum-agent
  +--------------------+       increments stage; updates Isaac env cfg
            |
            +-- stage < 6 ----------------> COLLECT  (next stage)
            |
            +-- stage == 6 ---------------> DONE
            |
            v
         [DONE]
```

### State Transition Table

| From State          | Condition                             | To State            | Agent/Module               |
|---------------------|---------------------------------------|---------------------|----------------------------|
| START               | always                                | COLLECT             | user / orchestrator init   |
| COLLECT             | dataset size >= min_episodes          | TRAIN_BASELINE      | lerobot-data-collection-agent |
| COLLECT             | quality_filter rejects all episodes   | COLLECT             | lerobot-dataset_quality skill |
| TRAIN_BASELINE      | max_steps reached or time budget      | EVAL_GATE           | lerobot-isaac-train         |
| EVAL_GATE           | pc_success < 0.5                      | COLLECT             | lerobot-evaluation-agent   |
| EVAL_GATE           | 0.5 <= pc_success < 0.80              | AUGMENT_SIM         | lerobot-evaluation-agent   |
| EVAL_GATE           | pc_success >= 0.80                    | ADVANCE_CURRICULUM  | lerobot-evaluation-agent   |
| AUGMENT_SIM         | DR episodes generated                 | TRAIN_FULL          | lerobot-isaac-synthetic    |
| TRAIN_FULL          | max_steps reached or time budget      | EVAL_GATE           | lerobot-isaac-train         |
| EVAL_GATE (2nd)     | pc_success < 0.80                     | TRAIN_FULL          | lerobot-evaluation-agent   |
| EVAL_GATE (2nd)     | pc_success >= 0.80                    | ADVANCE_CURRICULUM  | lerobot-evaluation-agent   |
| ADVANCE_CURRICULUM  | stage < 6                             | COLLECT             | lerobot-curriculum-agent   |
| ADVANCE_CURRICULUM  | stage == 6                            | DONE                | lerobot-curriculum-agent   |

Thresholds are configurable per stage in `packages/lerobot-isaac-configs/configs/isaac_so101_pickplace.yaml`.

---

## Cross-Package Coupling

Full coupling rules (from plan §11.6):

| Package | Allowed Imports | Forbidden Imports | Rationale |
|---------|----------------|-------------------|-----------|
| `lerobot-isaac-env` | `isaaclab.*`, `torch`, `numpy` | all siblings | Pure Isaac Lab; safe to spin out |
| `lerobot-isaac-configs` | `yaml`, `pathlib` only | all siblings | Leaf; all siblings may import it |
| `lerobot-isaac-adapters` | `lerobot-isaac-configs` | `lerobot-isaac-env` (direct) | Accesses env via `isaac_data_recorder.py` soft-import only |
| `lerobot-isaac-autoresearch` | `lerobot-isaac-configs` | all other siblings | Calls adapters as subprocess, never imports |
| `lerobot-isaac-synthetic` | `lerobot-isaac-env` (soft), `lerobot-isaac-configs` | `lerobot-isaac-adapters`, `autoresearch` | Isaac imports deferred to function bodies |
| `lerobot-isaac-meta` | all siblings | none | Umbrella; explicitly depends on all |
| `lerobot-isaac-dashboard` | `lerobot-isaac-meta` (soft) | all other siblings (code) | Read-only artefact consumer; accesses siblings only via file system |

**All heavy-dependency imports are soft** (wrapped in `try/except ImportError`):
- `isaaclab.*` — deferred; tests run without Isaac Sim
- `lerobot.*` — deferred; dry-run mode works without LeRobot
- `sheeprl.*` / `dreamer.*` — deferred; only needed for DreamerV3 target
- `transformers.*` (LeWorldModel) — deferred; only needed for LeWM target

---

## Repo-Workspace Contract

Full table of file/dir roles, locations, and ownership:

| File / Directory | Role | Location | Owner |
|-----------------|------|----------|-------|
| `agents/orchestrators/*.md` | Agent source of truth (edit here) | `/home/koen/tools/claude_code/agents/` | claude_code repo |
| `agents/workers/*.md` | Worker agent source of truth | `/home/koen/tools/claude_code/agents/workers/` | claude_code repo |
| `skills/*/` | Skill source of truth | `/home/koen/tools/claude_code/skills/` | claude_code repo |
| `~/.claude/agents/` | Installed agent copies (what Claude invokes) | `~/.claude/agents/` | `install.sh` |
| `plans/*.md` | Build plans and experiment plans | `/home/koen/tools/claude_code/plans/` | claude_code repo |
| `packages/*/` | Python implementation | `~/workspaces/lerobot-isaac-training/packages/` | this workspace |
| `packages/lerobot-isaac-configs/configs/` | YAML configs per target_arch | this workspace | this workspace |
| `datasets/` | LeRobot Parquet datasets (gitignored) | this workspace | this workspace |
| `outputs/` | Training checkpoints (gitignored) | this workspace | this workspace |
| `outputs/snapshots/` | Dashboard snapshots (gitignored) | this workspace | lerobot-isaac-dashboard |
| `outputs/reports/` | Static HTML reports (gitignored) | this workspace | lerobot-isaac-dashboard |
| `.agent-state/` | Orchestrator run state (gitignored) | this workspace | agents at runtime |
| `docs/` | Cross-package documentation | this workspace | this workspace |
| `pixi.toml` | Workspace environment definition | this workspace root | this workspace |
| `pyproject.toml` | uv workspace umbrella | this workspace root | this workspace |
| `CLAUDE.md` | Session orientation for any `cd` into workspace | this workspace root | this workspace |

To deploy agent edits: `cd /home/koen/tools/claude_code && ./install.sh`

---

## Pixi Workspace Layout

The root `pixi.toml` defines environments as combinations of features. Each `packages/*/pixi.toml` is **dormant** in monorepo mode — it only activates when the package is spun out.

| Environment | Features | Use Case | Heavy Deps |
|-------------|---------|---------|-----------|
| `default` | `dev` | Unit tests, lint, format | none |
| `train-policy` | `dev` + `lerobot` | Train ACT / SmolVLA / Diffusion | LeRobot |
| `train-dreamer` | `dev` + `lerobot` + `dreamerv3` | Train DreamerV3 world model | LeRobot + sheeprl |
| `train-lewm` | `dev` + `lerobot` + `leworldmodel` | Train HF LeWorldModel | LeRobot + transformers |
| `sim` | `dev` + `lerobot` + `isaaclab` | Isaac Lab simulation (post-install) | Isaac Sim + Isaac Lab |
| `dashboard` | `dev` + `dashboard` | Live + static metrics dashboard | streamlit + plotly |
| `full` | all features | All targets simultaneously | all of the above |

Feature composition is additive — `full` is the union of all individual features.
Isaac Lab is NOT installed by `pixi install`; run `pixi run install-isaac-lab` separately.

---

## Spinout Mechanics

Each package under `packages/` is independently pip-installable and can be extracted to a standalone Git repository.

### Why This Works

Each package has:
- Its own `pyproject.toml` with explicit dependency declarations
- No hardcoded workspace-relative paths
- Soft imports of sibling packages that become PyPI dependencies post-spinout

### Spinout Procedure

```bash
# Option A: git subtree split (preserves history for the package subtree)
git subtree split -P packages/lerobot-isaac-env -b spinout-env
git push git@github.com:yourorg/lerobot-isaac-env.git spinout-env:main

# Option B: git filter-repo (cleaner — rewrites all commits to remove other packages)
git clone ~/workspaces/lerobot-isaac-training /tmp/lerobot-isaac-env-repo
cd /tmp/lerobot-isaac-env-repo
pip install git-filter-repo
git filter-repo --path packages/lerobot-isaac-env/ --path-rename packages/lerobot-isaac-env/:

# After spinout: update pyproject.toml
# Change: lerobot-isaac-configs = {path = "../lerobot-isaac-configs", ...}
# To:     lerobot-isaac-configs = ">=0.1.0"   # PyPI dep
```

### Post-Spinout Steps

1. Update `pyproject.toml` in extracted repo: remove `packages/` prefix from package paths.
2. Sibling deps (`lerobot-isaac-configs`) become PyPI dependencies.
3. Update `pixi.toml` in extracted repo: activate it (remove dormant comment).
4. Create new remote repo and push.
5. In the monorepo, replace path dep with PyPI dep:
   ```toml
   lerobot-isaac-env = ">=0.1.0"  # was: {path = "../lerobot-isaac-env"}
   ```

---

## Failure Modes and Mitigations

From plan §7:

| Failure Mode | Impact | Mitigation |
|-------------|--------|-----------|
| SO-101 USD unavailable; URDF→USD conversion fails | Phase 1 blocked | Conversion script + manual fallback via `omni.isaac.urdf`; USD binary NOT vendored in git |
| Isaac Lab API churn (pre-1.0 instability) | All Isaac phases | Exact version pin in `pixi.toml`; upgrades as separate plan |
| RTX 3080 10 GB OOM for some configs | Training phases | `--num_envs 1`, AMP on by default, gradient checkpointing; OOM ladder in `lerobot-program.md` |
| LeWM HDF5 schema undocumented / mismatch | Phase 2 LeWM target | `lerobot_world_model_bridge` skill produces HDF5; schema-discovery script provided; inspect `quentinll/lewm-pusht` |
| MimicGen incompatible with Isaac Lab | Phase 4b | MimicGen runs in its own MuJoCo environment; output converted back via `lerobot_mimicgen_bridge` skill |
| DreamerV3 implementation choice (multiple impls) | Phase 2 | Default sheeprl; fallback dreamer-v3-pytorch; adapter abstracts import |
| Isaac Lab + LeRobot Parquet schema drift | Phase 1 / 4a | Both versions pinned; `isaac_data_recorder.py` fails loudly on schema mismatch |
| Workspace `CLAUDE.md` drifts from repo | Ongoing | Workspace CLAUDE.md *links to* repo CLAUDE.md; does not duplicate content |

---

## Glossary

| Term | Definition |
|------|-----------|
| `pc_success` | Pick-and-place success rate — fraction of eval episodes where the object reaches the target zone. Range [0, 1]. Higher is better. The primary curriculum advancement metric. |
| `recon_loss` | Reconstruction loss in DreamerV3's RSSM world model — measures how well the model can decode its latent state back to the observation. Lower is better. |
| `pred_loss` | Prediction loss in HF LeWorldModel — measures accuracy of next-embedding prediction. Lower is better. |
| `DR` | Domain Randomization — the technique of varying simulator parameters (object pose, friction, lighting) at episode reset so trained policies generalize better to the real world. |
| `MimicGen` | A data augmentation tool (Mandlekar et al. 2023) that generates synthetic manipulation demos from a small real set by replanning trajectories in novel configurations, internally using MuJoCo. |
| `ManagerBasedRLEnvCfg` | Isaac Lab's base configuration class for environments driven by modular "managers" (ObservationManager, ActionManager, RewardManager, EventManager, TerminationManager). |
| `EventTermCfg` | Isaac Lab's per-event configuration record; used to wire domain randomization functions with their parameters and triggering mode (`reset` / `interval` / `startup`). |
| `MDP` | Markov Decision Process — the formalism underlying RL. In Isaac Lab, "MDP terms" refer to the functions that compute observations, actions, rewards, and terminations. |
| `RSSM` | Recurrent State Space Model — DreamerV3's world model backbone; a GRU-based sequence model combined with a CNN encoder that maintains a compact latent representation of the environment. |
| `ArticulationCfg` | Isaac Lab's class for defining a robot as a USD articulation, specifying joint actuators, USD path, and default joint states. |
| `SAL` | Scene Anomaly Localization — a quality-filtering method that detects frames with camera anomalies or scene occlusions. Used by `lerobot_dataset_quality` skill. |
| `TED` | Trajectory Edit Distance — a quality-filtering method that measures how dissimilar an episode trajectory is from the median; used to remove low-quality demonstrations. |
| `source tag` | A `source` column in the merged Parquet dataset recording where each episode came from: `"real"`, `"sim_dr"`, or `"mimicgen"`. Used for per-source weighting during training. |
| `pixi` | A conda-compatible package manager with lock files and environment definitions; used here to manage heavy ML deps (Isaac Sim, LeRobot, etc.) without conda's verbosity. |
| `uv workspace` | A Python monorepo feature of the `uv` package manager; `pyproject.toml` at the root lists all `packages/*` as workspace members so they are all editable-installable together. |
| `spinout` | Extracting a single package from the monorepo into its own standalone Git repository using `git subtree split` or `git filter-repo`. |
| `program.md` | A Markdown-formatted configuration file consumed by `autoresearch-loop-orchestrator`; specifies the training command, metric, direction, mutation operators, and budget for an autoresearch run. |
| `lerobot-isaac-train` | The CLI entrypoint installed by `lerobot-isaac-adapters`; single command for all training targets (`--target_arch` selector). |
| `SO-101` | The Smart Robotics SO-101 6-DOF robot arm (7 joints including gripper). Hardware target for this workspace. Two units available. |
| `LeRobotDataset` | The canonical dataset format from the LeRobot library; stores episodes as Parquet files with metadata in `meta/info.json`, `meta/stats.json`, `meta/episodes.parquet`. |
| `Dashboard` | The `lerobot-isaac-dashboard` package — a read-only metrics surface that reads local pipeline artefacts and presents them as a live Streamlit UI or static HTML report. Never writes to training artefacts. |
| `Snapshot` | A point-in-time capture of the full dashboard loader state, stored under `outputs/snapshots/<id>/` as Parquet files + JSON. Replayable via `load_snapshot`; schema version is bumped on breaking format changes. |
