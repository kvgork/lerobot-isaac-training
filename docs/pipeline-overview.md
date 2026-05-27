# Pipeline Overview — Data Collection to Autoresearch

**Audience:** Anyone trying to understand what every piece of this workspace does
and how the pieces connect, end to end.
**Scope:** Every stage from raw teleop ingestion through autoresearch HP search.
**Authoritative single-command runner:** [`scripts/run_full_pipeline.sh`](../scripts/run_full_pipeline.sh).
**Status:** Reflects code state on 2026-05-14 after the synthetic-data integration
landed (`lerobot 0.5` API surface, Isaac Sim 6.0, Isaac Lab 0.54.3).

This is the **single source of truth for pipeline structure**. Runbooks (`docs/runbook/`)
give command recipes; internals docs (`docs/internals/`) give per-module specifics;
this file describes how the modules fit together.

---

## 0. End-to-End Dataflow at a Glance

```
                       +─────────────────────────────+
                       |  SO-101 robot (real arm)    |
                       |  ─ teleop session           |
                       +──────────────┬──────────────+
                                      | LeRobotDataset v3.0
                                      v
       +──────────────────────────────────────────────────+
       |  datasets/<org>/<name>/                          |
       |    data/chunk-000/file-000.parquet  (frames)     |
       |    meta/info.json + episodes/* + tasks.parquet   |
       +──────────────┬──────────────────┬────────────────+
                      |                  |
                      |    (1) bridge    | (2) DR replay
                      v                  v
   +──────────────────────────+  +──────────────────────────────+
   |  outputs/.../bridge/     |  |  outputs/.../synthetic/      |
   |    so101_dreamerv3.hdf5  |  |    data + meta (source=sim_dr|
   |    so101_lewm.hdf5       |  +──────────────┬───────────────+
   +──────────────┬───────────+                 |
                  |                             | (3) optional merge
                  |                             v
                  |                  +──────────────────────────+
                  |                  |  merged real + sim_dr DS |
                  |                  +──────────────┬───────────+
                  |                                 |
                  |                                 | (4) policy train
                  |                                 v
       (5) WM train|                  +──────────────────────────+
                  v                  |  outputs/.../policy-*/   |
   +──────────────────────────+      |    checkpoints/NNNN/...  |
   |  outputs/.../wm-*/       |      +──────────────┬───────────+
   |    sheeprl tensorboard   |                     |
   |    + checkpoints         |                     | (6) eval
   +──────────────┬───────────+                     v
                  |                       +──────────────────────────+
                  |                       |  outputs/eval/<run>.json |
                  |                       +──────────────┬───────────+
                  |                                      |
                  v                                      v
                  +────────────────────┬─────────────────+
                                       | (7) loaders
                                       v
                       +──────────────────────────────+
                       |  Streamlit dashboard         |
                       |  Static HTML report          |
                       |  Snapshot / Compare reports  |
                       +──────────────────────────────+

         autoresearch loop wraps (4)/(5) and re-runs them with
         mutated hyperparameters until plateau or budget exhaust
```

`run_full_pipeline.sh` walks stages (2) → (3) skipped → (4) → (5) → (6) → (7) in
order. The autoresearch loop is run separately via `/autoresearch` against a
`programs/*.md` file.

---

## 1. Package Map (live + spun-out)

| Package | GitHub repo (canonical) | Installed via | Responsibility |
|---------|----------------------|---------------|----------------|
| `lerobot-isaac-meta`            | `packages/lerobot-isaac-meta/` (live)                   | editable workspace member | Umbrella CLI + workspace path resolver. |
| `lerobot-isaac-env`             | [github.com/kvgork/lerobot-isaac-env](https://github.com/kvgork/lerobot-isaac-env)                   | editable from `src/`      | Isaac Lab env (`Isaac-SO101-Pick-v0`, `…-PickPlace-v0`). |
| `lerobot-isaac-adapters`        | [github.com/kvgork/lerobot-isaac-adapters](https://github.com/kvgork/lerobot-isaac-adapters)         | editable from `src/`      | Single training entrypoint + 5 backends + sheeprl `custom_hdf5` plugin + LeWM mini-trainer. |
| `lerobot-isaac-synthetic`       | [github.com/kvgork/lerobot-isaac-synthetic](https://github.com/kvgork/lerobot-isaac-synthetic)       | editable from `src/`      | Isaac DR replay → LeRobotDataset, merge utilities, MimicGen stub. |
| `lerobot-isaac-autoresearch`    | [github.com/kvgork/lerobot-isaac-autoresearch](https://github.com/kvgork/lerobot-isaac-autoresearch) | editable from `src/`      | `train_wrapper.py` shim + `programs/*.md` configs. |
| `lerobot-isaac-configs`         | [github.com/kvgork/lerobot-isaac-configs](https://github.com/kvgork/lerobot-isaac-configs)           | editable from `src/`      | YAML configs (policy, wm, batch). |
| `lerobot-isaac-dashboard`       | [github.com/kvgork/lerobot-isaac-dashboard](https://github.com/kvgork/lerobot-isaac-dashboard)       | editable from `src/`      | Streamlit live UI + static HTML report + snapshots + compare. |

Plus four external repos that the pipeline soft-imports at runtime:
- **lerobot 0.5+** (HuggingFace) — datasets + `lerobot-train` CLI + policy classes.
- **isaaclab 0.54+** (NVIDIA) — `ManagerBasedRLEnv` + DR event manager.
- **isaacsim 6.0+** (NVIDIA) — Kit framework, USD, physics, sensors.
- **sheeprl 0.5.8.dev** (Eclectic-Sheep) — DreamerV3 / DreamerV1/V2 algorithms.

The four claude_code skills used by the pipeline (`lerobot_world_model_bridge`,
`lerobot_mimicgen_bridge`, `lerobot_dataset_quality`, `autoresearch`) live in
`${CLAUDE_CODE_ROOT}/skills/`.

---

## 2. Pixi Environments

| Env | Features | Heavy deps that need `install_train_deps.sh` |
|-----|----------|----------------------------------------------|
| `default`       | dev + editable-siblings (path deps from `src/`)   | — |
| `frozen`        | dev + git-siblings (GitHub https URLs)            | — |
| `train-policy`  | dev + lerobot + editable-siblings                 | `pip install lerobot[smolvla]` |
| `train-dreamer` | dev + lerobot + dreamerv3 + editable-siblings     | `pip install --ignore-requires-python git+...sheeprl` |
| `train-lewm`    | dev + lerobot + leworldmodel + editable-siblings  | `pip install lerobot` |
| `sim`           | dev + lerobot + isaaclab + editable-siblings      | `bash scripts/install_isaac_lab.sh` (Isaac Sim 6.0 + Isaac Lab editable) |
| `dashboard`     | dev + dashboard + editable-siblings               | — |
| `full`          | every feature                                     | all of the above |

`feature.lerobot`, `feature.dreamerv3`, `feature.leworldmodel` are deliberately
empty in `pixi.toml` — see [ADR-0003](adr/0003-soft-import-discipline.md) and
[docs/runbook/00-install.md §Step 4](runbook/00-install.md). Pixi cannot
co-resolve `lerobot` + `sheeprl` (conflicting gymnasium pins), so the heavy
training libraries are pinned outside the pixi solver.

---

## 3. Stage-by-Stage

### Stage A — Data Collection (real teleop)

| What | Where |
|------|-------|
| Source | SO-101 arm, recorded via the `robot_data_recorder` opt-in package. |
| Output format | LeRobotDataset v3.0 — see [data-pipeline.md](internals/data-pipeline.md). |
| On-disk location | `datasets/<org>/<repo_id>/` |
| Runbook | [docs/runbook/02-collect-data.md](runbook/02-collect-data.md) |

LeRobotDataset v3.0 directory layout (the dataset shipped at
`datasets/kvgork/so101-pickplace1/` is the reference):
```
<repo_id>/
  data/chunk-000/file-000.parquet         # all frames concatenated (one or more shards)
  meta/info.json                          # total_episodes, total_frames, fps, features
  meta/stats.json                         # per-column statistics
  meta/episodes/chunk-000/file-000.parquet  # per-episode (length, dataset_from/to_index, stats, source)
  meta/tasks.parquet                      # task table (id, name)
  videos/<image_key>/chunk-000/file-000.mp4 # optional — for dtype=video features
```

LeRobotDataset v2.x used a single `meta/episodes.parquet`. lerobot 0.5+ writes
the sharded `meta/episodes/chunk-XXX/file-XXX.parquet` layout instead. The
dashboard's `load_parquet_dataset` and `load_synthetic` loaders auto-detect
both layouts (commit `ebdc393` and `356346b` in the dashboard bare repo).

`observation.images.*` features can be stored as either:
- `dtype: "video"` — MP4 files under `videos/<key>/...` (legacy).
- `dtype: "image"` — encoded PNG/JPG bytes inline in the data parquet under a
  `struct<bytes: binary, path: string>` column (current convention).

The bridge skill handles both formats — see `claude_code` commit `4e6e21c`.

---

### Stage B — Synthetic Data Generation (Isaac DR replay)

| What | Where |
|------|-------|
| Producer | `lerobot_isaac_synthetic.isaac_dr.replay_runner` |
| Env | `Isaac-SO101-PickPlace-v0` registered by `lerobot_isaac_env.tasks._register_envs()` |
| Per-frame writer | `lerobot_isaac_synthetic.isaac_dr.parquet_writer` |
| Output | LeRobotDataset tagged `source=sim_dr` |
| Runbook | [docs/runbook/05-augment-with-dr.md](runbook/05-augment-with-dr.md) |
| Internals | [docs/internals/synthetic-data.md](internals/synthetic-data.md) |

End-to-end flow:

```
1. PYTHONNOUSERSITE=1 python -m lerobot_isaac_synthetic.isaac_dr.replay_runner \
       --source_dataset datasets/kvgork/so101-pickplace1 \
       --n_variants 2 --max_episodes 3 \
       --output_path outputs/.../synthetic
2. replay_runner imports `lerobot.datasets.lerobot_dataset.LeRobotDataset`
   (lerobot 0.5; old `lerobot.common.*` namespace removed).
3. Boots `SimulationApp({"headless": True})` BEFORE importing isaaclab.envs
   (otherwise `omni.*` is missing).
4. Sets carb setting `/persistent/isaac/asset_root/cloud` to NVIDIA's S3 mirror
   so the Isaac Sim ground-plane USD resolves.
5. Imports `lerobot_isaac_env`, calls `_register_envs()` to land
   `Isaac-SO101-*-v0` in the gymnasium registry.
6. For each source episode:
     - read action sequence via `dataset.meta.episodes.dataset_from_index` /
       `dataset_to_index` (lerobot 0.5 dropped `episode_data_index`).
     - for each variant: `env.reset(seed=variant_seed)` → step through the action
       sequence (batched as `(num_envs=1, action_dim)`, cast to env device).
     - yield an `Episode` dataclass.
7. `parquet_writer.write_episodes_to_lerobot_dataset()`:
     - `LeRobotDataset.create(repo_id=..., root=..., fps=..., features=...)`
     - for each frame: `frame["task"] = "pick_and_place"`, `frame["next.done"] =
       np.array([done], dtype=bool)`, `dataset.add_frame(frame)`.
     - `dataset.save_episode()` per episode.
     - `dataset.finalize()` — flushes the data parquet so the file footer is
       written (without this the parquet has `PAR1` head but no footer →
       unreadable).
8. `_tag_source_column()` adds `source="sim_dr"` to every shard of
   `meta/episodes/chunk-XXX/file-XXX.parquet`. If lerobot didn't write any
   per-episode parquet at all (some configurations), the writer synthesises
   one from the data parquet's `episode_index` column.
```

The same `(num_envs, action_dim)` tensor wrapping logic in `replay_runner.py`
applies to any caller that wants to drive Isaac Lab through a pre-recorded
action sequence — see `lerobot-isaac-synthetic` commit `6cf3b22`.

Key failure modes (now fixed; documented for future debugging):
- `RuntimeError: Failed to find a rigid body when resolving '/World/envs/env_.*/TargetBin'`
  → spawned scene asset lacks `USD RigidBodyAPI`. Fixed by attaching
  `RigidBodyPropertiesCfg(kinematic_enabled=True)` + `MassPropertiesCfg` +
  `CollisionPropertiesCfg` to the `CuboidCfg` spawn — `lerobot-isaac-env`
  commit `4774690`.
- `TypeError: CreateShaderPrimFromSdrCommand.__init__() got an unexpected keyword
  argument 'name'` → Isaac Sim 6.0 renamed that constructor signature; the
  workaround is to drop `visual_material=PreviewSurfaceCfg(...)` from the
  `CuboidCfg` (commit `8ebe838`).
- `ValueError: Invalid action shape, expected: 0, received: 6` → action
  manager has no terms registered; `SO101EnvCfg.__post_init__` must assign the
  real `ActionsCfg()` (commit `b3c9d59`).

---

### Stage C — World-Model Bridge (Parquet → HDF5)

| What | Where |
|------|-------|
| Producer | `skills/lerobot_world_model_bridge` (claude_code repo) |
| Public API | `lerobot_to_worldmodel(dataset_path, output_path, output_format="hdf5", image_size=(64,64) or (96,96), window_size=16, stride=8, normalize_actions=True)` |
| Output | HDF5 with `windows/{frames,actions}` group plus `episodes/<idx>/{frames,actions,states,...}`. |
| Internals | [docs/internals/world-model-bridge.md](internals/world-model-bridge.md) |

The bridge reads a LeRobotDataset and emits per-window tensors suitable for
sheeprl/DreamerV3 (`(64,64)` images) or LeWorldModel (`(96,96)` images, window
16). It auto-detects:
- `dtype: video` features → decode MP4 via OpenCV.
- `dtype: image` features → decode inline parquet bytes via Pillow (no cv2 dep)
  — added in claude_code commit `4e6e21c`.

The bridge is also where action normalisation happens: `actions = (actions -
mean) / (std + 1e-6)`. Mean/std are recomputed per-bridge and saved into
the HDF5's root attrs.

---

### Stage D — Policy Training (LeRobot)

| What | Where |
|------|-------|
| Backend | `lerobot_isaac_adapters.targets.policy_lerobot` |
| Subprocess CLI | `lerobot-train` (lerobot 0.5+) |
| Output | `<output_dir>/checkpoints/NNNNNN/pretrained_model/{model,policy_processor*}.safetensors` |
| Runbook | [docs/runbook/03-train-policy.md](runbook/03-train-policy.md) |
| Internals | [docs/internals/training-dispatch.md](internals/training-dispatch.md) |

```
python -m lerobot_isaac_adapters.train \
  --target_arch diffusion \
  --dataset datasets/kvgork/so101-pickplace1 \
  --output_dir outputs/.../policy-diffusion \
  --steps 1000000 --batch_size 8 --lr 1e-4 --seed 42
```

`policy_lerobot.run()` builds a `lerobot-train` command line. The exact
flag spelling matches lerobot 0.5 (changed in 2026; previous releases used
`--training.batch_size` etc.):

```
lerobot-train
  --policy.type=diffusion
  --dataset.repo_id=<derived from path or hub id>
  --dataset.root=<absolute local path>     # added if --dataset is on-disk
  --dataset.video_backend=pyav             # avoid torchcodec libavutil mismatch
  --batch_size=N --steps=N --optimizer.lr=F --seed=N
  --output_dir=<run_dir>
  --policy.push_to_hub=false               # local-only by default
```

`_split_dataset_arg()` infers an `<org>/<name>` style `repo_id` from local
paths so caches / logs stay consistent.

Watchdog-killed runs are normal — `run_full_pipeline.sh` issues SIGTERM at
the `--train-minutes` budget. lerobot stops, the most recent
`checkpoints/NNNNNN/` survives, and eval picks it up.

---

### Stage E — World-Model Training

#### DreamerV3 (sheeprl)

| What | Where |
|------|-------|
| Backend | `lerobot_isaac_adapters.targets.wm_dreamerv3` |
| Subprocess | `python -m sheeprl …` (NOT `sheeprl.cli` — that one has no `__main__`) |
| Custom env | `lerobot_isaac_adapters.sheeprl_plugin.hdf5_env.HDF5ReplayEnv` |
| Hydra config | bundled at `lerobot_isaac_adapters/sheeprl_plugin/configs/env/custom_hdf5.yaml` |
| Runbook | [docs/runbook/04-train-world-model.md](runbook/04-train-world-model.md) |

```
python -m sheeprl \
  --config-dir=<plugin>/configs \
  exp=dreamer_v3 env=custom_hdf5 \
  +env.dataset_path=outputs/.../bridge/so101_dreamerv3_data.hdf5 \
  algo.per_rank_batch_size=8 algo.world_model.optimizer.lr=1e-4 \
  algo.total_steps=1000000 seed=42 \
  hydra.run.dir=outputs/.../wm-dreamerv3
```

`HDF5ReplayEnv` exposes a `Dict({"rgb": Box((C,H,W), uint8), "state":
Box((A,), float32)})` observation space and a continuous `Box((A,), float32)`
action space. It picks a random window per `reset()` and advances along the
recorded time axis per `step()`. Rewards are zero — the agent learns a world
model from the observation stream, not a policy.

The `+env.dataset_path=…` form (`+` prefix) is required because Hydra's
strict-set override would fail on env configs that don't predefine the key —
necessary if the caller overrides `env=dmc` etc. instead of `env=custom_hdf5`.

#### LeWorldModel (in-process fallback)

| What | Where |
|------|-------|
| Backend | `lerobot_isaac_adapters.targets.wm_leworldmodel` |
| Default in-process trainer | `lerobot_isaac_adapters.targets._lewm_minimal` |
| Opt-in upstream CLI | `python -m lerobot.scripts.train_world_model` (set `LEROBOT_ISAAC_LEWM_BACKEND=hf`) |

The HF `lerobot 0.5.x` package does NOT ship `lerobot.scripts.train_world_model`.
The default backend is now an in-process minimal trainer: 4-layer CNN encoder
→ 128-dim embedding → 2-layer MLP forward dynamics head trained with MSE on
next-embedding prediction. Stdout emits `pred_loss=<float>` lines every 50
steps for the autoresearch metric regex. Saves
`<output_dir>/lewm_minimal_last.pt` on clean exit.

If a future lerobot release adds the upstream CLI, set
`LEROBOT_ISAAC_LEWM_BACKEND=hf` to switch back to the subprocess path
without code changes.

---

### Stage F — Evaluation (open-loop action MSE)

| What | Where |
|------|-------|
| Script | `scripts/_open_loop_eval.py` |
| Output JSON keys | `run_id`, `task`, `ts`, `pc_success`, `n_episodes`, `intervention_rate`, `mean_ep_len`, `_metadata.{source,mse,n_frames_evaluated,policy_path,dataset_root}` |
| On-disk location | `outputs/eval/<run_id>.json` |

SO-101 has no registered gymnasium env (only the Isaac Lab one, which would
require booting Isaac Sim + ground-truth physics matching teleop). The eval
script therefore measures **open-loop action prediction error** on held-out
real episodes:

```
for each held-out frame:
    action_pred = policy.select_action(obs)
    mse_t = mean((action_pred - action_recorded) ** 2)
pc_success = 1 / (1 + mean(mse_t))      # bounded to [0, 1]
```

The `_metadata.source = "open_loop_action_mse"` flag in the JSON makes it
clear to any downstream consumer (dashboard, autoresearch executor) that
this is NOT closed-loop rollout success — just a proxy useful for tracking
policy quality across runs.

For closed-loop eval (real gym env rollouts), use `lerobot-eval --env.type=pusht`
with a pusht-trained checkpoint, or wait for the SO-101 Isaac env to land a
real reward function (currently `SO101RewardsCfg(success=None, progress=None)`).

---

### Stage G — Dashboard (read-only)

| What | Where |
|------|-------|
| Live mode | `pixi run -e dashboard dashboard` → Streamlit at http://localhost:8501 |
| Static report | `pixi run -e dashboard report --output-dir <dir>` → 4-5 MB HTML |
| Snapshot | `pixi run -e dashboard snapshot save --label <name>` |
| Compare | `pixi run -e dashboard compare --snapshots A B [C ...]` |
| Runbook | [docs/runbook/07-dashboard.md](runbook/07-dashboard.md) |

8 tabs, each backed by one or more loaders:

| Tab | Loader(s) | Sources scanned |
|-----|-----------|-----------------|
| Data Collection | `parquet_dataset` | `datasets/**/meta/{episodes.parquet,episodes/chunk-*/file-*.parquet,info.json}` |
| Synthetic Data | `synthetic` | same as above, filtered to datasets with a `source` column |
| Policy Training | `training_logs` + `checkpoints` | `outputs/checkpoints/<arch>/<run>/log.txt` (canonical) + `logs/**/*.log` (fallback) + recursive `*.{pt,safetensors}` under `outputs/` |
| World Model Training | `training_logs` + `checkpoints` | same as Policy |
| Evaluation | `eval_results` | `outputs/eval/*.json` matching `EVAL_SCHEMA` |
| Autoresearch | `autoresearch` | `.agent-state/<sess>/autoresearch/<slug>/{history.jsonl,best.json,plateau.json,program.json}` |
| Curriculum | `curriculum` | `outputs/curriculum_stage.json` + `outputs/curriculum_history.jsonl` |
| Pipeline Health | `events` + counts | `.agent-state/<sess>/events.jsonl` + cross-loader counts |

Top-level KPI banner (added in dashboard commit `d50ef57`): Episodes /
Training rows / Checkpoints / Eval runs / Events.

Auto-refresh (Live mode) uses `streamlit_autorefresh.st_autorefresh()` —
hard-pinned in `pyproject.toml` because the fallback meta-refresh tag does
full page reloads that wipe widget state (`d50ef57`+`ee0cef9`).

---

### Stage I — Hardware Deployment

| What | Where |
|------|-------|
| CLI | `lerobot-isaac-deploy` (entry from `lerobot-isaac-adapters`) |
| Module | `lerobot_isaac_adapters.deploy` |
| Robot driver (upstream) | `lerobot.robots.so_follower.SO101Follower` |
| Runbook | [`docs/runbook/10-deploy-to-hardware.md`](runbook/10-deploy-to-hardware.md) |

Single command runs a trained policy on the physical SO-101 follower.
Safety layers (6) are stacked: dry-run default, `max_relative_target`
server-side clip, fixed rate limit (30 Hz), stuck-action watchdog,
SIGINT clean exit + optional home-on-exit, and the always-available
physical power switch.

```bash
# DRY-RUN first
lerobot-isaac-deploy \
    --policy-path outputs/.../checkpoints/last/pretrained_model \
    --port /dev/ttyACM0 \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --camera d435_rgb=/dev/video0,640,480 \
    --duration-s 30 -v

# EXECUTE only after dry-run looks sane
lerobot-isaac-deploy --execute --max-relative-target 3.0 --home-on-exit ...
```

Observation / action conversion glue is in the same module
(`_obs_to_policy_input`, `_action_to_robot_dict`) — about 120 LOC end to end.

---

### Stage H — Autoresearch Loop

| What | Where |
|------|-------|
| Orchestrator | `${CLAUDE_CODE_ROOT}/agents/orchestrators/autoresearch-loop-orchestrator.md` |
| ML proposer (domain-aware) | `${CLAUDE_CODE_ROOT}/agents/workers/autoresearch-ml-proposer-worker.md` — patched to load `domain_knowledge:` ref card when present. |
| ML executor | `${CLAUDE_CODE_ROOT}/agents/workers/autoresearch-ml-executor-worker.md` |
| Skill | `${CLAUDE_CODE_ROOT}/skills/autoresearch/` |
| Workspace shim | `src/lerobot-isaac-autoresearch/src/lerobot_isaac_autoresearch/train_wrapper.py` (editable clone from [github.com/kvgork/lerobot-isaac-autoresearch](https://github.com/kvgork/lerobot-isaac-autoresearch)) |
| **Domain pack (this stack)** | [`programs/_domain_knowledge.md`](../programs/_domain_knowledge.md) — VRAM ceilings, lerobot 0.5+ flag shape, sheeprl Hydra paths, OOM recovery, operator priority |
| **Per-arch programs** | [`programs/`](../programs/) — diffusion, smolvla, act, dreamerv3, lewm + short smoke variant |
| Wrapper | [`scripts/run_autoresearch.sh`](../scripts/run_autoresearch.sh) — name → path resolver, env-var injection, deterministic bash fallback |
| State | `.agent-state/<session>/autoresearch/<slug>/{history.jsonl,program.json,best.json,plateau.json}` |
| Internals | [docs/internals/autoresearch-integration.md](internals/autoresearch-integration.md) |

Loop:
1. Read `program.md`: target metric (name, direction, regex), search space,
   budget (`seconds_per_experiment`, `max_experiments`, `plateau_limit`).
2. For each iteration:
   a. Proposer mutates the previous best config.
   b. Executor runs `train_wrapper.py` which spawns
      `lerobot_isaac_adapters.train` as a subprocess, captures stdout, and
      guarantees a final `pc_success=<float>` line (with a `0.0` sentinel
      if the metric regex finds nothing).
   c. Append a JSON record to `history.jsonl`:
      ```json
      {"trial_index": N, "metric_name": "pc_success", "metric_value": F,
       "config": {...}, "ts": "...", "duration_s": N, "status": "ok"}
      ```
   d. Update `best.json` if the new metric beats the previous best.
   e. Update `plateau.json` (`consecutive_non_improvements`).
3. Stop on plateau or `max_experiments`. Final report has best config +
   per-iter metric trace.

The dashboard's Autoresearch tab reads `history.jsonl` directly — see
`load_autoresearch` in `lerobot-isaac-dashboard/src/loaders/autoresearch.py`.

When the orchestrator agent is not invocable (e.g. permission gates in a
session), the same protocol can be driven by `scripts/_run_autoresearch_smoke.sh`
which is a deterministic bash loop producing the same on-disk artefacts.

---

## 4. One-Command Pipeline (`run_full_pipeline.sh`)

The recipe that ties stages B through G together is
[`scripts/run_full_pipeline.sh`](../scripts/run_full_pipeline.sh). It's also
exposed as `pixi run pipeline`.

```
bash scripts/run_full_pipeline.sh \
  [--train-minutes N]      # default 30; SIGTERM watchdog per training
  [--n-synthetic N]        # default 3 source episodes
  [--dataset DIR]          # default datasets/kvgork/so101-pickplace1
  [--skip-{synthetic,policy,worldmodel,eval,dashboard}]
```

Stage matrix (auto-skipped when watch-dogged dependencies aren't met, e.g.
no checkpoint → eval auto-skips with `SKIP(no_ckpt)` rather than failing):

| Stage         | Env           | Output                                                      |
|---------------|---------------|-------------------------------------------------------------|
| preflight     | bash          | logs/preflight.log                                          |
| synthetic     | `sim`         | `<run_dir>/synthetic/`, linked under `datasets/synthetic/`  |
| policy_train  | `train-policy`| `<run_dir>/policy-diffusion/checkpoints/NNNN/...`           |
| wm_train      | `train-dreamer` (bridge runs in `default`) | `<run_dir>/wm-dreamerv3/` + `bridge/dreamerv3_data.hdf5` |
| eval          | `train-policy`| `outputs/eval/<run>-policy.json`                            |
| dashboard     | `dashboard`   | `<run_dir>/dashboard/report.html` + snapshot                |

`save_freq` for the policy backend auto-scales with the training budget so
even a 2-min smoke run produces a checkpoint eval can load.

---

## 5. Where Each Recent Bugfix Lives

This pipeline went through a heavy integration debug pass on 2026-05-13/14.
The commits below are the canonical references for **why** the code looks
The commits below are in the relevant sibling repo on GitHub

| Symptom | Fix | Commit |
|---------|-----|--------|
| Bridge errors on `dtype: image` PNG bytes parquet | PIL-only decode path, no cv2 dep | claude_code `4e6e21c` |
| `lerobot-train` unknown arg `--training.batch_size` | Rename to `--batch_size` / `--steps` / `--optimizer.lr` / `--config_path` | adapters `bfef7e6` |
| `lerobot-train` requires `policy.repo_id` | Default `--policy.push_to_hub=false` | adapters `c7639ca` |
| `python -m sheeprl.cli` exits silently | Switch to `python -m sheeprl`, namespace algo keys | adapters `0fb5434` |
| Hydra `env.dataset_path=…` strict-set fails | Use `+env.dataset_path=…` | adapters `c798649` |
| LeWM real training (no upstream CLI) | In-process `_lewm_minimal` trainer | adapters `d87d677` |
| sheeprl needs HDF5 replay env | Bundled `custom_hdf5.yaml` + `HDF5ReplayEnv` plugin | adapters `d9e57db` |
| `Isaac-SO101-PickPlace-v0` not registered | `_register_envs()` at import (env package) | env `1e0350b` |
| SO-101 joint names mismatch new URDF | `shoulder_pan` / `_lift` / `elbow_flex` / `wrist_flex` / `_roll` / `gripper` | env `c1cd863` |
| Scene asset missing RigidBodyAPI | `RigidBodyPropertiesCfg(kinematic_enabled=True)` on `CuboidCfg` | env `4774690` |
| Isaac Sim 6.0 PreviewSurfaceCfg crash | Drop `visual_material` from spawn | env `8ebe838` |
| `total_action_dim=0` runtime | Wire real `ActionsCfg()` in `__post_init__` | env `b3c9d59` |
| `LeRobotDataset(path)` HFValidationError | Pass `repo_id` + `root` kwargs | synthetic `54ca242` |
| `episode_data_index` AttributeError | Use `dataset.meta.episodes.dataset_from/to_index` | synthetic `0de088d` |
| `env.step(1D_action)` IndexError | Batch + cast to env device | synthetic `6cf3b22` |
| Isaac Lab asset_root None | Set `/persistent/isaac/asset_root/cloud` carb setting | synthetic `954e970` |
| `add_frame()` got unexpected kwarg `task` | Put `task` in frame dict | synthetic `baadc47` |
| `next.done` type mismatch | `np.array([bool(done)], dtype=bool)` | synthetic `ab51636` |
| Parquet file truncated (no footer) | Call `dataset.finalize()` after save_episode loop | synthetic `49179f1` |
| `meta/episodes.parquet` not written by lerobot 0.5 | Synthesise from data parquet `episode_index` | synthetic `7b3d17a` |
| Dashboard `render_kpi_row(ctx)` TypeError | Build top-level KPI item list, pass container + items | dashboard `d50ef57` |
| Dashboard tabs empty | Loaders scan `logs/**/*.log` + `outputs/**/*.{pt,safetensors}` fallback | dashboard `d50ef57` |
| Dashboard mode resets to Live on refresh | Drop meta-refresh fallback, require `streamlit-autorefresh` | dashboard `ee0cef9` |
| Data Collection tab empty for v3.0 datasets | Read sharded `meta/episodes/chunk-*/file-*.parquet` + fallback to `info.json` counts | dashboard `ebdc393` |
| Synthetic tab empty for v3.0 datasets | Same fix in the synthetic loader | dashboard `356346b` |

A more exhaustive list (with workarounds for not-yet-fixed gaps) lives in
[`docs/internals/system-improvements.md`](internals/system-improvements.md).

---

## 6. Cross-Doc Index

| Topic | Doc |
|-------|-----|
| Install + heavy deps           | [runbook/00-install.md](runbook/00-install.md) |
| Data collection                | [runbook/02-collect-data.md](runbook/02-collect-data.md) + [internals/data-pipeline.md](internals/data-pipeline.md) |
| Synthetic DR replay            | [runbook/05-augment-with-dr.md](runbook/05-augment-with-dr.md) + [internals/synthetic-data.md](internals/synthetic-data.md) |
| MimicGen (deferred)            | [runbook/06-augment-with-mimicgen.md](runbook/06-augment-with-mimicgen.md) + [research/mimicgen-reference.md](research/mimicgen-reference.md) |
| Policy training                | [runbook/03-train-policy.md](runbook/03-train-policy.md) + [internals/training-dispatch.md](internals/training-dispatch.md) |
| World-model training           | [runbook/04-train-world-model.md](runbook/04-train-world-model.md) + [internals/world-model-bridge.md](internals/world-model-bridge.md) + [research/{dreamerv3,leworldmodel}-reference.md](research/) |
| Dashboard                      | [runbook/07-dashboard.md](runbook/07-dashboard.md) |
| Autoresearch                   | [internals/autoresearch-integration.md](internals/autoresearch-integration.md) |
| Isaac Lab specifics            | [research/isaac-lab-reference.md](research/isaac-lab-reference.md) + [internals/isaac-lab-integration.md](internals/isaac-lab-integration.md) |
| Adapter / modular target arch  | [concepts/modular-training-adapter.md](concepts/modular-training-adapter.md) + ADR-0005 |
| Soft-import discipline         | [concepts/soft-import-discipline.md](concepts/soft-import-discipline.md) + ADR-0003 |
| Pixi workspace                 | [concepts/pixi-workspace.md](concepts/pixi-workspace.md) + ADR-0002 |
| Multi-package monorepo         | [concepts/multi-package-monorepo.md](concepts/multi-package-monorepo.md) + ADR-0004 |
| Architecture & spinout         | [../ARCHITECTURE.md](../ARCHITECTURE.md) |
| API reference (all packages)   | [api-reference.md](api-reference.md) |
| Gap log / known issues         | [internals/system-improvements.md](internals/system-improvements.md) |
