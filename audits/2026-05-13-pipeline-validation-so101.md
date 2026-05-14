# Pipeline Validation against SO-101 Dataset — 2026-05-13/14

**Session id:** `20260513-pipeline-validation-so101`
**Dataset:** `datasets/kvgork/so101-pickplace1/` (LeRobotDataset v3.0, 20 eps × 7491 frames, 640×480 PNG inline) + `datasets/kvgork__so101-pickplace1.h5` (7.4 GB, DreamerV3-style schema).
**Goal:** End-to-end pipeline smoke + one 30-min real training per backend, fix all failures, commit.

---

## Final Stage Matrix

| Stage | Scope | Result | Notes |
|---|---|---|---|
| A  | 7-stage `pipeline_smoke.sh` (pusht baseline)  | PASS (1s)        | All stages PASS/DRYRUN. |
| A2 | SO-101 → HDF5 bridge (64×64 + 96×96)          | FIX + PASS       | Bridge missed `dtype=image` PNG bytes in parquet. Patched (claude_code 4e6e21c). Both schemas now built: 20 eps × 7491 frames × 907 windows. |
| A3 | Backend CLIs in train-policy / train-dreamer / train-lewm | PASS w/ gap | `lerobot 0.5.1` and `sheeprl 0.5.8.dev0` installed. `lerobot.scripts.train_world_model` does NOT exist in lerobot 0.5 → Stage D blocked. |
| B  | Real 30-min `--target_arch diffusion` on SO-101 | PASS           | 4562/200K steps. Loss 0.046 → 0.039. Watchdog cleaned up. Checkpoints at 2k/4k/last (5.9 GB). |
| C  | Real 30-min `--target_arch dreamerv3` on SO-101 HDF5 + `env=dummy` | PASS w/ gap | 7720 policy_steps on GPU. Runtime validated, but sheeprl ships no `custom_hdf5` env so the SO-101 HDF5 is not consumed during training — pure runtime smoke. |
| D  | Real 30-min `--target_arch le_world_model`     | BLOCKED          | Upstream gap: `lerobot.scripts.train_world_model` not in lerobot 0.5. Dry-run still works. Tracked in `system-improvements.md`. |
| E  | One `autoresearch/train_wrapper.py` iteration on SO-101 | PASS (in progress at write time) | Validates the metric-extraction shim. Dispatches to `lerobot_isaac_adapters.train` and tails stdout for `pc_success=…`. |
| F  | Dashboard static report + 2-way snapshot compare | PASS           | `outputs/pipeline-validation-so101/stage-f-dashboard/report.html` (4.7 MB) + `…/stage-f-compare/report.html`. Live: `pixi run -e dashboard dashboard`. |

---

## Failures Found and Fixed

### 1. Bridge skill silently rejected `dtype=image` parquet datasets
- Symptom: `OperationResult(success=False, error="No video directories found under …/videos/")` even though parquet had inline PNG bytes.
- Root cause: `lerobot_to_worldmodel()` only handled `dtype=video` (MP4 under `videos/`).
- Fix: detect `dtype: image` features in `meta/info.json`, decode inline bytes via PIL+numpy (no `cv2` dep). Surface real errors instead of silently returning None on missing dep.
- Commit: `claude_code` **4e6e21c**.
- Verified: SO-101 parquet → 64×64 + 96×96 HDF5, 20 eps × 7491 frames each.

### 2. Heavyweight training deps absent from pixi envs
- Symptom: `import lerobot` / `import sheeprl` raised `ModuleNotFoundError` in every train-* env after `pixi install`.
- Root cause: pixi.toml leaves `feature.lerobot` / `feature.dreamerv3` / `feature.leworldmodel` empty by design (gymnasium version pin conflicts).
- Fix: new helper `scripts/install_train_deps.sh` (idempotent, per-env flags). Sheeprl pins `python<3.12` in metadata but works on 3.12 — script auto-passes `--ignore-requires-python` when needed. Wired into `pixi run install-train-deps`, `docs/runbook/00-install.md §Step 4`, `scripts/README.md`, `CLAUDE.md` Common Pitfalls.
- Commit: workspace **c1c6b09**.

### 3. Adapter CLI command mis-targeted lerobot 0.5+
- Symptom: `lerobot-train: error: unrecognized arguments: --training.batch_size --training.num_steps --training.lr` plus required-`policy.repo_id` error.
- Root cause: lerobot 0.5 renamed CLI flags and made `policy.push_to_hub=true` the default. Adapter still emitted the legacy shape.
- Fix: `policy_lerobot.py` rewrites cmd builder — `--batch_size` / `--steps` / `--optimizer.lr` / `--config_path` / `--policy.push_to_hub=false`. New `_split_dataset_arg()` helper for local LeRobotDataset paths (synthesises a repo-id-like label + `--dataset.root=<path>`).
- Commits: adapters **bfef7e6** (rename), **c7639ca** (push_to_hub default).
- Verified: real `lerobot-train` reached training loop on SO-101 parquet.

### 4. Adapter dispatched DreamerV3 via wrong sheeprl entrypoint
- Symptom: `python -m sheeprl.cli exp=dreamer_v3 …` exited 0 immediately with no training and no files produced.
- Root cause: `sheeprl/cli.py` does not have an `if __name__ == "__main__"` block. The `@hydra.main`-decorated `run()` only dispatches via `sheeprl/__main__.py`. The adapter pointed at the wrong module. Also, sheeprl 0.5 namespaced config keys under `algo.*` (`algo.per_rank_batch_size`, `algo.world_model.optimizer.lr`, `algo.total_steps`) and the run dir lives at `hydra.run.dir`.
- Fix: `wm_dreamerv3.py` switched to `python -m sheeprl`, renamed every config key.
- Commit: adapters **0fb5434**.

### 5. Hydra `env.dataset_path=…` override rejected on built-in envs
- Symptom: `Could not override 'env.dataset_path'. Key 'dataset_path' is not in struct (full_key: env.dataset_path, object_type=dict)` when invoking adapter with `-- env=dummy` (overriding the `custom_hdf5` sentinel).
- Root cause: Hydra `=` is a set-only override; built-in env configs (dummy/atari/dmc) do not predefine `dataset_path`.
- Fix: prefix with `+env.dataset_path=…` so Hydra appends the key. Adapter still works for callers who register a real `custom_hdf5` env that pre-declares the key.
- Commit: adapters **c798649**.

---

## Outstanding Gaps (tracked in `docs/internals/system-improvements.md`)

1. **LeWorldModel real training is blocked.** `lerobot 0.5.x` does not ship `lerobot.scripts.train_world_model`. The `le_world_model` target's dry-run works but a real run will fail. Either rewire to HF research fork, build a minimal in-adapter trainer, or drop the target.
2. **No `custom_hdf5` sheeprl env.** DreamerV3 cannot consume the SO-101 HDF5 dataset out-of-box. `env=dummy` was used to validate runtime only — real DR/replay training requires a custom env plugin in `lerobot-isaac-adapters`.
3. **Cosmetic:** bridge metadata `action_dim` / `state_dim` count column names, not the underlying array width — confusing for downstream consumers. Actual ndarray shapes in HDF5 are correct.
4. **events.parquet `commits` column** mixed-type warning in dashboard auto-snapshot (non-fatal — file written empty).

---

## Commits (this session)

| Repo | SHA | Subject |
|---|---|---|
| claude_code | `4e6e21c` | fix(bridge): support dtype=image (inline PNG bytes in parquet) |
| lerobot-isaac-adapters (bare repo) | `bfef7e6` | fix(policy_lerobot): align with lerobot >= 0.5 CLI flags |
| lerobot-isaac-adapters | `c7639ca` | fix(policy_lerobot): default --policy.push_to_hub=false |
| lerobot-isaac-adapters | `0fb5434` | fix(wm_dreamerv3): use `python -m sheeprl` + new flag names |
| lerobot-isaac-adapters | `c798649` | fix(wm_dreamerv3): use Hydra `+env.dataset_path` append syntax |
| lerobot-isaac-training (workspace) | `c1c6b09` | docs+install: wire up train-deps script and document new gotchas |

---

## Lessons Learned (routed)

- **Project-specific** → already in `CLAUDE.md` "Common Pitfalls": train-deps not auto-installed, lerobot CLI rename, local dataset path splitting, LeWM gap, dtype=image bridge support.
- **Pipeline / orchestrator** → memory file `feedback-autonomous-progress`: never `ScheduleWakeup`-and-exit during long-running stages; poll background tasks every 60–120s and pipeline downstream prep work in parallel.
- **Systemic** → `docs/internals/system-improvements.md` new entries: upstream LeWM CLI absence, lerobot 0.5 CLI API drift, bridge `dtype=image` silently-broken case.
