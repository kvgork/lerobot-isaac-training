# System Improvements Log

Tracks systemic gaps discovered during orchestration runs. Project-specific patterns
go to `CLAUDE.md`. Pipeline improvements edit agent files directly. Only true
infrastructure/missing-agent gaps land here.

## Format

```
### YYYY-MM-DD — {short title}
- **Type:** missing-agent | missing-skill | infrastructure
- **Discovered in:** {sessionId}
- **Gap:** {what was missing}
- **Workaround applied:** {what we did instead}
- **Suggested fix:** {what to build}
```

## Entries

### 2026-05-08 — `outputs/eval/*.json` schema not specified
- **Type:** infrastructure (cross-package contract)
- **Discovered in:** `20260508-064654-metrics-dashboard`
- **Gap:** `lerobot-evaluation-agent` does not document or enforce the on-disk JSON shape it writes to `outputs/eval/*.json`. The dashboard's `eval_results` loader had to guess keys (`run_id`, `task`, `ts`, `pc_success`, `n_episodes`, `intervention_rate`, `mean_ep_len`).
- **Workaround applied:** Loader uses `df.get(col, default)` per field with NA fill + per-field warnings; documented as "contract pending" in module docstring + dashboard runbook 07.
- **Suggested fix:** Add a `docs/contracts/eval-results.md` (or co-locate with the evaluation agent) that pins the schema. Update `lerobot-evaluation-agent.md` to write that exact shape. Then the dashboard loader can drop its lenient mode.

### 2026-05-08 — `outputs/curriculum_stage.json` never written
- **Type:** missing-skill / agent-output gap
- **Discovered in:** `20260508-064654-metrics-dashboard`
- **Gap:** `lerobot-curriculum-agent` advances stages but does not persist the current stage state to disk. The dashboard's `curriculum` loader returns empty + UI banner, but the Curriculum tab can never populate without this file.
- **Workaround applied:** Loader returns empty canonical-shape DataFrame; runbook 07 troubleshooting points users at the agent contract.
- **Suggested fix:** Update `lerobot-curriculum-agent` to write `outputs/curriculum_stage.json` (`{stage, task_config, advancement_reason, ts}`) on every advancement, plus append a row to `outputs/curriculum_history.jsonl`.

### 2026-05-08 — No standardized run/snapshot registry across packages
- **Type:** infrastructure
- **Discovered in:** `20260508-064654-metrics-dashboard`
- **Gap:** Every package writes to `outputs/` with its own ad-hoc layout (eval, checkpoints, autoresearch, snapshots). Cross-package consumers (the dashboard, future agents) must hardcode glob patterns. A registry (e.g. `outputs/runs/<run_id>/manifest.json`) would let consumers iterate runs cleanly.
- **Workaround applied:** Dashboard loaders glob each output type independently; `outputs/snapshots/<id>/meta.json` introduces a partial pattern but only for snapshot state.
- **Suggested fix:** Define a `RunManifest` contract (probably in `lerobot-isaac-meta`) that every training/eval run writes. Migrate existing producers + the dashboard loaders to consume it.

### 2026-05-13 — `lerobot_world_model_bridge` skill fails on LeRobot v3.0 array-column action format
- **Type:** missing-skill (broken/incomplete for v3.0 format)
- **Discovered in:** `20260513-3phase-meta-audit-e2e`
- **Gap:** The skill's `lerobot_to_worldmodel()` calls `ep_df[action_keys].to_numpy().astype("float32")` which fails with `ValueError: setting an array element with a sequence` when `action` is stored as a single column of `np.ndarray` values (LeRobot v3.0 layout for multi-dim actions; observed on `lerobot/pusht`). The skill only supports the column-per-action-dim layout used by some older v2.1 datasets.
- **Workaround applied:** Step 2 of the Phase 3 e2e was skipped; policy training on `lerobot/pusht` used `lerobot-train` directly (which handles the array-column layout natively). Real HDF5 bridge for world-model training was never produced.
- **Suggested fix:** In `~/tools/claude_code/skills/lerobot_world_model_bridge/operations.py`, detect array-valued columns via `df[col].iloc[0].ndim >= 1` and use `np.stack(df[col].to_list())` instead of `to_numpy().astype(...)`. Same fix applies to states and rewards. Add a regression test that loads a v3.0 dataset with array-valued action column.

### 2026-05-13 — `torchcodec` lacks FFmpeg shared libraries on host system
- **Type:** infrastructure
- **Discovered in:** `20260513-3phase-meta-audit-e2e`
- **Gap:** Default LeRobot `dataset.video_backend` resolves to `torchcodec` when the package is importable, but `torchcodec` ships against FFmpeg 4–7 ABI and the system has FFmpeg 7 only via `/home/koen/.cache/rattler/cache/`. Result: `libavutil.so.59` not on `LD_LIBRARY_PATH`, training crashes before step 1.
- **Workaround applied:** Pass `--dataset.video_backend=pyav` to `lerobot-train` — pyav is installed and works.
- **Suggested fix:** Add `--dataset.video_backend=pyav` as a default in `packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/policy_lerobot.py` (with a comment pointing at this entry). Alternative: add a pixi task `install-ffmpeg-libs` that symlinks/installs `libavutil.so.59` from the rattler cache into the env's `lib/`. Document in `CLAUDE.md` "Common Pitfalls".

### 2026-05-13 — autoresearch `requires_workspace_root` gating mis-detects thin-meta-repo as standalone
- **Type:** infrastructure (test gating)
- **Discovered in:** `20260513-pipeline-smoke`
- **Gap:** `archive/packages/lerobot-isaac-autoresearch/tests/conftest.py::_in_monorepo()` checks for sibling packages under `<root>/packages/<sibling>`. Post-spinout the workspace lives in thin-meta mode: `packages/` only contains `lerobot-isaac-meta`, and the 6 siblings now live under `archive/packages/`. `_REQUIRED_SIBLINGS = ("lerobot-isaac-adapters", "lerobot-isaac-configs", "lerobot-isaac-env")` therefore evaluate to false, all `@pytest.mark.requires_workspace_root` tests auto-skip — including the 6-arch e2e metric-contract regression. CI silently passes with 6 skips instead of 6 passes.
- **Workaround applied:** Stage 5 of the pipeline smoke ran the `train_wrapper` subprocess directly (bypassing the marker) and verified all 5 archs emit the expected `<metric>=<float>` last line. Test gating itself was not touched (HARD STOP: no edits to `archive/packages/`).
- **Suggested fix:** Extend the `_in_monorepo()` heuristic to ALSO detect thin-meta-repo layout: if `<root>/archive/packages/<sibling>` exists alongside `<root>/packages/lerobot-isaac-meta/` AND a `[workspace]` table is present in root `pixi.toml`, the tree is still the active monorepo. Implement once the autoresearch package is re-imported back into a writable location (or fix in the spinout repo + re-publish).

### 2026-05-13 — `lerobot-isaac-dashboard` editable install points at non-existent path after spinout
- **Type:** infrastructure (install / pixi env staleness)
- **Discovered in:** `20260513-pipeline-smoke`
- **Gap:** `.pixi/envs/dashboard/lib/python3.12/site-packages/_editable_impl_lerobot_isaac_dashboard.pth` still records `/home/koen/workspaces/lerobot-isaac-training/packages/lerobot-isaac-dashboard/src` — the path before the rename to `archive/packages/`. Result: `python -m lerobot_isaac_dashboard.report` raises `ModuleNotFoundError`; only the `templates/` subdir (which `package_data` copies eagerly) is importable. The current `pixi.toml` resolves `lerobot-isaac-dashboard` from `file:///home/koen/workspaces/spinouts/lerobot-isaac-dashboard.git`, but the env was built before this swap and was never re-installed.
- **Workaround applied:** Stage 7 ran the report module with `PYTHONPATH=archive/packages/lerobot-isaac-dashboard/src`. The 4.6 MB HTML report rendered successfully (one minor `events.parquet commits` column-shape warning during the auto-snapshot).
- **Suggested fix:** Run `pixi install -e dashboard --frozen=false` (or `pixi clean && pixi install -e dashboard`) to rebuild the env against the current git+file:// URL spec. Add a CI smoke step that imports `lerobot_isaac_dashboard.report` from the dashboard env to catch this drift automatically. Also consider documenting "env rebuild required after spinout" in `docs/runbook/01-bootstrap.md`.

### 2026-05-13 — `lerobot 0.5.x` does not ship `lerobot.scripts.train_world_model`
- **Type:** missing-upstream (LeRobot package gap)
- **Discovered in:** `20260513-pipeline-validation-so101`
- **Gap:** `lerobot_isaac_adapters.targets.wm_leworldmodel` dispatches to `python -m lerobot.scripts.train_world_model`, but that module does not exist in any installed lerobot wheel through 0.5.1. Dry-run prints the resolved command and exits 0, but a real run fails with `ModuleNotFoundError: No module named 'lerobot.scripts.train_world_model'`. Therefore the entire `--target_arch le_world_model` path is unable to actually train, even with `bash scripts/install_train_deps.sh --lewm`.
- **Workaround applied:** Pipeline validation skipped the LeWM real run (Stage D) and used DreamerV3 (`--target_arch dreamerv3`) for the world-model side. Documented in `docs/runbook/00-install.md §Step 4 Known training-backend gap` and `CLAUDE.md` Common Pitfalls.
- **Suggested fix:** Either (a) rewire the adapter to call HF LeWorldModel's actual training script (which lives in a research fork, not main `lerobot`), (b) implement a minimal training loop in `lerobot-isaac-adapters` that consumes the `(96, 96)` HDF5 produced by the bridge, or (c) drop the `le_world_model` target until upstream publishes one. Track upstream at `https://github.com/huggingface/lerobot`.

### 2026-05-13 — `lerobot 0.5.x` CLI flag rename broke the adapter
- **Type:** API drift (downstream adapter went stale against upstream CLI)
- **Discovered in:** `20260513-pipeline-validation-so101`
- **Gap:** lerobot 0.5.x replaced `--training.batch_size` / `--training.num_steps` / `--training.lr` / `--config` with `--batch_size` / `--steps` / `--optimizer.lr` / `--config_path`, and now requires `policy.repo_id` when `policy.push_to_hub` is true (default). The pinned adapter still emitted the legacy flags. Smoke dry-run kept passing because it just `print(shlex.join(cmd))` — the bug only surfaces on a real `lerobot-train` invocation.
- **Workaround applied:** Patched `policy_lerobot.py` in `lerobot-isaac-adapters` (commits `bfef7e6` + `c7639ca` in the bare repo) to emit the new flag shape and default `--policy.push_to_hub=false`. Tests updated. Force-reinstalled in `train-policy`. Real `lerobot-train` on SO-101 now reaches the training loop.
- **Suggested fix:** Pin the adapter's lerobot version constraint in `pyproject.toml` (currently unpinned) and add a CI job that asserts the resolved subprocess command parses against the actually-installed `lerobot-train --help`. Otherwise the next CLI rename will silently break runs again. The adapter dry-run test should at minimum subprocess-execute `lerobot-train --help` and grep for each of its emitted flags.

### 2026-05-13 — bridge skill silently rejects LeRobotDataset v3.0 `dtype: image` columns
- **Type:** missing-skill (broken for inline image bytes)
- **Discovered in:** `20260513-pipeline-validation-so101`
- **Gap:** `lerobot_world_model_bridge.lerobot_to_worldmodel()` only handled the `dtype: video` layout (MP4 files under `videos/`). When `meta/info.json` declares `dtype: image` (encoded PNG/JPG bytes stored inline in parquet's struct column), the skill errored with "No video directories found". Additionally, the silent `cv2` ImportError fallback obscured the root cause.
- **Workaround applied:** Patched `skills/lerobot_world_model_bridge/operations.py` (commit `4e6e21c` in the claude_code repo) to auto-detect `dtype: image` features from `meta/info.json` and decode inline bytes via Pillow + numpy (no cv2 dep — works in any pixi env without opencv-python).
- **Suggested fix:** Regression-test the new `_load_episode_frames_from_parquet` helper. Add the SO-101 `dtype: image` dataset (or a 1-episode synthetic equivalent) to the skill's `tests/` fixtures.

### 2026-05-14 — `_lewm_minimal` trainer drops checkpoint on SIGTERM
- **Type:** code (downstream adapter)
- **Discovered in:** `20260513-pipeline-validation-so101` follow-up (Stage D2).
- **Gap:** `lerobot_isaac_adapters.targets._lewm_minimal.train()` writes `lewm_minimal_last.pt` only after the training loop's natural exit (`step > total_steps`). The 30-min watchdog SIGKILL skips that save, so a long real run produces metrics-only output and no usable checkpoint.
- **Workaround applied:** None — the metric stream itself proved convergence (pred_loss 0.02 → 0.0009). Re-run with `--steps` ≤ the watchdog budget to get a checkpoint, or wait for the fix below.
- **Suggested fix:** install `signal.signal(signal.SIGTERM, …)` (and `SIGINT`) handlers that flush a final checkpoint + final `pred_loss=` line to stdout before exiting. Mirror the same fix in `train_wrapper.py` so autoresearch executors see `FALLBACK_METRIC_LINE` instead of nothing when the wrapper is killed.

### 2026-05-14 — Dashboard N-way compare hits plotly duplicate-name error
- **Type:** infrastructure (dashboard package)
- **Discovered in:** `20260513-pipeline-validation-so101` follow-up (Stage F2).
- **Gap:** `lerobot_isaac_dashboard.compare --mode nway --snapshots A B C` raises `plotly.graph_objs._bar.Bar() got multiple values for keyword argument 'name'`. 2-way (`--mode 2way` default) is unaffected.
- **Workaround applied:** Stage F + F2 use 2-way compare only.
- **Suggested fix:** Audit how `compare.py` builds plotly Bar traces in N-way mode — likely passes `name=` both positionally and via kwargs when grouping by snapshot. Bare-repo `lerobot-isaac-dashboard`.
