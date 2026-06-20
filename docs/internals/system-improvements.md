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
- **Gap:** Default LeRobot `dataset.video_backend` resolves to `torchcodec` when the package is importable, but `torchcodec` ships against FFmpeg 4–7 ABI and the system has FFmpeg 7 only via `${HOME}/.cache/rattler/cache/`. Result: `libavutil.so.59` not on `LD_LIBRARY_PATH`, training crashes before step 1.
- **Workaround applied:** Pass `--dataset.video_backend=pyav` to `lerobot-train` — pyav is installed and works.
- **Suggested fix:** Add `--dataset.video_backend=pyav` as a default in `packages/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/policy_lerobot.py` (with a comment pointing at this entry). Alternative: add a pixi task `install-ffmpeg-libs` that symlinks/installs `libavutil.so.59` from the rattler cache into the env's `lib/`. Document in `CLAUDE.md` "Common Pitfalls".

### 2026-05-13 — autoresearch `requires_workspace_root` gating mis-detects thin-meta-repo as standalone
- **Type:** infrastructure (test gating)
- **Discovered in:** `20260513-pipeline-smoke`
- **Gap:** `src/lerobot-isaac-autoresearch/tests/conftest.py::_in_monorepo()` (when cloned) checks for sibling packages under `<root>/packages/<sibling>`. In the thin-meta-repo, `packages/` only contains `lerobot-isaac-meta`; siblings live in `src/<name>/`. `_REQUIRED_SIBLINGS = ("lerobot-isaac-adapters", "lerobot-isaac-configs", "lerobot-isaac-env")` therefore evaluate to false, all `@pytest.mark.requires_workspace_root` tests auto-skip — including the 6-arch e2e metric-contract regression. CI silently passes with 6 skips instead of 6 passes.
- **Workaround applied:** Stage 5 of the pipeline smoke ran the `train_wrapper` subprocess directly (bypassing the marker) and verified all 5 archs emit the expected `<metric>=<float>` last line. Test gating itself was not touched.
- **Suggested fix:** Extend the `_in_monorepo()` heuristic to also detect thin-meta-repo layout: if `<root>/src/<sibling>` exists alongside `<root>/packages/lerobot-isaac-meta/` AND a `[workspace]` table is present in root `pixi.toml`, the tree is still the active monorepo. Fix in `src/lerobot-isaac-autoresearch/` and push to GitHub.

### 2026-05-13 — `lerobot-isaac-dashboard` editable install points at non-existent path after spinout
- **Type:** infrastructure (install / pixi env staleness)
- **Discovered in:** `20260513-pipeline-smoke`
- **Gap:** `.pixi/envs/dashboard/lib/python3.12/site-packages/_editable_impl_lerobot_isaac_dashboard.pth` still records `${LEROBOT_ISAAC_WORKSPACE}/packages/lerobot-isaac-dashboard/src` — the legacy pre-spinout path. Result: `python -m lerobot_isaac_dashboard.report` raises `ModuleNotFoundError`; only the `templates/` subdir (which `package_data` copies eagerly) is importable. The current `pixi.toml` resolves `lerobot-isaac-dashboard` from `github.com/kvgork/lerobot-isaac-dashboard`, but the env was built before this swap and was never re-installed.
- **Workaround applied:** Stage 7 ran the report module with `PYTHONPATH=src/lerobot-isaac-dashboard/src`. The 4.6 MB HTML report rendered successfully (one minor `events.parquet commits` column-shape warning during the auto-snapshot).
- **Suggested fix:** Run `pixi install -e dashboard --frozen=false` (or `pixi clean && pixi install -e dashboard`) to rebuild the env against the current GitHub URL spec. Add a CI smoke step that imports `lerobot_isaac_dashboard.report` from the dashboard env to catch this drift automatically. Also consider documenting "env rebuild required after spinout" in `docs/runbook/01-bootstrap.md`.

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

### 2026-05-24 — `train-dreamer` pixi env has numpy/cv2 incompat
- **Type:** infrastructure
- **Discovered in:** `20260524-orchestrate-wm-isaac-trials-1to9`
- **Gap:** `.pixi/envs/train-dreamer` ships numpy 2.4.4 but cv2 (4.10.x) was built against numpy 1.x → `import cv2` → `numpy.core.multiarray failed to import` → `import sheeprl` cascade-fails (sheeprl.utils.env imports cv2). `.pixi/envs/sim` is unaffected (numpy 1.26.4 + cv2 4.8.0).
- **Workaround applied:** `scripts/_run_wm_isaac_overnight.sh` probes `sim` env first (works), only falls back to train-dreamer (would crash). Sweep launched in `sim` env.
- **Suggested fix:** Pin numpy<2 in `pixi.toml` `[feature.dreamerv3]` deps OR rebuild cv2 against numpy 2.x. Until fixed, do NOT run any sheeprl path via `train-dreamer` directly.

### 2026-05-24 — `lerobot_isaac_adapters.train` has no `--target_arch ppo`
- **Type:** missing-feature
- **Discovered in:** `20260524-orchestrate-wm-isaac-trials-1to9`
- **Gap:** `plans/2026-05-24-wm-isaac-hp-trials-1to9.md` trial 7 calls for a PPO baseline as a "DreamerV3-free reality check" (the key diagnostic for "is it the algo or the env?"). Adapter only supports `smolvla|act|diffusion|dreamerv3|le_world_model` — no `ppo` target.
- **Workaround applied:** Sweep script logs a `DEFERRED` warning and `continue`s past trial 7. The other 7 trials cover the reward-shape + entropy + replay axes.
- **Suggested fix:** Add `src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/targets/wm_ppo.py` (subprocess wrapper for `sheeprl exp=ppo` with the IsaacSO101Env wrapper). Metric: `Rewards/rew_avg`. Dispatch from `train.py`. Estimated effort: 4-6 h.

### 2026-05-24 — `_run_autoresearch_wm_isaac.sh` orphans children on SIGKILL
- **Type:** infrastructure (process management)
- **Discovered in:** `20260524-orchestrate-wm-isaac-trials-1to9` (sweep v2 → v3 transition)
- **Gap:** Killing the top sweep bash (`PGID 47529`) with `kill -KILL` did NOT propagate to grand-children `_run_wm_isaac_overnight.sh` + `timeout 10800` + `python`. The `_run_wm_isaac_overnight.sh` was re-parented to systemd (PPID 1) and continued running, spawning trial 1's python that conflicted with the next sweep launch's GPU. Required manual `kill -KILL` of every detached descendant by PID.
- **Workaround applied:** `kill -KILL <pid>` per descendant. Verified all v2 procs dead before launching sweep v3.
- **Suggested fix:** In `scripts/_run_autoresearch_wm_isaac.sh`, set `set -m` to enable job control and start the per-trial overnight script with `setsid` so it gets its own PGID. Then `trap 'kill -- -$$' EXIT TERM INT` on the sweep script propagates death to the entire tree. Pre-flight worker advisory B3 (2026-05-24 review) already flagged the `set -uo pipefail` without `-e` as risky — this confirms the corollary.

### 2026-05-24 — code-review didn't catch scene-key mismatch in success_termination
- **Type:** pipeline (code-review-orchestrator gap)
- **Discovered in:** `20260524-orchestrate-wm-isaac-trials-1to9` (sweep v3 trial 0 crash)
- **Gap:** Phase 1+2 code-review-orchestrator focused on Hydra plumbing + pkill scope + soft-import discipline but did NOT verify that `success_termination(env)` actually resolves the right scene-entity key for the active task. Sweep v3 hit `KeyError 'object'` because pick_and_place uses `source_object` (and PickEnvCfg uses `target_object` via scene.target_object) — neither matches the hardcoded `env.scene["object"]` in `terminations.py`.
- **Workaround applied:** Added `object_name` + `robot_name` kwargs to `success_termination`, set sensible defaults (`source_object`), per-task override in pick.py. Commit `811c2e2`.
- **Suggested fix:** Update `agents/orchestrators/code-review-orchestrator.md` checklist to include: "when a new Isaac Lab term function (reward, termination, observation) is added, grep all task subclasses' scene_cfg attributes for entity names and verify the function references them or accepts a parametrized `<entity>_name` kwarg with a verifiably-correct default." Pre-flight worker should also be updated to scan for hardcoded `env.scene["..."]` strings in newly-added term funcs.

### 2026-05-25 — motor-write safety needs review-orchestrator + `np.clip(action)` first
- **Type:** pipeline (code-review-orchestrator + implementation-executor patterns)
- **Discovered in:** `20260524-orchestrate-wm-isaac-trials-1to9` (deploy phase)
- **Gap:** First implementation of DreamerV3 motor-write adapter passed initial executor self-checks AND landed with 6 safety blockers, surfaced ONLY when the orchestration pipeline ran `code-review-orchestrator` post-hoc:
  1. No `np.clip(action, -1, 1)` before scaling — pathological actor logits → unbounded per-step motion
  2. `read_joint_limits()` returned cal-derived limits LOOSER than hardcoded safety floor (e.g. cal `[0, 4095]` → `±180°` overwriting `±90°` floor)
  3. `elbow_flex -10°` table-avoid floor lost when cal returned symmetric range
  4. `home_targets(0.0)` instant goto from arbitrary pose → high-velocity slam risk
  5. `max_relative_target` not passed to SO101FollowerConfig → server-side safety clamp OFF
  6. No NaN / range validation on `current_jp` → comm failure → garbage targets
- **Workaround applied:** `bc46c0f` — applied all 6 fixes after review. All 142 tests pass.
- **Suggested fix:** Update `agents/orchestrators/master-project-orchestrator.md` to require `code-review-orchestrator` AS A HARD GATE when the implementation touches:
  - Real-hardware motor writes (any path that imports `lerobot.robots.*`)
  - Physical-process control with safety clamps
  - Persistent storage with unbounded writes (e.g. training loops, sweep loops)
  Currently Step 7.5a is gated only on `RIGOR=agentic`, but the implementation-executor's self-review may miss safety bugs in narrow domains. Domain-specific review checklists (motor-write, training-loop, schema-migration) would improve consistency.

### 2026-06-19 — CLAUDE.md is 539 lines (2.7× over the 200-line ceiling)
- **Type:** systemic (CLAUDE.md hygiene)
- **Discovered in:** `20260619-161309-level3-pipeline` (Step 8 lessons-routing pre-check)
- **Gap:** Project `CLAUDE.md` is 539 lines vs the Karpathy 200-line ceiling. The `claude-md-update` skill will refuse any new rule append while over-ceiling, so behavioural-rule additions are currently blocked without a prune. A stale OOM-ladder ref (`dataloader-gpu-decode-plan.md` → moved to `plans/archive/`) was found + corrected in-place this run.
- **Workaround applied:** Surgical zero-growth stale-ref correction at CLAUDE.md:333 (added `archive/` prefix + pointer to the active `dali-gpu-decode-plan.md`). Did NOT prune mid-orchestration (out of scope; disruptive).
- **Suggested fix:** Run a dedicated `claude-md-prune apply` session to relocate verbose pitfall/runbook content (the bulk of lines ~250–400) into `docs/`, leaving cross-links. Target ≤200 lines so `claude-md-update` unblocks.

### 2026-06-20 — dry-run tests fail in the `default` env (torch absent) for lack of skip markers
- **Type:** systemic (test hygiene / CI marking)
- **Discovered in:** `20260620-084313-continue-level3-plan` (Step 7.5b full CPU sweep across 4 trees)
- **Gap:** `lerobot-isaac-deploy/tests/test_wm_dryrun.py::test_run_dryrun_missing_ckpt_raises` and `lerobot-isaac-autoresearch/tests/test_e2e_dry_run.py::{test_wrapper_dry_run_does_not_invoke_heavy_backend, ...[dreamerv3]}` fail in the `default` env with `ImportError: torch is required ...`. Heavy deps are intentionally not installed in `default` (see CLAUDE.md), but these tests are not marked `requires_dreamerv3`/`requires_lerobot`, so they run and fail instead of skipping. `test_e2e_dry_run.py` also uses an unregistered `@pytest.mark.requires_workspace_root` (PytestUnknownMarkWarning). Pre-existing — NOT caused by the Phase 1/2 changeset shipped this session.
- **Suggested fix:** mark the torch-dependent dry-run tests with the existing `requires_*` markers (or guard with `pytest.importorskip("torch")`), and register `requires_workspace_root` in the autoresearch `pyproject.toml`/`conftest.py` so the marker is honored and the warning clears.
