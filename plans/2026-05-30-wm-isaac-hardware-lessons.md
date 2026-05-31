# Lessons — Track A/B GPU bring-up + SO-101 hardware deploy + WM-Isaac (2026-05-30)

Hard-won during a long GPU+hardware session. Grouped by area. Each is a real bug
hit + the fix, so future sessions don't re-discover them.

## Isaac Lab boot / AppLauncher ordering
- **AppLauncher MUST be constructed before importing ANY `lerobot_isaac_env`
  symbol.** Importing the env package pulls module-level `isaaclab`/USD imports
  (`so101_env_cfg`); if those load before Kit boots, a stale `pxr` is loaded and
  Kit crashes with SIGSEGV ("extension class wrapper … not created yet"). Fix:
  build `AppLauncher(...).app` inline in a heavy-import-free module (e.g.
  `lerobot_isaac_meta.cli._cmd_env`) THEN import the env path. A helper that lives
  *inside* `lerobot_isaac_env` can't be used for this — importing it triggers the
  package.
- **Kit parses `sys.argv` at construction.** Leftover CLI subcommand flags crash
  it. Strip to `sys.argv[:1]` for the launch, then restore.
- **`SimulationApp({"enable_cameras": True})` is NOT enough** — only
  `isaaclab.app.AppLauncher(enable_cameras=True)` sets the carb flag
  `isaaclab`'s Camera spawn checks. Raw SimulationApp → "A camera was spawned
  without the --enable_cameras flag" at env reset. Use AppLauncher.
- **Enabling cameras needs BOTH** AppLauncher `enable_cameras=True` AND the env
  cfg `enable_cameras=True` (the cfg is what wires the `d435_rgb` obs term via
  `_wire_cameras`).
- **Isaac `kit.app` orphans don't die on SIGTERM/SIGKILL cleanly** — a stuck trial
  leaves 1–2 zombies holding ~1.6 GB VRAM. Check `nvidia-smi` before a sweep;
  reboot clears them.
- **Headless stdout is block-buffered + `app.close()` does `os._exit`** → buffered
  prints are lost. Use `PYTHONUNBUFFERED=1 python -u` + `flush=True` and print
  results BEFORE close (matches the existing smoke-script pitfall).

## Env config / observation wiring
- **A single-shape obs group flattens to a bare Tensor.** With an image term
  added, `obs['policy']` must stay a dict — set `concatenate_terms=False` when
  cameras are wired, else `obs['policy'].keys()` raises AttributeError.
- **`torch.Tensor` has `.values()` / `.keys()`** (sparse-layout API) → testing obs
  type with `hasattr(x,'keys')` is WRONG (calls `.values()` on a dense tensor →
  "expected sparse tensor layout"). Use `isinstance(x, dict)`.
- **Camera prim path:** the current `so101_new_calib` USD nests links under a
  `Geometry` Scope → `…/Robot/Geometry/base_link/.../wrist_link/d435` (the old
  `Payload`-derived path without `/Geometry` raised "Unable to find source prim").
- **Articulation data is only valid after `sim.reset()`** — accessing
  `articulation.num_joints` before reset raises
  `'Articulation' object has no attribute '_root_physx_view'`. Defer any data
  access (even a debug log) until after reset.

## isaaclab Camera + auto-scene USD
- **`CameraCfg` requires `spawn=None`** to attach to an existing USD camera prim
  (omitting `spawn` → "Missing values … spawn").
- **isaaclab requires canonical `[translate, orient, scale]` xform ops** on prims
  it wraps; a single `matrix4d xformOp:transform` is rejected ("not a xformable
  prim with standard transform operations"). Fixed in the auto-scene writer
  (`scene_gen.write_usd_stub`) to emit `translate`+`orient`+`scale` for the camera.
- **auto-scene camera prim is `/World/Scene/D435`**, not `/World/Scene/cameras/<obs_name>`.

## LeRobotDataset / data pipeline (the big one)
- **`add_frame` compares `array.shape` (tuple) to `feature['shape']`.** Features
  loaded from `info.json` have shapes as **lists** (`[6]`), so `(6,) != [6]` and
  EVERY frame is silently rejected ("feature 'X' of shape '(6,)' does not have the
  expected shape '[6]'"). **Always coerce feature shapes to tuples** before
  `LeRobotDataset.create`.
- **Synthetic (DR) datasets must match the real dataset's EXACT schema** to merge
  or co-train: image layout (real = CHW `[3,480,640]`, names `channels/height/
  width`), joint `names`, and which columns exist (real has NO `next.done`).
  Reuse the source dataset's `info.json` features rather than a hand-rolled
  `_DEFAULT` — the DR writer now does this via `_load_source_features`.
- **lerobot reads images as CHW float at `ds[i]`** regardless of declared layout;
  the declared `shape`/`names` is metadata the policy preprocessing keys on. Keep
  the synthetic declared schema identical to real.
- **`merge_utilities` was v2.x** (assumed per-episode `episode_NNNNNN.parquet`);
  LeRobotDataset v3.0 is **sharded** (`meta/episodes/chunk-*/file-*.parquet`,
  `data/chunk-*/file-*.parquet`). Read frames through the `LeRobotDataset` API
  (`dataset[i]` over `dataset_from_index..dataset_to_index`) instead of raw
  parquet — format-agnostic.

## sheeprl / DreamerV3 (Track C)
- **sheeprl 0.5.8 × gymnasium 1.2.1 incompatibility:** sheeprl calls
  `TransformObservation(env, func)` (2-arg) but gymnasium 1.2.1 (hard-pinned by
  isaaclab `==1.2.1`) made `observation_space` a required positional. Can't
  downgrade gymnasium → shim it (`observation_space=None` matches old inferred
  behaviour). Patch BEFORE importing sheeprl.
- **RESOLVED 2026-05-31 (GPU-verified).** It was never a stall — a **masked crash
  + Isaac teardown hang**. All three suspects above were WRONG (Isaac steps at
  52.7/s; obs is already coerced to 64×64; num_envs=1 no double-vec). Real causes:
  1. **`learning_starts < per_rank_sequence_length (64)`** → first grad update
     raises `ValueError: Cannot sample a sequence of length 64` (`buffers.py:432`).
     The §1b debug cmd's `learning_starts=16` was self-inflicted; prod default 1024
     is fine. Invariant: `per_rank_sequence_length ≤ learning_starts < total_steps`.
  2. **Wrapper `close()` tore down the process-global backing env**, but sheeprl
     calls `envs.close()` then `test()` which resets a fresh wrapper reusing that
     dead singleton → `'ManagerBasedRLEnv' object has no attribute 'scene'`. Fix:
     `IsaacSO101Env.close()` = no-op on the backing env.
  3. **Isaac atexit `SimulationApp.close()` hangs in `render()`** → any crash looks
     like a frozen, GPU-holding "stall" (`metric=-9999`). Fix: `_wm_isaac_entry.py`
     wraps `run()` + `os._exit(code)` to skip atexit.
  Verified: `Loss/world_model_loss` logs to TB + eval `Test - Reward` completes +
  process exits clean (GPU freed). Diagnosis tool: `faulthandler.dump_traceback_later`
  (no sudo/ptrace, unlike py-spy under `ptrace_scope=1`).
- **KNOWN GAP (not the stall):** the sheeprl wrapper feeds the CNN an **all-zero
  RGB** — `make_env()` doesn't enable cameras and `camera_key` is the pre-DR100
  `wrist_camera_rgb` (now `d435_rgb`). WM learns no vision until this is wired.
- **`metric.log_every=5000`, `learning_starts=1024`** (sheeprl defaults) → a
  <5000-step smoke logs NO `rew_avg`; don't conclude "broken" from a tiny run.

## Watchdog design
- **Report-only, never auto-kill on a frozen log.** The T+300s "log-frozen → kill"
  in `_run_wm_isaac_overnight.sh` killed slow-but-progressing trials at a fixed
  ~312s (quiet model-build/collection phase). Silence ≠ hung. Kill only on an
  explicit fatal pattern; otherwise emit status and let a decision step act.

## SO-101 hardware deploy
- **Feetech SDK:** the follower needs `scservo_sdk` from `lerobot[feetech]`
  (`feetech-servo-sdk`). Without it, robot setup fails at the dry-run loop with
  `No module named 'scservo_sdk'`. Added to `LEROBOT_EXTRAS` default.
- **D435 RGB stream = `/dev/video2`** (video0/1 = depth/metadata; not openable as
  RGB by OpenCV). The D435 enumerates 6 nodes; color is the 3rd.
- **Fixed calibration id** (`--id so101_follower` → `SO101FollowerConfig(id=...)`)
  → calibration persists at `so_follower/<id>.json` and is reused; without an id
  it saves to `None.json` and re-calibrates (range-of-motion prompt) each run.
- **HF request flood:** huggingface_hub re-validates the cached SmolVLM2 backbone
  on every policy load (hundreds of HEAD/GET). Set `HF_HUB_OFFLINE=1`
  (+ `TRANSFORMERS_OFFLINE=1`) BEFORE hf_hub imports, at each entrypoint.
- **lerobot `max_relative_target` clamp** clips per-step joint delta server-side.
  Too tight (2°) + heavy proximal joints under gravity → joints "stuck" at a
  constant clamped goal while light wrist joints track. Raise `--clamp-loose`
  (5°) for real motion; constant `safe goal_pos` across steps = not advancing.
- **Open-loop tooling added:** `robot-data-run --actions-out` records a trajectory;
  `robot-data-run-replay` plays it back; `robot-data-run-sweep` drives all joints
  on a sine through a safe % of calibrated range (normalized space, amplitude =
  % of range). All dry-run by default, `--execute` for motors.

## Process / scope
- Each track kept revealing genuine integration bugs (schema mismatch,
  sheeprl/gym, TB scrape, watchdog) — "finish B/C/D" was a multi-session effort,
  not one pass. B finished; C crash fixed but stall open; D mechanics proven,
  scored eval is operator-gated.
