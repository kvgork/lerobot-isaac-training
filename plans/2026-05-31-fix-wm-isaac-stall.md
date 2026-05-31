# Tomorrow Plan — Fix the WM-Isaac (Track C) training stall (2026-05-31)

> **STATUS 2026-05-31: Track 1 (stall) RESOLVED. Track 2 (watchdog) DONE.**
> The "stall" was never a stall — it was a **masked crash + Isaac teardown hang**.
> GPU-verified: DreamerV3 now reaches gradient updates (`Loss/world_model_loss`
> logged to TB at step 300) AND completes the eval phase (`Test - Reward` logged)
> AND exits cleanly (GPU freed, no orphans). See **§Resolution** below.
> Remaining: Track 3 sweep (GPU hrs), Track 4 SmolVLA-on-merged, Track 5 scored eval.

---

## Resolution (2026-05-31, GPU-verified)

Three layered bugs, all hidden behind one hang:

1. **`learning_starts < per_rank_sequence_length (64)` → crash.** The first grad
   update calls `rb.sample_tensors(sequence_length=64)`; if fewer than 64 steps
   were collected it raises `ValueError: Cannot sample a sequence of length 64.
   Data added so far: N` (`sheeprl/data/buffers.py:432`). The plan's own §1b debug
   cmd used `algo.learning_starts=16` → self-inflicted. **Prod default 1024 is OK.**
   Rule: keep `per_rank_sequence_length ≤ learning_starts < total_steps`.

2. **Shared-singleton env torn down before eval → crash.** `dreamer_v3.py:765`
   calls `envs.close()` then `:767 test(player,…)`, and `test()`
   (`dreamer_v3/utils.py:115`) builds a FRESH env via `make_env(…)()` + resets it.
   That fresh `IsaacSO101Env` reuses the process-global `_GLOBAL_BACKING_ISAAC_ENV`.
   The old `close()` called `self._isaac_env.close()` → `ManagerBasedRLEnv` deletes
   `.scene` → eval reset crashes `'ManagerBasedRLEnv' object has no attribute
   'scene'`. **Fix:** `IsaacSO101Env.close()` is now a **no-op** on the backing env.

3. **Isaac atexit hang masks every crash as a "stall".** On any exit, Isaac's
   `SimulationApp.close()` hangs forever in `render()`. Process stays alive holding
   ~1.6 GB VRAM, emits no metric → looks frozen, returns `metric=-9999`. **Fix:**
   `_wm_isaac_entry.py` wraps `run()` and calls `os._exit(code)` to bypass atexit —
   trials now die the instant `run()` returns/raises, freeing the GPU.

**Diagnosis method that worked:** §1a proved raw Isaac steps fast (52.7 steps/s) →
not Isaac. Then `faulthandler.dump_traceback_later()` (no sudo/ptrace needed, unlike
py-spy under `ptrace_scope=1`) caught both the `ValueError` and the teardown-hang
stack. The `make_env()` path runs cameras OFF → obs is flat `(1,18)`, so the CNN
encoder trains on a **synthesized all-zero RGB** (`camera_key="wrist_camera_rgb"`
was renamed `d435_rgb` in DR100 Phase 1, and `make_env` takes no `enable_cameras`).
That is a real obs-quality gap but NOT the stall — tracked separately below.

**Files changed:** `scripts/_wm_isaac_entry.py` (os._exit wrapper),
`src/lerobot-isaac-adapters/.../sheeprl_plugin/isaac_env.py` (close no-op — commit
inside the sibling repo to persist), `scripts/_run_wm_isaac_overnight.sh`
(watchdog → report-only, Track 2).

### Follow-ups
- **DONE — WM CNN now sees real RGB.** `make_env(enable_cameras=True)` wires the
  `d435_rgb` term; wrapper `camera_key="d435_rgb"` + `_resize_chw` 480×640→64².
  GPU-verified: `observation_loss≈85` on real frames (vs ~0 for zeros). Committed
  (adapters `05ad0b0` + env `61d26f3`).
- **DONE — `Rewards/rew_avg` now logs (3rd sheeprl×gymnasium shim).** sheeprl 0.5.8
  reads `infos["final_info"]` (gymnasium <1.0 vector API); gymnasium 1.2.1 removed
  it (stats now in `infos["episode"]`+`infos["_episode"]`). Without a bridge,
  `rew_avg`/`ep_len_avg` never log → sweep ratchets at `-9999` despite healthy
  training. Fix: `_patch_gym_vector_final_info()` in `_wm_isaac_entry.py` rebuilds
  `final_info` each vector step. Verified: `Rewards/rew_avg` + `Game/ep_len_avg`
  appear in TB. The full 20k trial (`wm-isaac-hp-…-133757`, dur 5057s) proved the
  pipeline (WM+actor+critic learning, eval, clean exit) — it returned `-9999` only
  because this shim hadn't landed yet; re-runs will scrape a real `rew_avg`.
- **Still open (overnight path only):** `wm_dreamerv3.py` parses `recon_loss=` from
  stdout, which sheeprl never prints. Harmless for the *sweep* (scrapes TB
  directly); the *overnight* single-trial emitted metric is meaningless. Scrape TB.

---

**Goal:** get DreamerV3 (sheeprl) actually *training* on the Isaac SO-101 env so a
Track C HP sweep produces real `rew_avg` metrics. Then run the sweep, train on
the merged DR100 dataset (Track B payoff), and the SO-101 scored eval (Track D).

---

## State at start of day (verified 2026-05-30)

**DONE:**
- ✅ **Track A** (sim verify) — GPU-verified: A.1 obs dict, A.2 `env smoke`, A.3 sim-deploy boot.
- ✅ **Track B** (DR100 + multi-dataset) — **complete**:
  - DR replay writes the **real dataset's exact schema** (CHW `[3,480,640]`, joint
    names, tuple shapes). DR100 = 80 ep / 23,372 frames.
  - Merge rewritten (v3.0 API, tuple-shape coercion) → merged = **100 ep / 30,863
    frames**, loads via `LeRobotDataset`. `datasets/kvgork/so101-pickplace1-dr100-merged`.
  - `--datasets` multi-dataset flag + meta CLI `train`/`dr-replay` wired.
- ✅ **Track D** mechanics — motor control proven (sweep + execute moved the arm);
  `robot-data-run-sweep` / `-replay` built; feetech SDK, camera `/dev/video2`,
  fixed calib id `so101_follower`, HF-offline all wired.

**BROKEN (the focus):**
- ⚠️ **Track C training stalls.** Two issues found:
  1. **Crash — FIXED:** sheeprl 0.5.8 × gymnasium 1.2.1 `TransformObservation`
     2-arg call. Shim in `scripts/_wm_isaac_entry.py::_patch_gym_transform_observation`.
  2. **Stall — OPEN:** after the gymnasium fix, training boots Isaac → builds the
     DreamerV3 model (`Encoder CNN keys: ['rgb']`) → sets up the training env →
     then **never reaches a gradient update**. TB logs only `hp_metric`; no
     `world_model_loss` / `rew_avg` / `Grads/actor` even with **no watchdog** and
     GPU active (~30%). Every trial returns `metric=-9999`.
- ⚠️ **Watchdog mis-design** (`scripts/_run_wm_isaac_overnight.sh`): auto-kills at
  T+300s on "log frozen" → false-positive killed slow/quiet-but-progressing runs
  at a fixed ~312s. Must become **report-only** (see memory `watchdog-report-only`).

---

## Track 1 — Diagnose the training stall (DO FIRST, ~1–2h)

The decisive question: **where** does sheeprl block after env/model setup? Likely
the env-collection loop (`learning_starts` random steps) or the first env
`reset()`/`step()` inside sheeprl's `SyncVectorEnv` wrapper around the Isaac env.

### 1a. Isolate: raw Isaac stepping vs sheeprl loop
Time raw `env.step()` on the SO-101 env (no sheeprl) — does Isaac itself step fast?
```bash
pixi run -e sim python -u -c "
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True, enable_cameras=True).app
import time, torch, gymnasium as gym, lerobot_isaac_env
from lerobot_isaac_env.tasks import _register_envs, PickAndPlaceEnvCfg
_register_envs()
env=gym.make('Isaac-SO101-PickPlace-v0', cfg=PickAndPlaceEnvCfg(enable_cameras=True), num_envs=1)
env.reset()
a=torch.zeros(env.action_space.shape, device=env.unwrapped.device)
t0=time.time()
for i in range(200): env.step(a)
print('200 steps in', round(time.time()-t0,1),'s =', round(200/(time.time()-t0),1),'steps/s')
import os; os._exit(0)"
```
- **Fast (>20 steps/s):** stall is in sheeprl's wrapper/loop → go to 1b.
- **Slow/hangs:** stall is the Isaac env (camera render at 1 env) → go to 1c.

### 1b. py-spy the live sheeprl process (needs sudo for ptrace)
Run a trial unbuffered + no watchdog (use the §Track-1 entry cmd below), then in
another shell:
```bash
PYPID=$(pgrep -f scripts/_wm_isaac_entry.py | head -1)
sudo env "PATH=$PATH" py-spy dump --pid $PYPID         # py-spy already installed in sim env
# repeat 'py-spy dump' 2–3× over 30s; the common top frame = the block site.
```
Look for: a frame in `sheeprl/algos/dreamer_v3` collection, `env.step`,
`isaaclab` render, a `torch` CUDA sync, or a `queue`/`recv` wait.
(If sudo unavailable: `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope`, or
add `import faulthandler, signal; faulthandler.register(signal.SIGUSR1)` to the
entry script and `kill -USR1 $PYPID` to dump the stack to stderr.)

### Direct entry cmd (no watchdog, unbuffered) — use for 1b/1c
```bash
PLUGIN=$(pixi run -e sim python -c "import lerobot_isaac_adapters.sheeprl_plugin as p,os;print(os.path.join(os.path.dirname(p.__file__),'configs'))")
PYTHONUNBUFFERED=1 pixi run -e sim python -u scripts/_wm_isaac_entry.py \
  --config-dir "$PLUGIN" exp=dreamer_v3 env=isaac_so101 \
  algo.total_steps=2000 algo.learning_starts=16 metric.log_every=200 \
  hydra.run.dir=/tmp/c_dbg 2>&1 | tee /tmp/c_dbg.log
# Watch for: "Rank-0" step lines, Loss/world_model_loss, sps. TB at
# logs/runs/dreamer_v3/isaac_so101/<latest>/version_0 (check tags != just hp_metric).
```

### 1c. Likely fixes by root cause
- **Isaac stepping slow with cameras:** the sweep image obs (`d435_rgb 480×640`)
  is heavy for DreamerV3 (which wants 64×64). Check the sheeprl_plugin env config
  — it should **resize obs to 64×64** and/or DreamerV3 `cnn_keys` should point at a
  downsized key. If the env feeds 480×640 to the CNN encoder, the first forward is
  huge → fix the plugin env wrapper to resize (or add a `Resize` obs wrapper).
- **sheeprl SyncVectorEnv reset hang:** Isaac `ManagerBasedRLEnv` is already
  vectorized (num_envs); sheeprl wrapping it in its own vector env can double-wrap
  / deadlock. Verify the plugin uses `env.num_envs=1` and sheeprl `env.num_envs=1`
  (no nested vectorization).
- **bf16/precision init stall:** try `fabric.precision=32` to rule out an AMP hang.
- **learning_starts collection never flushes a metric:** confirm with 1a that env
  steps happen; if they do but no TB, lower `metric.log_every` (done: 200) and
  confirm the FIRST gradient update is reached (`per_rank_pretrain_steps`).

**Acceptance for Track 1:** `Loss/world_model_loss` (and eventually `Rewards/rew_avg`)
appear in TB for the direct entry run → training genuinely progresses.

---

## Track 2 — Watchdog: report-only (~30 min)  [memory: watchdog-report-only]

In `scripts/_run_wm_isaac_overnight.sh`:
- **Remove** the "log frozen → kill" branch in `watchdog_check`. Silence ≠ hung
  (training is quiet during model build + collection).
- Keep ONLY: **report** process-alive, `train.log` age, and the **TB step/metric
  progress** (reuse `.tb_scrape.py` to print step + last loss) at intervals.
- Auto-kill ONLY on an explicit fatal pattern (`Traceback|RuntimeError`), not on
  mtime/no-output.
- Emit the status to stdout/state so the autoresearch loop (or operator) decides.

**Acceptance:** a deliberately-slow trial runs to its step budget without a
premature kill; status lines show increasing TB step count.

---

## Track 3 — Run the Track C sweep (after 1+2 green, GPU hours)

```bash
DRY_RUN=1 bash scripts/_run_autoresearch_wm_isaac.sh           # sanity
# 3 best-bet trials first (~9h): trial 0,4,5
MAX_TRIALS=6 SKIP_TRIALS=0 bash scripts/_run_autoresearch_wm_isaac.sh
```
**Acceptance (plan bar):** ≥1 trial reaches `rew_avg > 0` (sparse-success firing);
chosen trials reach ≥40k steps without actor collapse. Full 8 = optional.

---

## Track 4 — Train a policy on the merged DR100 dataset (Track B payoff, GPU)

The merged real+sim dataset exists. Validate it improves SmolVLA:
```bash
bash scripts/install_train_deps.sh   # ensure lerobot[smolvla] in train-policy
pixi run -e train-policy lerobot-isaac-train --target_arch smolvla \
  --dataset datasets/kvgork/so101-pickplace1-dr100-merged \
  --cache_frames --policy.load_vlm_weights=true --steps 20000   # via -- remainder
```
**Acceptance:** trains end-to-end on 100 ep; compare eval vs the real-only trial_7.

---

## Track 5 — SO-101 scored eval (Track D, operator + E-stop)

Mechanics proven. Scored closed-loop = motor writes → **human on E-stop**, not autonomous.
```bash
pixi run -e train-policy lerobot-isaac-deploy session \
  --policy-path outputs/.../trial_7/checkpoints/045000/pretrained_model \
  --dataset-root datasets/kvgork/so101-pickplace1 \
  --port /dev/ttyACM0 --camera d435_rgb=/dev/video2,640,480 \
  --robot-id so101_follower --clamp-tight 1.0 --clamp-loose 5.0 \
  --n-eval-episodes 3 --require-real-ckpt --execute
```

---

## Files touched 2026-05-30 (context for tomorrow)
- `scripts/_wm_isaac_entry.py` — gymnasium TransformObservation shim (crash fix).
- `scripts/_run_wm_isaac_overnight.sh` — **watchdog to fix** (report-only).
- `src/lerobot-isaac-synthetic/.../isaac_dr/replay_runner.py` — obs→LeRobot row (CHW), `_load_source_features`.
- `src/lerobot-isaac-synthetic/.../isaac_dr/parquet_writer.py` — source-features schema, conditional next.done.
- `src/lerobot-isaac-synthetic/.../merge_utilities.py` + `merge.py` — v3.0 API merge, tuple shapes.
- `src/robot-data-runner/.../` — `--id` calib, HF-offline, `run_replay`/`cli_replay`, `run_sweep`/`cli_sweep`.
- `src/lerobot-isaac-deploy/.../session.py`, `cli.py` — camera video2 default, robot-id, `--skip-dry-loop` (partial), HF-offline.

## Known orphan-process gotcha
Isaac `kit.app` procs don't die on SIGTERM/SIGKILL cleanly — a stuck trial can
leave 1–2 zombies holding ~1.6 GB VRAM. `cleanup_isaac_orphans` in the overnight
script + a reboot clears them. Check `nvidia-smi` before launching a sweep.
