# GPU + Hardware Execution Checklist (2026-05-30)

**Supersedes the runnable parts of** `plans/2026-05-27-tomorrow.md`.
This file is what an operator runs ON the GPU/hardware machine. The code-only
deliverables (Tracks A.1, A.2, B.1-P2, B.2) **landed + were unit/dry-run
verified on 2026-05-30** — see "What landed" below. The remaining steps need a
live GPU and/or the physical SO-101 and could not run in the headless session.

---

## What landed 2026-05-30 (code complete; GPU-verified where noted)

| Track | Change | Verification |
|-------|--------|------------------------|
| A.1 | `concatenate_terms=False` on policy obs group when cameras wired (`so101_env_cfg._wire_cameras`) | **GPU-VERIFIED**: `obs['policy']` is a dict `['d435_rgb','joint_pos','joint_vel','last_action']`, `d435_rgb (1,3,480,640)` |
| A.2 | `lerobot-isaac env smoke` subcommand + GPU runner `smoke.py` | **GPU-VERIFIED**: no-cam `(1,18)`; `--cameras=d435` dict obs, 20 steps finite, exit 0 |
| A.3 | sim-deploy boot fixes (articulation pre-reset, camera spawn/prim/xform) | **GPU-VERIFIED**: `booted num_joints=6 cameras=['d435_rgb'] stepped=100 OK` |
| B.1-P2 | replay `--camera_key`/`--source_tag`; parquet schema → `d435_rgb` PNG + state (6,) | `test_dry_run.py` 11/11 |
| B.1-P3 | replay camera capture (AppLauncher + cameras-enabled cfg + camera_key thread) | **GPU-VERIFIED**: 1-ep replay `n_obs=300 obs_keys=[d435_rgb,joint_pos,joint_vel,last_action]` |
| B.2 | `--datasets` multi-dataset flag + policy backend forwarding | `test_multi_dataset.py` 9/9; 0 new failures |

### Real bugs found + fixed during GPU verification (2026-05-30)

1. **A.1** policy obs group flattened to a bare Tensor (image+state can't concat) → `concatenate_terms=False` when cameras wired.
2. **Camera prim path** missing `/Geometry` scope — `so101_new_calib` USD nests links under a `Geometry` Scope; old `Payload`-derived path raised "Unable to find source prim path".
3. **AppLauncher import ordering** — any `lerobot_isaac_env` import pulls module-level USD imports; must construct `AppLauncher` BEFORE importing it (else Kit SIGSEGV). Fixed in `cli._cmd_env` (inline) + `smoke.run_env_smoke(simulation_app=...)`.
4. **sys.argv pollution** — Kit parses argv; leftover subcommand flags crash it → strip to argv[0] for the launch.
5. **smoke `.values()`** — torch tensors expose `.values()` (sparse) → use `isinstance(dict)` not `hasattr`.
6. **A.3 articulation** — `num_joints` read pre-`sim.reset()` → `_root_physx_view` AttributeError; deferred the log.
7. **A.3 camera** — `CameraCfg` needs explicit `spawn=None` to attach to an existing USD camera prim.
8. **A.3 camera prim path** — auto-scene bakes `/World/Scene/D435`, not `/World/Scene/cameras/<name>`; added obs-key→prim map.
9. **A.3 camera xform** — auto-scene wrote `matrix4d xformOp:transform`; isaaclab requires canonical `[translate,orient,scale]`. **Fixed in the auto-scene writer** (`scene_gen.write_usd_stub`), scene USD regenerated.
10. **B.1-P3 replay cameras** — raw `isaacsim.SimulationApp({"enable_cameras":True})` doesn't set the flag isaaclab's Camera check reads → switched to `isaaclab.app.AppLauncher`, + cameras-enabled cfg + `camera_key` thread.

**Pre-existing failures NOT touched today (out of plan scope):** 16 world-model
(dreamerv3 / le_world_model) dry-run + subprocess tests in
`lerobot-isaac-adapters` (stale vs the `_lewm_minimal` backend — consistent with
CLAUDE.md "LeWorldModel BLOCKED"), 1 MimicGen stub message-regex test, and
`test_cached_dataset.py` (numpy absent in `default` env).

---

## Track A — Sim verify (GPU)

### A.1 acceptance — obs regression fix
```bash
cd ~/workspaces/lerobot-isaac-training
pixi run -e sim python -c "
from isaaclab.app import AppLauncher
AppLauncher(headless=True, enable_cameras=True).app
import gymnasium as gym, lerobot_isaac_env
from lerobot_isaac_env.tasks import _register_envs; _register_envs()
env = gym.make('Isaac-SO101-PickPlace-v0', num_envs=1)
obs, _ = env.reset()
print(sorted(obs['policy'].keys()))   # must NOT raise; must include d435_rgb
"
```
PASS = prints a key list including `d435_rgb` (no `'Tensor' has no attribute 'keys'`).

### A.2 acceptance — env smoke CLI (now fully wired)
```bash
pixi run -e sim lerobot-isaac env smoke --steps=100                 # state-only (1,18)
pixi run -e sim lerobot-isaac env smoke --cameras=d435 --steps=100  # dict obs w/ d435_rgb
# dry-run works anywhere (no GPU):
lerobot-isaac env smoke --cameras=d435 --dry-run
```

### A.3 — sim-deploy boot smoke — **USD READY + configs-served (GPU run pending)**
The scene USD was **generated 2026-05-30** by `isaac-auto-scene` from the
existing real D435 calibration (no hardware — `write_usd_stub` is pure Python)
and **checked into the configs leaf** at
`src/lerobot-isaac-configs/src/lerobot_isaac_configs/scenes/so101_workspace.usd`
(tracked, ships in the wheel, resolved via `importlib.resources`). It
references the real `so101_new_calib` asset chain (Geometry + Physics payloads).

Resolve it in code (no hardcoded workspace path):
```python
from lerobot_isaac_configs import get_scene_path
get_scene_path("so101_workspace")   # -> absolute Path, works editable + wheel
```

Regenerate / refresh from a calib.json (output straight into the configs pkg):
```bash
cd ~/workspaces/isaac-auto-scene
pixi run -e default isaac-auto-scene generate \
  --calib ~/.config/isaac-auto-scene/calib.json \
  --out ~/workspaces/lerobot-isaac-training/src/lerobot-isaac-configs/src/lerobot_isaac_configs/scenes/so101_workspace.usd
# calib sources: ~/.config/isaac-auto-scene/{calib.json, calib_bundle.json,
#   manual-calibs/*.json, captures/20260527-180717/calib.json}
# NEW physical setup: re-capture via `isaac-auto-scene calibrate` (D435 + arm) first.
```
GPU boot smoke (now uses the configs-resolved scene by default):
```bash
pixi run -e sim python -c "
from lerobot_isaac_configs import get_scene_path
from lerobot_isaac_deploy.sim._isaac_runtime import IsaacSimRuntime
rt = IsaacSimRuntime(get_scene_path('so101_workspace'))
rt._boot(); print('booted, num_joints=', rt._articulation.num_joints); rt.close()
"
# IsaacSceneSession(usd_path=None, ...) now auto-resolves the same scene.
```
PASS = `_boot()` returns without raising; 100 zero-action steps give finite obs;
no `NotImplementedError` in phase2.1–2.9.

---

## Track B — DR100 + multi-dataset (GPU for B.1 P3/P4)

### B.1 Phase 3 — DR replay GPU run
> `lerobot-isaac dr-replay` is now wired to `replay_runner` (2026-05-30). The
> replay loop + camera capture is GPU-verified for 1 episode. Full DR100 below.
> Camera capture needs `--camera_key d435_rgb` (sets enable_cameras via AppLauncher).
```bash
pixi run -e sim lerobot-isaac dr-replay \
    --source_dataset datasets/kvgork/so101-pickplace1 \
    --output_path    datasets/kvgork/so101-pickplace1-dr100 \
    --n_variants 4 \
    --camera_key d435_rgb \
    --source_tag sim_dr
# verify flags first:  add --dry_run
# REMAINING WIRING: the parquet writer must map the env obs dict
# (policy.d435_rgb / joint_pos) -> LeRobot columns (observation.images.d435_rgb
# PNG + observation.state (6,)). Replay capture verified; writer mapping is the
# open item before a usable dr100 dataset.
```

### B.1 Phase 4 — merge + verify
```bash
pixi run -e sim python -m lerobot_isaac_synthetic.merge \
    --real datasets/kvgork/so101-pickplace1 \
    --sim  datasets/kvgork/so101-pickplace1-dr100 \
    --out  datasets/kvgork/so101-pickplace1-dr100-merged
# Expect ~100 episodes, ~37,400 frames.
```

### B.2 — multi-dataset training (dry-run already green)
```bash
pixi run -e train-policy lerobot-isaac-train \
    --target_arch smolvla \
    --datasets datasets/kvgork/so101-pickplace1,datasets/kvgork/so101-pickplace1-dr100 \
    --dry_run
# prints "multi-dataset (2)" + comma-joined --dataset.repo_id.
# NOTE: two LOCAL roots → merge first (B.1 P4) before a real run; the backend
# refuses multi-local-root real runs and points at merge.
```

---

## Track C — WM-Isaac HP sweep (GPU, long)

Script exists: `scripts/_run_autoresearch_wm_isaac.sh`.
```bash
DRY_RUN=1 bash scripts/_run_autoresearch_wm_isaac.sh   # print trial cmds, no run
bash scripts/_run_autoresearch_wm_isaac.sh             # launch (pick ≤3 trials)
```
**Caveats:**
- **No mid-run collapse killer.** The script's collapse handling is *post-hoc
  forensic* (filters winners on `Grads/actor`), NOT a watchdog that kills a
  collapsing trial at 15k steps. `_collapse_killer.sh` does **not** exist. Watch
  TensorBoard manually; Ctrl-C a trial if `Grads/actor < 0.005 AND rew_avg < -50`.
- Budget ≈ 3h15m/trial × ≤3 trials. Do not launch all 8.
- Recommended 3: `sparse-success-default`, `hybrid`, `hybrid-curriculum`.

Acceptance: ≥1 trial reaches `rew_avg > 0` (success terminations firing);
all chosen trials reach ≥40k steps without actor collapse.

---

## Track D — Hardware closed-loop (SAFETY-CRITICAL — operator only)

**HARD GATE:** Tracks A + B green first. **Never run autonomously.** Operator
hand on E-stop for every motor-write step.

> The plan's old flags (`--max-step-deg`, `--max-relative-target`,
> `--home-on-exit`) are **stale**. The implemented CLI is the confirm-gated
> `lerobot-isaac-deploy session` ladder. Corrected commands below.

> **Dependency (2026-05-30):** the SO-101 serial driver needs `scservo_sdk` from
> lerobot's `feetech` extra. Without it, the dry-run loop fails at robot setup
> with `No module named 'scservo_sdk'`. Fixed: `LEROBOT_EXTRAS` now defaults to
> `smolvla,feetech` in `scripts/install_train_deps.sh`. One-off:
> `pixi run -e train-policy python -m pip install "feetech-servo-sdk>=1.0.0,<2.0.0"`.

### D.1 — Bench dry-run (NO hardware, NO motor writes)
```bash
TRIAL=outputs/autoresearch-lerobot-policy-smolvla/trial_7/checkpoints/045000/pretrained_model
pixi run -e train-policy lerobot-isaac-deploy session \
    --policy-path "$TRIAL" \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --mock-hardware \
    --rate-hz 30 --duration-s 10 \
    --require-real-ckpt
# in-process synthetic-obs loop; proves ckpt loads + is a real (non-synthetic) ckpt.
```

### D.2 — Dry loop on connected arm (NO motor writes)
```bash
# connect + calibrate arm per robot-data-recorder docs first
pixi run -e train-policy lerobot-isaac-deploy session \
    --policy-path "$TRIAL" \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --port /dev/ttyACM0 \
    --camera d435_rgb=/dev/video2,640,480 \
    --rate-hz 30 --duration-s 30 \
    --require-real-ckpt
# (no --execute) → dry loop. Watch actions ∈ [-1,1], current_jp finite.
```

### D.3 — Closed-loop execute (MOTOR WRITES — E-stop ready)
```bash
pixi run -e train-policy lerobot-isaac-deploy session \
    --policy-path "$TRIAL" \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --port /dev/ttyACM0 \
    --camera d435_rgb=/dev/video2,640,480 \
    --rate-hz 30 --duration-s 60 \
    --clamp-tight 1.0 --clamp-loose 2.0 \
    --require-real-ckpt \
    --execute
# home-on-exit is the DEFAULT (pass --no-home-on-exit to disable — don't).
# --n-eval-episodes N to record scored episodes (replaces the old run-eval cmd).
```
**Stop conditions:** any joint near a boundary, repeat-warn fires, or NaN → Ctrl-C
(SIGINT → ramped home + disconnect). Do NOT raise `--clamp-loose` above 2.0.

---

## End-of-day push (4 repos)
```bash
# workspace (main)
cd ~/workspaces/lerobot-isaac-training && git add -A && git commit && git push
# siblings (feature/wm-isaac-env): env, adapters, synthetic
for p in lerobot-isaac-env lerobot-isaac-adapters lerobot-isaac-synthetic; do
  (cd src/$p && git add -A && git commit && git push origin feature/wm-isaac-env)
done
# deploy (feature/sim-deploy) — only if Track D ran
(cd src/lerobot-isaac-deploy && git push origin feature/sim-deploy)
```
> Today's code edits live in the **sibling** repos (`src/<pkg>/`, branch
> `feature/wm-isaac-env`) + the workspace repo (`main`, branch new test file in
> `packages/`). Commit inside each sibling separately.
