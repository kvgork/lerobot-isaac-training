# Next Steps — lerobot-isaac-training

Last updated: 2026-05-21

## Immediate (blocker — system-level)

### 1. Fix NVML driver-library version mismatch

GPU-required smoke tests (cameras / rendering / PhysX-on-GPU) fail with:

```
Could not initialize NVML: return code 18 (NVML_ERROR_LIB_RM_VERSION_MISMATCH:
  RM detects a driver/library version mismatch.)
Failed to create any GPU devices, including an attempt with compatibility mode.
RuntimeError: Expected all tensors to be on the same device, but found at least
  two devices, cuda:0 and cpu!
```

Cause: kernel was upgraded since the nvidia driver libs were installed.
Userspace `libnvidia-*` doesn't match the loaded kernel module.

**Fix (pick one):**

```bash
# Option A (easiest): reboot
sudo reboot

# Option B: hot-reload driver without reboot
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm

# Option C: reinstall the driver to match the running kernel
sudo apt install --reinstall nvidia-driver-<version>
```

Verify with:

```bash
nvidia-smi             # should print GPU table without errors
nvidia-smi --query-gpu=driver_version --format=csv
cat /sys/module/nvidia/version
# driver_version and /sys/module/nvidia/version MUST match
```

## Bundle C.1 — Camera Obs (verification post-driver-fix)

Camera scene wiring is already verified — Kit booted both cameras successfully
(see `05-Wiki/log.md` 2026-05-21 entry). Only blocker is the driver.

```bash
# After driver reload:
cd ~/workspaces/lerobot-isaac-training

# 1. No-cameras smoke (already passes — sanity check)
pixi run -e sim lerobot-isaac env smoke --steps=100
#   Expected: obs.policy shape (1, 18), exits clean in ~7-10s

# 2. Cameras smoke (Bundle C.1 acceptance)
pixi run -e sim lerobot-isaac env smoke --cameras=wrist,overhead --steps=100
#   Expected obs shapes:
#     policy.joint_pos             (1, 6)
#     policy.joint_vel             (1, 6)
#     policy.last_action           (1, 6)
#     policy.wrist_camera_rgb      (1, 3, 128, 128) uint8
#     policy.overhead_camera_rgb   (1, 3, 128, 128) uint8
#   Acceptance: latency p50 <= 30 ms on RTX 3080

# 3. Re-run the full test suite in sim env
pixi run -e sim pytest src/lerobot-isaac-env/tests/test_camera_obs.py \
                       src/lerobot-isaac-synthetic/tests/test_mimicgen_bridge.py \
                       src/lerobot-isaac-adapters/tests/test_loggers.py \
                       packages/lerobot-isaac-meta/tests/test_cli_env_smoke.py
#   Expected: 30 passed (no skips)
```

If latency p50 > 30 ms, options:
- Drop resolution to 96x96 (still LeRobotDataset compatible)
- Move cameras to wrist-only first
- Profile with `nvidia-smi dmon -s u -c 100` during the run

## Bundle sequencing — next phases of plan

The deferred-bundles plan lives at:
- `~/Documents/Vaults/Local/01-Projects/lerobot-isaac-deferred-bundles-plan.md`
- Per-bundle: `lerobot-isaac-bundle-{c,f}-plan.md`,
  `lerobot-isaac-{hil-serl,sim-to-real}-plan.md`

### Phase C.2 — DR Stage Scheduler 1→4 (~2-4 h)

Recommended next bundle. Independent of C.1 acceptance.

Files to touch (per plan):
- `src/lerobot-isaac-env/src/lerobot_isaac_env/randomization.py` (add `DRStageScheduler`)
- `src/lerobot-isaac-env/src/lerobot_isaac_env/curriculum.py` (new)
- `src/lerobot-isaac-configs/configs/dr/stage_{1..4}.yaml` (new — per
  `lerobot-isaac-sim-to-real-plan.md` Phase S.1 has draft budgets)

Acceptance:
- `DRStageScheduler(initial_stage=1, advance_every=10_000_steps)` advances 1→2→3→4
- Alt mode `advance_on_pc_success(threshold=0.7)`
- Unit test: stage transitions + param-set switches

Kickoff: `/orchestrate "Bundle C.2 — DR scheduler"` using bundle-c-plan §C.2 as brief.

### Phase C.3 — Insertion task (~6-10 h)

Higher-difficulty (contact-rich, sub-mm tolerance). Curriculum stage 5.
Blocked on C.1 acceptance (cameras working) + C.2 (DR scheduler).

Kickoff: `/orchestrate "Bundle C.3 — Insertion task"`.

### Phase C.4 — End-to-end smoke (~1-2 h)

Integrates C.1 + C.2 + C.3. Final acceptance for Bundle C.

Command after C.4 lands:
```bash
lerobot-isaac e2e-smoke \
  --task=insertion --cameras=wrist,overhead \
  --dr-stage=1 --policy=smolvla --steps=1000
```

### Bundle F — State-machine CLI

Needs A + B (done) + C (in progress) + E (scaffold done).
Plan: `lerobot-isaac-bundle-f-plan.md`.

### HIL-SERL — research-class

Multi-month direction. Plan: `lerobot-isaac-hil-serl-plan.md`.
Blocked on: real SO-101 hardware operational + Bundle C insertion task.

### Sim-to-real protocol scripts

Reproducible eval + safety gates + ledger.
Plan: `lerobot-isaac-sim-to-real-plan.md`.
Blocked on: real SO-101 hardware + first sim baseline policy.

## Workflow reminders

- **Edit in `src/<sibling>/`** — changes reflect immediately (editable install).
- **Push from inside `src/<sibling>/`** — each is its own git checkout.
- **Fresh clone bootstrap**: `bash scripts/setup.sh -e <env>` (handles src/ clone
  + USD asset mirror + `pixi install`).
- **All standard envs use editable-siblings** (default, sim, train-*, full, ...).
  `frozen` and `frozen-sim` envs pull from GitHub for reproducibility checks.

## Recently landed (2026-05-21)

- ✅ Bundle D — MimicGen bridge un-stub (delegates to `lerobot_mimicgen_bridge` skill)
- ✅ Bundle E — W&B logger scaffold + dashboard template + alerts YAML
- ✅ Bundle C.1 — Camera obs wiring (verified at scene level; full smoke blocked on driver)
- ✅ Pixi workspace canonicalized — src/-editable by default; `frozen` env preserves GitHub source path
- ✅ Setup script with USD asset bootstrap

## Quick commands

```bash
# Setup from fresh clone
bash scripts/setup.sh -e sim                    # default sim env
bash scripts/setup.sh -e full --recorder        # everything + recorder

# Tests
pixi run -e sim pytest src/ packages/           # all in sim env
pixi run -e editable pytest src/ packages/      # no torch / no isaaclab

# Smoke
pixi run -e sim lerobot-isaac env smoke --dry-run                  # safe pre-check
pixi run -e sim lerobot-isaac env smoke --steps=100                # no cameras
pixi run -e sim lerobot-isaac env smoke --cameras=wrist,overhead   # full

# Sync siblings (idempotent — also part of setup.sh)
pixi run sync           # ensure src/ has all 6 siblings
pixi run sync-update    # pull latest in each src/<sibling>/

# Frozen reproducibility check
pixi install -e frozen
pixi run -e frozen test
```
