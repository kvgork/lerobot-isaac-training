# Sim-Deploy Phase 2 — Isaac Sim Runtime Bodies

**Date:** 2026-05-23
**Parent plan:** [`2026-05-23-sim-deploy-pipeline.md`](2026-05-23-sim-deploy-pipeline.md)
**Status:** SKELETON LANDED — 9 TODO bodies remain.
**Target file:** `src/lerobot-isaac-deploy/src/lerobot_isaac_deploy/sim/_isaac_runtime.py`
**Pixi env:** `.pixi/envs/sim/` (Isaac Sim 6.0 + Isaac Lab v2.3.2)

---

## Prerequisites

1. **USD scene on disk.** Produced separately by
   `~/workspaces/isaac-auto-scene` (the laptop with D435). Drop into
   `assets/sim_scenes/<scene-name>.usd` + sibling
   `assets/sim_scenes/<scene-name>.meta.json`.
2. **Sim env installed.** `pixi install -e sim && pixi run install-isaac-lab`.
3. **Scene preflight green.** `bash scripts/check_sim_scene.sh <usd>` exits 0.
4. **Existing sibling APIs available.**
   - `lerobot_isaac_env.so101_articulation.build_articulation_cfg(usd_path=...)` — returns
     `ArticulationCfg` or `None` when Isaac Lab missing. Reuse — do NOT
     rebuild.
   - `lerobot_isaac_deploy.policy_loader.load_policy(path, dataset_root, device)` —
     dispatches per `detect_policy_kind`. Already handles dreamerv3 +
     lerobot kinds.

---

## Auto-Scene Pitfalls (inherit from `~/workspaces/isaac-auto-scene/CLAUDE.md`)

Carry these constraints verbatim — same Isaac Sim 6.0 build, same gotchas:

- **WARM_UP_FRAMES = 30** mandatory after `sim.reset()` before camera obs
  are valid. Skipping returns black frames. Constant already exported.
- **`SimulationApp.close()` deadlocks** on this build. After rollout
  completes, caller should `os._exit(0)` instead of relying on clean
  shutdown. `IsaacSimRuntime.close()` already logs a warning on the
  `app.close()` exception.
- **`Camera.set_world_poses_from_view`** must be called **after**
  `sim.reset()` (else `_ALL_INDICES` is unset). Hook lives at
  `_post_reset_camera_init()`.
- **`PreviewSurfaceCfg` incompatible** with Isaac Sim 6.0's
  `CreateShaderPrimFromSdrCommand` — `TypeError` on `name` kwarg. Don't
  spawn cuboids with `visual_material`. The auto-scene USD already ships
  geometry with correct materials so this likely doesn't bite us — flag
  if a TODO needs to spawn anything extra.
- **OmegaConf + Hydra interpolation:** if you load a sheeprl cfg here for
  any reason, register the `now:` resolver first via
  `hydra.core.utils.setup_globals()` (mirrored from the wm_loader patch).

---

## TODO Bodies (9 markers)

Each `TODO(phase2.<n>)` in `_isaac_runtime.py` maps to one of these.

### phase2.1 — `_load_usd()`

**Goal:** mount the USD under `/World/Scene`.

**API:** `isaacsim.core.utils.stage.add_reference_to_stage`

```python
from isaacsim.core.utils.stage import add_reference_to_stage
add_reference_to_stage(usd_path=str(self.usd_path), prim_path="/World/Scene")
```

**Constraint:** `check_sim_scene.sh` already verified the USD has
required prims (`/World/SO101`, `/World/object`, `/World/basket`,
`/World/cameras/{overhead,wrist}`). After referencing, those prims live
at `/World/Scene/SO101` etc. UPDATE all hard-coded prim paths in
phase2.2/2.3 to include the `/World/Scene/` prefix.

**Acceptance:** `stage.GetPrimAtPath("/World/Scene/SO101").IsValid()`.

---

### phase2.2 — `_attach_articulation()`

**Goal:** wrap the SO-101 prim in an Isaac Lab `Articulation`.

```python
from isaaclab.assets import Articulation

cfg = build_articulation_cfg(usd_path=self.usd_path)
# Override prim_path because the USD is now under /World/Scene/SO101
# (instead of the default /World/envs/env_0/SO101 the cfg ships).
cfg = cfg.replace(prim_path="/World/Scene/SO101")
self._articulation = Articulation(cfg)
self._scene.add_asset("so101", self._articulation)  # OR direct register
```

**Constraint:** `build_articulation_cfg` returns `None` when Isaac Lab
is absent. Raise a clear error in that branch (already wired).

**Acceptance:** `self._articulation.num_joints == 6`,
`self._articulation.dof_names == ["shoulder_pan", "shoulder_lift",
"elbow_flex", "wrist_flex", "wrist_roll", "gripper"]`.

---

### phase2.3 — `_attach_cameras()`

**Goal:** register Camera sensors for each name in `self.render_cameras`.

```python
from isaaclab.sensors import Camera, CameraCfg

for name in self.render_cameras:
    cam_cfg = CameraCfg(
        prim_path=f"/World/Scene/cameras/{name}",
        width=64, height=64,
        data_types=["rgb"],
        update_period=1.0 / self.rate_hz,
    )
    self._cameras[name] = Camera(cam_cfg)
```

**Constraint:** image_size MUST match what the policy was trained on.
LeRobotDataset `info.json` declares 640×480 raw frames, but the
SmolVLA / DreamerV3 pipelines downsample to 96×96 / 64×64 respectively.
Read the policy's `train_config.json` `input_features.observation.images.*.shape`
to autodetect; for tonight assume 64×64 (matches the bridged HDF5).

**Closes:** the open item in workspace CLAUDE.md "Build Status Checklist
— Camera observation wiring".

**Acceptance:** `self._cameras["overhead_camera_rgb"].data.output["rgb"].shape == (1, 64, 64, 3)`.

---

### phase2.4 — `_post_reset_camera_init()`

**Goal:** any camera setup that needs `_ALL_INDICES` populated.

**Default:** no-op. The auto-scene USD already encodes camera world poses
(verified by `check_sim_scene.sh`). Leave as a documented hook.

**If a camera ever needs runtime repositioning:**
```python
self._cameras[name].set_world_poses_from_view(eyes=..., targets=...)
```

**Acceptance:** function returns without raising; cameras produce
non-zero RGB after WARM_UP_FRAMES.

---

### phase2.5 — `reset_episode(seed)`

**Goal:** arm to home, randomize object xy, zero buffers.

```python
import torch
from omegaconf import OmegaConf  # if using a randomization config

# 1. Reset articulation to home pose.
home_q = torch.zeros(self._articulation.num_joints, device=self.device)
# Read home from build_articulation_cfg.init_state.joint_pos when available.
self._articulation.write_joint_state_to_sim(
    position=home_q.unsqueeze(0),
    velocity=torch.zeros_like(home_q).unsqueeze(0),
)
self._articulation.reset()

# 2. Randomize object position using meta.basket_bounds (XY rectangle).
basket = self._meta.get("basket_bounds")
rng = np.random.default_rng(seed)
obj_xy = [
    rng.uniform(basket["xmin"], basket["xmax"]),
    rng.uniform(basket["ymin"], basket["ymax"]),
]
obj_prim = stage.GetPrimAtPath("/World/Scene/object")
# Set translate attribute via Xformable; preserve original Z.

# 3. Re-warm camera buffers.
for _ in range(WARM_UP_FRAMES):
    self._sim.step(render=True)
```

**Constraint:** `seed` is the only knob; same seed → identical
randomization. Caller passes the episode index.

**Acceptance:** after reset, `get_obs()["observation.state"]` is within
1e-3 of home; `obs["object.pose"][:2]` lies in basket_bounds.

---

### phase2.6 — `get_obs()`

**Goal:** dict matching the LeRobotDataset schema the policy expects.

```python
import torch

# Joint state (positions only — SmolVLA + DreamerV3 ignore velocities).
joint_pos = self._articulation.data.joint_pos[0].cpu().numpy()  # (6,)

# Camera RGB → float32 in [-0.5, 0.5], CHW.
imgs = {}
for name, cam in self._cameras.items():
    rgb = cam.data.output["rgb"][0]  # (H, W, 3) uint8
    rgb = rgb.permute(2, 0, 1).float().div_(255.0).sub_(0.5)
    imgs[f"observation.images.{name}"] = rgb.cpu().numpy()

# Object pose (xyz + xyzw quat) read from USD prim.
obj_prim = stage.GetPrimAtPath("/World/Scene/object")
obj_xform = UsdGeom.Xformable(obj_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
obj_pose = np.concatenate([
    np.array(obj_xform.ExtractTranslation()),
    np.array(obj_xform.ExtractRotation().GetQuat().GetReal()),  # xyzw
    # (Use Gf.Rotation -> Quaternion conversion; placeholder pseudo-code.)
])

return {
    "observation.state": joint_pos.astype(np.float32),
    **imgs,
    "object.pose": obj_pose.astype(np.float32),  # for success criterion
    "basket.bounds": self._meta.get("basket_bounds"),  # for success criterion
}
```

**Constraint:** if `get_obs` is called BEFORE `reset_episode`, callers
see stale or zero buffers. `IsaacSceneSession.run()` orders them
correctly already.

**Acceptance:** dict keys match the policy's expected obs schema (from
`train_config.json`); image shape is `(3, 64, 64)` float32.

---

### phase2.7 — `apply_action(action)`

**Goal:** write joint position targets.

```python
import torch

# action: (6,) float32 from policy.select_action(obs). Already in the
# trained action space (radians, joint positions, no normalization).
action_t = torch.from_numpy(action).to(self.device).unsqueeze(0)

# Optional safety clamp — mirror robot-data-runner's --max-relative-target.
if self._max_relative_target is not None:
    current = self._articulation.data.joint_pos[0]
    delta = action_t - current
    delta_clamped = torch.clamp(delta, -self._max_relative_target, self._max_relative_target)
    action_t = current + delta_clamped

self._articulation.set_joint_position_target(action_t)
self._articulation.write_data_to_sim()
```

**Constraint:** action_dim MUST match articulation.num_joints (6 for SO-101).
Raise ValueError if mismatch.

**Acceptance:** after step(), `self._articulation.data.joint_pos[0]` moves
toward `action` (not necessarily there in one step — PD controller).

---

### phase2.8 — `step()`

**Goal:** advance physics one tick.

```python
self._sim.step(render=True)
```

**Constraint:** `render=True` is mandatory — without it, cameras don't
refresh. There's a faster `render=False` path for headless rollouts that
don't need images, but our policy always needs RGB obs.

**Acceptance:** wall-clock per call ≈ `1 / rate_hz` (i.e. 33 ms @ 30 Hz)
on RTX 3080.

---

### phase2.9 — `get_info()`

**Goal:** per-step termination flags.

```python
# Read contact sensors if registered. For now we trust the success
# criterion to handle "task done" termination; contact_terminal is for
# safety-aborts (arm hits table, joint at limit).

info = {"episode_done": False, "contact_terminal": False}

# If we register a contact sensor on the arm base or table:
#     contact_data = self._sensors["table_contact"].data.net_forces_w
#     info["contact_terminal"] = bool((contact_data.norm(dim=-1) > 50.0).any())

return info
```

**Constraint:** keep cheap — called every step. Heavy reads (full force
buffers) only when a sensor is registered for them.

**Acceptance:** runtime overhead < 1 ms.

---

## Order of Implementation

```
phase2.1 (USD mount)
   │
   ▼
phase2.2 (Articulation)        — depends on USD prim path
   │
   ▼
phase2.3 (Cameras)              — depends on USD prim path
   │
   ▼
phase2.4 (post-reset cam init)  — likely no-op, validate
   │
   ▼
phase2.5 (reset_episode)        — depends on Articulation + cameras
   │
   ▼
phase2.6 (get_obs)              — depends on Articulation + cameras
   │
   ▼
phase2.7 (apply_action)         — depends on Articulation
   │
   ▼
phase2.8 (step)                 — trivial
   │
   ▼
phase2.9 (get_info)             — depends on step
```

Bottom-up testing:
1. After phase2.1–4: smoke `IsaacSimRuntime._boot()`; verify
   `app.update()` runs, no crashes, scene shows expected prim count.
2. After phase2.5–6: run 1 episode of zeros — verify obs shape +
   non-trivial RGB after WARM_UP_FRAMES.
3. After phase2.7–9: run 1 episode driven by a constant action (e.g.
   `np.zeros(6)`) — arm should stay at home, no NaN, no contact
   terminal.
4. Real ckpt rollout — drive `IsaacSceneSession.run()` with the LoRA-best
   ckpt + the auto-scene USD.

---

## Acceptance Tests

```bash
# 1. Boot smoke (no policy needed)
pixi run -e sim python -c "
from lerobot_isaac_deploy.sim._isaac_runtime import IsaacSimRuntime
from pathlib import Path
rt = IsaacSimRuntime(Path('assets/sim_scenes/so101_workspace.usd'))
rt._boot()
print('booted, num_joints=', rt._articulation.num_joints)
rt.close()
"

# 2. Zero-action rollout (no policy)
pixi run -e sim python -c "
import numpy as np
from lerobot_isaac_deploy.sim._isaac_runtime import IsaacSimRuntime
from pathlib import Path
rt = IsaacSimRuntime(Path('assets/sim_scenes/so101_workspace.usd'))
rt._boot()
rt.reset_episode(seed=0)
for _ in range(100):
    rt.apply_action(np.zeros(6, dtype=np.float32))
    rt.step()
obs = rt.get_obs()
print('joint_pos:', obs['observation.state'])
print('rgb shape:', obs['observation.images.overhead_camera_rgb'].shape)
rt.close()
"

# 3. End-to-end rollout
bash scripts/sim_deploy.sh \
    --policy-path outputs/autoresearch-lerobot-policy-smolvla-lora/trial_12/checkpoints/merged/pretrained_model \
    --usd assets/sim_scenes/so101_workspace.usd \
    --n-episodes 3 \
    --output-dir /tmp/sim_deploy_phase2_e2e
cat /tmp/sim_deploy_phase2_e2e/rollout_summary.json
```

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| Articulation prim path mismatch between cfg + USD | Test phase2.2 BEFORE 2.3; print all prim paths on boot |
| Camera obs distribution shift (simple PBR ≠ D435 capture) | Phase 4 DR. For tonight: accept the gap, log it. |
| Gripper friction wrong → object slips | Tune in USD via auto-scene; if persistent, set friction explicitly in articulation cfg |
| sim.step too slow (< 30 Hz) | Drop image_size to 32×32 OR `render=False` between policy queries |
| Memory leak on long sweeps (Isaac Sim known issue) | Cap rollout at 10 episodes, restart subprocess per autoresearch trial |

---

## Estimated Effort

| Phase | Time |
|-------|------|
| phase2.1 USD mount | 0.5 h |
| phase2.2 Articulation | 1 h |
| phase2.3 Cameras (closes CLAUDE.md open item) | 1.5 h |
| phase2.4 post-reset hook | 0.25 h |
| phase2.5 reset_episode | 1.5 h |
| phase2.6 get_obs | 1.5 h |
| phase2.7 apply_action | 0.75 h |
| phase2.8 step | 0.25 h |
| phase2.9 get_info | 0.5 h |
| Smoke acceptance tests (3 above) | 1 h |
| **Total** | **~9 h ≈ 1 working day** |

---

## Exit Criteria

Phase 2 is "done" when ALL hold:

- `bash scripts/check_sim_scene.sh <usd>` exits 0 against the real
  auto-scene USD.
- `IsaacSimRuntime._boot()` returns without error.
- 100 zero-action steps produce non-trivial, non-NaN obs.
- `bash scripts/sim_deploy.sh --policy-path <real-ckpt> --usd <usd>
  --n-episodes 3` writes a `rollout_summary.json` with `pc_success`
  reported (any value, including 0.0).
- No `NotImplementedError` raised anywhere in `_isaac_runtime.py` during
  the above.
- `rate_hz=30` is sustained on RTX 3080 (steady-state wall-clock per
  `step()` ≤ 35 ms).
