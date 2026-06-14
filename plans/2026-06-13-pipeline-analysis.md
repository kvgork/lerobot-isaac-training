# Demo-warmstart pipeline — full analysis (2026-06-13)

Goal of this doc: a **manually-checkable** walkthrough of every step, module, and data hop
in the scripted-grasp → sim-demos → DreamerV3 warm-start pipeline. Each section gives the
file, the inputs, the transform, the outputs, and a **VERIFY** pointer (file:line / command)
so you can confirm it against the code yourself. Built during the 2026-06-13 session.

> Status at write time: Stages 1–3 done + committed + GPU-verified. Stage 4 run #1
> (warmstart-v1) plateaued ~−30 (reward-0 poisoning + 4h timeout). Reward-fix applied;
> demos regenerating; warm-start v2 pending.

---

## 0. Pipeline map (information flow)

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │ Isaac Lab env: SO101 ManagerBasedRLEnv (pick_and_place)       │
                    │  robot articulation (so101.usd) + die (CuboidCfg) + d435 cam  │
                    └─────────────────────────────────────────────────────────────┘
                          ▲ action (6)              │ obs(state,rgb)+reward
                          │                         ▼
 STAGE 1   scripts/_scripted_pickplace.py  ── DifferentialIKController (pose/DLS) ──┐
 (grasp)   straight-down GRASP_QUAT, grip sign, ramped close                        │
                          │ drives env through waypoints → SUCCESS                  │
                          ▼                                                          │
 STAGE 2   scripts/_gen_sim_demos.py  ── per-step (state12, rgb 3×64×64, action6,   │
 (demo-gen) reward) → LeRobotDataset + reward sidecar; jitter; success-filter       │
                          │                                                          │
                          ▼  datasets/local/so101-sim-pickplace-demos/              │
 STAGE 3   sheeprl_plugin/demo_buffer.py::load_sim_demos  ── parquet+sidecar →       │
 (seed)    per-episode step-data (rgb,state6,actions,rewards,terminated,…)          │
                          │                                                          │
                          ▼  _wm_isaac_entry.py::_patch_seed_demo_buffer (monkeypatch)│
 STAGE 4   sheeprl dreamer_v3.main() → EnvIndependentReplayBuffer (seeded) →         │
 (train)   train() world-model + actor + critic ───────────────────────────────────┘
                          │ checkpoints
                          ▼  logs/runs/dreamer_v3/isaac_so101/...
 STAGE 4b  scripts/_sim_eval.py → pc_success (closed-loop rollouts)
```

---

## 1. The Isaac Lab environment (the substrate everything rides on)

**Package:** `lerobot-isaac-env` (sibling at `src/lerobot-isaac-env/`).
**Entry:** `make_env(task="pick_and_place", num_envs=1, headless=, enable_cameras=)` →
gym-wrapped `ManagerBasedRLEnv`.

### 1.1 Robot articulation
- **File:** `src/lerobot_isaac_env/so101_articulation.py::build_articulation_cfg`
- USD: `assets/usd/so101.usd` → references `so101_new_calib/so101_new_calib.usda`
  → `payloads/{robot,instances,Physics/*}.usda`. **VERIFY:** `grep -r "references" assets/usd/so101.usd`.
- `fix_root_link=True` (env `LEROBOT_ISAAC_FIX_BASE=1`) — anchors base; enables num_envs>1
  + simplifies IK jacobian. **VERIFY:** so101_articulation.py:192–197.
- Actuators: one group `.*`, stiffness 80, damping 4, effort_limit 10, velocity_limit 10.
  **VERIFY:** so101_articulation.py:217–223.
- 6 joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
- **Gripper joint limits:** [−0.1745, +1.745] rad. CLOSE action −1 → joint −0.1745 (hard
  stop); OPEN +1 → +0.5. **VERIFY:** `_gripper_physics_probe.py`.

### 1.2 Collision geometry (the Stage-1 blocker, now fixed)
- **File:** `assets/usd/so101_new_calib/payloads/instances.usda`
- Finger colliders `moving_jaw_so101_v1` + `wrist_roll_follower_so101_v1`:
  `physics:approximation = "convexDecomposition"` (was `convexHull` — filled the jaw
  V-mouth with a solid wedge that shoved the die out). **VERIFY:** instances.usda:447–490;
  backup `instances.usda.bak.20260613`.

### 1.3 The die (source_object)
- **File:** `src/lerobot_isaac_env/tasks/pick_and_place.py:248–270`
- Primitive `sim_utils.CuboidCfg` (true PhysX box collider — solid). Edge =
  `_OBJECT_SCALE*0.06` (0.267→16 mm). mass 0.01, friction 1.0. Was DexCube USD (hollow
  mesh collider; finger passed through). NO `visual_material` (PreviewSurfaceCfg crashes
  this Isaac build: `CreateShaderPrimFromSdrCommand 'name'`). **VERIFY:** that block.
- Spawn pose env-driven: `LEROBOT_ISAAC_OBJECT_X/Y/Z` (default 0.22/0.05/0.05). Target bin
  `LEROBOT_ISAAC_TARGET_X/Y/Z` (0.22/−0.13/0.01). pick_and_place.py:45–57.
- **Rest z = 0.008** (half of 16 mm) — proves the collider is correctly sized. Spawns at
  z≈0.048 and SETTLES to 0.008 (important: read pose AFTER settling).

### 1.4 Observations (what the policy/WM sees)
- **File:** `src/lerobot_isaac_env/observations.py` + `so101_env_cfg.py` PolicyObsGroup.
- Terms: `joint_pos` (6), `joint_vel` (6), `last_action` (6), `d435_rgb` (3,480,640 uint8).
- **d435 camera:** `env.scene['d435_camera'].data.output['rgb']` → (1,480,640,3) uint8,
  permuted to (1,3,480,640). **VERIFY:** observations.py `def d435_rgb`.
- **Reachability (key constraint):** pointing STRAIGHT DOWN, the arm reaches forward only
  to ee_x≈0.218; tilted it reaches ~0.29. So the graspable straight-down die zone is
  r≲0.20 — demos use OBJECT_X=0.18. **VERIFY:** `_grasp_joint_diag.py`.

### 1.5 Reward (staged shaping)
- **File:** `src/lerobot_isaac_env/rewards.py`; wired in pick_and_place.py.
- 6 active terms (RewardManager). Includes `progress` (EE→object distance), closure,
  lift_shaping (grip×ee_height), place, place_success. Env-gated by
  `LEROBOT_ISAAC_STAGED_REWARD=1`, weights via env vars. **VERIFY:** RewardManager line in
  any train.log: "Reward Manager: contains 6 active terms".
- The old RL plateau: reward −10.6 (grip+lift, no carry) — exploration limit.

---

## 2. STAGE 1 — scripted grasp controller

**File:** `scripts/_scripted_pickplace.py`. **Run:** `--gui --interactive` for hands-on,
or one-shot. Drives the env open-loop through Cartesian waypoints via IK.

### 2.1 IK control
- `DifferentialIKController` (`command_type="pose"`, `ik_method="dls"`) — tracks
  gripper_link position + orientation. Jacobian indexing handles fixed-base
  (row=ee_idx−1, cols=arm_ids, no +6 offset). **VERIFY:** _scripted_pickplace.py `step_to`.
- Per step: read ee pose (base frame) → ik.compute → q_des_arm → **action =
  (q_des − q_default)/0.5** (env JointPositionAction scale=0.5, use_default_offset=True,
  clip=None → reproduces q_des exactly). **VERIFY:** so101_env_cfg.py:259–264.

### 2.2 Grasp parameters (eyeball-derived)
- `GRASP_QUAT = [1,0,0,0]` (identity) = straight down: gripper_link local −z is the
  finger axis, so identity points fingers at world −Z. **VERIFY:** `_grasp_joint_diag.py`
  (down_dot=1.0, no joint at limit at die 0.18).
- **Grip sign:** `GRIP_OPEN=+1` (joint→+0.5), `GRIP_CLOSE=−1` (joint→−0.1745). (Old
  docstring had it reversed.)
- `grasp_z≈0.106` = gripper_link height (fingertips ~0.10 below it, at the die).
- Slow **ramped close** (`grip` ramps OPEN→CLOSE over `close_steps`) — pinch, not bat.

### 2.3 Waypoint sequence (do_grasp + do_place)
1. above_obj (z=0.17, open) → 2. descend (grasp_z, open) → 2b. dwell (open) →
3. ramped close → 3b. seat → 4. lift → 5. carry to bin → 6. lower → 7. release.
- SUCCESS = object xy within 6 cm of target. **VERIFY:** do_grasp/do_place in the script.
- Interactive cmds (stdin, no reboot): r/z/x/y/o/f/p/q. Must run in YOUR terminal (not
  via Claude `!` — stdin must be your keyboard).

---

## 3. STAGE 2 — demo generation

**File:** `scripts/_gen_sim_demos.py`. **Run:**
```
LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_STAGED_REWARD=1 \
  .pixi/envs/sim/bin/python scripts/_gen_sim_demos.py --episodes 40 \
    --out datasets/local/so101-sim-pickplace-demos
```

### 3.1 Per-episode loop (`rollout`)
1. `env.reset()` → teleport die to jittered (ox,oy) via `obj.write_root_state_to_sim`
   (spawn xy is fixed at build; teleport = per-episode jitter with NO reboot).
2. Settle 30 steps (not recorded; lets die fall to z=0.008).
3. Run the do_grasp+do_place sequence (same IK as Stage 1), enable_cameras=True.
4. SUCCESS check (obj xy near bin). Only SUCCESS episodes saved (~48% under ±0.03 jitter).

### 3.2 Per-step recording (`grab_frame` + reward capture)
- `observation.state` (12,) float32 = joint_pos(6) ⊕ joint_vel(6).
- `observation.images.d435_rgb` (3,64,64) uint8 = d435 (3,480,640) bilinear-resized to 64².
- `action` (6,) float32 = the normalized action sent to env.step.
- `task` str (per-frame, lerobot 0.5.1 requirement).
- **reward** (per step) captured from `env.step(...)[1]` → **sidecar** (NOT a LeRobot
  feature — lerobot 0.5.1 crashes on a (1,) reward feature). **VERIFY:** _gen_sim_demos.py
  step_to (`out = env.step`) + the SAVED line printing reward sum.

### 3.3 Output dataset
- **Path:** `datasets/local/so101-sim-pickplace-demos/` (gitignored).
- `data/chunk-000/file-000.parquet` — all frames (images are PNG bytes in-parquet).
- `meta/{info.json,stats.json,tasks.parquet,episodes/...}` — `finalize()` flushes episode
  metadata (without it, episodes.parquet is missing). **VERIFY:** info.json total_episodes.
- `meta/demo_rewards/ep_NNNN.npy` — per-episode reward arrays (the sidecar).
- lerobot 0.5.1 API: `from lerobot.datasets.lerobot_dataset import LeRobotDataset` (NOT
  `lerobot.common...`); `add_frame(frame)` with `task` in the dict; `finalize()`.

---

## 4. STAGE 3 — replay-buffer seeding

### 4.1 Loader: parquet+sidecar → sheeprl step-data
- **File:** `src/lerobot_isaac_adapters/sheeprl_plugin/demo_buffer.py::load_sim_demos`
- Per episode → dict: `rgb`(T,3,64,64)uint8, `state`(T,12)f32, `actions`(T,6)f32,
  `rewards`(T,1)f32, `terminated`(T,1)bool, `truncated`(T,1)bool, `is_first`(T,1)bool.
- Episode grouping via `hf_dataset["episode_index"]` (lerobot 0.5.1 dropped
  `episode_data_index`). **VERIFY:** `_episode_frames`.
- `terminated[-1]=True` (success = true terminal, not time-out → WM won't bootstrap past).
- **Reward injection:** reads `meta/demo_rewards/ep_NNNN.npy` if present (real reward),
  else zeros. **VERIFY:** the rew_dir block in load_sim_demos.

### 4.2 Seeding monkeypatch
- **File:** `scripts/_wm_isaac_entry.py::_patch_seed_demo_buffer` (called in main() before
  `from sheeprl.cli import run`). Env-gated by `LEROBOT_ISAAC_DEMO_DATASET`.
- Wraps `EnvIndependentReplayBuffer.add` → **lazy-seeds on first call** (so it learns the
  exact online obs schema from the live step_data: keys, state_dim, image size).
- **Schema adaptation:** demo `state` 12 → sliced to the env's 6 (joint_pos) to match.
  rgb 64² matches. **VERIFY:** the `_seed` fn; train.log line "SEEDED 38 demo episodes
  (17290 transitions)".
- Each demo episode added as `(T,1,...)` to env-0's sub-buffer via `rb.add(data,
  indices=[0])`. **VERIFY:** buffers.py:627 `EnvIndependentReplayBuffer.add` (indices →
  per-env sub-buffer; data shape [seq, n_envs, ...]).

### 4.3 sheeprl replay-buffer mechanics (for cross-checking)
- **File:** `.pixi/envs/sim/.../sheeprl/algos/dreamer_v3/dreamer_v3.py`
- rb created `main():479` `EnvIndependentReplayBuffer(buffer_size, buffer_cls=
  SequentialReplayBuffer)`. Online add `:587`. Sample `:664` `rb.sample_tensors(...)` →
  `train(...)` `:682`. Actor `policy_loss` `:297`.
- Seeding puts demo transitions in rb so the WM learns carry→place DYNAMICS + the actor's
  imagination starts include demo states.

---

## 5. STAGE 4 — DreamerV3 training (sheeprl on Isaac)

**Launcher:** `scripts/_run_wm_isaac_overnight.sh` → wraps `lerobot_isaac_adapters.train
--target_arch dreamerv3` → `targets/wm_dreamerv3.py` → `scripts/_wm_isaac_entry.py` (boots
Isaac SimulationApp FIRST, applies gym+seed patches, then sheeprl `run()`).

### 5.1 Entry patches (order matters)
- `_patch_gym_transform_observation`, `_patch_gym_vector_final_info`,
  `_patch_gym_vector_isaac` (num_envs>1 → one IsaacSO101VectorEnv), **`_patch_seed_demo_buffer`**.
- `os._exit(code)` at the end — skips Isaac's hanging atexit `SimulationApp.close()`
  (the "WM-Isaac stall"). **VERIFY:** _wm_isaac_entry.py main().

### 5.2 sheeprl env wrapper
- **File:** `sheeprl_plugin/isaac_env.py::IsaacSO101Env`. obs = `{"rgb":(3,H,W),
  "state":(6,)}` (joint_pos; expands to 13 with object_pose). image_size=64 →
  rgb resized internally (`_resize_chw`). action Box(6). **VERIFY:** isaac_env.py:137–151,
  `_translate_obs` :439.
- **MUST run num_envs=1** (the wrapper collapses env spacing; >1 broke earlier — memory
  `wm-isaac-num-envs-bug`).

### 5.3 Key config (launcher env vars)
| var | default | note |
|-----|---------|------|
| STEPS | 50000 | total_steps |
| BATCH_SIZE | 16 | drop to 8 on OOM (9.6 GB at 16) |
| NUM_ENVS | 1 | MUST be 1 |
| IMAGE_SIZE | 64 | must match demo 64² |
| REPLAY_RATIO | 2 | WM grad steps/env step |
| **LEROBOT_TRAIN_TIMEOUT** | **4h (14400s)** | **train_wrapper hard kill — separate from SECONDS_PER_EXP; raise to 40000 for 50k** |
| LEROBOT_ISAAC_DEMO_DATASET | (unset) | set → seeding ON |
| CHECKPOINT_EVERY | 10000 | ckpt cadence |

### 5.4 Metric
- `train_wrapper` greps `recon_loss=` from stdout; sheeprl logs `Loss/observation_loss`
  → currently emits sentinel `recon_loss=0.0` (cosmetic for autoresearch; the warm-start
  is judged by reward/pc_success, not this). **VERIFY:** train.log tail.

---

## 6. STAGE 4b — evaluation
- **File:** `scripts/_sim_eval.py` → closed-loop rollouts → pc_success. sim env needs
  transformers==5.3.0 + num2words (memory `sim-policy-eval`). Loads a checkpoint from
  `logs/runs/dreamer_v3/isaac_so101/<run>/version_0/checkpoint/`.

---

## 7. End-to-end data-schema table (how one observation morphs)

| Hop | state | image | action | reward |
|-----|-------|-------|--------|--------|
| Isaac env obs | joint_pos 6 (+vel/last_action terms) | d435 (3,480,640) u8 | — | staged (6 terms) |
| Stage-2 recorded frame | (12,) pos⊕vel f32 | (3,64,64) u8 | (6,) f32 | sidecar .npy |
| LeRobotDataset row | observation.state (12,) | observation.images.d435_rgb (3,64,64) | action (6,) | (in sidecar) |
| load_sim_demos | state (T,12) | rgb (T,3,64,64) | actions (T,6) | rewards (T,1) ← sidecar |
| seeded into rb | sliced → (T,1,6) | (T,1,3,64,64) | (T,1,6) | (T,1,1) |
| sheeprl online env | state (6,) | rgb (3,64,64) | Box(6) | staged |

**Critical match:** seeded demo state is sliced 12→6 to equal the online env's 6-dim
joint_pos. If the env runs with object_pose (state=13) the slice target changes — the
monkeypatch reads it from the live step_data, so it auto-adapts.

---

## 8. Known issues / gotchas to check

1. **warmstart-v1 failure (run #1):** reward-0 demos poisoned the reward model (plateau
   ~−30) + 4h timeout killed it at step 17.5k with NO checkpoint. FIXED: reward sidecar +
   `LEROBOT_TRAIN_TIMEOUT=40000`.
1b. **warmstart-v2 (run #2): STALLED at −27, flat 45k steps.** Full 50k, rc=0, ckpt_50000
   saved (`logs/runs/dreamer_v3/isaac_so101/2026-06-13_19-59-52_.../version_0/checkpoint/`).
   **ROOT CONFOUND found post-run:** both warm-start runs were UNDER-SHAPED —
   `LEROBOT_ISAAC_CLOSURE_WEIGHT` and `LEROBOT_ISAAC_LIFT_SHAPING_WEIGHT` default **0.0**,
   and these are the terms that drove the OLD plateau −55→−10.6. The runs set only
   `STAGED_REWARD=1` → closure OFF, lift_shaping OFF, place_std 0.05 (old used 0.15). So
   −27 ≠ a fair seeding test. **Old −10.6 config:** `CLOSURE_WEIGHT=4 LIFT_SHAPING_WEIGHT=14
   PLACE_STD=0.15`. **AND** the demo reward sidecars were recorded under the weak config →
   a correct v3 must REGEN demos with the tuned weights so seeded rewards match online.
   DECISION PENDING (user reviewing): regen+v3 with tuned weights (± no-seed control) vs
   BC-loss. `_sim_eval.py` is for lerobot policies, NOT sheeprl ckpts (use dreamer_v3
   evaluate.py to eval the warm-started agent).
2. **Reward-0 vs real-reward:** seeding demos into a reward-based buffer needs CONSISTENT
   rewards or the reward model learns demo-states→0, conflicting with online shaping.
3. **Camera-bound speed:** ~1.2 steps/s with d435 render → 50k ≈ 11h. Budget the timeout.
4. **num_envs=1 only** (articulation instability at >1 even with the vector wrapper —
   physics explodes; memory `wm-isaac-num-envs-bug`).
5. **No PreviewSurfaceCfg** on the die (shader-create crash this Isaac build).
6. **Plain box die** (no DexCube texture) — fine for sim demos; revisit if the WM needs
   realistic texture.
7. **If seeding still plateaus:** escalate to the BC-loss path —
   `demo_buffer.behavior_cloning_loss` + `bc_weight` (DreamerFD decay + virtual clutch)
   are built but NOT yet wired into the dreamer_v3 train loop (that's the next monkeypatch).

---

## 9. File index (for manual review)

| File | Role |
|------|------|
| `scripts/_scripted_pickplace.py` | Stage 1 grasp controller (IK + interactive) |
| `scripts/_gen_sim_demos.py` | Stage 2 demo recorder (+reward sidecar) |
| `scripts/_wm_isaac_entry.py` | Stage 4 entry; gym patches + `_patch_seed_demo_buffer` |
| `scripts/_run_wm_isaac_overnight.sh` | Stage 4 launcher (env-var config + watchdog) |
| `scripts/_sim_eval.py` | Stage 4b pc_success eval |
| `scripts/_grasp_*_probe.py`, `_grasp_joint_diag.py`, `_gripper_physics_probe.py`, `_jaw_width_probe.py` | grasp diagnostics |
| `src/lerobot-isaac-env/.../tasks/pick_and_place.py` | env task: die spawn, reward wiring, poses |
| `src/lerobot-isaac-env/.../so101_articulation.py` | robot cfg, joint limits, fix_root_link |
| `src/lerobot-isaac-env/.../observations.py` | obs terms incl d435_rgb |
| `src/lerobot-isaac-env/.../rewards.py` | staged reward terms |
| `src/lerobot-isaac-env/assets/usd/so101_new_calib/payloads/instances.usda` | finger collider approximation |
| `src/lerobot-isaac-adapters/.../sheeprl_plugin/demo_buffer.py` | loader + DemoBuffer + BC loss |
| `src/lerobot-isaac-adapters/.../sheeprl_plugin/isaac_env.py` | sheeprl↔Isaac env wrapper |
| `src/lerobot-isaac-adapters/.../targets/wm_dreamerv3.py` | dreamerv3 dispatch |
| `.pixi/envs/sim/.../sheeprl/algos/dreamer_v3/dreamer_v3.py` | sheeprl train loop (read-only) |

## 10. Commands quick-reference
```bash
# Stage 1 grasp (your terminal, interactive)
LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_OBJECT_X=0.18 \
  LEROBOT_ISAAC_OBJECT_Y=0.05 LEROBOT_ISAAC_STAGED_REWARD=1 \
  .pixi/envs/sim/bin/python scripts/_scripted_pickplace.py --gui --interactive --grasp_z 0.106

# Stage 2 demo-gen
LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_STAGED_REWARD=1 \
  .pixi/envs/sim/bin/python scripts/_gen_sim_demos.py --episodes 40 \
    --out datasets/local/so101-sim-pickplace-demos

# Stage 3+4 warm-start (seeding ON, timeout fixed)
STEPS=50000 BATCH_SIZE=16 SESSION_ID=warmstart-v2 LEROBOT_TRAIN_TIMEOUT=40000 \
  LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_OBJECT_X=0.18 \
  LEROBOT_ISAAC_OBJECT_Y=0.05 LEROBOT_ISAAC_STAGED_REWARD=1 \
  LEROBOT_ISAAC_DEMO_DATASET=datasets/local/so101-sim-pickplace-demos \
  bash scripts/_run_wm_isaac_overnight.sh

# Verify (find a ckpt first)
find logs/runs/dreamer_v3/isaac_so101 -name "*.ckpt" | tail
.pixi/envs/sim/bin/python scripts/_sim_eval.py --checkpoint <ckpt>
```

## Related
- memory: `demo-warmstart-pipeline`, `scripted-grasp-infeasible`, `so101-sim-reach-envelope`,
  `wm-isaac-num-envs-bug`, `sim-policy-eval`, `wm-isaac-stall-resolved`
- plans: `2026-06-11-demo-warmstart-plan.md`, `2026-06-13-scripted-grasp-manual-investigation.md`
- Commits (2026-06-13): env-collider `6569587`; grasp+probes `53f6113`; demo-gen `ec2402b`;
  demo_buffer lerobot-0.5.1 fix; seed patch; reward-sidecar fix.
```
