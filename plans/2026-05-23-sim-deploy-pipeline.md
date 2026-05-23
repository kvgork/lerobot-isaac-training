# Sim-Deploy Pipeline — Plan

**Date:** 2026-05-23
**Parent context:** `plans/2026-05-22-lora-sweep-next-steps.md` Phase 1
(closed-loop sim eval) + `plans/2026-05-22-wm-autoresearch-plan.md`.
**Trigger:** deploy package now lives in-tree under
`src/lerobot-isaac-deploy/` (editable sibling). Closed-loop sim eval is
finally feasible: we have a working deploy session, a working
DreamerV3 actor loader, and a USD-scene generator
(`~/workspaces/isaac-auto-scene/`).

---

## Current State (verified 2026-05-23)

| Asset | Path | Status |
|-------|------|--------|
| Deploy pkg | `src/lerobot-isaac-deploy/` | editable sibling — module resolves from in-tree path |
| Deploy entry | `li-deploy-session`, `lerobot-isaac-deploy` | console scripts |
| Real-arm path | `--execute` + `robot-data-runner` | hardware required |
| Mock path | `--mock-hardware` | zero-obs synthetic, no env feedback |
| WM loader | `wm_loader.load_dreamerv3()` | working (sheeprl 0.5.8 API) |
| WM rollout | `wm_rollout.rollout()` | offline (HDF5 frames, no env step) |
| Scene generator | `~/workspaces/isaac-auto-scene/` | live pixi workspace, borrows Isaac Sim from training env |
| Isaac Sim | `.pixi/envs/sim/` | installed via `pixi run install-isaac-lab` |
| SO-101 USD | `assets/usd/` (downloaded by training workspace) | available |

**Gap:** no closed-loop sim deploy. The deploy session can drive a real arm
OR a mock (zero obs). Neither closes the loop with a physics simulator that
would give us a real `pc_success` without hardware.

---

## Goal

A single command runs the trained policy against an Isaac Sim scene that
mirrors the real workspace (`isaac-auto-scene` USD), produces rendered RGB
observations, sends predicted joint targets back into the simulator, and
emits a rollout JSON with:

- `pc_success` (task completion rate over N episodes)
- `mean_ep_len`
- `intervention_rate` (always 0 in sim — no human teleop overrides)
- per-step action chunks, joint trajectories, contact events

No hardware. No motor writes. Closed-loop signal we can ratchet against in
the autoresearcher.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  scripts/sim_deploy.sh        ← entry script                     │
│      ↓                                                           │
│  li-deploy-session --sim-backend isaac-auto-scene                │
│      ↓                                                           │
│  lerobot_isaac_deploy.sim.IsaacSceneSession  (NEW)               │
│      ├── load USD via isaac-auto-scene                           │
│      ├── add SO-101 articulation (urdf → ArticulationCfg)        │
│      ├── attach cameras (wrist + overhead, 64×64 RGB)            │
│      ├── tick (30 Hz):                                           │
│      │     obs = render() + joint_state()                        │
│      │     act = policy.select_action(obs)                       │
│      │     sim.apply_action(act)                                 │
│      │     sim.step()                                            │
│      └── on episode end:                                         │
│            check success criterion (object in basket?)           │
│            log rollout summary                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Phase 0 — Foundation Already Landed (this session)

✅ `lerobot-isaac-deploy` moved into `src/` as editable sibling.
✅ `pixi.toml` updated: `feature.editable-siblings` adds path-dep entry +
   `feature.git-siblings` adds GitHub URL entry.
✅ `scripts/sync/sync_siblings.sh` updated to include deploy in the 7-sibling
   list.
✅ `train-dreamer` env now resolves `lerobot_isaac_deploy` from
   `src/lerobot-isaac-deploy/src/lerobot_isaac_deploy/__init__.py`.

Verify after rerun on a fresh host:
```bash
bash scripts/setup.sh -e train-dreamer
.pixi/envs/train-dreamer/bin/python -c \
  "import lerobot_isaac_deploy; print(lerobot_isaac_deploy.__file__)"
# Expect: …/src/lerobot-isaac-deploy/src/lerobot_isaac_deploy/__init__.py
```

---

## Phase 1 — Consume Pre-Saved USD Scenes (0.5 day)

**Goal:** load a USD scene that the user already produced via
`isaac-auto-scene` (separately, on the laptop with the D435). Sim-deploy
NEVER invokes `isaac-auto-scene` directly — decoupled by design so the
scene-capture pipeline and the policy-eval pipeline have independent
lifecycles.

**Convention:**
- Canonical drop location: `assets/sim_scenes/<scene-name>.usd` in the
  training workspace.
- Each USD ships with a sibling `assets/sim_scenes/<scene-name>.meta.json`
  describing: capture timestamp, source workspace (laptop hostname), arm
  pose, camera intrinsics, basket bounds, object IDs.
- USD + meta are the ONLY interface contract. Sim-deploy reads both.

**Tasks:**
1. Add `scripts/check_sim_scene.sh` — preflight that validates a USD +
   meta pair before sim-deploy boots Isaac Sim. Checks:
   - USD file exists, parseable by `usdrt` / `pxr.Usd.Stage.Open`.
   - Required prims present (`/World/SO101`, `/World/object`,
     `/World/basket`, `/World/cameras/overhead`, `/World/cameras/wrist`).
   - meta.json schema fields all present.
2. Document expected USD layout in `docs/sim-scenes.md` so the user knows
   what `isaac-auto-scene` needs to emit (cross-reference, not a hard
   requirement enforced from this workspace).
3. `IsaacSceneSession.__post_init__` raises a clear error pointing at
   `check_sim_scene.sh` when the USD or meta is missing.

**Out of scope for this workspace:**
- USD capture / generation. That's `isaac-auto-scene`'s job — runs
  wherever the camera is plugged in.
- USD authoring. We treat the file as opaque + read-only.

**Acceptance:**
- `bash scripts/check_sim_scene.sh assets/sim_scenes/so101_workspace.usd`
  exits 0 when a valid USD+meta pair is present, exits 2 with a clear
  "missing prim" / "missing meta" message otherwise.
- `IsaacSceneSession` can pre-flight-validate before booting Isaac Sim.

**Risks:**
- USD schema drift between auto-scene versions: pin a `usd_schema_version`
  field in meta.json so sim-deploy can refuse incompatible scenes early.

---

## Phase 2 — IsaacSceneSession Class (skeleton landed; ~1 day to finish bodies)

**Status (2026-05-23):** scaffold + soft-imports + class shape committed.
`_isaac_runtime.py` has 9 explicit `TODO(phase2.<n>)` markers for the
methods that still need bodies. Each TODO points at the relevant Isaac
Lab API and the auto-scene pitfall that applies.

**Deliverable:** `src/lerobot-isaac-deploy/src/lerobot_isaac_deploy/sim/isaac_scene_session.py`

```python
class IsaacSceneSession:
    def __init__(
        self,
        usd_path: Path,
        policy_path: Path,
        dataset_root: Path,           # for obs schema reference
        n_episodes: int = 10,
        max_steps: int = 600,         # 20 s @ 30 Hz
        render_cameras: list[str] = ("overhead_camera_rgb", "wrist_camera_rgb"),
        rate_hz: float = 30.0,
        device: str = "cuda",
        success_criterion: Callable[[dict], bool] = ...,  # default: object Z-height > basket_z
    ):
        ...

    def run(self) -> Path:
        """Returns path to rollout_summary.json.

        Schema:
            {
                "pc_success": float,
                "mean_ep_len": float,
                "n_episodes": int,
                "per_episode": [{"len": int, "success": bool, "fail_reason": str}, ...],
                "policy_path": str,
                "usd_path": str,
                "wall_clock_s": float,
            }
        """
```

**Internal architecture:**
1. `_build_sim()` — spawn `SimulationApp` + `SimulationContext` + load USD.
2. `_add_so101_articulation()` — attach SO-101 URDF as ArticulationCfg,
   reuse the cfg from `lerobot-isaac-env/.../so101_cfg.py`.
3. `_add_cameras()` — wrist + overhead `CameraCfg` (the open item from
   `CLAUDE.md §Build Status Checklist` — finally close it here).
4. `_step(action)` — apply action via
   `articulation.set_joint_position_target(action)`; `sim.step()`;
   render cameras; return obs dict matching the LeRobotDataset schema
   (`observation.state` + `observation.images.<name>`).
5. `_success_criterion(obs, info)` — default: gripper holding object AND
   object position within basket bounds. Pluggable.
6. `_log_episode()` — append to `rollout_summary.json`.

**Tasks:**
1. Wire `SimulationApp(headless=True, enable_cameras=True)` plus the
   30-frame warm-up dance documented in `isaac-auto-scene/CLAUDE.md`.
2. Camera obs format must MATCH the dataset schema the policy was trained
   on (float32 in [-0.5, 0.5] per channel, NCHW, no leading T dim — same
   as `_open_loop_eval.py` does for hold-out).
3. Action clamping inside sim — mirror the deploy session's
   `--max-relative-target` safety. Sim has no consequences for clamp
   violations but still useful for parity with hardware traces.
4. Episode reset — for each new episode, randomise object position in
   the basket bounds, reset arm to home pose, zero camera buffers.

**Acceptance:**
- Smoke run: 1 episode × 100 steps with the synthetic-marker DreamerV3
  fixture from deploy pkg's test fixtures — exits 0, rollout JSON has
  `n_episodes=1`, `mean_ep_len≈100`, `pc_success∈{0.0, 1.0}` (single ep).
- Real run: 10 episodes with the LoRA-best SmolVLA ckpt
  (`outputs/autoresearch-lerobot-policy-smolvla-lora/trial_12/checkpoints/merged/pretrained_model`)
  — exits 0, produces `pc_success` value.

**Risks:**
- Sim-to-real gap: rendered camera obs distribution will NOT match real
  D435 captures (synthetic textures, simple PBR materials). Phase 4 wires
  domain randomisation to bridge.
- Isaac Sim startup is slow (~30 s cold). Cache the SimulationApp
  instance across episodes; only reset state.
- USD physics tuning needed for gripper friction + object inertia — first
  rollouts will see the gripper slip. Reuse params from
  `lerobot-isaac-env` synthetic data pipeline.

---

## Phase 3 — Entry Script + CLI Plumbing (0.5 day)

**Deliverable:** `scripts/sim_deploy.sh` + new subcommand on
`li-deploy-session`.

```bash
# Smoke (synthetic marker, no Isaac Sim required)
bash scripts/sim_deploy.sh \
    --policy-path src/lerobot-isaac-deploy/tests/fixtures/dreamerv3_synthetic/ \
    --n-episodes 1

# Real DreamerV3 sim closed-loop
bash scripts/sim_deploy.sh \
    --policy-path outputs/autoresearch-wm-dreamerv3/trial_11/staged_ckpt/ \
    --usd assets/sim_scenes/so101_workspace.usd \
    --n-episodes 10

# Real LoRA SmolVLA
bash scripts/sim_deploy.sh \
    --policy-path outputs/autoresearch-lerobot-policy-smolvla-lora/trial_12/checkpoints/merged/pretrained_model \
    --usd assets/sim_scenes/so101_workspace.usd \
    --n-episodes 10 \
    --success-criterion pickplace_basket
```

**CLI flags:**

| Flag | Default | Use |
|------|---------|-----|
| `--policy-path` | required | dir for any supported kind (lerobot/dreamerv3/lewm) |
| `--usd` | `assets/sim_scenes/so101_workspace.usd` | scene to load |
| `--n-episodes` | 10 | rollout count |
| `--max-steps` | 600 | per-episode horizon |
| `--rate-hz` | 30 | control rate |
| `--success-criterion` | `pickplace_basket` | name resolved against a registry |
| `--output-dir` | `outputs/sim_deploy/<ts>/` | rollout JSON destination |
| `--dr-randomize` | off | enable domain randomisation (Phase 4) |

**Acceptance:**
- Dry-run (`--dry-run`) prints the resolved sim args without booting Isaac
  Sim.
- Real run writes `rollout_summary.json` + dashboard-discoverable state to
  `.agent-state/<session>/sim_deploy/<slug>/`.

---

## Phase 4 — Domain Randomisation (1 day, optional)

**Goal:** reduce sim-to-real gap; bring sim rollouts closer to real-robot
performance distribution.

Reuse the existing `lerobot-isaac-synthetic` DR pipeline (already wired
per CLAUDE.md). Each episode reset randomises:
- Lighting (HDR env-map exposure + sun direction)
- Camera position (±3 cm jitter from calibrated pose)
- Object texture + color
- Object friction + mass
- Arm joint friction

Wire the DR config under `assets/sim_scenes/dr_configs/<name>.yaml`.

**Acceptance:**
- DR-on rollouts produce a 10-30 % wider success-rate distribution than
  DR-off. Hardware deploy should track DR-on rollouts within ±0.15
  absolute pc_success.

---

## Phase 5 — Integration with Autoresearch (0.5 day)

**Goal:** every LoRA or WM trial's `pc_success` can be sourced from sim
deploy instead of (or alongside) open-loop MSE.

**Plumbing:**

1. Extend `scripts/_run_autoresearch_lora.sh` with `EVAL_MODE=sim` knob.
   When `sim`, post-train hook calls `sim_deploy.sh` instead of
   `_open_loop_eval.py`.
2. Extend `scripts/_run_autoresearch_wm.sh` similarly.
3. Dashboard Autoresearch tab gets a `metric_kind` filter: `open_loop_mse`
   / `wm_rollout_holdout_mse` / `sim_closed_loop_pc_success`.

**Cost considerations:**
- Sim eval ≈ 3-5 min per trial at 10 episodes × 600 steps. At 12-trial
  sweep → +60 min wall. Acceptable for overnight sweeps; skippable for
  fast iteration.
- For sim-eval-gated sweeps, scale STEPS down (no point training to 100k
  if sim disagrees with held-out MSE at step 10k).

**Acceptance:**
- `EVAL_MODE=sim` produces `pc_success` rows in history.jsonl that
  rank-order LoRA trials differently from open-loop MSE → real signal.
- Best trial under sim-eval matches OR diverges from open-loop best —
  either result is informative.

---

## Critical Path

```
Phase 0 (move + tomls, DONE)
     │
     ▼
Phase 1 (USD scene gen, 1 d) ─────────────┐
     │                                    │
     ▼                                    │
Phase 2 (IsaacSceneSession class, 2 d)  ──┤
     │                                    │
     ▼                                    │
Phase 3 (CLI plumbing, 0.5 d) ◄───────────┘
     │
     ├──▶ Phase 4 (DR, 1 d, optional)
     │
     ▼
Phase 5 (autoresearch hook, 0.5 d)
```

Total: **~2.5 working days** active engineering (Phase 0 + 1 + part of 2
landed this session) + ~2 h compute for first real sim sweep.

---

## Cross-References

| Asset | Path |
|-------|------|
| Deploy pkg in-tree | `src/lerobot-isaac-deploy/` |
| Scene generator | `~/workspaces/isaac-auto-scene/` |
| Isaac Sim env | `.pixi/envs/sim/` (training workspace) |
| Existing camera obs gap | `CLAUDE.md` §Build Status Checklist — "Camera observation wiring" |
| SO-101 USD | `assets/usd/so101_new_calib.urdf` |
| Open-loop eval | `scripts/_open_loop_eval.py` |
| WM rollout (offline) | `lerobot_isaac_deploy.wm_rollout.rollout()` |
| Hardware deploy session | `lerobot_isaac_deploy.session.run_session()` |

---

## Exit Criteria

The pipeline is "landed" when ALL hold:

- `bash scripts/sim_deploy.sh --policy-path <X> --usd <Y>` runs end-to-end
  without manual Isaac Sim launching.
- 10 episodes complete in < 5 min on RTX 3080.
- Rollout JSON includes `pc_success`, `mean_ep_len`, per-episode breakdown.
- Same ckpt evaluated twice produces `pc_success` within ±10 % rel (low
  noise floor).
- Sim `pc_success` for `trial_12` LoRA-best ckpt is > 0.0 (the policy
  produces non-degenerate joint trajectories in sim).
- Dashboard Autoresearch tab can ratchet against sim metric.
- Sim-deploy entry point is callable from the autoresearcher
  (`EVAL_MODE=sim` in sweep scripts).
