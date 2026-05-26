# DreamerV3 Actor Deploy — Implemented

**Date:** 2026-05-16
**Owner:** Koen van Gorkom
**Status:** **implemented** — 2026-05-25, deploy commit `5d95358` on `feature/sim-deploy`

## What was built (2026-05-25)

### Ckpt structure (ckpt_40000_0.ckpt, iter 40k)

```
top-level keys: world_model, actor, critic, target_critic,
                world_optimizer, actor_optimizer, critic_optimizer,
                moments, ratio, iter_num=40000, batch_size=16, rb
encoder.mlp_encoder...weight: (512, 13)  → state_dim=13 (joint_pos[6]+object_pose[7])
actor.mlp_heads.0.weight:     (12, 512)  → action_dim=6 (TanhNormal: 12 = 2×6)
config layout: version_0/config.yaml (Lightning CSV logger, NOT .hydra/)
```

### Fixes applied to wm_loader.py

1. **Config discovery** — `_discover_config_yaml()` probes 4 candidate paths:
   `version_N/.hydra/config.yaml`, `version_N/config.yaml` (Lightning layout),
   `<run>/.hydra/config.yaml`, `<run>/config.yaml`. Resolves the deferred
   "config path not found" failure.

2. **state_dim from weights** — `_state_dim_from_weights()` reads
   `encoder.mlp_encoder.model._model.0.weight.shape[1]` directly from
   the saved checkpoint, bypassing cfg. Fixes `RuntimeError: size mismatch
   (shape (512,13) vs (512,6))`.

3. **action_dim from weights** — `_action_dim_from_weights()` reads
   `actor.mlp_heads.0.weight.shape[0] // 2` (TanhNormal = 2× concatenated
   mean+logstd). Returns 6 for the SO-101 config.

4. **PlayerDV3 as select_action** — `LoadedWMActor.select_action` now
   calls `player.get_actions(obs_dict)` via sheeprl's `PlayerDV3` instead
   of the manual `actor.init_state / actor.step` stub (which doesn't exist
   in this API version).

5. **Obs tensor shape** — CNN keys shaped `(T=1, num_envs=1, C, H, W)`
   normalized `[-0.5, 0.5]` (sheeprl's `prepare_obs` convention); MLP keys
   shaped `(T=1, num_envs=1, D)`.

6. **cnn_keys field** — `LoadedWMActor.cnn_keys` stores the encoder CNN
   key list from cfg for correct normalization routing in `select_action`.

### New: wm_dryrun.py + `wm-dryrun` CLI

```bash
# From the training workspace (desktop):
.pixi/envs/sim/bin/python -m lerobot_isaac_deploy.wm_dryrun \
    --policy-path logs/runs/dreamer_v3/isaac_so101/<run>/version_0/checkpoint/ckpt_40000_0.ckpt \
    --n-samples 100

# Writes outputs/wm-dryrun-<ts>/report.json
# Acceptance checks: all_finite=True, in_range=True, shape_ok=True
```

### Dry-run results (ckpt_40000_0.ckpt, N=20)

```
joint 0: mean=+0.497  std=0.531  [-1.000, +1.000]
joint 1: mean=+0.684  std=0.477  [-0.692, +1.000]
joint 2: mean=+0.717  std=0.452  [-0.553, +1.000]
joint 3: mean=-0.342  std=0.696  [-1.000, +1.000]
joint 4: mean=+0.721  std=0.470  [-0.418, +1.000]
joint 5: mean=+0.146  std=0.827  [-1.000, +1.000]
checks: finite=True  in[-1,1]=True  shape_ok=True  PASS
```

Note: Policy is unconverged (only ~40k steps on a sparse reward task).
Actions are saturated (±1.0) which is typical for an unconverged continuous
actor. Use for integration testing only, not for real arm motion yet.

## Acceptance criteria status

- [x] `load_dreamerv3` returns a working `LoadedWMActor` on the real sheeprl ckpt
- [x] `LoadedWMActor.select_action(obs)` returns shape `(6,)` float32, values in [-1,1]
- [x] `wm-dryrun` CLI produces `report.json` with acceptance checks (all pass)
- [x] `lerobot-isaac-deploy session --policy-path <dreamer-ckpt>` routes correctly,
      preflight skipped for dreamerv3 kind
- [x] Safety clamps verified: `--max-relative-target` (default 3°), `home_on_exit=True`
      (default), `safety_critical=True` gate on all motor-write steps
- [x] 86/86 tests pass (10 new tests in `test_wm_dryrun.py`)

## Command for user when ready to test on real arm

```bash
# Dry-run (no motors): reads real obs, prints predicted actions
pixi run -e sim lerobot-isaac-deploy session \
    --policy-path logs/runs/dreamer_v3/isaac_so101/2026-05-25_12-57-32_dreamer_v3_isaac_so101_42/version_0/checkpoint/ckpt_40000_0.ckpt \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --port /dev/ttyACM0 \
    --dry-run-loop \
    --duration-s 30

# Safety knobs (active by default — no flags needed to enable):
#   --max-relative-target 3.0 (deg per step, step_execute_* only)
#   --no-home-on-exit         (disable only if explicitly wanted)
# To require real ckpt gate (not synthetic): --require-real-ckpt
```

**DO NOT add `--execute`** until you have reviewed the dry-run action output and
confirmed the arm is in a safe position with a hand on the e-stop.

## Still out of scope

* DreamerV3 training on laptop (VRAM limited)
* MPC planning on top of LeWM
* Camera obs wiring (training ran with zero-RGB; real cameras will be OOD)

## Why deferred (original)

The deploy package routes hardware deploy through `robot-data-runner`'s
subprocess CLI which loads policies via lerobot's `make_policy` factory.
DreamerV3 (sheeprl) checkpoints are NOT lerobot policies — they have a
different file layout (`ckpt_*.ckpt` + `config.yaml`) and a different
forward-pass API (`encoder + RSSM + actor`, recurrent state across steps).

As of 2026-05-16 no DreamerV3 run had produced a real `ckpt_*.ckpt`.
The first real ckpt (2026-05-25) revealed the exact sheeprl API and obs
shapes needed to wire everything up.
