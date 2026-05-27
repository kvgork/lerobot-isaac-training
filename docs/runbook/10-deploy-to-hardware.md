# Runbook 10 — Deploy a Trained Policy on the Real SO-101 Arm

**Audience:** anyone with a real SO-101 follower arm, a checkpoint trained
on its data, and a desire to make the arm move.
**Outcome:** the trained policy drives the arm in closed loop at 30 Hz, with
safety clamps + e-stop, against the same dataset schema used for training.
**CLI:** `robot-data-run` (standalone `robot-data-runner` package at
`src/robot-data-runner/` (cloned from [github.com/kvgork/robot-data-runner](https://github.com/kvgork/robot-data-runner))). The earlier
`lerobot-isaac-deploy` entry-point still works as a backward-compat alias.
**Install:** `pixi run sync-runner && pixi run -e train-policy pip install -e src/robot-data-runner`.
**Cross-references:** [`docs/pipeline-overview.md §Stage I`](../pipeline-overview.md), [`docs/runbook/02-collect-data.md`](02-collect-data.md), [`docs/runbook/03-train-policy.md`](03-train-policy.md).

---

## TL;DR

```bash
# 1. Identify the port (one-time)
pixi run -e train-policy lerobot-find-port

# 2. Calibrate the arm (one-time, or after motor swaps)
pixi run -e train-policy lerobot-calibrate \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0

# 3. DRY-RUN — read obs, print predicted actions, NO motor writes
pixi run -e train-policy lerobot-isaac-deploy \
    --policy-path outputs/.../checkpoints/last/pretrained_model \
    --port /dev/ttyACM0 \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --camera d435_rgb=/dev/video0,640,480 \
    --duration-s 30 -v

# 4. EXECUTE — real motor writes, tight safety clamp, home on exit
pixi run -e train-policy lerobot-isaac-deploy \
    --policy-path outputs/.../checkpoints/last/pretrained_model \
    --port /dev/ttyACM0 \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --camera d435_rgb=/dev/video0,640,480 \
    --execute \
    --max-relative-target 3.0 \
    --home-on-exit \
    --duration-s 60
```

**The default is DRY-RUN.** You must pass `--execute` to allow motor
writes. Always run dry first against a fresh checkpoint.

---

## 1. Hardware Prerequisites

### 1.1 What you need plugged in

| Item                   | Notes                                                                 |
|------------------------|-----------------------------------------------------------------------|
| SO-101 follower arm    | 6 DYNAMIXEL motors (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`) wired in daisy chain. |
| U2D2 (USB → DYNAMIXEL) | OR built-in USB adapter (some SO-101 kits ship with one).             |
| 12 V power supply      | Must be on BEFORE you call `robot.connect()`.                         |
| Camera (optional)      | RealSense D435 / USB webcam if the policy was trained on images.      |
| Workspace clear of obstacles | The first execute-mode run will move; clear ≥0.7 m radius.       |

### 1.2 Identify the serial port

```bash
pixi shell -e train-policy
lerobot-find-port
# Unplug the U2D2 when asked. Note the device that disappears (e.g.
# /dev/ttyACM0 or /dev/ttyUSB0). That's --port.
```

### 1.3 Identify the camera (if needed)

```bash
ls /dev/video*
v4l2-ctl --device=/dev/video0 --all | head -5     # name + native resolutions
```

If your training dataset used `observation.images.d435_rgb`, the `--camera`
spec MUST use `d435_rgb` as the name so the obs key matches.

### 1.4 First-time motor calibration

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0
```

Follow the on-screen prompts. The calibration writes
`~/.cache/huggingface/lerobot/calibration/robots/so101_follower/main.json`.
**Re-run after** any motor swap, any new arm, or visible motion drift.

---

## 2. Safety Model

The `SO101Follower` driver and `lerobot-isaac-deploy` together enforce
**five** independent safety layers. None of them are sufficient on their
own; together they make the arm hard to crash.

### Layer 1 — DRY-RUN by default

`lerobot-isaac-deploy` does NOT command motors unless you pass `--execute`.
The control loop still:
- connects to the arm,
- reads observations every step,
- runs the policy,
- prints the predicted action,
- BUT does not call `robot.send_action()`.

Use dry-run for at least one full episode (30-60 s) on every new checkpoint.
If predicted actions look wild (NaN, motors stuck at 0, oscillating ±90°),
the policy is broken — DO NOT proceed.

### Layer 2 — `--max-relative-target`

Server-side clip enforced inside the `FeetechMotorsBus.send_action()` call.
Any action that would move a joint more than this delta in one step is
clipped silently. **Smaller = safer.** Recommended ladder:

| Trust level                                   | `--max-relative-target` |
|-----------------------------------------------|-------------------------|
| First execute-mode run on a brand-new policy  | `2.0` deg               |
| Policy validated for a few seconds, no drift  | `5.0` deg               |
| Mature policy, known-safe environment         | `10.0` deg              |
| Replay of recorded teleop (no policy)         | `30.0` deg              |

### Layer 3 — Rate limit

`--rate-hz 30` matches the SO-101 dataset recording rate. Higher = more
commands per second = more chance to fight the motors. Don't exceed 60 Hz.

### Layer 4 — Stuck-action watchdog

If the policy emits the same action (within ε) for `--repeat-warn-steps`
consecutive steps (default 30 ≈ 1 s), the CLI logs a WARNING. Means the
policy is either at a stable fixed point or has crashed (NaN, deadlock).
If you see this and the arm is NOT at a sensible target pose, SIGINT.

### Layer 5 — SIGINT clean-exit + optional home

Ctrl-C / SIGTERM is caught. The CLI:
1. Stops the control loop.
2. If `--home-on-exit` AND `--execute`: sends a zero-position command, then
   sleeps 0.5 s.
3. Calls `robot.disconnect()`, which disables motor torque (per
   `disable_torque_on_disconnect`, default True).

**Disable `--home-on-exit`** when homing through the zero pose would hit a
fixture, the table, or a workspace object. The default zero pose is
"shoulder pan = 0°, shoulder lift = 0°, elbow = 0° …", which often goes
straight through the table on first calibration.

### Layer 6 (out-of-band) — Physical e-stop

If the run starts to crash motors, **kill power**. The arm has no torque
without power. Software safety is never a substitute for being able to
hit the power switch.

---

## 3. CLI Reference

```
lerobot-isaac-deploy [-h] --policy-path P [--port DEV] [--dataset-root DIR]
                     [--camera SPEC] [--rate-hz N] [--duration-s N]
                     [--max-relative-target N] [--use-degrees]
                     [--execute] [--home-on-exit]
                     [--repeat-warn-steps N] [--seed N] [-v]
```

### Required

| Flag             | Meaning |
|------------------|---------|
| `--policy-path`  | `pretrained_model/` dir produced by training (the dir with `model.safetensors` + `policy_preprocessor.json` etc). |

### Robot

| Flag                       | Default     | Meaning |
|----------------------------|-------------|---------|
| `--port`                   | `/dev/ttyACM0` | DYNAMIXEL serial device. |
| `--use-degrees`            | false       | Bus reads/writes in degrees instead of normalized [-100, 100]. Match the units of the training dataset. |
| `--max-relative-target`    | `5.0`       | Per-joint max delta per step (degrees or normalized; see `--use-degrees`). |
| `--camera`                 | _none_      | Repeatable. `name=device,W,H`. e.g. `d435_rgb=/dev/video0,640,480`. The `name` MUST match the `observation.images.<name>` key in the policy's input schema. |

### Loop control

| Flag                | Default | Meaning |
|---------------------|---------|---------|
| `--rate-hz`         | `30.0`  | Control loop frequency. |
| `--duration-s`      | `60.0`  | Hard wall-clock cap. |
| `--repeat-warn-steps`| `30`   | Stuck-action warning threshold. |

### Policy

| Flag              | Default | Meaning |
|-------------------|---------|---------|
| `--dataset-root`  | _none_  | LeRobotDataset root used to derive obs/action feature shapes when the checkpoint doesn't carry them. **Required for older lerobot checkpoints** — pass the same dataset used for training. |
| `--seed`          | `42`    | Torch seed for non-deterministic policy layers. |
| `--task`          | _none_  | **REQUIRED for VLA policies (SmolVLA, OpenVLA).** Natural-language task instruction (e.g. `"pick and place cube"`). Must match the string recorded in the training dataset's `meta/tasks.parquet`. Without it the SmolVLA preprocessor cannot tokenise the language input and `select_action` crashes with `KeyError: 'observation.language.tokens'`. Ignored by ACT / Diffusion. |

### Safety

| Flag             | Default  | Meaning |
|------------------|----------|---------|
| `--execute`      | **off**  | Required to enable real motor writes. Default is dry-run. |
| `--home-on-exit` | off      | Send zero-position before disconnect. Disable if zero collides with the workspace. |
| `-v` / `--verbose` | off    | Log every predicted action. |

---

## 4. Step-by-Step Recipe

### Step 4.1 — Confirm policy + dataset shape match

```bash
pixi shell -e train-policy
python -c "
from lerobot.configs.policies import PreTrainedConfig
cfg = PreTrainedConfig.from_pretrained('outputs/.../pretrained_model')
print('type:', cfg.type)
print('input shapes:', cfg.input_features if hasattr(cfg,'input_features') else 'n/a')
print('output shapes:', cfg.output_features if hasattr(cfg,'output_features') else 'n/a')
"
```

The output should list `observation.state` (6,) and any `observation.images.*`
the policy needs. If the camera names don't match what you'll pass via
`--camera`, training and deployment are inconsistent — re-train or rename.

### Step 4.2 — Bench dry-run

Arm POWERED but BOTH FREE OF OBSTACLES and **physically restrained**
(strap, clamp, vise, or held by a human). On first run we don't trust the
policy yet.

```bash
lerobot-isaac-deploy \
    --policy-path outputs/full-pipeline-2026-05-14-XXXXXX/policy-diffusion/checkpoints/last/pretrained_model \
    --port /dev/ttyACM0 \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --camera d435_rgb=/dev/video0,640,480 \
    --duration-s 30 -v
```

Watch the action log. Each step should show the 6 joint targets close to
the current joint positions (because the dataset starts near home). A
healthy log looks like:

```
step 0 action={'shoulder_pan.pos': -0.31, 'shoulder_lift.pos': -42.0, ...}
step 1 action={'shoulder_pan.pos': -0.30, 'shoulder_lift.pos': -42.1, ...}
```

Pathological:
- All zeros → policy returning identity, no learning.
- Massive jumps (>30° between steps) → policy is wild; do not execute.
- `nan` → checkpoint corrupted or normalisation broken.

### Step 4.3 — Bench execute, tight clamp

If dry-run looked sane:

```bash
lerobot-isaac-deploy \
    --policy-path .../pretrained_model \
    --port /dev/ttyACM0 \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --camera d435_rgb=/dev/video0,640,480 \
    --execute \
    --max-relative-target 2.0 \
    --duration-s 10 \
    --home-on-exit
```

10 seconds, 2° max delta, home on exit. Stand next to the e-stop / power
strip. Watch the arm. If it moves smoothly toward a sensible target pose,
gradually relax the clamp on subsequent runs.

### Step 4.4 — Real task, larger budget

After a few short runs confirm the policy behaves:

```bash
lerobot-isaac-deploy \
    --policy-path .../pretrained_model \
    --port /dev/ttyACM0 \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --camera d435_rgb=/dev/video0,640,480 \
    --execute \
    --max-relative-target 5.0 \
    --duration-s 60
```

Place the source object at a position similar to the training distribution.
If the policy was trained on a fixed start pose, replicate that pose
before launch.

---

## 5. Output Schema (observations + actions)

The control loop converts between three coordinate systems:

```
SO101Follower.get_observation()          # robot driver
  → {"shoulder_pan.pos": float, ..., "d435_rgb": ndarray (H, W, 3) uint8}

_obs_to_policy_input()                   # adapter glue
  → {"observation.state":  (1, 6)  float32,
     "observation.images.d435_rgb": (1, 3, H, W) float32 ∈ [0, 1]}

policy.select_action(obs_in)             # lerobot policy
  → torch.Tensor  shape (1, 6)  float32  (action targets)

_action_to_robot_dict()                  # adapter glue
  → {"shoulder_pan.pos": float, ..., "gripper.pos": float}

SO101Follower.send_action(dict)          # robot driver
```

The 6 joint order matches `SO101_JOINT_NAMES` in `lerobot_isaac_env`:
`shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll / gripper`.
A checkpoint trained with `--target_arch diffusion` on a dataset
recorded by `robot_data_recorder` uses this same order.

---

## 6. Troubleshooting

| Symptom                                          | Most likely cause |
|--------------------------------------------------|-------------------|
| `robot.connect() failed — is the arm plugged in?` | Wrong `--port`, U2D2 unpowered, or USB cable. Try `lerobot-find-port` again. |
| `policy load failed: KeyError 'observation.images.d435_rgb'` | Camera name mismatch. Re-check `--camera <name>=...` against `cfg.input_features`. |
| `policy inference failed: shape mismatch` | Camera resolution mismatch. Training used 640×480 but you passed 1280×720. |
| `send_action failed: Joint out of range`         | `--max-relative-target` too high. Halve it. |
| Arm twitches but doesn't go anywhere             | `--max-relative-target` too low (server clips every command). Raise it. |
| Stuck-action warning fires                       | Policy NaN or stuck. SIGINT, re-train with `--seed` change. |
| Camera frame is all black                        | `/dev/videoN` wrong, or the camera needs a different colorspace. Try `v4l2-ctl --device=/dev/video0 --set-fmt-video=pixelformat=MJPG`. |
| `lerobot-find-port` says no candidate device     | U2D2 driver not loaded (`sudo modprobe ftdi_sio`) OR USB permissions (`sudo usermod -aG dialout $USER`, log out + in). |

---

## 7. Programmatic API

The CLI is a thin shell around the importable module. Use it from Python
when integrating into a larger control system:

```python
from lerobot_isaac_adapters.deploy import main
exit_code = main([
    "--policy-path", "/abs/path/to/pretrained_model",
    "--port", "/dev/ttyACM0",
    "--dataset-root", "/abs/path/to/datasets/kvgork/so101-pickplace1",
    "--camera", "d435_rgb=/dev/video0,640,480",
    "--execute",
    "--max-relative-target", "3.0",
    "--duration-s", "30",
    "--home-on-exit",
])
```

For finer control (custom obs preprocessing, custom safety predicates),
copy `_obs_to_policy_input`, `_action_to_robot_dict`, and the main loop
body from `deploy.py` and adapt — it's ~120 LOC end-to-end.

---

## 8. What This Does NOT Do (yet)

- **Closed-loop success metric.** SO-101 has no reward function on the
  real hardware. The eval proxy (`pc_success` = `1 / (1 + action_mse)`)
  only applies during open-loop dataset eval.
- **Multi-arm coordination.** Single follower only. For bimanual work,
  see `lerobot.robots.bi_so_follower` and adapt this module.
- **Leader-follower teleop.** This module replaces the teleop loop with a
  policy. For human-in-the-loop record, use `robot_data_recorder` or
  `lerobot-teleoperate`.
- **Isaac Lab simulation deploy.** A simulated rollout (run the policy in
  Isaac Sim with the same MDP terms) is a separate path — wait for the
  SO-101 Isaac env to land closed-loop reward terms (currently
  `SO101RewardsCfg(success=None, progress=None)`).

---

## 9. SmolVLA on so101-pickplace1 — concrete deploy recipe

Best-known checkpoint after the **2026-05-17 12-h overnight sweep** is
**trial_7 (`batch_up`, bs=8, lr=3e-5)** at step 45 000:

```
outputs/autoresearch-lerobot-policy-smolvla/trial_7/checkpoints/045000/pretrained_model
```

Re-rank eval (`outputs/eval/overnight-smolvla-2026-05-16T184411-rerank/winner.json`):

| Rank | Run                 | pc_success | MSE   | Op                |
|------|---------------------|-----------:|------:|-------------------|
| **1**| **trial_7**         | **0.1532** |  5.53 | batch_up (bs=8)   |
| 2    | anchor (49.5 k)     | 0.1128     |  7.86 | reference         |
| 3    | trial_6             | 0.0987     |  9.13 | lr_mid (2e-5)     |
| 4    | trial_4             | 0.0809     | 11.35 | weight_decay      |
| 5    | trial_2             | 0.0808     | 11.38 | lr_up (5e-5)      |
| 6    | trial_5             | 0.0743     | 12.46 | seed_swap         |
| 7    | trial_0             | 0.0725     | 12.79 | baseline          |
| 8    | trial_1             | 0.0614     | 15.28 | lr_down (1e-5)    |
| 9    | trial_3             | 0.0467     | 20.42 | batch_down (bs=2) |

### Easy mode (laptop) — pixi run wrapper

Once `winner.json` + ckpt are synced to the laptop, the
`lerobot-isaac-deploy` package runs the full confirm-gated ladder:

```bash
cd ~/workspaces/lerobot-isaac-deploy

# One-shot bootstrap (first time only).
pixi run bootstrap

# Pull winner + ckpt from desktop.
pixi run sync-ckpt -- --from desktop --winner <desktop-winner.json>

# Confirm-gated ladder: pre-flight → 1° clamp → 3° clamp → 10-ep eval.
pixi run session -- --winner <local-winner.json> --execute
```

`pixi run session` calls `li-deploy-session` (console entry from
`lerobot_isaac_deploy.cli:session_main`), which wraps the
`robot-data-run*` ladder below. See `~/workspaces/lerobot-isaac-deploy/`
for source.

### Manual mode — `robot-data-run` ladder

Below is what `pixi run session` invokes under the hood. Use this when
debugging a single stage or when the wrapper is unavailable.

Deploy commands, in escalating risk order:

```bash
# Step A — Pre-flight: load checkpoint, dump I/O schema (NO motor writes).
robot-data-run-check \
    --policy-path outputs/autoresearch-lerobot-policy-smolvla/trial_7/checkpoints/045000/pretrained_model \
    --dataset-root datasets/kvgork/so101-pickplace1

# Step B — Bench dry-run: full inference loop, NO motor writes, 30 s.
robot-data-run \
    --policy-path outputs/autoresearch-lerobot-policy-smolvla/trial_7/checkpoints/045000/pretrained_model \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --port /dev/ttyACM0 \
    --camera d435_rgb=/dev/video0,640,480 \
    --rate-hz 30 --duration-s 30 \
    --task "pick and place cube" \
    -v
# Expect: action lines logged every step; NO arm movement.

# Step C — Bench execute, tight clamp (1° per step), 30 s.
robot-data-run \
    --policy-path outputs/autoresearch-lerobot-policy-smolvla/trial_7/checkpoints/045000/pretrained_model \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --port /dev/ttyACM0 \
    --camera d435_rgb=/dev/video0,640,480 \
    --rate-hz 30 --duration-s 30 \
    --max-relative-target 1.0 \
    --task "pick and place cube" \
    --execute --home-on-exit \
    -v
# Keep finger on physical e-stop. Stop if motion is jerky or off-target.

# Step D — Real task, larger clamp (3°), 60 s.
robot-data-run \
    --policy-path outputs/autoresearch-lerobot-policy-smolvla/trial_7/checkpoints/045000/pretrained_model \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --port /dev/ttyACM0 \
    --camera d435_rgb=/dev/video0,640,480 \
    --rate-hz 30 --duration-s 60 \
    --max-relative-target 3.0 \
    --task "pick and place cube" \
    --execute --home-on-exit
```

Closed-loop multi-episode eval (recorded `pc_success`):

```bash
robot-data-run-eval \
    --policy-path outputs/autoresearch-lerobot-policy-smolvla/trial_7/checkpoints/045000/pretrained_model \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --port /dev/ttyACM0 \
    --camera d435_rgb=/dev/video0,640,480 \
    --task "pick and place cube" \
    --task-spec prompt_user_observer \
    --n-episodes 10 \
    --duration-per-episode-s 15 \
    --output-json outputs/eval/anchor-closed-loop.json \
    --i-have-read-the-safety-runbook
```

Notes:

- **First closed-loop run on a machine requires the
  `--i-have-read-the-safety-runbook` flag once** (stores a marker at
  `~/.config/robot-data-runner/safety_ack`).
- **`--task` is mandatory** for the SmolVLA anchor — without it,
  `select_action` crashes on the missing `observation.language.tokens`
  key (preprocessor builds it from `task` string).
- Winner updated 2026-05-17 from 12-h overnight sweep. trial_7
  (batch_up, bs=8, lr=3e-5, 45k steps) reached pc_success=0.153,
  +36 % over the prior 49.5k-step anchor (0.113). To swap winners,
  edit `winner.json`'s `winner_policy_path` and re-run `pixi run session`.

---

## 10. Related Files

- Adapter source: `src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/deploy.py`
- Console entry: `lerobot-isaac-deploy` (declared in adapters `pyproject.toml`)
- Robot driver: `lerobot.robots.so_follower.SO101Follower`
- Dataset utilities: `lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata`
- Open-loop eval (proxy metric on dataset, not hardware): `scripts/_open_loop_eval.py`
- Recording (the other direction — teleop → dataset): `robot_data_recorder`
  (opt-in package; `pixi run sync-recorder`)
