# Runbook 12 — Hybrid (desktop+laptop) vs Single-System Deploy

**Audience:** anyone who trained on one machine and wants to run closed-loop
hardware eval on a different machine connected to the SO-101.
**Outcome:** trained policy from desktop → arm on laptop, eval JSONs flow
back, dashboard sees everything.
**Sibling runbooks:**
[`10-deploy-to-hardware.md`](10-deploy-to-hardware.md) (CLI safety),
[`11-closed-loop-eval.md`](11-closed-loop-eval.md) (eval protocol),
[`00-install.md`](00-install.md) (desktop install).

---

## TL;DR — Hybrid Path

```bash
# ONCE on laptop:
bash scripts/laptop_bootstrap.sh                            # creates ~/workspaces/lerobot-isaac-deploy
pixi shell -e deploy
lerobot-find-port                                           # note the /dev/ttyACM*
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0

# EACH DEPLOY CYCLE — on DESKTOP:
LAPTOP_HOST=laptop bash scripts/sync_ckpt_to_laptop.sh \
    --run-dir outputs/long-train-2026-05-14-diffusion-dreamerv3-4h

# On LAPTOP (dry-run first):
robot-data-run \
    --policy-path ~/workspaces/lerobot-isaac-deploy/checkpoints/<run-name>/<ckpt>/pretrained_model \
    --port /dev/ttyACM0 \
    --dataset-root ~/workspaces/lerobot-isaac-deploy/datasets/kvgork/so101-pickplace1 \
    --camera d435_rgb=/dev/video0,640,480 \
    --duration-s 30 -v
# … then runbook 11's closed-loop eval.

# AFTER eval session — on DESKTOP:
LAPTOP_HOST=laptop bash scripts/sync_eval_from_laptop.sh
```

Dashboard's Evaluation tab on the desktop will show the closed-loop runs
once `sync_eval_from_laptop.sh` finishes.

---

## 1. When to pick Hybrid vs Single

| Choice | Pick when |
|---|---|
| **Hybrid** | Desktop is busy training the next AR run AND laptop has a GPU AND laptop is physically near the arm. |
| **Single** | One machine has both the training time AND the arm cabling. Simpler, no sync overhead. |

User's laptop = 6 GB Nvidia GPU. Plenty for diffusion-policy inference
(model is ~250 MB; one denoising step ~50 ms on 6 GB GPU). Same
software stack as `robot_data_recorder` (lerobot + pyrealsense2),
so no new bring-up.

The user said: "I will switch to single setup later probably." → both
paths documented here so the switch is cheap.

---

## 2. Hybrid Setup

### 2.1 Laptop pre-reqs

| Item | Value |
|---|---|
| OS | Ubuntu 22.04 / 24.04 (same as recorder host) |
| Python | 3.10–3.12 (pixi handles this) |
| pixi | https://pixi.sh/install |
| GPU | NVIDIA driver ≥535 (user has 6 GB GPU) |
| USB | U2D2 / DYNAMIXEL adapter for SO-101 + USB cable for D435 |
| SSH | desktop can `ssh laptop` (configure `~/.ssh/config` for short alias) |
| Disk | ≥10 GB free under `$HOME/workspaces/` (lerobot wheel + ckpts) |

### 2.2 One-time bootstrap

```bash
# On laptop:
cd ~  # any dir
curl -fsSL https://raw.githubusercontent.com/kvgork/lerobot-isaac-training/main/scripts/laptop_bootstrap.sh -o laptop_bootstrap.sh
# OR copy the script from a desktop-shared path:
scp desktop:lerobot-isaac-training/scripts/laptop_bootstrap.sh .
bash laptop_bootstrap.sh
```

`laptop_bootstrap.sh`:

1. Creates `~/workspaces/lerobot-isaac-deploy/` with a minimal `pixi.toml`
2. Runs `pixi install -e deploy` (pulls Python 3.12 + numpy)
3. Pins `lerobot==0.5.1` (matches desktop's installed version — read at
   bootstrap time; override with `LEROBOT_VERSION=...` env var)
4. Clones `robot_data_runner` from the local spinout dir (default) or
   the GitHub repo (`RUNNER_REPO=...`) and installs it editable
5. Verifies `robot-data-run`, `robot-data-run-check`,
   `robot-data-run-eval` console entries exist
6. Detects `/dev/ttyACM*` and prints the next-step calibration command

The script is idempotent — re-running it just refreshes the
robot-data-runner git checkout.

### 2.3 Calibrate on the laptop (one-time per motor swap)

```bash
pixi shell -e deploy
lerobot-find-port                    # note the device
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0
```

Writes `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/main.json`
on the laptop. **Re-run after any motor swap.**

### 2.4 Pre-flight on the laptop (every session)

```bash
robot-data-run-check \
    --policy-path ~/workspaces/lerobot-isaac-deploy/checkpoints/<run>/<ckpt>/pretrained_model \
    --dataset-root ~/workspaces/lerobot-isaac-deploy/datasets/kvgork/so101-pickplace1
```

Prints expected `observation.images.*` keys. If your camera name doesn't
match, training + deployment are inconsistent — fix before continuing.

### 2.5 Sync from desktop (each new checkpoint)

On the DESKTOP:

```bash
bash scripts/sync_ckpt_to_laptop.sh \
    --host laptop \
    --remote-base '~/workspaces/lerobot-isaac-deploy' \
    --run-dir outputs/long-train-2026-05-14-diffusion-dreamerv3-4h
```

Transfers (rsync over SSH):
- Latest `policy-diffusion/checkpoints/<NNNN>/pretrained_model/` → `laptop:<base>/checkpoints/<run-name>/<NNNN>/pretrained_model/`
- `dashboard/manifest.json` → `laptop:<base>/checkpoints/<run-name>/manifest.json`
- `datasets/<repo>/meta/` only (NOT the data parquet — too large; metadata
  + info.json is what the dataset-root flag needs)

Add `--full-dataset` if you need the parquet shards too (e.g. for closed-loop
fall-back to open-loop preflight on the laptop). Not implemented yet —
edit the script if needed.

### 2.6 6 GB GPU inference budget

Diffusion policy + 6 GB GPU at 30 Hz:

- Model + activations: ~1.5–2 GB.
- Denoising loop: ~50 ms per `select_action` call on a 30 series 6 GB.
- 30 Hz → 33 ms / step. **Inference at default `policy.n_action_steps`
  may not finish in time.** Mitigation:

  ```bash
  robot-data-run-eval ... \
      -- --policy.n_action_steps=16
  ```

  Policy emits a 16-step action chunk per inference. Runner replays the
  chunk at 30 Hz between inferences — effective inference cost drops to
  ~3 ms / step (chunk replay is free).

  Default in `robot_data_runner` is no chunking. Set `n_action_steps` to
  match what the policy was trained with (typically 8 or 16).

ACT and SmolVLA already chunk natively; diffusion needs the flag.

### 2.7 Closed-loop eval on the laptop

Same recipe as runbook 11, with `--policy-path` pointing at the laptop's
checkpoint copy.

```bash
robot-data-run-eval \
    --policy-path ~/workspaces/lerobot-isaac-deploy/checkpoints/<run>/<ckpt>/pretrained_model \
    --port /dev/ttyACM0 \
    --dataset-root ~/workspaces/lerobot-isaac-deploy/datasets/kvgork/so101-pickplace1 \
    --camera d435_rgb=/dev/video0,640,480 \
    --task-spec prompt_user_observer \
    --n-episodes 10 \
    --duration-per-episode-s 10 \
    --max-relative-target 3.0 \
    --home-on-exit \
    --output-json ~/workspaces/lerobot-isaac-deploy/outputs/eval/closed-loop-<run>-N17.json \
    --n-train-eps 17 \
    --i-have-read-the-safety-runbook
```

JSON lands in `laptop:~/workspaces/lerobot-isaac-deploy/outputs/eval/`.

### 2.8 Sync eval JSONs back to desktop

On the DESKTOP:

```bash
bash scripts/sync_eval_from_laptop.sh
```

rsync pulls every `outputs/eval/*.json` from the laptop into the
desktop's `outputs/eval/`, then re-renders the dashboard report. Live
dashboard on port 8501 refreshes within ~30 s.

---

## 3. Single-System Setup

When you switch to running both training and arm on the SAME machine
(typically the laptop once the desktop is no longer needed for AR runs):

### 3.1 Migrate workspace

```bash
# On the laptop (full workspace clone, not the deploy-only one)
git clone <lerobot-isaac-training-repo> ~/workspaces/lerobot-isaac-training
cd ~/workspaces/lerobot-isaac-training
pixi install                                          # default env
bash scripts/install_train_deps.sh                    # adds lerobot+sheeprl
bash scripts/install_isaac_lab.sh                     # Isaac Sim 6.0 (10 GB)
```

The 6 GB GPU is borderline for Isaac Lab DR replay (the bundled
`Isaac-SO101-PickPlace-v0` env wants ~3 GB free during runtime — at 6 GB
with sim assets it'll work for `--num_envs 1`). For policy + WM training
on the SO-101 dataset it's fine.

### 3.2 Copy the dataset

```bash
# Either rsync from the desktop:
rsync -av desktop:~/workspaces/lerobot-isaac-training/datasets/kvgork/ \
          ~/workspaces/lerobot-isaac-training/datasets/kvgork/

# OR re-collect via robot_data_recorder.
```

### 3.3 Same commands work

Once on a single machine, ALL the existing pipeline / autoresearch /
deploy commands work — no sync wrappers needed. Eval JSONs already
land in `outputs/eval/`, dashboard already reads them. Skip steps 2.5
+ 2.8 of the hybrid recipe.

### 3.4 What you lose by going single

- Desktop can no longer train concurrently with eval.
- AR runs that wanted overnight 8 h budgets must wait for arm time.

### 3.5 What you gain

- Zero sync overhead.
- One environment to maintain.
- Dashboard, eval, training, calibration all colocated.

---

## 4. Switching Hybrid → Single

Migration is just a one-time copy:

```bash
# On laptop (now serving as the single workstation):
rsync -av desktop:~/workspaces/lerobot-isaac-training/outputs/ \
          ~/workspaces/lerobot-isaac-training/outputs/
rsync -av desktop:~/workspaces/lerobot-isaac-training/datasets/ \
          ~/workspaces/lerobot-isaac-training/datasets/
rsync -av desktop:~/workspaces/lerobot-isaac-training/.agent-state/ \
          ~/workspaces/lerobot-isaac-training/.agent-state/

# Stop using the deploy-only workspace
rm -rf ~/workspaces/lerobot-isaac-deploy
```

The deploy-only workspace and the hybrid sync scripts become unused —
keep them if you might switch back, delete if not.

---

## 5. Reference Tables

### 5.1 Files this runbook references

| File | Role |
|---|---|
| `scripts/laptop_bootstrap.sh` | One-time laptop setup (pixi env + lerobot pin + runner editable install). |
| `scripts/sync_ckpt_to_laptop.sh` | Push latest checkpoint + dataset meta from desktop → laptop. |
| `scripts/sync_eval_from_laptop.sh` | Pull eval JSONs from laptop → desktop + refresh dashboard. |
| `docs/runbook/10-deploy-to-hardware.md` | Hardware setup + 6-layer safety. Read first. |
| `docs/runbook/11-closed-loop-eval.md` | Per-session eval protocol + task specs. |
| `plans/2026-05-15-closed-loop-eval.md` | Design + spec. |
| `~/.ssh/config` | Set `Host laptop \n HostName 192.168.x.y \n User koen` for short aliases. |

### 5.2 Version pinning

| Component | Pin to | Set by |
|---|---|---|
| `lerobot` | exactly the desktop's version | `LEROBOT_VERSION` env var passed to `laptop_bootstrap.sh` |
| `robot_data_runner` | `main` branch HEAD or a tagged release | manual `git checkout` if needed |
| `torch` | whatever lerobot pulls | follows lerobot |
| Calibration JSON | host-specific — DO NOT sync | each host calibrates once |

### 5.3 Hybrid topology

```
+──────────────+ rsync over SSH +──────────────+
|  desktop     |───── ckpt ────>|  laptop      |
|              |                |              |
|  - training  |                |  - arm USB   |
|  - AR runs   |<──── eval ─────|  - inference |
|  - dashboard |  JSONs         |  - eval CLI  |
+──────────────+                +──────────────+
```

Single arrow each direction. Each script is one rsync invocation. No
NFS / mounts / databases needed.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `Permission denied (publickey)` from rsync | Add desktop's pubkey to laptop's `~/.ssh/authorized_keys`. Or use `ssh-copy-id laptop`. |
| `lerobot version mismatch — checkpoint load failed` | Re-run `laptop_bootstrap.sh` with `LEROBOT_VERSION=<desktop's>`. |
| Inference > 33 ms / step on laptop | Add `-- --policy.n_action_steps=16` to the runner call. |
| Calibration `main.json` not found | Re-run `lerobot-calibrate` on the laptop. Calibration is host-specific. |
| Eval JSON appears on dashboard but `pc_success` is None | Older `robot_data_runner` (no `EVAL_SCHEMA` fields). Re-run `bash laptop_bootstrap.sh` to update. |
| Dashboard shows stale data | Run `bash scripts/sync_eval_from_laptop.sh` — re-renders manifest. |
