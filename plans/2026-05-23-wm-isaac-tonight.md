# WM Isaac Tonight — Resume Plan

**Date:** 2026-05-23
**Branch:** `feature/wm-isaac-env` (across all 3 repos)
**Parent:** `plans/2026-05-23-wm-isaac-env-plan.md`
**Status:** Track C UNBLOCKED — Isaac Lab env runs end-to-end. Training
verified active (3h12m baseline run reached step 37800/50000 before kill).
Plateaued at reward=-2.37 because reward function uses wrong EE body.

---

## What happened today

1. ✅ Patched lerobot-isaac-env / lerobot-isaac-adapters / lerobot-isaac-training to land Track C scaffold + bodies.
2. ✅ Fixed 5 cascading boot/runtime bugs:
   - sheeprl-with-deps broke `libgobject` → `pip install --no-deps`.
   - AppLauncher must boot BEFORE sheeprl imports → `scripts/_wm_isaac_entry.py`.
   - Double AppLauncher call deadlocked → `_boot` detects existing app.
   - `Cannot re-initialize CUDA in forked subprocess` → `sync_env: True`.
   - `target_bin` kinematic RigidObjectCfg hung sim.reset → AssetBaseCfg.
3. ✅ Reward pipeline functional: dense `progress_reward` term wired, gradient signal alive end-to-end.
4. ❌ Reward TARGET is wrong: `body_pos_w[:, -1, :]` picks the last
   articulation body (`moving_jaw_so101_v1_link`), not the actual gripper
   tip. Distance hits a physical-offset floor at ~0.95 m. Actor learned
   "max-extend arm" then plateaued.

---

## Tonight smoke result (2026-05-23 evening)

Patched `progress_reward` to use named `gripper_link` body (`ee_body_name`
kwarg, default "gripper_link"). Live smoke:

```
RESET OK
reward: min=-0.00864 max=-0.00837 mean=-0.00846
  per-step dist estimate: 1.015 m   ← STILL ~1 m for random actions
SMOKE GREEN
```

**Diagnosis updated:** body-lookup change is semantically correct but
NOT the root cause of yesterday's plateau. Random actions genuinely
fling the gripper across ~1 m of workspace; the prior -2.37 plateau was
the actor converging to "thrash randomly" because the per-step reward
gradient is too weak to dominate DreamerV3's policy loss.

Real blocker: **reward signal magnitude**, not body index.

| Issue | Symptom | Fix |
|-------|---------|-----|
| `weight=1.0` × Isaac Lab `dt=1/120` → per-step ≈ -0.008 | gradient too weak vs noise | bump to weight=10 or 100 |
| `distance_scale=1.0` (raw metres) | dist 1 m → reward -1 | set `distance_scale=0.4` (reach) |
| 50k steps only | DreamerV3 converges at 200k+ | more compute OR BC pre-train |

## Tomorrow (≤ 3.5 h compute, ~15 min eng)

### Step 1 — Bump reward signal in PickAndPlaceEnvCfg

File: `src/lerobot-isaac-env/src/lerobot_isaac_env/tasks/pick_and_place.py`

```python
self.rewards.progress = RewardTermCfg(
    func=_rewards_mod.progress_reward,
    params={
        "distance_scale": 0.4,                # normalise by SO-101 reach
        "object_cfg": SceneEntityCfg("source_object"),
        "ee_body_name": "gripper_link",       # already-defaulted, explicit for clarity
    },
    weight=10.0,                              # was 1.0
)
```

Effect: per-step reward becomes ~(-dist / 0.4) × 10 × dt = (-1.0 / 0.4)
× 10 × (1/120) ≈ -0.21 → still small but **25× the prior magnitude**.
Optionally raise weight to 100 for another 10×.

### Step 2 — Fix EE body lookup in `progress_reward`

File: `src/lerobot-isaac-env/src/lerobot_isaac_env/rewards.py`

Replace:
```python
ee_pos = robot.data.body_pos_w[:, -1, :]   # WRONG — picks gripper jaw
```

With named-body lookup:
```python
# Use a real EE body. Candidates (in SO-101 articulation order):
#   gripper_link        — preferred, the EE midpoint
#   wrist_link          — fallback, base of wrist before gripper
ee_body_name = "gripper_link"   # or pass via SceneEntityCfg.body_names
ee_idx = robot.find_bodies(ee_body_name)[0][0]
ee_pos = robot.data.body_pos_w[:, ee_idx, :]
```

Optionally accept `body_name` as a `params` field so the task config can
choose the link explicitly without code changes.

**Acceptance:**
- Smoke run prints per-step reward ≈ -0.2 to -0.4 at random init (vs
  -2.37/300 ≈ -0.008/step today → 0.95 m off-target).
- After 5k steps of training, reward floor moves toward -0.05 (5 cm of
  cube) when actor learns to reach.

### Step 2 — Smoke test the fix (no training, just env)

Reuse the `wm_smoke5.txt` pattern:

```bash
.pixi/envs/sim/bin/python - <<'PY' > /tmp/wm_reward_smoke.log 2>&1
import os, numpy as np
LOG = open('/tmp/wm_reward.txt', 'w', buffering=1)
def p(*a): print(*a, file=LOG, flush=True)
from lerobot_isaac_adapters.sheeprl_plugin.isaac_env import IsaacSO101Env
env = IsaacSO101Env(num_envs=1, image_size=64, headless=True, device='cuda')
obs, info = env.reset(seed=0)
rewards = []
for i in range(60):
    a = (env.action_space.sample()*0.3).astype(np.float32)
    obs, r, term, trunc, info = env.step(a)
    rewards.append(r)
p(f'rewards: min={min(rewards):.4f} max={max(rewards):.4f} mean={sum(rewards)/len(rewards):.4f}')
LOG.close()
os._exit(0)
PY
cat /tmp/wm_reward.txt
```

**Pass criterion:** `mean` in [-0.005, -0.002] range (= per-step dist 0.2–0.5 m × dt 1/120).

### Step 3 — Relaunch with the perf knobs

Script already has the user-pinned defaults (`bs=16, num_envs=2,
replay_ratio=2, bf16-mixed, fabric.precision=bf16-mixed`). Per the
script comment: expected 2-3× speedup vs today (SM 44 % → 70-80 %).

```bash
SESSION_ID="wm-isaac-$(date +%Y%m%d-%H%M%S)" \
    bash scripts/_run_wm_isaac_overnight.sh > /tmp/wm_isaac_v6.log 2>&1 &
```

Expected wall: 50k steps × ~5-7 step/s (vs 3.3 today) = 2–3 h.

### Step 4 — Watch the curve

| Phase | Per-step reward | Per-episode (300 steps) | EE-cube gap |
|-------|----------------|------------------------|-------------|
| Random init | -0.004 to -0.006 | -1.2 to -1.8 | 50–70 cm |
| Reaching | -0.002 to -0.003 | -0.6 to -0.9 | 20–35 cm |
| Touching | -0.0003 | -0.1 | < 5 cm |
| Grasped (no success_bonus yet) | 0 | 0 | 0 |

Stop ratchet when per-episode return < -0.5 for 5k consecutive steps =
strong actor. Or run full 50k for the baseline.

### Step 5 — Sync to laptop + dry-run

```bash
.pixi/envs/sim/bin/li-deploy-sync-wm \
    --sheeprl-run-dir logs/runs/dreamer_v3/isaac_so101/<run-dir> \
    --hydra-cfg-dir outputs/wm-isaac-prod-<session>/.hydra \
    --label wm-isaac-trial1

# On laptop (after pull feature/sim-deploy on deploy repo):
li-deploy-session \
    --policy-path ~/workspaces/lerobot-isaac-deploy/checkpoints/wm/wm-isaac-trial1 \
    --dataset-root ~/workspaces/lerobot-isaac-deploy/datasets/so101-pickplace1 \
    --port /dev/ttyACM0 \
    --duration-s 30 -v
```

DRY-RUN. Observe joint targets; do NOT pass `--execute` until trajectories
look sane.

---

## Open follow-ups (not blocking tonight)

1. **Wire a `success` termination** so `success_bonus` can fire — currently disabled (RewardsCfg sets success_bonus=None). Need a `terminations.py` term that detects "object lifted > 5 cm AND inside basket". Once that exists, re-enable the +5 terminal reward.
2. **Camera obs term still NotImplementedError** in `lerobot-isaac-env/observations.py` (`wrist_camera_rgb` / `overhead_camera_rgb`). The wrapper falls back to zero RGB. DreamerV3 will train with state-only obs effectively. Phase C3 from main plan.
3. **HP sweep** (`programs/wm-dreamerv3-isaac.md`) — kick off after baseline is verified working. 8-trial sweep × ~2.5 h each = ~20 h compute.
4. **Vault wiki note:** if perf experiment write-up at `05-Wiki/sources/2026-05-23-wm-gpu-perf-experiment.md` (referenced in script header) doesn't exist yet, capture it post-v6.

---

## Three repos, three branches — pushed state

| Repo | Branch | Tip |
|------|--------|-----|
| lerobot-isaac-training | `feature/wm-isaac-env` | program + plans (this file pending) |
| lerobot-isaac-adapters | `feature/wm-isaac-env` | IsaacSO101Env scaffold + bodies + double-boot fix |
| lerobot-isaac-env | `feature/wm-isaac-env` | progress_reward wired, target_bin AssetBaseCfg fix, RewardsCfg.success_bonus=None |

`sheeprl --no-deps` install in `.pixi/envs/sim` is local-only — note in
the script header documents the install but NOT pinned in `pixi.toml`
yet. Add as a tomorrow follow-up before the autoresearch sweep.

---

## Exit criteria for tomorrow's run

- Per-step reward ≥ -0.003 mean over last 5k steps (= reaching).
- At least one ckpt produced (step 10000 or higher).
- Sync to laptop succeeds via `li-deploy-sync-wm`.
- Dry-run on real SO-101 (if hardware reachable) shows smooth joint
  targets within calibrated limits.

If reward stays flat AND wrong-link-floor disappears → the bug fix
landed but actor needs longer train OR HP tuning. Either way the
pipeline becomes diagnosable, not blocked.
