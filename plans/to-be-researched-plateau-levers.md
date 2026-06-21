# Carry-place plateau — lever queue (easiest-first, autonomous)

**Created:** 2026-06-21 (master-project-orchestrator, autonomous mode)
**Context:** the carry-place run learns reach/lift (reward −71→−24) but never places
(`Game/ep_len_avg` pinned at 300, zero `place_termination` in 10000 steps) — a sparse-reward
exploration failure (see memory `carryplace-place-wall-plateau`). Root cause: from the 6.6cm
die-start the policy never randomly carries the die into the bin, so the place reward never
fires → no gradient toward placing.

**Directive:** run the levers EASIEST-FIRST on the (single) GPU; auto-advance to the next if the
current one doesn't break the plateau; when one SOLVES it (ep_len_avg drops below 300 / places
happen + reward climbs past ~−20), STOP and archive the remaining levers here as
"researched / not needed". No arm motion (sim only).

**Solve criterion (per lever):** `Game/ep_len_avg` < ~290 (min-ever < 300) AND `Rewards/rew_avg`
climbing past ~−20, sustained over a trailing window. = the policy places.

---

## Queue

### 0. Lever A — DreamerFD demo-seed  ← IN FLIGHT (2026-06-21)
`cp-stage1-seed-20260621`, replay 8, +25 demos (5412 transitions) seeded. Running. If this
breaks the plateau, levers 1–4 below become research backlog (do NOT run).

### 1. Lever B — easier curriculum step-0 (EASIEST; env-var only)
Make a place reachable by chance so the reward locks on, then curriculum-harden outward.
Resume `ckpt_10000` (keep learned reach/lift) with the die ~3.5cm from the bin.
```bash
SESSION_ID=cp-stage1-easyB-$(date +%H%M%S) STEPS=20000 BATCH_SIZE=8 REPLAY_RATIO=8 PRECISION=bf16-mixed \
NUM_ENVS=1 CHECKPOINT_EVERY=5000 SECONDS_PER_EXP=72000 LEROBOT_TRAIN_TIMEOUT=70000 \
LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1 LEROBOT_ISAAC_STAGED_REWARD=1 LEROBOT_ISAAC_CLOSURE_WEIGHT=4 \
LEROBOT_ISAAC_LIFT_SHAPING_WEIGHT=14 LEROBOT_ISAAC_PLACE_STD=0.15 LEROBOT_ISAAC_FIX_BASE=1 \
LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_OBJECT_X=0.185 LEROBOT_ISAAC_OBJECT_Y=-0.13 \
LEROBOT_ISAAC_TARGET_X=0.22 LEROBOT_ISAAC_TARGET_Y=-0.13 \
EXTRA_HYDRA="algo.horizon=25 algo.actor.ent_coef=1e-3 metric.log_every=500 algo.mlp_keys.encoder=[state] checkpoint.resume_from=<logs/.../checkpoint/ckpt_10000_0.ckpt>" \
bash scripts/_run_wm_isaac_overnight.sh
```
Die (0.185,−0.13) is ~3.5cm from the bin (0.22,−0.13) — within a single carry of the existing
reach/lift policy. (Resume path: newest `logs/runs/dreamer_v3/isaac_so101/*/version_0/checkpoint/ckpt_10000_0.ckpt`.)

### 2. Lever C — terminal place bonus (EASY; env-var; combine with B)
Once places occur, make them dominate the return so credit propagates hard. Add to lever B's env:
```
LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT=10
```
Useless ALONE (no place to reward); only run as B+C. So lever 2 = re-run lever B with this added.

### 3. More demos at higher-yield start (MEDIUM; regen + re-seed)
The (0.16,−0.10) regen yielded only 25 demos (~18-28% scripted success — grasp struggles at low
obj_x near the base). Regen ~40 demos at the higher-yield `(0.18,-0.10)` start (x=0.18 is in the
scripted grasp's success zone), then re-seed a fresh run.
```bash
# regen (higher yield):
LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_STAGED_REWARD=1 \
LEROBOT_ISAAC_CLOSURE_WEIGHT=4 LEROBOT_ISAAC_LIFT_SHAPING_WEIGHT=14 LEROBOT_ISAAC_PLACE_STD=0.15 \
LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1 LEROBOT_ISAAC_TARGET_X=0.22 LEROBOT_ISAAC_TARGET_Y=-0.13 \
.pixi/envs/sim/bin/python scripts/_gen_sim_demos.py --episodes 40 --max_attempts 120 \
  --obj_x 0.18 --obj_y -0.10 --out datasets/local/so101-sim-pickplace-demos-op2
# then relaunch the seeded run with LEROBOT_ISAAC_DEMO_DATASET=...demos-op2
```

### 4. Wire the explicit DreamerFD BC-loss (HARDEST; code)
`demo_buffer.behavior_cloning_loss` is DEFINED but NOT called in the training step (scaffold
only). Wire it as an explicit actor BC term (with the DreamerFD decay + virtual-clutch) in the
sheeprl dreamer_v3 actor update via the `_wm_isaac_entry.py` monkeypatch path. Needs code + a
GPU run to validate. Do only if levers A–C + more-demos all fail to place. Risk: touches the
sheeprl training loop; high correctness bar.

---

## Notes / open research
- The env's `place_termination` is **XY-only** (no z-gate, no gripper-open gate) — "place" = die
  carried over the bin XY, even while gripped/lifted. If real release-into-bin is wanted, that's a
  separate `place_termination` redesign (z-gate + gripper-open) — research item, not a plateau lever.
- num_envs MUST stay 1 (the is_first crash; no parallel-env speedup lever exists).
- Throughput ~0.33 steps/s at replay 8; each lever run is ~14-16h for 20k steps (ckpts at 5k/10k/15k).
