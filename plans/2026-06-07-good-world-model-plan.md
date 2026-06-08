# Plan: Getting a *good* world model for SO-101 pick-place (2026-06-07)

## TL;DR
Your recorded data is **fine** — not wasted. The offline `custom_hdf5` DreamerV3 path
is a thin proof-of-concept, not a real WM setup. For a good world model **you do need
failures + exploration**, and the cleanest source is the **online Isaac sim you already
have** (`env=isaac_so101`), which generates them itself and provides proprio + a reward.
Recommendation: **switch to online DreamerV3 in Isaac sim**; keep the offline path only
as a dynamics-only smoke / optional warm-start.

---

## What we verified today

| Item | State |
|------|-------|
| Recorder data (parquet + HDF5) | ✅ Correct: state (T,12), action (T,6), frames, reward, done. No re-record. |
| Bridge meta `state_dim/action_dim=1` | ✅ **Fixed** — was counting keys not feature width; now reads array last-axis (12 / 6). Skill tests 15/15. |
| Offline `custom_hdf5` env (`hdf5_env.py`) | ⚠️ Thin POC — see below. |
| Sim reward | ✅ Exists: `rewards.success_reward` (Gaussian EE→object distance, [0,1]); online `isaac_env.py` returns it. Staged shaping (grasp/lift/place) is a TODO, not a blocker. |

### Why the offline `custom_hdf5` run was never going to give a good WM
`hdf5_env.py` (`_load_h5` + `_get_obs`):
1. **Loads only `frames` + `actions`** — the 12-dim proprio `states` array is **never read**.
2. The `"state"` obs key is filled with the **action**, not joint state (so Dreamer's
   mlp-keys saw the action, no proprioception).
3. **Only the first 16 timesteps per episode** are used → ~800 of 18804 frames.
4. `step()` ignores the action, **reward = 0** always → the actor-critic gets no signal;
   only the world model's reconstruction trains (the obs_loss 2767→46 we saw).
5. Data is **success-only** (50 expert demos) → narrow "success manifold"; the WM never
   sees the consequences of wrong/exploratory actions.

Net: that path yields a pixels-only reconstruction model on a sliver of narrow data — good
for proving the pipeline, useless as a controllable WM.

---

## Do I need failures? — Yes (and more)

A world model used for **control/imagination** must cover more than expert successes.
Dreamer imagines off-distribution actions; if the WM never observed exploratory/failed
transitions, its predictions there are garbage and the imagined policy is worthless.
Required coverage for a good WM:
- **Failures + recoveries** (not just successes).
- **Exploratory / off-policy actions** (random, perturbed).
- **Varied initial states** (object pose, arm config) — domain randomization.

Expert teleop demos give almost none of this. The two ways to get it: online RL
exploration (Path A) or deliberately recording/ generating diverse data (Path B helpers).

---

## Path A — ONLINE DreamerV3 in Isaac sim  ✅ recommended (sim already set up)

`env=isaac_so101` (`sheeprl_plugin/isaac_env.py` + `lerobot_isaac_env`):
- **Obs:** rgb `(3,H,W)` + real proprio `state` (joint_pos[6], +object_pose[7] optional).
- **Reward:** `success_reward` distance kernel (add staged grasp/lift/place shaping for
  better credit assignment).
- **Dynamics + exploration:** Dreamer acts → fails/succeeds → learns WM **and** policy by
  imagination. This is what DreamerV3 is designed for; coverage is generated for free.
- GPU-verified path (CLAUDE.md WM-Isaac 2026-05-31; camera `d435_rgb` wired).

Deliverable: a usable world model **and** a pick-place policy.

### Path A steps
1. [x] Fix bridge meta (done).
2. Boot/accept checks on GPU: `lerobot-isaac env smoke --cameras=d435` + A.1/A.3 from
   `plans/2026-05-30-gpu-hw-execution-checklist.md`.
3. (Optional but high-value) Add **staged shaping** to `SO101RewardsCfg` / `rewards.py`
   (reach → align → grasp → lift → place) — biggest lever for manipulation RL.
4. Short online Dreamer smoke: `env=isaac_so101`, `num_envs=4`, image 64,
   `learning_starts ≥ num_envs×seq_len (≥256)`, ~few-k steps → confirm reward signal +
   WM loss + policy return trending up.
5. Full online run under the report-only watchdog; track `Loss/observation_loss` +
   episode return / success rate.
6. (Optional) Warm-start: offline-pretrain the WM on the recorded demos, then continue online.

### Path A constraints (RTX 3080)
- `num_envs` 4–8, image 64 (CLAUDE.md OOM ladder). VRAM is not the limit for Dreamer.
- `learning_starts ≥ num_envs × per_rank_sequence_length` (=256 for 4×64) — see
  memory `dreamerv3-learning-starts-rule`. Prod default 1024 safe.
- Isaac teardown: `IsaacSO101Env.close()` no-op + entry `os._exit` (already handled;
  memory `wm-isaac-stall-resolved`).

---

## Path B — improve the OFFLINE WM from recorded data (dynamics-only)

Only if you want a *passive* dynamics / video-prediction model (NOT a policy). Even fixed,
it stays narrow (success-only) and reward-free. Required `hdf5_env.py` fixes:
1. Load `states` from the HDF5 and expose **real 12-dim proprio** as `"state"` (stop using
   the action).
2. Window across **all** frames per episode, not the first 16.
3. Record **failures** (recorder `f` key — now supported) for coverage; optionally add DR /
   synthetic data via `lerobot-isaac-synthetic`.
4. Accept: no reward → no policy; WM is reconstruction-only.

Best use of Path B: produce a warm-start WM to seed Path A.

---

## Recommendation
Go **Path A**. You have the sim; it supplies proprio, reward, and — crucially — its own
failures/exploration, which expert demos can't. Use the teleop demos only to seed/pretrain.
Treat the offline `custom_hdf5` path as a pipeline smoke, not the training plan.

Open dependency to decide: invest in **staged reward shaping** before the full online run
(recommended) vs run with the existing distance reward first.

---

## Online smoke result (2026-06-07) + the num_envs bug

Online Isaac DreamerV3 smoke (`env=isaac_so101`):
- **`num_envs=2` CRASHES** — `dreamer_v3.py:608 is_first IndexError (size 1)` +
  `Invalid action shape expected 6 received 3`. `IsaacSO101Env` is a **single-env wrapper**:
  it collapses Isaac's batch to env-0 (`_scalar`) and declares single-env spaces, but sheeprl
  is told `num_envs` and expects that many env-slots → mismatch.
- **`num_envs=1` runs clean (20-min cap, rc=124):** reward improving
  `-86.5 → -81.5 → -66.5` (actor learning to approach object), GPU **8.7 GB / 51% util**,
  0 errors. ✅ This is the working baseline.

**Fix 1 (done):** `_run_wm_isaac_overnight.sh` default `NUM_ENVS` 2 → 1 (the broken default
would crash any overnight run). Memory: `wm-isaac-num-envs-bug`.

**Fix 2 (follow-up — true vectorization, throughput win):** to use Isaac's native N-parallel
sim, make `IsaacSO101Env` a real vectorized env instead of collapsing to env-0:
1. `isaac_env.py` — stop collapsing: `reset()/step()` return **batched** `(num_envs, …)` obs
   dict, reward, terminated, truncated; remove the `_scalar()` reduction; per-env arrays.
2. Declare `single_observation_space`/`single_action_space` + batched `observation_space`
   (subclass `gymnasium.vector.VectorEnv`, or expose an `autoreset` batched API).
3. sheeprl side — bypass its `SyncVectorEnv` wrapping so it consumes the **already-vectorized**
   env (it currently builds a vector wrapper of `num_envs` copies; Isaac can't be multi-
   instanced — `SimulationContext` is a singleton). Needs a sheeprl env-build hook or a
   thin "pre-vectorized" adapter.
4. Verify on GPU (each boot ~1–2 min) at num_envs=4/8; watch VRAM + is_first/done shapes.

Not a correctness blocker — DreamerV3 trains fine at num_envs=1 (sample-efficient); Fix 2 is a
wall-clock/data-throughput optimization. Estimated: medium refactor + several GPU iterations.

### Fix 2 — full implementation spec (2026-06-08 deep-dive)

Root cause (exact): `sheeprl/algos/dreamer_v3/dreamer_v3.py:384-398`:
```python
vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
envs = vectorized_env([make_env(..., rank*num_envs+i, ...) for i in range(cfg.env.num_envs)])
```
- Builds `num_envs` SEPARATE `IsaacSO101Env` instances → each `_boot()` wants the Isaac
  `SimulationContext` **singleton** → 2nd instance fails / collapses to 1 → sheeprl's per-env
  buffers (sized `num_envs`, lines 478/543/634) mismatch the 1-env returns → `is_first`
  IndexError.
- `SyncVectorEnv` also steps sub-envs **sequentially** (`env0.step`, `env1.step`, …) — there is
  no way to map that onto ONE batched Isaac `env.step((N,6))`. So SyncVectorEnv must be
  **replaced**, not satisfied.

Required changes:
1. **New `IsaacSO101VectorEnv(gymnasium.vector.VectorEnv)`** (new file
   `sheeprl_plugin/isaac_vector_env.py`). Boots ONE Isaac `ManagerBasedRLEnv` with
   `num_envs=N` (reuse the existing `_boot` / `_translate_obs` logic from `isaac_env.py` —
   factor the shared parts out). API:
   - `num_envs=N`, `single_observation_space` (rgb (3,H,W)+state), `single_action_space` (6,),
     batched `observation_space`/`action_space`.
   - `reset()` → `(obs_dict batched (N,…), info)`; `step(actions (N,6))` →
     `(obs (N,…), reward (N,), terminated (N,), truncated (N,), info)`.
   - **Autoreset**: Isaac `ManagerBasedRLEnv` auto-resets terminated sub-envs internally; expose
     gymnasium autoreset semantics (return the reset obs + flag) so sheeprl's `dones_idxes`
     handling (dreamer_v3.py:640+) works.
   - Preserve the **singleton + `close()` no-op + os._exit** discipline from `isaac_env.py`
     (memory `wm-isaac-stall-resolved`) — the eval/test phase (dreamer_v3.py:765-767) rebuilds
     an env and reuses the backing singleton.
2. **Construction intercept** in `scripts/_wm_isaac_entry.py` (it already monkeypatches
   `gymnasium.vector` before importing sheeprl): patch `gymnasium.vector.SyncVectorEnv` so that
   when the env_fns build the Isaac env AND `num_envs>1`, it returns a single
   `IsaacSO101VectorEnv(num_envs=N)` instead of N copies. (dreamer_v3 references
   `gym.vector.SyncVectorEnv` at call time, so the patch takes.)
3. Keep `num_envs=1` on the **untouched** single-env path (zero regression). The vector path
   activates only at `num_envs>1`.

Verification protocol (GPU, ~3–6 boot cycles):
- `NUM_ENVS=2` smoke → confirm: no `is_first` IndexError; obs/action/reward shapes
  `(2,…)`; reward flows for both envs; a grad step completes; the train→`close()`→`test()`
  lifecycle survives (no `'scene'` attr crash, no atexit hang).
- Then `NUM_ENVS=4`; watch VRAM (Isaac N-parallel scales VRAM ~linearly — may need image 64 /
  batch tuning) + step-rate gain vs num_envs=1.

Risk: medium-high. Touches the same Isaac-singleton/sheeprl-lifecycle surface that produced the
2026-05-31 stall bug. **Do NOT commit unverified** — land only after the lifecycle survives on
GPU. Until then, `num_envs=1` (Fix 1) is the safe production path.
