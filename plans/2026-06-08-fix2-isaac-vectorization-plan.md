# Plan: Fix 2 — true num_envs>1 vectorization for online Isaac DreamerV3

**Status:** spec'd, NOT implemented. `num_envs=1` (Fix 1, committed `5789100`) is the safe
production path until this lands + is GPU-verified.
**Goal:** use Isaac Lab's native N-parallel sim so DreamerV3 collects ~N× faster (the
throughput unlock for multi-hour staged-reward runs — see
`plans/2026-06-08-staged-reward-tuning-plan.md`).
**Risk:** medium-high — touches the Isaac-singleton + sheeprl-lifecycle surface that caused the
2026-05-31 stall bug (memory `wm-isaac-stall-resolved`). Do NOT commit unverified.

---

## Root cause (exact)

`sheeprl/algos/dreamer_v3/dreamer_v3.py:384-398`:
```python
vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
envs = vectorized_env([make_env(..., rank*num_envs+i, ...) for i in range(cfg.env.num_envs)])
```
1. Builds `num_envs` SEPARATE `IsaacSO101Env` instances. Each `_boot()` needs the Isaac
   `SimulationContext` **singleton** → the 2nd instance can't boot a 2nd sim → effectively 1
   env materializes while sheeprl sized its per-env buffers (dreamer_v3.py:478/543/634) for N →
   crash `dreamer_v3.py:608 is_first IndexError: index 1 out of bounds, size 1`.
2. `SyncVectorEnv` steps sub-envs **sequentially** (`env0.step`, `env1.step`, …). There is no
   mapping from that onto ONE batched Isaac `env.step((N,6))`. So `SyncVectorEnv` must be
   **replaced**, not satisfied.

Current `IsaacSO101Env` is single-env by design: non-batched spaces, `step()` collapses Isaac's
batch to env-0 via `_scalar()`. `num_envs` is passed to Isaac internally but the output is
collapsed — so to sheeprl it is always 1 env.

---

## Required changes

### 1. New `IsaacSO101VectorEnv(gymnasium.vector.VectorEnv)`
New file `src/lerobot-isaac-adapters/src/lerobot_isaac_adapters/sheeprl_plugin/isaac_vector_env.py`.
Boots ONE Isaac `ManagerBasedRLEnv` with `num_envs=N` and returns batched results. Reuse the
boot + obs-translation logic from `isaac_env.py` — **factor the shared parts out** (boot,
camera probe, `_translate_obs` per-env) into a helper both classes import, to avoid drift.

API contract:
- attrs: `num_envs=N`, `single_observation_space` (Dict rgb (3,H,W) + state (state_dim,)),
  `single_action_space` (Box (6,)), batched `observation_space` / `action_space`.
- `reset()` → `(obs_dict batched (N,…), info)`.
- `step(actions (N,6))` → `(obs (N,…), reward (N,), terminated (N,), truncated (N,), info)`.
  Do NOT `_scalar()`-collapse — return full per-env tensors as numpy.
- **Autoreset:** Isaac `ManagerBasedRLEnv` auto-resets terminated sub-envs internally each
  step. Expose gymnasium VectorEnv autoreset semantics so sheeprl's `dones_idxes` reset
  handling (dreamer_v3.py:640+) sees the right obs/flags.
- **Preserve the singleton discipline** from `isaac_env.py`:
  - `close()` = NO-OP on the backing singleton (sheeprl calls `envs.close()` then `test()` at
    dreamer_v3.py:765-767, which rebuilds an env reusing `_GLOBAL_BACKING_ISAAC_ENV`).
  - AppLauncher-once probe (don't double-launch Kit).
  - real teardown only at process exit via `os._exit()` in `_wm_isaac_entry.py`.

### 2. Construction intercept (`scripts/_wm_isaac_entry.py`)
It already monkeypatches `gymnasium.vector` BEFORE importing sheeprl (for 0.5.8 compat). Add a
patch so `gymnasium.vector.SyncVectorEnv`, when its env_fns build the Isaac env AND
`num_envs>1`, returns a single `IsaacSO101VectorEnv(num_envs=N)` instead of N copies.
dreamer_v3 references `gym.vector.SyncVectorEnv` at call time → the patch takes. Gate strictly
on the isaac_so101 env + num_envs>1 so no other env path is affected.

### 3. Zero-regression gate
`num_envs=1` keeps using the **untouched** single-env `IsaacSO101Env`. The vector path
activates only at `num_envs>1`. The launcher default stays `NUM_ENVS=1` until this is verified;
flip to a higher default only after sign-off.

---

## Verification protocol (GPU, ~3–6 boot cycles, ~1h)

1. `NUM_ENVS=2` smoke (`scripts/_run_wm_isaac_overnight.sh`, short STEPS, SECONDS_PER_EXP cap):
   - no `is_first` IndexError; obs/action/reward shapes are `(2,…)`.
   - reward flows for BOTH envs; a grad step completes; `Loss/*` logged.
   - **train → `envs.close()` → `test()` lifecycle survives** — no
     `'ManagerBasedRLEnv' object has no attribute 'scene'`, no atexit Kit hang, clean
     `os._exit`.
2. `NUM_ENVS=4`: watch VRAM (Isaac N-parallel scales VRAM ~linearly on the RTX 3080 10 GB — may
   need image 64 / batch tuning) and measure step-rate gain vs num_envs=1 (baseline ~35
   steps/min single-env).
3. Only then: commit + raise the launcher default; update memory `wm-isaac-num-envs-bug`.

---

## Gotchas / watch-list
- `learning_starts ≥ num_envs × per_rank_sequence_length` (memory
  `dreamerv3-learning-starts-rule`) — at num_envs=4, seq 64 → ≥256; default 1024 fine.
- VRAM is the likely num_envs ceiling on the 3080 (cameras × N). Probe 2 → 4 → 8.
- Keep all changes behind the num_envs>1 branch; never regress the working single-env path.
- If sheeprl's VectorEnv expectations fight the single-batched-env model beyond a few
  iterations, fall back: keep `num_envs=1` and revisit with a sheeprl version that supports
  pre-vectorized envs natively.

## Not a blocker
DreamerV3 trains fine at num_envs=1 (sample-efficient). Fix 2 is purely a wall-clock/throughput
win. Prioritise only when staged-reward runs need faster data collection.
