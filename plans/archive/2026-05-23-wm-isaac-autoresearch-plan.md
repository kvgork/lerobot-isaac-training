# WM Isaac Autoresearch — HP Sweep Plan

> **Status: SUPERSEDED (2026-05-26).** Replaced by `plans/2026-05-24-wm-isaac-hp-trials-1to9.md`
> after v8 plan ran + 4 actor-collapse runs forced rewrite. Kept for history only.

**Date:** 2026-05-23
**Branch:** `feature/wm-isaac-env`
**Parent context:** v7 plateau diagnosis (`Grads/actor → 0.009`, policy collapsed before learning task) → v8 launched with patched entropy knobs (`ent_coef=0.01, min_std=0.3, replay_ratio=1`). Auto-research extends this single-trial intuition into a real HP sweep.

---

## Problem Statement

DreamerV3 on `env=isaac_so101` failed once (v5: weak reward) and converged
to a wrong fixed point once (v7: actor collapse). One manual fix is in
flight (v8) but covers a single point in the HP space. We need a sweep
that varies the four knobs most likely to control "does the actor escape
the random-policy attractor and learn task control".

---

## Search Space (10–12 trials, ~2.5 h each)

| Axis | Default | Sweep values | Rationale |
|------|---------|--------------|-----------|
| `algo.actor.ent_coef` | 3e-4 | **1e-4, 3e-4, 3e-3, 1e-2** | Direct entropy regularisation. v7 collapse at 3e-4 → bias toward higher values. |
| `algo.replay_ratio` | 0.5 | **0.25, 0.5, 1, 2** | Train compute per env step. Lower = more env data, slower convergence. Higher = more grad updates, risk of actor saturation. |
| `algo.actor.min_std` | 0.1 | **0.1, 0.3, 0.5** | Minimum action variance — protects against deterministic collapse late in training. |
| `algo.world_model.optimizer.lr` | 1e-4 | **3e-5, 1e-4, 3e-4** | Sanity check; v6 perf had `bf16-mixed` which sometimes needs lower lr. |

**Frozen** (out of sweep, fixed at v8 values):
- `algo.per_rank_batch_size=16`
- `env.num_envs=1` (Isaac Lab SimulationContext singleton constraint)
- `env.image_size=64`
- `env.max_episode_steps=300`
- `fabric.precision=bf16-mixed`
- `algo.world_model.discrete_size=32`
- `algo.world_model.stochastic_size=32`
- reward terms: `weight=10 distance_scale=0.4 ee_body_name=gripper_link`

**Out of scope for v1 sweep** (defer to v2 if v1 doesn't find a winner):
- World-model capacity (D, S) — orthogonal axis
- Episode length — likely needs paired tuning with reward shape
- Image size — RTX 3080 already at edge with 64
- Reward weight scaling — fixed at 25× v5 default

---

## Metric

**Primary:** `tb_Rewards/rew_avg` post-training-completion, scraped from
TensorBoard.

**Direction:** maximize (less negative = closer to task success).

**Baseline reference:**
- Random policy: ~-62 (verified v7)
- Reaching: ~-19 to -25
- Touching: ~-3 to -10
- Grasped: 0

A trial that ends with `rew_avg ≥ -50` (any improvement over random by
≥10) is a "promising" run. ≥-20 is a "strong actor". ≥-5 is "ready for
robot deploy".

**Secondary metrics (forensics, not ratchet):**
- `Loss/observation_loss` — should drop ≥100× by end of training; if
  not, WM didn't learn images.
- `Grads/actor` — if collapses to <0.01 before step 30k, actor saturated
  early (v7 failure mode).
- `State/post_entropy` — should decrease but not below ~5.0 (below =
  WM over-confident).
- `Time/sps_train` — wall-clock budget tracker.

---

## Per-Trial Budget

- Wall: 2.5 h ceiling
- Steps: 60 000 (reachable at ~7 step/s with replay_ratio=0.5; 30k at
  replay_ratio=2)
- Eval: post-train TB scrape (no separate eval rollout — env reward IS
  the eval signal here)
- Checkpoint: every 15 000 steps → 4 ckpts per trial

10 trials × 2.5 h = 25 h compute = 1 day budget.

---

## Sweep Schedule (pinned trial pool)

```
Trial 0  baseline                            ent=3e-4 rr=0.5 min_std=0.1 lr=1e-4
Trial 1  high entropy                        ent=1e-2 rr=0.5 min_std=0.1 lr=1e-4
Trial 2  mid entropy + min_std bump          ent=3e-3 rr=0.5 min_std=0.3 lr=1e-4
Trial 3  v8 config (high ent + min_std)      ent=1e-2 rr=1.0 min_std=0.3 lr=1e-4
Trial 4  low replay_ratio                    ent=3e-4 rr=0.25 min_std=0.1 lr=1e-4
Trial 5  high replay_ratio                   ent=3e-4 rr=2.0 min_std=0.1 lr=1e-4
Trial 6  high entropy + high min_std         ent=1e-2 rr=0.5 min_std=0.5 lr=1e-4
Trial 7  lower wm lr (bf16 safety)           ent=1e-2 rr=0.5 min_std=0.3 lr=3e-5
Trial 8  higher wm lr                        ent=1e-2 rr=0.5 min_std=0.3 lr=3e-4
Trial 9  ablate min_std at high entropy      ent=1e-2 rr=0.5 min_std=0.1 lr=1e-4
Trial 10 (optional) repeat winner alt seed
Trial 11 (optional) repeat second-best alt seed
```

**Ordering rationale**: spend first 4 trials on the entropy axis (likely
biggest effect), then test interaction with replay_ratio + min_std, then
wm lr sanity. Last two slots reserved for seed repeats of the top
performers.

---

## Implementation Plan

### Files to add

1. `programs/wm-dreamerv3-isaac-hp.md` — sister to
   `wm-dreamerv3-isaac.md` but with the search space + trial pool above
   encoded in the canonical autoresearch program format.

2. `scripts/_run_autoresearch_wm_isaac.sh` — generic bash sweep, mirror
   of `_run_autoresearch_wm.sh` (HDF5-replay-env variant) but targets
   `env=isaac_so101`. Differences from the existing wm sweep script:
   - Uses `_run_wm_isaac_overnight.sh` per trial (which already wires
     the AppLauncher-first entry + sync_env + reward knobs).
   - Sets per-trial `EXTRA_HYDRA` from the trial pool to drive the four
     swept knobs.
   - Reads back `Rewards/rew_avg` from TB after each trial completes,
     writes to `history.jsonl`.

3. `scripts/_scrape_tb_rew_avg.py` (or extend `_scrape_tb_to_history.py`)
   — pull `Rewards/rew_avg` (final value) from each trial's sheeprl run
   dir at `logs/runs/dreamer_v3/isaac_so101/<run_name>/version_0/`.

### Files to modify

- `scripts/_run_wm_isaac_overnight.sh` already supports `EXTRA_HYDRA` (added 2026-05-23 evening). No further changes needed.

### Per-trial cleanup

Each trial leaves:
- `outputs/wm-isaac-prod-<session>-trial<N>/` — Hydra run dir + .hydra/config.yaml
- `logs/runs/dreamer_v3/isaac_so101/<run_name>/version_0/checkpoint/ckpt_*.ckpt` — 4 ckpts
- `.agent-state/<session>/autoresearch/wm-isaac-prod/trial_<N>.log`
- TB events file

Disk per trial ≈ 2 GB. Sweep ≈ 20–25 GB. Verify free disk before launch.

---

## Execution Workflow

```
Phase A — baseline measurement (1 trial, separate from sweep)
   Run current v8 config to completion (in flight; 12 h).
   Use the resulting rew_avg as the autoresearch "baseline" anchor.

Phase B — sweep launch (10–12 trials × 2.5 h ≈ 25–30 h)
   bash scripts/_run_autoresearch_wm_isaac.sh \
       --program programs/wm-dreamerv3-isaac-hp.md
   Runs all trials sequentially (GPU is shared, can't parallelise).
   Persists state to .agent-state/<session>/autoresearch/wm-isaac-hp/

Phase C — analysis (30 min)
   Read history.jsonl, identify top-3 rew_avg trials.
   Cross-check Grads/actor + State/post_entropy:
     ✓ Grads/actor stays > 0.05 throughout = no premature collapse
     ✓ State/post_entropy floors > 5.0 = WM not over-confident
   Confirm winner is genuine, not metric exploit (e.g. reward hack).

Phase D — winner refinement (1 trial × 4 h)
   Re-run winner config with 2× steps (240 000) to verify the
   improvement isn't seed-noise.

Phase E — deploy candidate (1 trial × 2 h on hardware)
   `li-deploy-sync-wm` the winner ckpt to laptop.
   Dry-run + execute-mode 10-episode rollout.
```

---

## Stopping Rules

- **Per-trial timeout:** 2.5 h hard ceiling. Truncated trials still
  log whatever rew_avg they had at last TB scrape.
- **Plateau in sweep:** if 4 consecutive trials post rew_avg in same
  ±5 of baseline, abort sweep — likely all in the same failure regime
  + try a different search space.
- **Compute exhaustion:** track wall clock; abort if total > 30 h.

---

## Risks

| Risk | Mitigation |
|------|------------|
| All 10 trials show same -62 random plateau | Sweep didn't escape v7-class failure → widen search (bigger ent_coef, longer per-trial) |
| TB scrape fails on truncated trial | Fall back to last `policy_step=N, reward_env_0=X` line from train.log |
| Disk fills during sweep | Pre-flight: `df -h /home`; abort if < 30 GB free |
| sheeprl env factory hangs on some HP combo | Per-trial timeout catches it (rc=124); record + skip |
| Winner is overfit to single seed | Phase D reruns at seed=1337 for cross-check |

---

## Concrete Acceptance Criteria

The sweep is "successful" when ALL hold:

- ≥ 1 trial achieves `Rewards/rew_avg > -50` (a 10-point improvement
  over random baseline).
- The winner trial's `Grads/actor` stays ≥ 0.05 through step ≥30k
  (no premature collapse).
- The winner trial's `Loss/observation_loss` drops ≥ 50× from start
  (WM dynamics learning is healthy).
- Phase D re-run at alt seed reproduces rew_avg within ±10 of winner
  (not seed-noise).
- A merged-format checkpoint (after `li-deploy-sync-wm` staging) loads
  cleanly via `lerobot_isaac_deploy.wm_loader.load_dreamerv3` on the
  laptop.

---

## Out of Scope

- HP search via Optuna or learned proposer — deterministic 10-trial
  grid is sufficient for first iteration. Karpathy-style learned proposer
  for future v2 sweep when the right axes are confirmed.
- Multi-task curriculum (pick + place + insertion).
- Domain randomisation tuning — separate axis, handle in a follow-up
  sweep after a strong baseline exists.
- Closed-loop sim eval — env reward IS the eval; no need for the
  separate `_open_loop_eval.py`-style scoring.

---

## Cross-References

- `plans/2026-05-23-wm-isaac-env-plan.md` — parent C-phase plan
- `plans/2026-05-23-wm-isaac-tonight.md` — v6/v7 diagnosis + v8 fix list
- `programs/wm-dreamerv3-isaac.md` — single-trial program (current single config)
- `scripts/_run_autoresearch_wm.sh` — sister sweep for HDF5 replay env (template to copy from)
- `scripts/_run_wm_isaac_overnight.sh` — single-trial runner (sweep calls it per trial)
