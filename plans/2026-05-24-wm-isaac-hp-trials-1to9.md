# WM Isaac HP Sweep — Trials 1-9 Resume Plan

**Date:** 2026-05-24
**Branch:** `feature/wm-isaac-env`
**Parent plan:** `plans/2026-05-23-wm-isaac-autoresearch-plan.md`
**Status:** Trial 0 (`baseline`) launched 2026-05-24 05:59 — in flight as of
this writing. Session id: `wm-isaac-hp-20260524-055912`. ETA 9 AM.

---

## What trial 0 gives us

| Item | Where |
|------|-------|
| Trial-0 metrics | `.agent-state/wm-isaac-hp-20260524-055912/autoresearch/wm-isaac-hp/{history,best,plateau}.json` |
| Trial-0 ckpts | `logs/runs/dreamer_v3/isaac_so101/2026-05-24_05-*_dreamer_v3_isaac_so101_42/version_0/checkpoint/` |
| Trial-0 TB | same dir, `events.out.tfevents.*` |
| Trial-0 log | `.agent-state/.../trial_0_baseline.log` |

The interesting reading: does sheeprl DreamerV3 default (`ent_coef=3e-4`)
escape the actor-collapse at full 60 k steps, or does it collapse like
v7/v8? Trial-0 is the ground-truth for "is the failure HP-fixable at
all, or is it deeper".

---

## Remaining trial pool (9 trials, 27 h compute)

Pool stored in `scripts/_run_autoresearch_wm_isaac.sh:TRIAL_POOL`. Order
and rationale repeated here for resume planning:

| # | Label | ent | rr | min_std | wm.lr | STEPS | Why |
|---|-------|-----|-----|---------|-------|-------|-----|
| 1 | high-entropy | 1e-2 | 0.5 | 0.1 | 1e-4 | 60000 | bumps ent_coef 33× vs default — direct test of v8-style fix at default rr |
| 2 | mid-ent-min_std | 3e-3 | 0.5 | 0.3 | 1e-4 | 60000 | 10× ent + min_std bump combined |
| 3 | v8-config | 1e-2 | 1.0 | 0.3 | 1e-4 | 60000 | exact v8 config but with 60 k steps not 36 k (v8 hit timeout early) |
| 4 | low-replay | 3e-4 | 0.25 | 0.1 | 1e-4 | 60000 | replay_ratio↓: more env data, slower gradient saturation |
| 5 | high-replay | 3e-4 | 2.0 | 0.1 | 1e-4 | 30000 | replay_ratio↑ — opposite test; 30 k steps because more compute per step |
| 6 | high-ent-high-min_std | 1e-2 | 0.5 | 0.5 | 1e-4 | 60000 | max-exploration combination |
| 7 | low-wm-lr | 1e-2 | 0.5 | 0.3 | 3e-5 | 60000 | bf16-mixed sometimes needs lower lr |
| 8 | high-wm-lr | 1e-2 | 0.5 | 0.3 | 3e-4 | 60000 | opposite direction |
| 9 | ablate-min_std | 1e-2 | 0.5 | 0.1 | 1e-4 | 60000 | isolate the min_std effect by repeating trial 1 with default min_std (already in trial 1; this is the duplicate slot — drop if trial 1 collapses cleanly) |

---

## Resume command (one-shot, 27 h budget)

```bash
# Read trial 0's metric to seed the ratchet.
TRIAL0_BEST=$(jq -r .metric_value \
    .agent-state/wm-isaac-hp-20260524-055912/autoresearch/wm-isaac-hp/best.json \
    2>/dev/null || echo "")

cd ~/workspaces/lerobot-isaac-training
SESSION_ID=wm-isaac-hp-20260524-055912 \
SKIP_TRIALS=1 \
MAX_TRIALS=10 \
PLATEAU_LIMIT=4 \
RESUME_BEST_METRIC="$TRIAL0_BEST" \
  bash scripts/_run_autoresearch_wm_isaac.sh \
       > /tmp/wm_hp_resume.log 2>&1 &
```

Same session id reuses trial-0's `history.jsonl` (appends rows 1-9).
`SKIP_TRIALS=1` skips the already-completed trial 0. Plateau counter
inherits from trial 0's seed via `RESUME_BEST_METRIC`.

---

## Sub-budget splits (if 27 h block isn't available)

### Option A — split across 3 nights (9 h each)

```bash
# Night 1 (trials 1-3 — entropy axis)
SESSION_ID=wm-isaac-hp-20260524-055912 SKIP_TRIALS=1 MAX_TRIALS=4 \
  bash scripts/_run_autoresearch_wm_isaac.sh

# Night 2 (trials 4-6 — replay_ratio + max-exploration axes)
SESSION_ID=wm-isaac-hp-20260524-055912 SKIP_TRIALS=4 MAX_TRIALS=7 \
  RESUME_BEST_METRIC=<best after night 1> \
  bash scripts/_run_autoresearch_wm_isaac.sh

# Night 3 (trials 7-9 — wm.lr axis + ablation)
SESSION_ID=wm-isaac-hp-20260524-055912 SKIP_TRIALS=7 MAX_TRIALS=10 \
  RESUME_BEST_METRIC=<best after night 2> \
  bash scripts/_run_autoresearch_wm_isaac.sh
```

### Option B — drop a trial if the early ones already show a clean winner

If trials 0-3 produce one trial with `rew_avg > -50` AND
`Grads/actor ≥ 0.05`, stop the sweep early. Skip directly to Phase D
of the parent plan (winner refinement at 240 k steps, alt seed).
Saves 18+ h.

### Option C — switch to a learned proposer when 4+ trials are in

When `history.jsonl` has ≥4 rows, the autoresearch infrastructure can
also be driven by an LLM proposer (the existing
`autoresearch-ml-proposer-worker` agent) instead of the deterministic
pool. Use this when the grid feels exhausted and the next mutation
should be axis-discovered, not pre-encoded.

---

## What to do after each batch finishes

1. **Check the dashboard's Autoresearch tab.** It auto-discovers the
   session at `.agent-state/wm-isaac-hp-20260524-055912/`.
2. **Look at the `Grads/actor` column in history.** Any trial where it
   collapses to < 0.05 is a v7/v8-class failure. Note the (ent_coef,
   replay_ratio) combo that DIDN'T collapse.
3. **Cross-check rew_avg vs forensic columns.** A trial with rew_avg
   = -45 but Grads/actor = 0.001 is a fake winner (saturated policy
   that happened to land in a slightly-less-bad fixed point).
4. **Write findings to** `docs/research/wm-isaac-hp-sweep-results.md`
   (new file). Include the rew_avg curve, the actor-grad curve, and
   the winner's combo. Stable artifact for paper / blog.

---

## If ALL 10 trials collapse (worst case)

The plan's "Hard-stop early" criterion fires at trial 5 — but if the
sweep completes anyway and no trial achieves `rew_avg > -58`, the
sweep regime itself is wrong. Next moves:

1. **Wider HP grid** (v2 sweep). Try `ent_coef ∈ [0.03, 0.1]`,
   `min_std ∈ [0.5, 0.8]`, longer trials (200 k steps).
2. **Sparse reward instead of dense.** Remove progress_reward, add
   `success_bonus` with a terminal condition (object lifted > 5 cm +
   inside basket). DreamerV3 sometimes does better with sparse signals
   because they force exploration. Requires `terminations.py` work.
3. **Auxiliary exploration bonus** (RND, curiosity). Not in default
   sheeprl DreamerV3; would need a small impl.
4. **Switch off DreamerV3** to a different baseline (e.g. PPO from
   `sheeprl.algos.ppo`) to check whether the issue is task or actor
   algorithm.
5. **Reduce horizon.** `max_episode_steps=300` is long for a static
   pick-place. Dropping to 100 makes each episode terminal sooner →
   actor sees terminal reward differences more.

---

## Tie-in to the deploy ladder

When a winner emerges:

```bash
li-deploy-sync-wm \
    --sheeprl-run-dir <winner trial's run dir> \
    --hydra-cfg-dir   <winner trial's output dir>/.hydra \
    --label wm-isaac-hp-winner-trialN

# On laptop:
li-deploy-session \
    --policy-path ~/workspaces/lerobot-isaac-deploy/checkpoints/wm/wm-isaac-hp-winner-trialN \
    --dataset-root ~/workspaces/lerobot-isaac-deploy/datasets/so101-pickplace1 \
    --port /dev/ttyACM0 \
    --duration-s 30 -v
```

DRY-RUN first. If joint targets look sensible, add `--execute
--max-relative-target 1.0 --home-on-exit` for the actual robot test.

---

## Exit Criteria

Sweep is "successful" when ALL hold:

- ≥ 1 trial achieves `rew_avg > -50` (10-point improvement vs random).
- That trial's `Grads/actor ≥ 0.05` (not a saturated-policy false-win).
- That trial's `Loss/observation_loss ≤ 0.5` (WM dynamics healthy).
- Re-run at alt seed reproduces within ±10 rew_avg.
- Resulting ckpt loads cleanly via `wm_loader.load_dreamerv3` on the
  laptop (no API breakage).

---

## Cross-References

- Parent plan: `plans/2026-05-23-wm-isaac-autoresearch-plan.md`
- Trial-0 session: `.agent-state/wm-isaac-hp-20260524-055912/`
- Sweep script: `scripts/_run_autoresearch_wm_isaac.sh`
- Single-trial runner (called per trial): `scripts/_run_wm_isaac_overnight.sh`
- AppLauncher-first entry: `scripts/_wm_isaac_entry.py`
- Multi-dataset (orthogonal axis, separate plan): `plans/2026-05-23-multi-dataset-training.md`
