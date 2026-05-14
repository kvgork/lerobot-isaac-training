# Post-AR Diffusion Saturation — Next Steps Plan

**Created:** 2026-05-14
**Trigger:** 3rd diffusion autoresearch run (6 trials × 30 min) confirmed
the open-loop action-MSE proxy is saturated. Best `pc_success` stuck at
~3.42e-4 across baseline / lr_down / lr_up / batch_up / weight_decay /
seed_swap. Loss floor 0.040 on the SO-101 dataset.
**Goal:** find a metric / model / data axis that actually discriminates,
then iterate.

---

## Status (anchor)

| Trial | Op | bs | lr | seed | loss | pc_success |
|---|---|---|---|---|---|---|
| 0 | baseline | 8 | 1e-4 | 42 | **0.040** | **0.00034158** |
| 1 | lr_down | 8 | 5e-5 | 1337 | 0.046 | 0.00034140 |
| 2 | batch_up | 16 | 1e-4 | 42 | 0.040 | 0.00034151 |
| 3 | weight_decay | 8 | 1e-4 | 42 | 0.040 | 0.00034153 |
| 4 | lr_up | 8 | 3e-4 | 42 | 0.046 | 0.00034155 |
| 5 | seed_swap | 8 | 1e-4 | 7 | 0.044 | 0.00034141 |

Plateau 5/3 (would auto-stop next iteration). Best config = baseline.

Conclusions:
- `lr=1e-4` confirmed near-optimum (both ±3× regress).
- `bs=8` ≈ `bs=16` — capacity not the bottleneck.
- `weight_decay=1e-4` no effect on this dataset.
- Open-loop action-MSE proxy is dominated by the 6-DOF action scale (~50° range); residual differences invisible.

---

## Next-step Options (priority order)

### A. Closed-loop real-arm eval (highest signal)

The only remaining discriminator. Open-loop MSE plateaued; closed loop
will surface failure modes (overshoot, oscillation, stuck states) that
MSE can't.

```bash
pixi run sync-runner
pixi run -e train-policy pip install -e src/robot-data-runner

# Pre-flight first
robot-data-run-check \
    --policy-path outputs/autoresearch-lerobot-policy-diffusion/trial_0/checkpoints/last/pretrained_model \
    --dataset-root datasets/kvgork/so101-pickplace1

# Dry-run on bench (clamp-mounted SO-101)
robot-data-run \
    --policy-path .../trial_0/checkpoints/last/pretrained_model \
    --port /dev/ttyACM0 \
    --dataset-root datasets/kvgork/so101-pickplace1 \
    --camera d435_rgb=/dev/video0,640,480 \
    --duration-s 30 -v

# Execute, tight clamp
robot-data-run --execute --max-relative-target 3.0 --home-on-exit ...
```

**Predicate for "discriminating":** if the arm reaches the source-object
pick in ≥30% of 10 trials, the proxy was lying — train more / mutate more.
If 0% success across all 6 AR-best checkpoints, the dataset/architecture
itself is the limit, not hyperparameters.

**Output schema for hardware eval:**

Write per-trial JSON to `outputs/hardware-eval/<run>/<trial>.json`:
```json
{
    "trial": "ar-v3-trial-0",
    "policy_path": "...",
    "n_episodes": 10,
    "n_success": 3,
    "pc_success_closed_loop": 0.30,
    "mean_episode_length_s": 4.2,
    "failure_modes": {"overshoot": 4, "stuck": 2, "wrong_object": 1}
}
```

Add a new `Hardware Evaluation` tab variant to the dashboard later.

---

### B. World-model program (different metric)

Switch axis: train the DreamerV3 / LeWM world model on the same data.
Different metric (`recon_loss`, `pred_loss`) — different surface,
potentially still discriminating.

```bash
bash scripts/run_autoresearch.sh --program dreamerv3 --bash
# or interactive:
/autoresearch programs/wm-dreamerv3.md --type ml_model
```

DreamerV3 budget: 1 h × 8 trials ≈ 8 h. Run overnight.

**Why this matters:** if the world model also saturates at low loss
without behaviour improvement, we know the dataset (not the algo) is the
bottleneck. If WM shows clear loss spread, the diffusion ceiling is
specific to the policy class.

---

### C. Sim-augmented dataset (sim2real axis)

Use the Isaac DR replay path to 5×-augment the dataset, re-train
diffusion, re-eval.

```bash
# 1. Generate synthetic
bash scripts/run_full_pipeline.sh \
    --train-minutes 0 --skip-policy --skip-worldmodel --skip-eval \
    --n-synthetic 20    # 20 source eps × 5 variants = 100 synthetic eps

# 2. Merge real + sim_dr into one dataset
.pixi/envs/train-policy/bin/python -c "
from lerobot_isaac_synthetic.merge_utilities import merge_datasets
merge_datasets(
    real_path='datasets/kvgork/so101-pickplace1',
    sim_paths=['outputs/full-pipeline-.../synthetic'],
    output_path='datasets/kvgork/so101-pickplace1-sim-aug',
    sim_weight=0.5,
)
"

# 3. Re-run diffusion AR on the merged dataset
DATASET=datasets/kvgork/so101-pickplace1-sim-aug \
    SECONDS_PER_EXP=1800 TRIALS=6 \
    bash scripts/_run_autoresearch_diffusion.sh
```

**Predicate:** sim2real-augmented dataset should reduce action MSE on
held-out REAL episodes if the synthetic data is on-distribution.

---

### D. Architecture swap (different policy family)

Try `smolvla` and `act` programs separately. Each has its own hyperparam
sweet spot (per `programs/_domain_knowledge.md` §9).

```bash
/autoresearch programs/lerobot-policy-smolvla.md --type ml_model
/autoresearch programs/lerobot-policy-act.md --type ml_model
```

Order: SmolVLA first (pretrained = fast convergence), then ACT (chunked
attention = faster inference). Compare best closed-loop success across
the three policy families before declaring a winner.

---

### E. Longer per-trial budget

If options A–D don't move the needle and you want more autoresearch
signal: bump per-exp budget to 2h (per the canonical
`programs/lerobot-policy-diffusion.md` spec). 6 trials × 2h = 12h.

```bash
SECONDS_PER_EXP=7200 TRIALS=6 \
    bash scripts/_run_autoresearch_diffusion.sh
```

Train loss usually drops another 30-50% past 4500 steps. If still
saturated → the dataset is the ceiling.

---

## Recommended Sequence

1. **A** (closed-loop hardware) — fastest signal on whether AR ranking is meaningful at all. **Today / tomorrow.**
2. **C** (sim augmentation) — if A shows non-zero success but variance high, augmentation is the next lever. **This week.**
3. **D** (SmolVLA / ACT) — orthogonal axis. **Next week.**
4. **B** (DreamerV3 AR) — overnight while A/C iterate. Different surface for sanity check.
5. **E** (2h trials) — only if A says ranking is meaningful and we want stronger checkpoints.

---

## Open Questions to Resolve

- Does `lerobot-train` save the final-step checkpoint when watchdog
  SIGKILLs? (Verify by checking that the highest-numbered checkpoint
  matches the last `Saving …` line in `policy_train.log`.)
- Does `robot-data-run-check` correctly load all 6 AR-best checkpoints
  on the train-policy env? (Run as part of step A pre-flight.)
- Is there a way to inject the closed-loop `pc_success` back into the
  autoresearch `history.jsonl` so the loader can rank by it? (Patch
  `_run_autoresearch_diffusion.sh` to call `robot-data-run` after each
  trial; gate on `--execute=false` for safety.)

---

## Artefacts Produced So Far

- `.agent-state/20260514-114628-autoresearch-diffusion/...` (5-min smoke)
- `.agent-state/20260514-121101-autoresearch-diffusion/...` (1st 30-min × 3)
- `.agent-state/20260514-141950-autoresearch-diffusion/...` (2nd 30-min × 3)
- `.agent-state/20260514-161705-autoresearch-diffusion/...` (3rd 30-min × 6, plateau)
- `outputs/autoresearch-lerobot-policy-diffusion/trial_{0..5}/checkpoints/` (6 best checkpoints)
- 3 snapshots under `outputs/snapshots/2026-05-14T*ar-diffusion*` for compare
- Dashboard Autoresearch tab has all 12 trials accumulated

---

## Files Touched / To Watch

- `programs/lerobot-policy-diffusion.md` (canonical program; update if budget changes)
- `programs/_domain_knowledge.md` (operator priority — if findings invalidate priority order, edit here)
- `scripts/_run_autoresearch_diffusion.sh` (grid is hardcoded; extend if option E chosen)
- `docs/runbook/10-deploy-to-hardware.md` (step A walkthrough)
- `docs/pipeline-overview.md` §Stage H, I, F (autoresearch / deploy / eval)
