# Phase 3 — 3-way real-arm model comparison (ACT vs SmolVLA vs vla_jepa)

**Created:** 2026-07-12 · **Gate:** HUMAN + physical SO-101 (not automatable headless).
**Parent:** `plans/2026-07-11-combined-today-plan.md` (Phase 3). **Branch context:** deploy runs
from `src/robot-data-runner` (branch `feat/deploy-runner-hardening`, commit `40d3198`).

## Goal
Rank the three real-data deploy candidates by **verified real-arm success rate** on the SO-101
pick-and-place task, and answer the project question: **does the world-model policy (`vla_jepa`)
beat straight BC (ACT / SmolVLA) on the real arm?** Output = a ranked result + a decision on the
next data/technique lever (Run 3a vs 3b).

## Candidates (all trained, checkpoints on disk)
| Arch | Loss | Checkpoint | Open-loop MSE proxy (2026-07-11) | HW status |
|------|------|-----------|----------------------------------|-----------|
| ACT-15k | 0.168 @15k | `outputs/act_real_so101_15k/checkpoints/015000/pretrained_model` (206 MB) | 52.24 | ✅ succeeded 2026-07-04 (baseline) |
| SmolVLA | 0.119 @20k | `outputs/smolvla_real_so101/checkpoints/020000/pretrained_model` (906 MB) | 40.89 | not yet HW-tested |
| vla_jepa | @20k | `outputs/vla_jepa_real_so101/checkpoints/020000/pretrained_model` | 41.10 | not yet HW-tested |

> Open-loop MSE is a proxy only (per-frame, no temporal alignment, under-rates ACT's
> chunking+ensembling, eval eps were seen in training). It ties SmolVLA≈vla_jepa > ACT — the
> real arm is the decider. Data: `outputs/ol_eval_{act,smolvla,vla_jepa}.json`.

## Preconditions (do before any episode)
1. **Runner installed from the committed branch.** `git -C src/robot-data-runner status` → on
   `feat/deploy-runner-hardening` (merge to main first if desired). `pixi run sync-runner` +
   `pip install -e src/robot-data-runner`; needs `lerobot[feetech]` (scservo_sdk).
2. **Pre-flight hardware check.** `robot-data-run-check` (or `robot-data-check --connect`) —
   servos respond, calibration present under the ids the runner uses (`so101_follower`), overhead
   camera enumerates at `/dev/video4` (D435 RGB).
3. **Physical scene matches training.** Overhead-camera framing identical to `so101-pickplace-new`
   recording; die + cup at the trained positions; workspace clear.
4. **Safety armed.** Clamp ladder starts at `--max-relative-target 1.0` for the first live episode
   per policy, ramp 1→3→5° once motion looks sane. `--home-on-exit` always. E-stop reachable.

## Protocol — paired, fair, verified
- **N ≥ 20 episodes per candidate**, `--duration-s 20`, `--rate-hz 30` (matches training fps).
- **Paired die positions.** Use the SAME ordered set of ~20 die start positions for all three
  candidates (record positions once; re-place per episode). Removes position luck from the ranking.
  Given the 50-demo narrow-tolerance finding, keep the set centered on the trained position with a
  modest spread (do NOT start with a hard out-of-distribution sweep — that measures generalization,
  a separate question deferred to Run 3a).
- **Between-episode reset** via `physical-reset-agent` (homes arm, verifies reset, e-stops on
  fault) — or manual re-home + re-place.
- **Verdict per episode = binary** via the `binary_success_verifier` skill / the runner's
  `--execute` success signal: object released resting inside the cup = success. Log
  success/fail + any human intervention + a one-line note (where it failed: reach / grasp / carry /
  release).
- Optionally record each rollout to a LeRobotDataset (`robot-data-run` logs episodes) for later
  DAgger / diagnosis.

**Per-candidate command (template):**
```bash
robot-data-run \
  --policy-path outputs/<candidate>/checkpoints/<step>/pretrained_model \
  --dataset-root datasets/local/so101-pickplace-new \
  --port /dev/ttyACM0 --camera overhead=/dev/video4,640,480 --id so101_follower \
  --execute --max-relative-target 5.0 --rate-hz 30 --duration-s 20 --home-on-exit
```
(First live episode of each: `--max-relative-target 1.0`, watch, then ramp.)

## Metrics + how to read them
- **Primary:** verified success rate = successes / N per candidate. With N=20, the 95% CI is ±~22pp
  at 50% — treat gaps < ~3/20 episodes as a tie, not a ranking.
- **Secondary:** failure-stage histogram (reach/grasp/carry/release), intervention count, mean
  episode length to success.
- **The WM question:** does `vla_jepa` ≥ the better BC (SmolVLA)? A tie still matters — it shows the
  WM policy is deploy-viable at 10 GB, the stated long-term goal.

## Decision gates
| Outcome | Next |
|---------|------|
| Any candidate ≥ ~60% | **Run 3a** — scale to **150+ diverse demos** (25/bin, ±45° yaw, multi-object) on that technique — the standing top lever toward the HF-blog ~90% regime. |
| All < ~40%, narrow-tolerance confirmed | **Run 3a** anyway (data diversity is the diagnosed bottleneck), OR **Run 3b** = DAgger / residual-RL if the failure is control-precision not coverage. |
| vla_jepa ≫ BC | WM-policy path validated → invest there (online DreamerV3 / HIL-SERL, Route C). |
| ACT/SmolVLA ≫ vla_jepa | BC is enough for this task → prioritize data scale over WM. |

## Risks / notes
- **50-demo narrow spatial tolerance** — candidates replay ~the trained-mean trajectory; success is
  position-sensitive. The paired centered-position protocol controls for this; generalization is a
  separate Run 3a question.
- **vla_jepa inference** drops the WM (train-only); open-loop eval already confirmed the
  WM-dropped checkpoint loads + runs, but real closed-loop is unverified.
- **Camera framing drift** — a shifted overhead cam silently tanks every candidate equally; verify
  against a training frame before starting.
- **Sim-trained policies are NOT in this bake-off** — sim2real ≈ 0 for this task; only the three
  real-data candidates compete.

## Session 1 (2026-07-25) — stopped mid-ACT, state saved

- **Preconditions all passed** after fixes: record env upgraded to lerobot 0.6.0 (pins mirrored
  from train-policy: transformers 5.5.4, num2words 0.5.14, accelerate 1.14.0 — open-ended pip
  constraint spins the resolver, use exact pins); all 3 candidates load (ACT 51.6M / SmolVLA
  450M / vla_jepa 2.28B WM-dropped); follower + D435 green (leader /dev/ttyACM1 unplugged —
  only needed for demo recording); overhead = /dev/video4.
- **CAMERA HAD DRIFTED MASSIVELY vs training: 62px/−98px + 40% darker.** Caught only after 3
  scattered-stage failures. Realigned iteratively (ORB median-feature-shift metric) to
  dx −5.4 / dy +4.9; brightness 78/101 accepted (uniform across candidates). **NEW MANDATORY
  PRECONDITION: quantitative framing gate (feature shift <10px) BEFORE episode 1 — the
  eyeball blend composite was not acted on and 6 episodes were burned.**
- **Clamp ladder result: scoring clamp = 5.0.** At 1.0 the arm cannot complete reach in 20s;
  at 3.0 reach+grasp work but carry starves. Shakedown (non-scoring): 1.0 ×2, 3.0 ×2, all FAIL.
- **ACT-15k score so far: 0/4** (grasp ×3, carry/release ×1) + ep 5 ran clean but UNVERIFIED
  (stopped before verdict). Mixed stages at center spots even post-realign — consistent with
  the narrow-tolerance risk; brightness deficit may contribute.
- Scoreboard: `outputs/bakeoff-20260725/scoreboard.md`. Spots: 1-8 center, 9-16 ±2cm NSEW,
  17-20 ±3cm diag. **Resume: verify/rerun ep5 → ACT 6-20 → SmolVLA ×20 → vla_jepa ×20,
  clamp 5.0, same spot order; re-run the framing gate at session start.**

## Related
- `plans/2026-07-11-combined-today-plan.md` (Phase 3 origin) · `plans/2026-06-28-act-real-campaign-plan.md`
- memory: `[[act-real-campaign-result]]` (deploy cmd + first HW success + mapper fixes) ·
  `[[vla-jepa-rtx3080-finetune-recipe]]` · `[[detach-long-training-jobs]]`
- agents: `binary_success_verifier` (skill), `physical-reset-agent`, `rollout-executor-agent`
