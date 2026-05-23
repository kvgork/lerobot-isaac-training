# LoRA Sweep — Next Steps Plan

**Date:** 2026-05-22
**Parent plan:** [`2026-05-19-lora-autoresearch-plan.md`](2026-05-19-lora-autoresearch-plan.md)
**Status:** Phase 7 (GPU sweep) done. Hardware deploy NOT yet validated.

---

## Current State (verified 2026-05-22)

| Artifact | Value |
|----------|-------|
| Anchor ckpt | `outputs/overnight-smolvla-2026-05-15T210257-anchor/policy-smolvla/checkpoints/last/pretrained_model` (step 49500) |
| Anchor pc_success (open-loop) | **0.0824** (MSE 11.13 on 2-ep held-out) |
| Best LoRA trial | `outputs/autoresearch-lerobot-policy-smolvla-lora/trial_12/checkpoints/merged/pretrained_model` |
| Best config | `r=32, α=64, dropout=0.05, attn_qv, lr=3e-5, warmup=500, steps=5500` |
| Best pc_success (open-loop, merged ckpt) | **0.1492** (+82 % rel vs anchor) |
| Trainable params | 4.1 M / 454 M (0.9 %) |
| Sweep state dirs (3 sessions) | `.agent-state/lora-bash-20260522-{065242,091531-resume,124020-resume2}/` |
| Dashboard | `http://localhost:8501` — Autoresearch tab shows all 3 sessions |

**Key finding:** HF SmolVLA PEFT default (r=64, α=r) is NOT optimal. `r=32, α=2r` wins by 3 % rel. `alpha=2r` helps at low/mid ranks, hurts at high ranks.

**Key non-finding:** Closed-loop success rate. Open-loop MSE on held-out frames is a teacher-forced proxy — does NOT capture compounding rollout error. Absolute MSE = 5.7 ≈ ~60° per-joint avg error → real-robot rollout likely fails.

---

## Goals

1. Get a closed-loop success signal (sim or robot).
2. Lift absolute pc_success — current best is +82 % relative but still low absolute.
3. Establish a reproducible hardware deploy path so any future ckpt can be validated in ≤30 min.

---

## Phase 0 — Hardware Deploy Smoke (1 day)

**Goal:** validate that `trial_12/merged` runs at 30 Hz on a real SO-101 in DRY-RUN, producing sensible joint targets.

**Tasks:**
1. Install missing serial SDK:
   ```bash
   pixi run -e train-policy pip install scservo-sdk dynamixel-sdk
   ```
   Or pin into `pixi.toml` under `train-policy` feature.
2. Verify `robot-data-runner` is installed editable:
   ```bash
   pixi run sync-runner
   pixi run -e train-policy pip install -e src/robot-data-runner
   ```
3. Plug in SO-101 + U2D2 → identify `/dev/ttyACM0`. Run `lerobot-find-port`.
4. Calibrate arm (one-time): `lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0`.
5. DRY-RUN at 30 Hz for 30 s. Capture stdout log — look for:
   - Smooth action sequences (no NaN, no joint-limit hits, jitter < 0.05 rad between steps)
   - Inference latency < 33 ms/step (else rate-throttled)
   - Gripper command varies (not stuck open/closed)
6. If dry-run looks sensible: try `--execute --max-relative-target 3.0 --home-on-exit --duration-s 30` with workspace clear. Observe whether the arm tracks a pick-place trajectory.

**Acceptance:**
- Dry-run completes 900 steps (30 s × 30 Hz) without crash.
- Predicted actions stay within calibrated joint range.
- Inference rate ≥ 30 Hz steady-state.

**Risks:**
- SmolVLA inference on RTX 3080 may run < 30 Hz → drop control rate or run `--use_cache` style inference batching.
- Calibration mismatch between recorded dataset and live arm → joint offsets cause arm to move to wrong absolute pose.

**Time:** 4–6 h (mostly env install + calibration). No further training.

---

## Phase 1 — Closed-Loop Sim Eval (2–3 days)

**Goal:** replace open-loop MSE with a real `pc_success` from rollout. Required to compare future LoRA sweeps honestly.

**Why now:** CLAUDE.md open item — "Camera observation wiring deferred (`wrist_camera_rgb` / `overhead_camera_rgb` need `CameraCfg` in scene, Isaac Lab tutorial 04)". This is the gate to closed-loop scoring.

**Tasks:**
1. Wire `CameraCfg` into the SO-101 Isaac Lab scene per Isaac Lab tutorial 04. Two cameras: wrist + overhead.
2. Hook the scene into `tasks/pickplace.py` so rollouts emit RGB obs that match the dataset schema (`observation.images.<name>` keys).
3. Write `scripts/_closed_loop_eval.py` (sibling to `_open_loop_eval.py`):
   - Spin up Isaac Lab env headless.
   - Roll out the policy for N=20 episodes.
   - Score: task completion + episode length + intervention rate.
4. Replace `_open_loop_eval.py` call in `scripts/_run_autoresearch_lora.sh` with `_closed_loop_eval.py` (gated by `EVAL_KIND=closed_loop`).
5. Re-eval the 16 sweep ckpts with closed-loop scoring → real rank curve.

**Acceptance:**
- Closed-loop eval runs in < 5 min per ckpt.
- Anchor and trial_12 both produce closed-loop `pc_success` > 0 (i.e. complete at least 1 episode).
- Rank curve from closed-loop matches open-loop within ±20 % relative ordering (or surfaces a real disagreement → important finding).

**Risks:**
- Sim-to-real gap means closed-loop sim success may not predict real-robot success.
- Camera rendering in Isaac Lab is slow on RTX 3080 → 20 episodes may take 10 min not 5.
- USD asset needs the gripper friction + object physics tuned to match dataset.

**Time:** 2–3 days (Isaac Lab MDP wiring is the long pole).

---

## Phase 2 — Data Scaling (1–2 weeks elapsed, mostly idle time)

**Goal:** lift the absolute ceiling. 20 demos / 7491 frames is far below the typical VLA fine-tune dataset (100s of demos).

**Tasks:**
1. Collect 30 more SO-101 pick-place teleop demos with:
   - 3 object types (currently single)
   - 4 starting positions (varied)
   - 2 lighting conditions (natural + LED)
   - 2 backgrounds
2. Apply SAL/TED quality filter (skill `lerobot_dataset_quality`) → drop bottom 20 %.
3. Merge with existing `kvgork/so101-pickplace1` → new dataset `kvgork/so101-pickplace2`.

**Acceptance:**
- ≥ 25 high-quality episodes added (post-filter).
- New dataset passes `pixi run -e dashboard dashboard` Data Collection tab health-check (FPS, schema, episode-length distribution).

**Risks:**
- Teleop is human-time-bound; can't parallelize.
- Reality drift between sessions (calibration changes day to day).

**Time:** 5–10 h teleop + same-day processing.

---

## Phase 3 — Stronger Base + Re-LoRA (2–3 days)

**Goal:** LoRA can only adapt; if base is weak, ceiling is low. Push base SmolVLA further before re-doing the LoRA sweep.

**Tasks:**
1. Resume base SmolVLA full-FT from anchor step 49500. Push to 150k steps on `so101-pickplace2`.
   ```bash
   pixi run -e train-policy lerobot-isaac-train --target_arch smolvla \
     --dataset datasets/kvgork/so101-pickplace2 \
     --output_dir outputs/smolvla-anchor-v2 \
     --steps 150000 --batch_size 4 --cache_frames \
     -- --policy.pretrained_path=<anchor> --optimizer.lr=1e-5
   ```
   Estimated wall time: ~8 h on RTX 3080 (cached path, lr=1e-5 to avoid catastrophic forgetting).
2. Eval new anchor closed-loop. Acceptance: `pc_success_closed_loop ≥ 0.20`.
3. Update `programs/lerobot-policy-smolvla-lora.md` `entry_args` to point to new anchor.
4. Re-run full 16-config sweep (`scripts/_run_autoresearch_lora.sh`, 10 h budget).
5. Document rank curve on stronger anchor — expected: r=64 may regain optimum once base has more headroom.

**Acceptance:**
- New base anchor closed-loop pc_success > old LoRA best.
- New best LoRA trial closed-loop pc_success ≥ 0.30 (target: deployable threshold for unattended pick-place).

**Risks:**
- Catastrophic forgetting from longer FT — mitigated by lower lr (1e-5) and early-stop on val loss spike.
- New dataset distribution shift makes new anchor incompatible with old LoRA configs.

**Time:** 1 day train + 1 day eval + 10 h LoRA sweep ≈ 3 days.

---

## Phase 4 — DR Augmentation (1–2 days, parallelizable with Phase 3)

**Goal:** synthetic data via Isaac Lab DR replay. Already wired (`lerobot-isaac-synthetic` Phase 4a green per CLAUDE.md). Multiply real demos ~10×.

**Tasks:**
1. Identify the 5 best real demos via SAL/TED score.
2. Run `lerobot-isaac-synthetic` DR replay: 5 demos × 20 DR variants = 100 synthetic episodes.
3. Merge synthetic + real (skill `lerobot_mimicgen_bridge` handles dedup + tagging).
4. Re-train anchor (Phase 3 step 1) on merged dataset.

**Acceptance:**
- 100+ synthetic episodes generated.
- Merged dataset round-trip-loads in `LeRobotDataset` without schema errors.
- Anchor trained on merged set has closed-loop pc_success ≥ Phase 3 anchor (synthetic data should not regress).

**Risks:**
- DR variant range too aggressive → policy learns invalid distribution.
- USD physics drift between original demo recording and DR replay.

**Time:** ~1 day generate + roll into Phase 3 retraining.

---

## Phase 5 — Hardware Validation + Deploy (1–2 days)

**Goal:** ship a checkpoint usable for unattended pick-place on real SO-101.

**Tasks:**
1. Rebuild final ckpt (`Phase 3 best LoRA on Phase 3 anchor on Phase 2 data + Phase 4 synthetic`).
2. Dry-run on hardware (Phase 0 procedure) — capture stdout + intervention markers.
3. Execute-mode rollout, 10 trials, with safety clamp.
4. Measure: completion rate, intervention rate, mean episode length.
5. Compare hardware result vs Phase 1 closed-loop sim result — quantify sim-to-real gap.

**Acceptance:**
- 10 hardware trials, ≥ 5 complete pick-place without human intervention.
- Mean episode length within 1.5× of teleop demos.
- No safety incidents (no joint-limit collision, no E-stop trip).

**Risks:**
- Sim-to-real gap larger than expected → may need Phase 4 DR retune or real-world fine-tune step.
- Hardware drift between sessions invalidates calibration mid-rollout.

**Time:** 1 day calibrate + execute + 1 day analyze.

---

## Phase 6 — Tech Debt (parallel, 0.5 day)

Bug list discovered during this sweep:

1. **`best.json` not written on seeded-resume** when no trial beats `RESUME_BEST_METRIC`.
   Fix: write a stub `best.json` from the seed on script start when `SKIP_TRIALS > 0`.
2. **Plateau-stop too aggressive** for exploration sweeps. Default `PLATEAU_LIMIT=3` ended sweep 1 at trial 3. Bumped to 6 then 10 manually.
   Fix: make default in `programs/lerobot-policy-smolvla-lora.md` = 6.
3. **`attn_qkvo` doesn't fit in 35-min/trial budget.** Trials 10 & 11 hit timeout, gave partial-ckpt evals.
   Fix: either auto-extend `SECONDS_PER_EXP` for qkvo trials, OR drop qkvo from the pool given attn_qv consistently wins.
4. **CLAUDE.md SmolVLA throughput table is wrong for LoRA path.** Documented 10.1 step/s; actual = 2.85 step/s. PEFT gradient routing on frozen base adds ~3.5× overhead.
   Fix: append note to CLAUDE.md SmolVLA throughput table — "LoRA path is ~3.5× slower; budget 0.35 s/step not 0.1 s/step."
5. **Loss-proxy metric removed in favor of real eval.** No code remaining; close the loop_proxy fallback path or mark it deprecated in the script comment.

---

## Critical Path

```
Phase 0 (hardware smoke, 1 day)
     │
     ▼
Phase 1 (closed-loop sim, 2-3 days)
     │
     ├─► Phase 4 (DR augmentation, 1-2 days) ─┐
     │                                         │
Phase 2 (data scaling, idle calendar time)─────┤
     │                                         │
     ▼                                         ▼
Phase 3 (stronger base + re-LoRA, 3 days) ◄────┘
     │
     ▼
Phase 5 (hardware deploy, 1-2 days)
```

Total active engineering time: **~8 working days** to deploy-ready, assuming Phase 2 teleop is parallel with code work.

Phase 6 tech-debt items are <30 min each; bundle into a single commit.

---

## Exit Criteria

A checkpoint is considered "robot-ready" when ALL hold:

- Closed-loop sim pc_success ≥ 0.30 (Phase 1 metric).
- Hardware dry-run produces smooth, in-range actions for 30 s.
- Hardware execute-mode: ≥ 50 % task completion across 10 trials.
- Safety clamp (`--max-relative-target`) never trips during normal operation.

Until ALL four are met: only run with `--execute` under human supervision with hand on E-stop.
