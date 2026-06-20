# Plan — Toward a Level-3 (Agentic / Self-Directing) Training Pipeline

**Date:** 2026-06-19
**Status:** proposal / roadmap — claims ground-truthed against the codebase 2026-06-19 (master-project-orchestrator audit; see "Audit verification" below)
**Author:** orchestration pipeline (master-project-orchestrator)
**Companion plan:** `~/tools/claude_code/plans/2026-06-19-autonomous-training-skills-agents-plan.md` (the harness skills/agents that drive this)

---

## Context & framing

Research synthesis (Second Brain): `05-Wiki/synthesis/2026-06-19-autonomous-training-pipeline-agentic-layer.md` — built on NVIDIA GEAR's **ENPIRE** + ASL / AgentEvolver / AceGRPO (2026).

**Thesis.** A *fully autonomous* training pipeline = today's pipeline-automation (MLOps L1/L2) **plus an agentic recipe-designer layer** that closes the loop on the three jobs a human still owns here: (1) curriculum/task generation, (3) reward / outcome verification, (4) recipe revision. The **trust anchor** that makes overnight operation safe is **verifiable reward (RLVR)** — binary, hard-to-game outcome checks — not a learned reward model. ENPIRE's distinctive add is **physical grounding** (real-hardware auto-reset + binary verify) and **fleet parallelism**.

This plan maps that lens onto *this* workspace and orders the work by **expected value ÷ feasibility on a single RTX 3080 + 2 SO-101 arms** — explicitly NOT a clone of ENPIRE's datacenter-scale setup.

## The 4-stage loop vs. current state (grounded)

| Stage | Target (autonomous) | Current state (file refs) | Gap |
|---|---|---|---|
| **1 Curriculum gen** | auto-advance + adapt task difficulty | ladder *spec* in `lerobot-curriculum-agent.md`; only stages 2–4 coded in `pick_and_place.py` (env var `LEROBOT_ISAAC_STAGE`, manual); Bundle C.2 DR-scheduler **deferred** (`NEXT_STEPS.md:88`) | no auto-advance controller; no `curriculum_state.json` persistence |
| **2 Rollout / explore** | parallel, unattended | sim `make_env()` wired; isaac_dr replay autonomous; **deploy is per-step manual-gated** (`lerobot-isaac-deploy/session.py:1-98`); single arm | no auto rollout on HW; no fleet |
| **3 Verifiable reward** | binary outcome, real + sim | scripted shaped reward (`rewards.py`); `place_termination()` binary but **SIM-ONLY** (`terminations.py:158-182`); HW = no outcome measurement | **no hardware-grounded binary verify** ← critical |
| **4 Recipe update** | agentic recipe design + prune | autoresearch HP mutation only (`autoresearch/train_wrapper.py`); static `programs/*.md` | no curriculum/recipe generation; no hypothesis tree |

---

## Phased roadmap (EV ÷ feasibility ordered)

### Phase 0 — Unblock current loop (prereq, ~1 day)
Not new capability; clears known blockers so later phases have a working baseline.
- **object_pose lever is already wired** as an opt-in env var `LEROBOT_ISAAC_INCLUDE_OBJECT_POSE` (default OFF, `so101_env_cfg.py:117`; obs 6→13 when set). Phase-0 action = decide whether to flip the default ON or set it per-run; **no new code needed** (highest-EV existing lever, per `plans/2026-06-16-fresh-training-run-plan.md:30`).
- Resolve NVML driver mismatch (`NEXT_STEPS.md:7`, "Immediate"). **Re-verify it is still broken first** — `NEXT_STEPS.md` is ~4 weeks stale (last touched 2026-05-25); the mismatch may already be fixed. Run `nvidia-smi` + `python -c "import pynvml; pynvml.nvmlInit()"` before treating it as a live blocker.
- **Accept:** sim smoke test passes with object_pose in obs; `pixi run pipeline` runs end-to-end (requires working GPU + Isaac Lab + NVML — not testable on a CPU-only box).

### Phase 1 — Close Stage 3: a reusable **binary outcome verifier** (highest leverage)
This is the RLVR trust anchor — everything autonomous depends on a ground-truth "did it succeed?" signal that is the *same* in sim and on hardware.
- Generalize the sim-only `place_termination()` into a task-agnostic `outcome_verifier` (predicate registry: object-in-bin, pose-within-ε, gripper-closed-on-object) usable from (a) sim eval and (b) a real-hardware reader.
- Add a hardware outcome reader: extend `lerobot-isaac-deploy/arm_state_reader.py` (currently joint-only) to also surface the verifier inputs (object pose via D435 detection or fiducial). Where vision is unavailable, fall back to a scripted physical check + a single human-confirm.
- Emit `task_success=0|1` alongside `pc_success` so eval gates on *verified* success, not just the training metric.
- **Driven by harness:** new `binary-success-verifier` skill (companion plan, Phase 1).
- **Accept:** sim eval and a real (read-only/dry-run) rollout both emit `task_success` from the *same* predicate; unit tests on the predicate registry.

### Phase 2 — Close Stage 1: **curriculum auto-advance** (Bundle C.2, already scoped)
- Implement the deferred DR Stage Scheduler: advance stage when `pc_success ≥ threshold AND verified task_success rate ≥ threshold` (uses Phase 1 signal), resume from prior checkpoint, finer steps (6→9→12→15→18 cm per `2026-06-16` plan #3).
- **Grounding (current code):** only stages 2–4 are implemented (`pick_and_place.py`, validated against `(2,3,4)` in `__post_init__`; stage 3=±2 cm, 4=±5 cm — the finer 6→18 cm ladder is new work here). `lerobot-curriculum-agent` advances *logically* but does **not** persist `curriculum_state.json` (documented gap, `system-improvements.md` / dashboard `loaders/curriculum.py`) — Phase 2 must add that persistence for the acceptance criterion below to hold.
- Wire to the existing `lerobot-curriculum-agent` (harness) so advancement is agent-decided, not env-var.
- **Accept:** a multi-stage run advances stages with zero manual env-var edits; `outputs/curriculum_stage.json` + `outputs/curriculum_history.jsonl` (the dashboard's existing contract — NOT `.agent-state/curriculum_state.json`) reflect verified-success gating.

### Phase 3 — Stage 4: **agentic recipe-designer** (the Level-3 step)
- Add a hypothesis-tree controller *above* the existing autoresearch loop: proposes recipes = {curriculum config + HP set + data mix}, runs them through autoresearch, prunes losing branches, reuses winning recipes — the ENPIRE "Evolution" analogue. Reuse `autoresearch-loop-orchestrator` for the inner HP search; the new layer designs the *outer* recipe.
- Persist the tree as git-like branches in `.agent-state/{sessionId}/hypotheses/` with success-curve + wall-clock + token cost per node (so the user "reads the reports in the morning").
- **Driven by harness:** new `recipe-designer` orchestrator (companion plan, Phase 3).
- **Accept:** controller runs ≥3 recipe branches unattended, prunes, and emits a ranked report; human audits post-hoc.

### Phase 4 — Stage 2 on hardware: **safe auto-reset + closed-loop rollout** (safety-critical)
- Replace the per-step stdin gate with an autonomous loop guarded by: hard joint clamps (keep existing `arm_motor_writer.py` limits — two-layer: action-clip to `[-1,1]` then cal∩hardcoded joint floors incl. elbow −10° table-avoidance), watchdog/e-stop, ramped home-on-fault (`ramped_home()`, per-step delta cap), and an auto-reset routine (homing + scene-reset confirm via the Phase-1 verifier). Keep a kill-switch and an opt-in `--autonomous` flag that is OFF by default. **Preserve the existing dual-flag execute contract** (`session.py`: `safety_critical=True` steps require both `--yes` AND `--execute`, and refuse the `--assume-yes` auto-path) — the autonomous loop lets a *vetted* policy through the gate; it does NOT remove the gate's safety layers.
- **Driven by harness:** new `physical-reset` + `rollout-executor` worker agents (companion plan, Phase 4).
- **Accept:** N unattended rollouts on real SO-101 with auto-reset between episodes and zero manual gates, with safety tests (clamp, watchdog, e-stop) passing first. **Do not enable autonomy on hardware until Phases 1+ verifier and safety tests are green.**

### Phase 5 — Research-class (flag, do NOT start near-term)
Explicitly deferred — low feasibility on current hardware budget:
- **Fleet parallelism** across the 2 SO-101 units (ENPIRE physical scaling). Only after Phase 4 is stable; the ENPIRE finding is that token cost grows superlinearly with fleet size — not worth it at N=2 until the single-arm loop is proven.
- **Learned RLVR reward model** (replace brittle binary predicates on edge cases). Research-class; binary verifiable reward is strictly better for *autonomy safety* until proven necessary. Aligns with the existing multi-month HIL-SERL item in `NEXT_STEPS.md`.

---

## Anti-hypertrophy / explicit non-goals
- **Do NOT** build the fleet coordinator or learned-reward model first — they are Phase 5, gated on a working single-arm loop.
- **Do NOT** introduce Hydra / a new config system — extend the existing YAML + dataclass loader.
- **Reuse** existing agents (curriculum, evaluation, autoresearch, verification-loop) — the new pieces are the *verifier*, the *recipe-designer* layer, and the *hardware* EN module; not a rewrite.
- Verifiable binary reward **>** learned reward for autonomy, per the synthesis — keep the trust anchor simple.

## Global constraints (carry into every phase)
- Single **RTX 3080 10 GB**: respect the documented OOM ladder (`plans/2026-05-15-dali-gpu-decode-plan.md`); no multi-GPU assumptions.
- Safety gates in `lerobot-isaac-deploy` are load-bearing — every hardware phase preserves clamp + dual-flag execute + ramped home; autonomy is opt-in and OFF by default.
- All new orchestration writes to `.agent-state/{sessionId}/` only (wake/resume contract).

## Dependency order
Phase 0 → 1 → 2 → 3 (1 also unblocks 4). 4 needs 1. 5 needs 4.

## Audit verification (2026-06-19)
Every phase claim was checked against the codebase by a 7-agent parallel audit. All file-refs and current-state claims **verified** except:
- **Phase 2 file-ref corrected** — the 6-stage ladder *spec* lives in `lerobot-curriculum-agent.md` (harness), not `so101_env_cfg.py`; only stages 2–4 are coded (`pick_and_place.py`, `LEROBOT_ISAAC_STAGE`). `curriculum_state.json` is a documented missing-persistence gap (correctly Phase-2 work).
- **Phase 0 object_pose** — already wired as an opt-in env var (default OFF), not absent; the action is a default-flip/decision, not new code.
- **NVML** — likely stale: `NEXT_STEPS.md` predates this plan by ~4 weeks; re-verify before treating as blocker.
- **OOM-ladder ref OK** — `plans/2026-05-15-dali-gpu-decode-plan.md` (cited in this plan) exists and is canonical (Approach D). The project `CLAUDE.md` instead names `…dataloader-gpu-decode-plan.md`, which is the *archived* predecessor — **CLAUDE.md is the stale one**, not this plan (fix tracked separately).
- Companion harness pieces (`binary-success-verifier` skill, `recipe-designer` orchestrator, `physical-reset-agent` + `rollout-executor-agent` workers) confirmed **net-new**; existing lerobot/autoresearch agents confirmed reusable.

## Implementation status (2026-06-19)
Progress landed this session (master-project-orchestrator):
- **Phase 1 (sim) — DONE + runtime-verified.** `lerobot_isaac_env.outcome_verifier` (pure-numpy predicate registry: `object_in_bin` / `pose_within_eps` / `gripper_closed_on_object`); `place_termination` delegates to it; `_sim_eval.py` emits verifier-grounded `task_success` sourcing geometry from the live env cfg. **Closed-loop sim eval confirmed emitting `task_success`** (3-ep ACT run, `geom_source=env.cfg.terminations.success.params`).
- **Phase 1 (hardware half) — SEAM BUILT.** `lerobot_isaac_deploy.outcome_reader` (`read_object_in_bin` / `read_task_success` / `manual_confirm`) uses the SAME predicate via dependency injection (decoupled; canonical predicate lazily loaded when `lerobot-isaac-env` present, else injected). Live D435/fiducial vision + on-arm verification still need the hardware box.
- **Phase 2 — CONTROLLER BUILT.** `lerobot_isaac_autoresearch.curriculum_controller` (`decide_advance` gated on `pc_success ≥ thr AND task_success ≥ thr`; persists `outputs/curriculum_stage.json` + `curriculum_history.jsonl` per the dashboard schema; `stage_env_value` → `LEROBOT_ISAAC_STAGE`). Wiring into `lerobot-curriculum-agent` + a real multi-stage GPU run remain.
- **Remaining:** GPU multi-stage curriculum run (Phase 2 accept); hardware vision + on-arm `task_success` (Phase 1 accept); then Phase 3 (recipe-designer) / Phase 4 (hardware autonomy, safety-gated).

## Implementation status (2026-06-20)
Stranded Phase 1/2 code (built 2026-06-19, uncommitted across 4 trees) reviewed + shipped this session (master-project-orchestrator, session `20260620-084313-continue-level3-plan`). Adversarial grill (3 attackers) found 1 real correctness bug + hardening gaps; user chose the full fix-then-ship pass:
- **FIXED — `_sim_eval.py` `task_success` spawn-pose bug.** The verifier read the object pose *after* `env.step()`, but Isaac `ManagerBasedRLEnv.step()` auto-resets terminated envs internally → it was measuring the next episode's spawn pose (≈0 regardless of policy). Now sources `task_success` from the env's `success` termination-term verdict (cached pre-reset; for pick_and_place that term IS `place_termination`→`object_in_bin`, so still the canonical predicate). New helper `_read_success_term(env)` + 9 CPU tests. **[P0] NOT yet GPU-verified — confirm on the next Isaac run before relying on `task_success` (and before the Phase-2 curriculum GPU run, which gates on it).**
- **FIXED — `outcome_verifier` hardening.** Shape errors now fail-loud (`ValueError` on non-`(3,)`/`(N,3)`, fixes the silent `(6,)`-joint-array→XY bug); non-finite (NaN/inf) is fail-safe (returns False, no raise — safe on the GPU termination hot path); negative gripper width no longer reports "closed". 30→49 tests.
- **FIXED — `curriculum_controller` atomic write.** Stage JSON now written via temp-file + `os.replace` (was non-atomic `write_text`; a torn write silently reset the curriculum to MIN_STAGE). 30→33 tests.
- **Geometry-source + soft-import "blockers" were FALSE POSITIVES** (refuted by direct code inspection: task remaps `terminations.success`→`place_termination` with the exact params read; numpy ≠ GPU dep).
- **[P2] Deferred (Phase-4 hardware, autonomy OFF):** `outcome_reader.read_object_in_bin` lets `object_pose_source()` exceptions propagate uncaught and `manual_confirm` blocks on stdin with no timeout → hang risk under unattended rollout. Harden before enabling Phase-4 `--autonomous`.

## Related
- Research: `05-Wiki/synthesis/2026-06-19-autonomous-training-pipeline-agentic-layer.md`, `05-Wiki/entities/ENPIRE.md`
- Existing plans: `NEXT_STEPS.md`, `plans/2026-06-16-fresh-training-run-plan.md`, `plans/2026-06-13-pipeline-analysis.md`
- Companion: `~/tools/claude_code/plans/2026-06-19-autonomous-training-skills-agents-plan.md`
