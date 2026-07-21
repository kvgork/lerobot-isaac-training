# Post-Campaign Wrap Plan — 2026-07-19

**Follows:** `plans/2026-07-15-three-day-autonomous-gpu-plan.md` (campaign COMPLETE 2026-07-18T02:40Z,
43 scored trials / 5 families, lessons vault-ingested 2026-07-19).
**Branch:** `feature/residual-rl`. **GPU:** idle since Jul 18 — Phases 2–3 below use it.

---

## Phase 1 — Infra merge-back (CPU, ~1 h, nothing is running so all files are safe to edit)

> **Status 2026-07-20:** items 1–3 + 5 DONE (`9eccfca` merge/thresholds/tail-loop, `41b0b4d` grill
> hardening: restart clobber guard + TAIL_MIN_FREE_GB disk floor; verification 8/8, meta 71 green).
> Item 4 CLOSED 2026-07-20: `feat/autoresearch-deterministic-runner` merged to claude_code main
> (`4048d5c` contains it); local checkout synced, engine `AR_TRIAL_START` present, dispatcher
> `--print-cmd` contract smoke green. **Phase 1 complete.**
> Grill accepted-without-fix list: `docs/.grill-accepted.log`. New knobs: `TAIL_HOPS` (default 2,
> 0=unlimited), `SEED_OFFSET`, `AR_OUT_ROOT` (env-overridable).

1. **Merge `save_freq` fix into the real dispatcher.** Apply the `SECONDS_PER_EXP/4` change from
   `scripts/run_autoresearch_policy_fixed.sh` to `scripts/run_autoresearch_policy.sh`, then DELETE the
   `_fixed` copy and repoint `scripts/gpu_campaign_ext2_diffusion.sh`/`ext3` (or delete the ext scripts
   too — they are one-shots; keep only if the pattern is wanted as reusable infra).
   Accept: `grep -c 'SECONDS_PER_EXP / 4' scripts/run_autoresearch_policy.sh` = 1; `_fixed` gone.
2. **Merge the stale-threshold rule into `scripts/gpu_campaign.sh`.**
   Rule: `STALE_KILL[job] ≥ SECONDS_PER_EXP(job) + eval(≤300 s) + margin(600 s)` because AR dispatchers
   print stdout once per FINISHED trial. Fix the act_F line (2700 → ≥3600), diffusion_E (2400 → ≥2700),
   wm_offline_G (2700 → ≥3300). Also add a **loop-extend tail hop**: when the chain exhausts, re-launch
   the top offline-scoreable sweep with fresh seeds instead of idling (GPU sat idle 40 h this run).
3. **Fix the smolvla hop wrapper too:** `_run_autoresearch_smolvla.sh` save_freq line has the same
   `*25/30` formula (worked at 10 step/s cached; fragile) — normalize to `/4`.
4. **Push the autoresearch engine branch.** `~/tools/claude_code` local branch
   `feat/autoresearch-deterministic-runner` holds the engine + `AR_TRIAL_START` resume commits — unpushed.
   Push branch (NOT main), open PR.
5. Commit all of the above; tests green (`pixi run test` meta 71 + sibling suites in `sim` env).

## Phase 2 — Candidate evaluation wrap (GPU eval-only, ~2 h)

1. **Refresh open-loop MSE on one common protocol** (`scripts/_open_loop_eval.py`) across ALL deploy
   candidates: ACT-15k (`act…015000`), SmolVLA-020k, vla_jepa-020k, **campaign SmolVLA winner**
   (`outputs/autoresearch-lerobot-policy-smolvla/trial_7/checkpoints/041660/pretrained_model`,
   pc 0.129/MSE 6.73), LoRA best (rank64 α128 attn_qv, 0.144), ACT sweep best (trial 0, 0.0208).
2. **Dashboard N-way compare + snapshot** (`docs/runbook/08-batch-train-and-compare.md`,
   `pixi run -e dashboard dashboard`). Save snapshot for the arm session.
3. Output: single ranked candidate table in this plan + dashboard snapshot path. Honest caveat stays:
   offline MSE/pc proxies are SOFT rankings; the arm is the only real gate.

> **Phase 2 DONE 2026-07-20.** Protocol: `_open_loop_eval.py`, n_episodes=4, held-out slice of each
> candidate's OWN training dataset — a single common dataset is impossible (pickplace1 camera key
> `observation.images.d435_rgb` vs pickplace-new `observation.images.overhead`), so the two groups
> below are NOT cross-comparable. Raw JSONs: `outputs/phase2-eval-20260720/` (+ copies in
> `outputs/eval/` for the dashboard loader).
>
> | rank | candidate | checkpoint | dataset (held-out) | MSE | pc_success |
> |---|---|---|---|---|---|
> | B1 | SmolVLA campaign winner | `autoresearch-lerobot-policy-smolvla/trial_7/checkpoints/041660` | pickplace1 (1673 fr) | **6.35** | **0.1360** |
> | B2 | LoRA r64 α128 attn_qv | `…smolvla-lora/trial_4/checkpoints/merged` | pickplace1 (1673 fr) | 6.45 | 0.1342 |
> | B3 | ACT sweep trial_0 | `…policy-act/trial_0/checkpoints/last` | pickplace1 (1673 fr) | 48.79 | 0.0201 |
> | A1 | vla_jepa-020k | `vla_jepa_real_so101/checkpoints/020000` | pickplace-new (1584 fr) | **40.96** | **0.0238** |
> | A2 | ACT-15k | `act_real_so101_15k/checkpoints/015000` | pickplace-new (1584 fr) | 52.24 | 0.0188 |
> | A3 | SmolVLA-020k | `smolvla_real_so101/checkpoints/020000` | pickplace-new (1584 fr) | 58.33 | 0.0169 |
>
> Consistency vs prior numbers: trial_7 6.35/0.136 (was 6.73/0.129), LoRA 0.134 (was 0.144), ACT
> sweep trial_0 pc 0.0201 (was 0.0208 — NB the plan's "0.0208" was pc_success, not MSE; on this
> proxy ACT sweep trails both SmolVLA candidates by ~7×). Dashboard N-way report:
> `outputs/reports/2026-07-20T161827-no-session/report.html`; **arm-session snapshot:**
> `outputs/snapshots/2026-07-20T161828-2026-07-20T161827-no-session/` (known wart: events.parquet
> written empty — mixed-type `commits` column; eval/training loaders intact).
> Arm bake-off short-list per group: **SmolVLA trial_7 + LoRA trial_4** (B) and **vla_jepa-020k** (A),
> keeping ACT-15k as the prior first-HW-success reference. Soft rankings; the arm decides.

## Phase 3 — Residual grasp unblock (GPU, ~3 h total, gated)

1. **Diff the two scripted controllers** — demo-gen (`scripts/_gen_sim_demos.py`, LIFTS+PLACES at the
   campaign's action scale) vs adapter `scripted_grasp_phases`
   (`src/lerobot-isaac-adapters/.../sheeprl_plugin/`, slips at oz 0.008, regrasp loop). Compare frame
   sequences: z_high, close depth/target, per-phase step caps, hold criteria, gripper command shape.
   Campaign evidence: `.agent-state/c1-residual-smoke/autoresearch/wm-isaac-prod/train.log` vs
   demo regen post-hoc lines in `outputs/gpu_campaign/campaign.log`.
2. **Port the demo-gen grasp sequence** into `scripted_grasp_phases`; unit-test phase progression
   (existing per-phase step-cap tests as template).
> **Items 1–2 DONE 2026-07-20** (adapter `1a134ec` on `feature/wm-isaac-env`, NOT pushed).
> Controller diff found 4 deltas vs the working demo: z_high 0.17→0.19 (CARRY_Z parity, cup-rim
> clearance), rate-limited LIFT → direct z_high command (slip-at-oz-0.008 suspect), close cradle
> 40→80-step ramp + new 25-step HOLD phase (demo's 80+25), schedule caps to demo durations
> (DESCEND 90 / LIFT 60 / CARRY 60 / LOWER 40, STABILIZE 30); RELEASE ramps to PLACE_PART_OPEN.
> No-stall invariant + bounded regrasp kept. Unit tests 25/25 (was 19), meta 71 green.
> **Unit-test gate PASSED → re-gate is next (GPU).**

3. **Re-gate** (~1.5 h GPU): `bash scripts/_residual_smoke_gate.sh` — PASS = oz>0.07 + CARRY reached.
> **Re-gate run 2026-07-20 (~50 min GPU): VERDICT FAIL — but the verdict is UNRELIABLE; two
> instrumentation/env bugs invalidate it, NOT the ported controller.** Gate line:
> `phases=['CLOSE'] min_ez=0.106 max_oz=0.015 lifted=False reached_lift=False descended=True`.
> The descent criterion PASSED (ee at grasp depth 0.106 exactly, xy_to_tgt 0.001 — old ez~0.30
> blocker gone; C1 scale fix works).
>
> Root causes found by log forensics (`.agent-state/c1-residual-smoke/.../train.log`):
> 1. **Backing Isaac env still truncates at 300 steps** (episode ends at policy_step 300/601/902/
>    1203 = its own `episode_length_s=10s×30Hz` time_out). The wrapper's `max_episode_steps=700`
>    only ADDS a cap (`isaac_env.py:270`) — it never raises the backing cap. Demo-gen disables it
>    (`_gen_sim_demos.py:66-70`, `episode_length_s=1e6`) precisely because the working sequence
>    needs ~485 steps. Under 300 the ported schedule can reach LIFT at best — never CARRY/PLACE.
> 2. **Trace aliasing:** `[script-dbg]` prints every 150 compute calls; episodes are 301 steps →
>    every sample lands at episode-step ~150 (mid-CLOSE). `phases`/`max_oz` in the gate verdict
>    are blind past that point — a lift at episode-step 200+ would be invisible.
>
> **Prescribed fixes (adapter, CPU, before any re-gate):** (a) wrapper raises backing
> `env.cfg.episode_length_s` when `max_episode_steps` exceeds it (mirror demo-gen); (b) de-alias
> dbg — print on every phase TRANSITION (+ episode-relative step), not a fixed 150 cadence;
> (c) optionally have the gate parser also read a per-episode max-phase line. Then re-gate
> (~1 h GPU). Escalated per step 5 — human decision to proceed.
>
> **Fixes (a)+(b) landed 2026-07-21** (adapter `634d910`, user-approved): `_boot()` raises backing
> `cfg.episode_length_s` when wrapper `max_episode_steps` exceeds backing `max_episode_length`
> (demo-gen's approach; wrapper cap stays sole authority); `[script-dbg]` now prints per phase
> TRANSITION with episode-relative `t=` via pure `format_phase_transition()` — gate-regex contract
> unit-tested against `_residual_smoke_gate.sh`'s parser. Phase tests 28/28. Re-gate round 2
> launched 2026-07-21T17:04Z → `outputs/gpu_campaign/c1_gate_20260721.log`.
>
> **Round 2 result (2026-07-21): VERDICT PASS (rc=0) — but treat as SOFT.**
> `phases=[APPROACH,DESCEND,STABILIZE,CLOSE,HOLD,LIFT,CARRY,LOWER] min_ez=0.118 max_oz=0.008
> lifted=False reached_lift=True descended=True`. Both fixes verified working: episodes run full
> length, all 8 phases traced (de-aliasing confirmed). HOWEVER `max_oz=0.008` = the die never
> physically lifted — `reached_lift` fired via the phase machine's force-advance caps, which the
> stall fix made unconditional, so that gate criterion is now vacuous. The plan's stricter bar
> (oz>0.07 + CARRY) is NOT met. The scripted base still doesn't grip under the residual blend
> (demo-gen at identical geometry lifts ~80%). Remaining suspects: pre-`learning_starts` random
> phase (first 200 steps) displacing the die before the scripted controller engages; residual
> actor perturbation on the gripper channel during CLOSE; a still-unfound grip-sequence delta —
> needs the step-5 frame diff (grip trajectory demo vs smoke ep-2) before authorising the 13 h
> full run. **Full run NOT launched (user hold + soft-PASS caveat). Human decision.**

4. **If PASS → full residual run** (13 h GPU):
   `LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS=15000 bash scripts/launch_residual_rl.sh` (setsid detach,
   stale backstop ≥3600 s; read result from training TB place-rate, NOT `_sim_eval.py`).
5. If FAIL again with the ported controller → stop; escalate to human with the frame diff.

## Phase 4 — Commit-debt review (CPU, human-judgment items)

1. `src/lerobot-isaac-configs` (5 modified files, main) — review, commit to a feature branch, PR.
2. `src/lerobot-isaac-autoresearch` (2 modified files + 2 FAILING tests in
   `test_curriculum_campaign.py::TestDerivePlaceSuccess` — mock patch-target bug: module lazily imports
   `event_accumulator`, tests patch it as module attr). Fix tests, commit to `feature/auto-wm`, PR.
3. `robot-data-recorder feat/deploy-runner-hardening` (885 insertions, 2 new CLIs) — **human-review PR**;
   do NOT auto-merge (frozen env installs siblings from `git+github@main`).

## Phase 5 — Arm session (hardware, when arm returns)

1. Run the **3-way real-arm bake-off** per `plans/2026-07-12-phase3-model-comparison-plan.md`, extended
   with the Phase-2 winners (SmolVLA trial_7 + LoRA best are new candidates).
2. Real-arm eval is the ONLY ranking that counts (sim2real ≈ 0 for sim metrics).
3. Then: 150+ diverse demos (top lever per ACT-real campaign), DAgger loop.

## Order / gates
| Gate | Pass → | Fail → |
|------|--------|--------|
| Phase 1 tests green | Phase 2 | fix before touching GPU |
| Phase 3 controller port unit tests | re-gate | debug port, don't burn GPU |
| Re-gate exit 0 | full residual 13 h | stop, escalate with frame diff |
| Runner PR human-approved | merge | leave on branch |

## Related
- `plans/2026-07-15-three-day-autonomous-gpu-plan.md` (campaign + incident log)
- `plans/2026-07-12-phase3-model-comparison-plan.md` (arm bake-off)
- Vault: `05-Wiki/sources/2026-07-19-gpu-campaign-3day-lessons.md`
- Memory: `three-day-autonomous-gpu-plan-2026-07-15`, `lerobot-060-cache-frames-break`
