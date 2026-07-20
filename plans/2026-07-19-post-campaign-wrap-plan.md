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

## Phase 3 — Residual grasp unblock (GPU, ~3 h total, gated)

1. **Diff the two scripted controllers** — demo-gen (`scripts/_gen_sim_demos.py`, LIFTS+PLACES at the
   campaign's action scale) vs adapter `scripted_grasp_phases`
   (`src/lerobot-isaac-adapters/.../sheeprl_plugin/`, slips at oz 0.008, regrasp loop). Compare frame
   sequences: z_high, close depth/target, per-phase step caps, hold criteria, gripper command shape.
   Campaign evidence: `.agent-state/c1-residual-smoke/autoresearch/wm-isaac-prod/train.log` vs
   demo regen post-hoc lines in `outputs/gpu_campaign/campaign.log`.
2. **Port the demo-gen grasp sequence** into `scripted_grasp_phases`; unit-test phase progression
   (existing per-phase step-cap tests as template).
3. **Re-gate** (~1.5 h GPU): `bash scripts/_residual_smoke_gate.sh` — PASS = oz>0.07 + CARRY reached.
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
