# 3-Day Autonomous GPU Plan — 2026-07-15 (v2, critique-hardened)

**Created:** 2026-07-15 · **Branch:** `feature/residual-rl` · **Mode:** `/orchestrate`, rigor=agentic.
**User-locked constraints:** NO arm for the 3 days → **fully autonomous/unattended**. Bias = **breadth**.
**Two decisions taken after a 4-lens adversarial critique (blindspot/nightmare/xray/domain):**
1. **Build an auto-chain supervisor FIRST** — the critique's headline finding (2× RECONSIDER) is that
   "autonomous 24/7" is impossible today: nothing chains one GPU job to the next on exit, watchdogs are
   report-only, and the residual gate needs human judgment. So Phase 0 builds the missing autonomy.
2. **Sweeps lead, residual is a gated stretch** — no arm + sim2real≈0 means the residual sim checkpoint
   can't rank as a real deploy candidate. Fill the unattended chain with offline-scoreable sweeps;
   attempt residual only if its fix passes a *scripted* gate.

## Governing constraint: one GPU, serial, unattended
Single RTX 3080 (10 GB), free/idle now. Jobs run one at a time. "Unattended" now means a **foreground
supervisor** (`scripts/gpu_campaign.sh`, itself under one `setsid`) launches each job on the previous
one's exit, evaluates *scripted* gates, kills wedged jobs on a hard stale-log timeout, and advances a
pre-decided fallback chain — so a crash/finish at hour 4 does NOT idle the GPU (the #1 critique BLOCKER).

## What the arm would have done (EXCLUDED — packaged as a runbook, not run)
3-way real-arm bake-off (ACT/SmolVLA/vla_jepa — the only real ranking + the WM question), 150+ diverse
demos, DAgger, deploy verification. None autonomous. The 3 days *bank* candidates + tee this up.

## Verified readiness (2026-07-15)
- **Isaac Sim + Lab WORK** in `.pixi/envs/sim` (`isaacsim: True`, `isaaclab 0.54.3`, last ran 07-12).
- `lerobot` in train-policy/train-lewm/sim; `sheeprl` in train-dreamer/sim; SmolVLM2 weights cached.
- 3 real candidates on disk (`act…015000`, `smolvla…020000`, `vla_jepa…020000`); WM-offline HDF5 (490 MB).
- Real dataset `so101-pickplace-new` (50 ep) intact — untouched by any sim action-scale work.

---

## PHASE 0 — Attended setup (~half day, BEFORE the unattended window)
Everything here is the price of real autonomy. Done once, with a human present, then the 3 days run hands-off.

### 0a — Build the supervisor `scripts/gpu_campaign.sh` (NEW, the core deliverable)
A foreground driver that runs a **fixed fallback chain** and never lets the GPU idle:
```
for job in CHAIN:            # A → C1-gate → [B] → C → E → F → G
  launch job (setsid, own logfile)
  while running:
    if log stale > STALE_KILL[job]:   kill -TERM; sleep; kill -KILL   # HARD kill (not report-only)
    sleep poll
  record exit code + tail
  run job.gate.sh (if any) → exit 0/1 decides the next hop
```
- `STALE_KILL[job]` = N× the job's longest *expected* quiet gap (e.g. residual `learning_starts`
  collection is legitimately quiet — size the timeout above it, per `[[wm-isaac-stall-resolved]]`).
- Distinct from the existing report-only heartbeat: this one **acts** (kills + advances).
- Emits one status line per transition to `.agent-state/gpu-campaign/events.jsonl` for cost/introspection.

### 0b — Make the C1 residual gate machine-checkable (NEW scripts)
The residual "does the ee descend?" gate must be a script, not a human eyeball:
- `scripts/_probe_action_scale.py` (NEW — the plan's old `--log_actions` flag does **not** exist;
  adapt `scripts/_scripted_arm_audit.py` / `_grasp_joint_diag.py`). Runs one scripted pick→place,
  writes per-joint `max|q_used − q_default|` to `outputs/action_scale.json`. **Branch built in:** if
  `|action|≤1` everywhere, the clamp is innocent → emit `SCALE_OK` and skip the rescale.
- Refactor `_ACTIONS_SCALE_DICT` in `so101_env_cfg.py` to **read `outputs/action_scale.json`** (env var
  `LEROBOT_ISAAC_ACTION_SCALE_JSON`) so the supervisor wires probe→env with **no human edit**. Mirror the
  normalization change in BOTH `isaac_env.py compute_scripted_action` and `_gen_sim_demos.step_to`
  (keep byte-identical). Re-tune `PHASE_STEP_CAP` for the new in-range (slower) motion; unit-test it.
- `scripts/_residual_smoke_gate.sh` (NEW): runs the smoke (`STEPS=700 algo.learning_starts=200`), greps
  its log for `ez`→≈0.106 **and** `oz>0.07` (`obj_lifted=True`) **and** phase reaching CARRY **and**
  reward ≫ −32 → `exit 0` (PASS → run Job B) else `exit 1` (FAIL → skip B, continue chain).
  **Honest caveat:** a PASS validates the ee-descent *mechanism* only, not place-learning at the real
  `learning_starts=1024` — it does NOT de-risk the documented place-wall plateau `[[carryplace-place-wall-plateau]]`.

### 0c — Fix Job B parameters (confirmed bug)
`launch_residual_rl.sh` defaults `DECAY_STEPS=30000` but only ~23k steps are reachable in the 13h cap
(@0.5 step/s) → the run dies with the scripted weight still ≈0.22, checkpoint is base-dominated, and the
"RL retains as w0 decays" gate is unevaluable. **Set `LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS=15000`** so
the pure-RL retention phase actually runs inside budget. Keep `CHECKPOINT_EVERY=5000`; add an early-abort
gate at 10k/15k (if `obj_lifted` rate ≈0 → kill, reclaim GPU to the next chain job).

### 0d — Make the tail jobs real
- `scripts/_run_autoresearch_act.sh` (NEW — does not exist; copy the `_run_autoresearch_diffusion.sh`
  pattern) so Job F is genuinely autonomous. Dry-run it.
- **Drop Job H (P2E)** from the chain — its metric extraction is a broken sentinel; not scoreable.

### 0e — Sim-eval dependency isolation (confirmed risk)
Do NOT `pip install transformers==5.3.0 num2words` into the shared `sim` env (carries lerobot 0.6.0
Qwen + isaaclab; can break Jobs B/D). Instead: create a **dedicated `eval` env** (or verify the resolved
requirement first) and gate behind `pixi run -e sim python -c 'import isaacsim, lerobot'`.
> **Correction to old deliverable #4:** `_sim_eval.py` loads **LeRobot policies only** — it CANNOT score
> the residual (a sheeprl/DreamerV3 checkpoint). The residual's in-sim result is read from its own
> training TB (`Rewards/rew_avg` + the env `is_placed` termination rate in the run log), NOT `_sim_eval.py`.
> A proper sheeprl sim-eval path (reuse the DreamerV3 player + Isaac env in eval mode) is a **follow-up**,
> not in this cycle.

### 0f — Clear commit debt SAFELY (not the old rushed hour-0 push)
The `frozen` env installs siblings from `git+github@main`, so a bad push to a sibling **main** poisons
every future reproducible install (nightmare BLOCKER). Rules:
- **Push feature branches only; never push a sibling `main` unattended.** Open PRs for main merges.
- Gate every adapter/env push on `pixi run test` **green this session** (659 pass baseline).
- **Exclude `robot-data-runner feat/deploy-runner-hardening` (885 insertions / 2 new CLIs) from any
  auto-push** — it needs a real human review PR.
- Debt: root `feature/residual-rl` (ahead 7 → push), adapter `feature/wm-isaac-env` (ahead 3 → push
  after tests), env `feature/wm-isaac-env` (ahead 2 → push after tests), configs `main` (5 M + new
  `scenes/`,`task/` → commit to a branch + PR), autoresearch `feature/auto-wm` (2 M → commit).
- This is CPU work in the setup window — it does NOT block the GPU (the supervisor starts after).

**Phase-0 exit gate:** supervisor dry-runs the chain end-to-end with `--dry-run` (each job prints its
launch cmd, gates return deterministically); tests green; debt pushed (feature branches). THEN start the
unattended window.

---

## PHASE 1 — Unattended 3-day chain (supervisor-driven, hands-off)
Fixed chain, sweeps-lead. Every job self-terminates on a wall budget; the supervisor launches the next on
exit or on a stale-kill. **All "wall" figures are watchdog CAPS, not runtimes.**

| Hop | Job | Command (via supervisor) | Cap | Gate → next | Scoreable offline? |
|-----|-----|--------------------------|-----|-------------|--------------------|
| 1 | **A · SmolVLA sweep** | `bash scripts/_run_tonight_smolvla_12h.sh` (anchor + AR + rerank + dashboard) | ≤12 h (**≤7 trials** — 8×6000 s = 13.3 h > cap; or `--ar-seconds 5000` to fit 8) | always → hop 2 | ✅ MSE proxy / pc_success |
| 2 | **C1 gate** | `_probe_action_scale.py` → apply scale → demo regen (~30 m) → `_residual_smoke_gate.sh` | ~1.5 h | `exit 0` → hop 3 (B); `exit 1` → hop 4 (skip B) | gate only |
| 3 | **B · residual-RL full** (gated stretch) | `LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS=15000 bash scripts/launch_residual_rl.sh` | ≤13 h (~23k steps); early-abort at 10k/15k if no lift | → hop 4 | via training TB place-rate (NOT `_sim_eval.py`) |
| 4 | **C · LoRA sweep** | `MAX_TRIALS=16 STEPS=20000 bash scripts/_run_autoresearch_lora.sh` | ~10 h | → hop 5 | ✅ MSE proxy |
| 5 | **E · diffusion AR** | `SESSION_ID=diff-d3 TRIALS=6 SECONDS_PER_EXP=1800 bash scripts/_run_autoresearch_diffusion.sh` | ~3 h | → hop 6 | ✅ MSE proxy |
| 6 | **F · ACT sweep** | `MAX_TRIALS=8 bash scripts/_run_autoresearch_act.sh` (NEW) | ~6 h | → hop 7 | ✅ MSE proxy |
| 7 | **G · WM-offline AR** (tail-fill, low value) | `MAX_TRIALS=12 STEPS=200000 bash scripts/_run_autoresearch_wm.sh` | ~9 h | → done / repeat | recon_loss (weak for control) |

Budget: hops 1+2+4+5+6+7 ≈ 43.5 h guaranteed autonomous work; + hop 3 (~13 h if gate passes) ≈ 56.5 h of
the 72. **If the chain exhausts before 72 h**, the supervisor extends the top offline-scoreable sweep (A
or C) with more trials/seeds rather than idling. Residual failing its gate simply removes ~13 h of *sim*
work — the chain continues on sweeps (this is the "gated stretch" contract).

**Isaac hard rules baked into the launchers (do not override):** `num_envs=1` (`[[wm-isaac-num-envs-bug]]`),
`REPLAY_RATIO=4` not 16 (`[[replay-ratio-wallclock-online-isaac]]`), `setsid` detach only
(`[[detach-long-training-jobs]]`).

---

## PHASE 2 — Attended wrap (when you return)
- Dashboard N-way compare of all candidates (`docs/runbook/08-batch-train-and-compare.md`); refresh
  open-loop MSE (`scripts/_open_loop_eval.py`) across ACT/SmolVLA/vla_jepa + the new sweep winners.
- Read the residual result from its training TB (place-rate / `Rewards/rew_avg`) — honest sim signal.
- Vault write-back (`05-Wiki`), update the plan files with results.
- Surface the **arm bake-off runbook** (`plans/2026-07-12-phase3-model-comparison-plan.md`) front-and-center
  for the next hardware session — plus the runner PR awaiting review.

## Decision gates
| Gate | Pass → | Fail → |
|------|--------|--------|
| Phase-0 dry-run + tests green | start unattended window | fix supervisor/gate/tests first |
| C1 `_residual_smoke_gate.sh` exit 0 | run Job B (residual) | skip B, continue sweep chain (no GPU wasted) |
| Job B early-abort (10k/15k lift-rate>0) | let residual run to cap | kill, advance to Job C |
| chain exhausts < 72 h | extend top sweep (more trials/seeds) | — |

## Risks / mitigations
- **Supervisor is now load-bearing** — a bug there idles/wedges the whole campaign. Mitigate: `--dry-run`
  the full chain in Phase 0; hard stale-kill with generous per-job timeouts; each hop independent.
- **C1 fix is a real design tension, not a one-liner** (highest risk). Mitigate: it's a *gated stretch* —
  a scripted FAIL costs only the ~1.5 h gate, then the chain continues on sweeps.
- **OOM** — residual batch 8→4; vla_jepa freeze_qwen+batch 2; `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- **MSE proxy is weak** (per-frame, eval eps seen in training) — "ranked" sweep winners are *soft* orderings
  banked for the arm, not verified success rates. Say so in the report.
- **No sibling-main pushes unattended** — feature branches + PRs only; runner branch is human-review-gated.
- **Sim2real ≈ 0** — no sim metric ranks a real deploy candidate; the arm is the only real gate (excluded).

## Deliverables at end of 3 days
1. `scripts/gpu_campaign.sh` supervisor + machine-checkable C1 gate scripts (reusable infra).
2. Commit debt cleared on feature branches + PRs (runner PR flagged for human review).
3. SmolVLA / LoRA / diffusion / ACT autoresearch winners — offline-MSE-ranked candidates banked for the arm.
4. Residual-RL: either a sim carry-place checkpoint + training place-rate (if C1 gate passed), OR a
   documented scripted-FAIL with the probe evidence — no wasted 13 h either way.
5. N-way candidate comparison + dashboard snapshot; vault write-back; arm bake-off runbook teed up.

## PHASE 0 EXECUTION (2026-07-15, `/orchestrate`)
- **Supervisor `scripts/gpu_campaign.sh` built** — dry-run green (full chain + events). Hard stale-kill + fallback chain + C1-conditional residual.
- **Autoresearch generalized** (user directive): loop engine → `~/tools/claude_code/skills/autoresearch/run_deterministic.sh` (arch-agnostic, sourced); workspace `scripts/run_autoresearch_policy.sh --arch act|diffusion|smolvla` supplies train/eval/grid. Bespoke `_run_autoresearch_act.sh` deleted; supervisor uses the dispatcher for diffusion+act.
- **Action-scale probe + fix DONE + GPU-VERIFIED.** `--measure_scale` → `outputs/action_scale.json`: `clamp_innocent=false`, wrist_flex max delta **2.20 rad** (4.4× over ±1 = root cause). Fix = shared `load_action_scale_dict()` in `lerobot_isaac_env.so101_env_cfg` read by env + BOTH scripted normalisers; **opt-in via `LEROBOT_ISAAC_ACTION_SCALE_JSON`** (default 0.5 = no-op). Unit 3/3, env-cfg regression 22/22, full suite green.
  - **C1 gate GPU result: ee-descent RESOLVED** — `[script-dbg] ez=0.106` (was ~0.30), `xy_to_tgt≈0.000`, phase APPROACH→DESCEND→STABILIZE→CLOSE. `scripts/_residual_smoke_gate.sh` machine-checks it (log-path fixed to `.agent-state/<session>/autoresearch/wm-isaac-prod/train.log`).
  - **NEXT WALL (separate, pre-existing): grasp does not HOLD/LIFT** — `oz=0.008` through CLOSE, regrasp loop; ~15% demo success here. Gripper closure at scale 0.5 too weak; historical firm grip used `LEROBOT_ISAAC_GRIPPER_ACTION_SCALE≈3.0` (`[[so101-gripper-kinematic-floor]]`). Residual full run gated on this — campaign auto-skips HOP 3 until fixed (no GPU wasted).
- **0c/0d/0e:** Job-B decay=15000 (supervisor); P2E dropped; sim-eval deps already in `sim` env (transformers 5.3.0 + num2words) — no isolation needed.

## PHASE 1 LAUNCH (2026-07-16, `/orchestrate`)
- **Unattended window STARTED 2026-07-16T14:51:53Z.** `setsid bash scripts/gpu_campaign.sh` (supervisor
  pid 11073, own session), hop 1 `smolvla_A` launched.
- Pre-launch gate re-verified this session: chain `--dry-run` green (7 hops + C1 gate), all hop scripts +
  shared engine present (`claude_code` on `feat/autoresearch-deterministic-runner`), `action_scale.json`
  present, disk 3.0 T free, meta tests 71/71, sibling tests in `sim` env **641 passed / 2 failed** —
  both failures isolated to `test_curriculum_campaign.py::TestDerivePlaceSuccess` (mock patch-target bug
  in the left-for-review curriculum driver dc22eea; nothing in the campaign chain imports it).
- **Fix applied pre-launch:** supervisor hop 1 called the SmolVLA script bare → default 8×6000 s = 13.3 h
  overshot the 12 h cap. Now passes `--ar-seconds 5000` (8 trials, ~11.1 h). Commit `9d6079b` (pushed,
  `feature/residual-rl`).
- Monitor: `tail -f outputs/gpu_campaign/campaign.log` · transitions in
  `.agent-state/gpu-campaign/events.jsonl` · per-job logs `outputs/gpu_campaign/<hop>.log`.
- Expected: C1 gate ~T+12 h will likely FAIL (grasp HOLD/LIFT wall, gripper scale) → hop 3 auto-skipped,
  chain continues on sweeps — that is the designed gated-stretch behaviour, not a campaign failure.

## INCIDENT + RECOVERY (2026-07-16, ~T+2 min)
- **Hop 1 (smolvla_A) exited rc=0 after 2 min** — every AR trial died in ~4 s. Root cause: lerobot
  0.6.0 removed the train script's local `make_dataset` binding (now `make_train_eval_datasets`), so
  `cli_train_cached` (the `--cache_frames` monkey-patch) exited 2 before training. Three masking layers
  made it look green: train_wrapper emitted sentinel `pc_success=0.0`; the AR history heredoc crashed on
  bare `null` (NameError → record dropped); the 12h wrapper + supervisor treated rc=0 as success.
- **Fixes (all pushed):** adapters `c018aeb` (dual-path cache patch: 0.5.x local binding / 0.6.0
  `factory.make_dataset`, verified live — wrapper banner + warmup on pickplace1; tests 20/20);
  workspace `25cc467` (`gpu_campaign_ext_smolvla.sh` reruns hop 1 after the chain exits + AR heredoc
  `null`→`None` so failed trials keep their history records).
- **Hop 4 (LoRA) was exposed to the same crash** — uses `--cache_frames`; the editable-install fix
  lands automatically for its subprocesses. LoRA `make_policy` patch verified present on 0.6.0.
- **Rerun waiter bug (caught immediately):** v1 waiter greped events.jsonl for `campaign done` and
  matched stale dry-run events → launched the sweep alongside the C1 gate (~2 min GPU contention,
  gate unaffected). Killed; waiter now waits on supervisor **process liveness only** (pid 19211).
- Chain state during recovery: C1 gate demo regen healthy — early regen'd demos at the new action
  scale show FULL place success (lifted+resting+released), so the gate may genuinely PASS.

## C1 GATE RESULT (2026-07-16T16:06Z): FAIL — hop 3 skipped, chain on sweeps
- Evidence (`campaign.log` + `.agent-state/c1-residual-smoke/.../train.log`): **descent fix holds**
  (`min_ez=0.106`, `xy_to_tgt≈0.000`) but `max_oz=0.057 < 0.07`, `phases=[DESCEND,STABILIZE,CLOSE]`
  regrasp loop, never LIFT. Scripted weight was ≈1.0 throughout (step 1000 → script_frac 0.967), so
  residual noise is NOT the cause.
- **Key discrepancy for follow-up:** the SAME gate's demo regen (via `_gen_sim_demos.py` controller)
  produced FULL places (maxz 0.103–0.110, lifted+resting+released) — while the adapter's
  `scripted_grasp_phases` controller (residual path) closes on the die (oz 0.008) and slips. Same env,
  same action scale, gripper scale shared (no `GRIPPER_ACTION_SCALE` env var anywhere). ⇒ the two phase
  controllers differ in behaviour (z_high / close depth / hold criteria / step caps), NOT the C1
  normalisation. Next lever: port the demo controller's grasp sequence into `scripted_grasp_phases`
  (frame-by-frame diff), then re-gate (~1.5 h) — decision left for the attended wrap.
- LoRA hop 4 confirmed healthy post-fix: `cache patch applied (0.6.0 factory make_dataset)` in trial 0.

## HOP RESULTS LOG (running)
- **Hop 4 LoRA (16:06→23:36Z, rc=0): HEALTHY.** 11 trials × full 2400 s (plateau early-stop at 11/16),
  real metrics; best 0.144 @ rank=64 alpha=128 attn_qv lr=3e-5.
- **Hop 5 diffusion (23:36→02:36Z, rc=0): trained but banked NOTHING.** 6 full 1800 s trials,
  ~0.8 step/s ⇒ ~1400 steps < save_freq 1500 (`SECONDS_PER_EXP*25/30` assumes ≥1 step/s) ⇒ zero
  checkpoints ⇒ zero evals ⇒ all sentinel-0.0 metrics. Fix = `save_freq=SECONDS_PER_EXP/4` in
  `run_autoresearch_policy_fixed.sh` (COPY — original not edited while ACT hop executes it; **merge back
  at wrap**). Re-run queued: `gpu_campaign_ext2_diffusion.sh` (diff-camp2, fires after supervisor+ext1).
  Commit `f70e619`.
- **Hop 6 ACT** launched 02:36Z (8×2700 s); checkpoint sentinel armed (ACT ~2-4 step/s should clear
  save_freq 2250 easily — alert if trial 0 shows no ckpt by +50 min).
- **Hop 6 ACT STALE-KILLED at trial 3/8 (05:37Z)** — false positive: supervisor stale threshold 2700 s
  == SECONDS_PER_EXP, but dispatchers only print a stdout line per FINISHED trial → hop log legitimately
  silent for trial+eval. 3 real trials banked (pc≈0.0197–0.0208). **Class-fixed everywhere reachable**
  (commit `7293e5f`): ext1 stale→6300, ext2→3000, NEW ext3 finishes ACT trials 3-7 into the same
  act-camp session (engine `AR_TRIAL_START` resume, claude_code commit on
  `feat/autoresearch-deterministic-runner`); NEW `_wm_keepalive.sh` shields hop 7 (2400+300 vs 2700 —
  forwards real trial-log liveness to the hop log; real hangs still die). **Merge the stale-threshold
  rule into `gpu_campaign.sh` itself at wrap** (can't edit while running): stale ≥ trial+eval+margin.
- GOTCHA (cost 2 relaunches): `setsid <cmd> &` forks — `$!` is the dead parent; pgrep the child's exact
  cmdline for the real waiter pid before wiring dependent waiters.
- Extended chain order: hop 7 (WM, keepalive-shielded) → smolvla_A2 (ext1 184800, ~11 h) →
  diffusion_E2 (ext2 184813, ~3 h) → act_F2 (ext3 184826, ~4 h). Projected end ~13:30Z Jul 18; window
  ends 14:51Z Jul 19. Fits.

## CAMPAIGN COMPLETE (2026-07-18T02:40Z) — extension chain all green
- smolvla_A2 (10:10→21:17Z, 8 trials): rerank winner **trial_7 pc 0.129 / MSE 6.73** vs May anchor
  0.060/15.7 — ckpt `outputs/autoresearch-lerobot-policy-smolvla/trial_7/checkpoints/041660`.
- diffusion_E2 (6 trials, save_freq fix): real ckpts+evals this time; pc 0.0014–0.0021 (weak proxy).
- act_F2 (trials 3-5 via AR_TRIAL_START): grid is 6 configs → **ACT sweep complete 6/6**; best remains
  trial 0 (0.0208, baseline cfg — every HP variant scored worse).
- Totals: **43 scored trials / 5 families**, zero human interventions post-launch. GPU idle from
  02:40Z Jul 18 (chain had no further extend hop — add a loop-extend to the supervisor next time).
- **Vault ingested 2026-07-19**: source `05-Wiki/sources/2026-07-19-gpu-campaign-3day-lessons.md`,
  updates to Autonomous-ML-Training-Loop / SO-101 / Residual-RL-on-Scripted-Controllers / LeRobot,
  glossary v81 (+8 terms), log+index verified, capture archived.
- Wrap remaining (attended): dashboard N-way compare + `_open_loop_eval` refresh across ACT/SmolVLA/
  vla_jepa + new winners; merge `save_freq` fix into `run_autoresearch_policy.sh` and the
  stale-threshold rule into `gpu_campaign.sh` (both now safe to edit); `scripted_grasp_phases` port +
  re-gate (~1.5 h); surface arm bake-off runbook + runner PR.

## Related
- `plans/2026-07-12-residual-ee-descent-blocker-plan.md` (C1 fix detail) ·
  `plans/2026-07-12-phase3-model-comparison-plan.md` (excluded arm bake-off) ·
  `plans/2026-07-11-combined-today-plan.md` (vla_jepa candidate origin)
- memory: `[[carryplace-cup0-warmstart-r4-result]]` · `[[scripted-grasp-infeasible]]` ·
  `[[carryplace-place-wall-plateau]]` · `[[wm-isaac-stall-resolved]]` · `[[watchdog-report-only]]` ·
  `[[detach-long-training-jobs]]` · `[[replay-ratio-wallclock-online-isaac]]` · `[[wm-isaac-num-envs-bug]]` ·
  `[[vla-jepa-rtx3080-finetune-recipe]]` · `[[act-real-campaign-result]]`
