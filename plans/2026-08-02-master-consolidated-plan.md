# Master Consolidated Plan — 2026-08-02 (rev 2, post-critique)

**Single entry point.** Source plans stay authoritative for their own history; none were modified:
`plans/2026-07-12-phase3-model-comparison-plan.md`, `plans/2026-07-19-post-campaign-wrap-plan.md`
(Phase 5 only), `plans/2026-07-12-residual-ee-descent-blocker-plan.md` (CLOSED),
`plans/2026-06-28-act-real-campaign-plan.md`.

**Session:** `.agent-state/20260802-084858-gpu-resume-and-master-plan/`
**Rev 2** incorporates an adversarial critique (verdict RECONSIDER) that caught a live invalid GPU
run, 8 factual errors in rev 1, and ~20 genuinely-open items rev 1 had dropped.

---

## 0. INCIDENT — residual-rl-v7 was invalid and has been killed

**What happened.** v7 was launched at 09:18 resuming `ckpt_25000_0.ckpt`. Five verification checks
passed (`resume_from` present, `script_frac` collapsed to 0, `SEEDED`=0, `policy_step=25700`, no
crash). **All five are plumbing checks. None verified behaviour.** The run was killed at 10:03.

**Root cause.** `src/lerobot-isaac-env` had been switched from `feature/wm-isaac-env` → `main`:
```
cedfd19 HEAD@{2026-07-25 09:43:15 +0200}: checkout: moving from feature/wm-isaac-env to main
```
That is **after** v6 finished (2026-07-25 02:45), so v6 is valid and v7 was not. `main` is 24 commits
behind the branch and lacks `place_termination` (grep count 0), `FIX_BASE` (0), `outcome_verifier.py`,
and the staged place/carry/lift reward terms.

**Consequence.** Six exported `LEROBOT_ISAAC_*` vars (`PLACE_CUP`, `PLACE_SUCCESS_WEIGHT`,
`STAGED_REWARD`, `CARRY_Z`, `FIX_BASE`, `OBJECT_FRICTION`) were **silent no-ops**. Measured:

| | v6 (valid) | v7 (invalid) |
|---|---|---|
| `fixed_base` | `True` | `False` |
| `obj_lifted=True` occurrences | 36 | **0** |
| `ez` floor | 0.106 (grasp depth) | 0.125 |
| `oz` | rises on lift | constant 0.001 |

**Collateral.** The restored replay buffer writes into v6's own directory (`MemmapArray` pickles by
filename). ~45 min ≈ 800–1000 wrong-env transitions were written into
`.../2026-07-24_13-46-16_.../version_0/memmap_buffer/` — roughly **3–4 % of a 25 000 buffer**.
**All five checkpoint files are untouched** (mtimes still Jul 24/25). A snapshot of the run dir is at
`outputs/backup/v6_run_dir_20260802/` (1.8 G).

**Fixes applied.** Branch restored to `feature/wm-isaac-env` @ `5dcceaa`; features verified present.
`scripts/launch_residual_rl.sh` gained (a) a `RESUME_FROM` knob + EXTRA_HYDRA clobber guard, and
(b) a **sibling-SHA manifest that fails closed** — it writes `sibling_manifest.txt` per run and
refuses to resume a run whose manifest is missing (`ALLOW_UNPINNED_RESUME=1` to override) or drifted
(`ALLOW_SIBLING_DRIFT=1`). This makes today's failure structurally impossible to repeat.

**Standing lesson (route to CLAUDE.md):** *a resume is not verified until a **behavioural** assertion
passes. Plumbing checks pass on physically dead runs.*

---

## 1. Corrections to prior text (rev-1 errors included — do not trust either older version)

| Claim | Reality (verified) |
|---|---|
| rev 1: "Track A resume VERIFIED / reproducing v6's terminal state" | **False** — see §0. |
| rev 1: "~32 h retrain at 0.53 step/s, per architecture" | **~2× too high, and not per-arch.** Measured from finished logs: ACT `15000 steps / 4:05:06` = **1.02 step/s** → 60 k ≈ **16.3 h**; SmolVLA `20000 / 1:28:23` = 3.77 step/s → ≈ **4.4 h**; vla_jepa `20000 / 1:17:34` = 4.30 step/s → ≈ **3.9 h**. Full sweep ≈ **25 h, not ~96 h.** The 0.53 figure was an in-flight estimate the finished run refuted. |
| rev 1: "~6.9 h at 2196 steps/h" for v7 | 2196 steps/h is v6's rate. v7 measured ~1080–1355 steps/h (on the wrong env). **Re-measure after the gate; do not reuse either number.** |
| rev 1: "sheeprl pops `total_steps` + `learning_starts`" | Pops **five**: `root_dir`, `run_name`, `algo.total_steps`, `algo.learning_starts`, `checkpoint.resume_from` (`sheeprl/cli.py:47-52`). The practical conclusion still holds. |
| rev 1: "`lerobot_isaac_synthetic.merge_datasets`" | No such symbol. The module is **`merge_utilities.py`**. The v2.x-hardcode substance is correct (`:210` `meta/episodes.parquet`, `:264-269` `episode_{i:06d}.parquet`). |
| rev 1: "496 KB/frame" | **441 KB/frame** (8.3 G / 18 804). |
| rev 1: "the runner branch is the sole only-on-disk work" | **Three** items: the runner branch, `autoresearch feature/auto-lora` (9 commits, `[origin/...: gone]`, no merge-base), and the **uncommitted `scripts/launch_residual_rl.sh`** that Track A depends on. |
| "3 open draft PRs: configs#1, autoresearch#1, adapters#1" | **All MERGED.** Zero open PRs across all 10 repos. |
| "`robot-data-recorder feat/deploy-runner-hardening`" | Branch is in **`robot_data_runner`**. Remotes use underscores; `src/` dirs use hyphens. |
| "5-candidate bake-off" | **Not runnable.** trial_7/LoRA: `d435_rgb`, `state[6]`, CHW. ACT/SmolVLA/vla_jepa: `overhead`, `state[12]`, HWC. Its MSE 6.35 vs 40–58 is measured on a **different dataset** — not evidence. |
| "frames stored as av1 video" | **False.** PNG bytes inline in parquet (`struct<bytes,path>`). No `videos/` dir. |
| "quantitative framing gate is a mandatory precondition" | **The script does not exist** (0 hits in `scripts/`, `src/robot-data-runner/`). |

---

## Track A — Residual-RL, relaunch under a verified env

### A0 — Env pinning (DONE this session)
1. ✅ `src/lerobot-isaac-env` → `feature/wm-isaac-env` @ `5dcceaa`.
2. ✅ Sibling-SHA manifest + fail-closed resume guard in `launch_residual_rl.sh`.
3. ⬜ **Commit the launcher change** — currently only-on-disk and load-bearing.
4. ⬜ **PR `lerobot-isaac-env feature/wm-isaac-env`** (Track E2). Until merged, *the correct code lives
   only on a branch*, and any `setup.sh` / checkout will silently strip it again. Merging the remote
   without also fixing the local checkout fixes nothing — do both.
5. ⬜ Generalise the fix: `LEROBOT_ISAAC_ACTION_SCALE_JSON` is exported by only
   `launch_residual_rl.sh`, `_residual_smoke_gate.sh`, `_gen_sim_demos.py`.
   `launch_warmstart_full.sh`, `launch_warmstart_cur.sh`, `_run_autoresearch_wm_isaac.sh` and the
   probes still silently take the 0.5 default — **the same trap that burned four 13 h runs is live on
   every other sim entrypoint.**

### A1 — Behavioural smoke gate (RUNNING)
`bash scripts/_residual_smoke_gate.sh` → `outputs/gpu_campaign/c1_gate_20260802.log`.
Demo-regen and scale-measure are skipped (cached); `clamp_innocent=False` so the gate proceeds.

**This gate existed the whole time and was not used before the 13 h launch. That is the process error.**

⚠️ **The shipped PASS bar is too weak — do not use it as-is.** `_residual_smoke_gate.sh:104` computes
`ok = descended and max_oz > 0.07 and ("CARRY" in phases)`, where `max_oz` is a **max over the entire
log**. One lucky attempt out of the ~6–9 the smoke observes clears it. The script's own comment block
(`:69-76`) records that this signal "swung PASS/FAIL on identical mechanics" across rounds 3-6 — the
response then was to raise `STEPS`, not to fix the aggregation, so the effective power is still ≈ n=1.

**Revised PASS bar (rate, not max):**
- `fixed_base=True` (env-integrity assertion — this is what v7 failed), AND
- **≥ 2 distinct grasp attempts** reach `oz > 0.07`, AND
- CARRY entered **on one of those attempts, evaluated on that attempt's own CARRY line**
  (`obj_lifted=True` or `oz>0.07` there) — v6's log contains force-advanced
  `phase=CARRY obj_lifted=False oz=0.008` lines, so bare `"CARRY" in phases` is vacuous, AND
- `min_ez ≤ 0.110` (grasp depth actually reached — the currently-running smoke floors at 0.120).

A single-attempt pass is **INCONCLUSIVE → re-run**. Two consecutive inconclusive = FAIL; diagnose
rather than launching 13 h on a coin flip.

**"Distinct attempt" is well-defined and parseable.** `[script-dbg]` prints only on phase transitions
(`scripted_grasp_phases.py:182`) and carries `t=` and `prev=` plus a ` REGRASP` tail the current regex
discards. **Every attempt opens with a `phase=APPROACH` line** (`prev=SETTLE` at episode start, or
REGRASP-tagged; `MAX_REGRASPS=2` ⇒ ≤3 attempts/episode × 3 episodes = the "~6-9"). Segment on APPROACH
entries and take a per-segment max `oz`. **This grouping is mandatory** — one successful grasp prints
`oz>0.07` on up to three lines (CARRY/LOWER/RELEASE entries), so naive line-counting double-counts a
single grasp as two "attempts" and re-creates the very bug being fixed.

**Implementation notes for the parser swap** (replaces the heredoc at `:87-109`):
- Return **exit 2** for INCONCLUSIVE. The shipped script and `gpu_campaign.sh` treat any non-zero as
  FAIL, so the caller needs updating too or every inconclusive silently becomes a hard fail.
- Re-running the gate **`rm -rf`s the session dir** (`:77`) and overwrites `c1_smoke.log`. The
  "two consecutive inconclusive" protocol therefore **requires copying the train log out per attempt** —
  only the gate's stdout survives, in distinct `c1_gate_*.log` files.

**Own the trade-off:** with ≤9 attempts and historical per-attempt lift rates in the ~10–30 % band,
"≥2 lifting attempts" will land INCONCLUSIVE fairly often on genuinely-working mechanics. That is a
deliberate **false-FAIL bias** — it buys protection against a 13 h coin-flip. It is not strictly more
powerful than the old bar, and should not be described as such.

### A1-REF — the empirical reference distribution (measured from v6, 2026-08-02)

Segmenting v6's own trace by the APPROACH rule above gives the reference rates the gate must be read
against. **Without these, "the smoke looks bad" is not a verdict.**

| metric | v6 (110 attempts, valid run) |
|---|---|
| attempts reaching `ez ≤ 0.110` (grasp depth) | **47 / 110 = 42.7 %** |
| attempts achieving a lift (`oz > 0.07`) | 23 / 110 = 20.9 % |
| per-attempt `min_ez` | median 0.114, min 0.098 |

**Use the depth rate, not the lift rate, as the primary signal** — it is ~2× more frequent and
therefore ~2× more powerful at small n:
- P(0 depth-reaches in 5 attempts | 42.7 %) = **6.2 %** — suggestive, NOT conclusive.
- P(0 depth-reaches in 9 attempts | 42.7 %) = **0.6 %** — conclusive.
- By contrast P(0 lifts in 5 | 20.9 %) = **30.9 %** — no evidence at all.

⇒ **The gate must observe ≥ 9 attempts before a zero-result may be called FAIL.** A zero-result at
5 attempts is INCONCLUSIVE by construction. Note also that v6 degrades within a run — its attempts
1-3 hit `ez` 0.106/0.106/0.107 and later ones drift to ~0.120 — so early attempts are the most
informative and a gate that only samples late attempts will under-read.

### A2 — Full run (only on A1 PASS)
```bash
CKPT=<abs path>/2026-07-24_13-46-16_dreamer_v3_isaac_so101_42/version_0/checkpoint/ckpt_25000_0.ckpt
setsid env SESSION_ID=residual-rl-v8 STEPS=40000 RESUME_FROM="$CKPT" \
  ALLOW_UNPINNED_RESUME=1 `# v6 predates manifest pinning; A0.2 pins everything from here on` \
  EXTRA_HYDRA_APPEND='algo.learning_starts=0' \
  LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT=1.0 LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS=1 \
  LEROBOT_ISAAC_BC_WEIGHT=0.0 \
  bash scripts/launch_residual_rl.sh > outputs/gpu_campaign/residual_full_20260802_v8.log 2>&1 &
```

**Why those env vars (traced to source):**
1. The residual `w` clock is a **per-process counter** (`_wm_isaac_entry.py:600-607`, `:660-662`), not
   sheeprl's `policy_step`. A naive resume re-runs the whole scripted→actor handoff on CARRY and
   LOWER — the only phases the actor had won. `DECAY_STEPS=1` collapses it.
   **Never `WEIGHT=0`** (returns before patching, `:587`, disarming the scripted base everywhere).
   Note `BLEND_SAFE_PHASES = {CARRY, LOWER}` (`scripted_grasp_phases.py:203`) — all other phases are
   script-forced at 1.0 regardless, so v6 never ran "pure actor" end-to-end.
2. The BC-weight clock resets identically (`_wm_isaac_entry.py:295-299`) → `BC_WEIGHT=0.0`.
3. `algo.learning_starts` default 1024 becomes `+= start_iter` → 26 025 ⇒ ~28 min of zero-gradient
   collection. `=0` reclaims it.

**Cannot be changed on resume** (re-read from the checkpoint's `config.yaml`): `batch_size`,
`replay_ratio`, `checkpoint.every` (stays 5000), `max_episode_steps`, `ent_coef`, `horizon`.

### A3 — Abort criteria (NEW — the check that was missing today)
- **T+40 min:** assert `fixed_base=True`, ≥1 `obj_lifted=True`, `ez` reaches ≤0.110, and at least one
  episode ends before the 701 cap. **Any failure → kill immediately.**
- **Wall-clock:** re-measure steps/h in the first hour. If the projected finish exceeds the 13 h
  `timeout`, cut `STEPS` rather than lose up to ~4 h of tail to a SIGTERM at the cap.
- Leave `COLLAPSE_KILL=0` (report-only): `COLLAPSE_MIN_STEP=15000` is already exceeded at 25 001.
- **Do not delete or move** `logs/runs/.../2026-07-24_13-46-16_.../` — the restored buffer lives there.

### A4 — Readout
Per-episode rewards in `train.log` are canonical (TB is too coarse; the trace under-counts placements
because place-termination ends the episode before `RELEASE` prints). v6 reference: 42 episodes,
4 placements ≈ 9.5 %, **place-rate survived the full handoff**.

> **SIM/REAL FIREWALL (restored — rev 2 dropped it).** Track A's output is **sim-only**. `ckpt_25000`
> and any v8 checkpoint are a sim bake-off candidate and a resume point — **never** a Track-B or
> hardware candidate. sim2real ≈ 0 for this task. The source plan said so explicitly; the
> consolidated plan had zero occurrences of "sim2real"/"sim-only" and used the bare word "candidate"
> for both tracks 25 lines apart.

**⚠️ The thresholds below are not decidable as written** — Wilson 95 % CI on v6's 4/42 is
**[3.8 %, 22.1 %]**, so any rate in that band is indistinguishable from "flat", and detecting a real
move 9.5 % → 20 % at 80 % power needs ~150–200 episodes per arm (v6 had 42). Two readers can reach
opposite verdicts on the same number. **Fix the gate before reading the run** — see the decision
pending below.

| Outcome | Next |
|---|---|
| rising vs 9.5 % | extend to 60 k from the v8 final ckpt (reuse `scripts/gpu_campaign.sh` chaining — do not hand-roll a supervisor) |
| flat ~9.5 % | design holds, does not improve at this scale. Bank the ckpt (**sim-only**); free the GPU for Track C retrains |
| collapses | check the sibling manifest **first**, then the config diff. Under the old plan this would have been misread as "residual-RL regresses on resume" and retired the one approach that worked |

---

## Track B — Real-arm bake-off (HARDWARE, human-gated)

> ## ⛔ TRACK B IS BLOCKED — DO NOT RUN IT AS WRITTEN
> A domain safety review returned **12 correctness-blockers**, each verified at source. Rev 1 and
> rev 2 of this plan contained **zero** physical-safety content (keyword counts across the whole
> document: e-stop 0, power 0, torque 0, operator 0, dry-run 0, workspace-clear 0). The six-layer
> safety stack exists — in `docs/runbook/10-deploy-to-hardware.md`, which this "single entry point"
> never cited. **Clear B-SAFETY below before any hardware session.**

### B-SAFETY — blockers that must be closed first

> ## Status 2026-08-02 — S1/S3/S4/S5/S7 + C1 IMPLEMENTED
> Committed as **`703f62f`** on `src/robot-data-runner` branch `feat/deploy-runner-hardening`
> (local only — not pushed; see E1).
>
> | Item | What landed |
> |---|---|
> | S3 | `safety_limits.py` — two-layer joint-limit clamp ported from `arm_motor_writer.py`, cal ∩ hardcoded floor, `elbow_flex [-10,90]` preserved. Applied at **all four** `send_action` sites via `_clamp_and_warn`; warns once per joint per episode |
> | S4 | `ramped_home()` ported; used at **all four** home sites before torque-off. False "ramps" docstrings removed |
> | S1 | `use_degrees` defaults **True** on all three real-motor entry points (`--as-normalized` reverts); `cli_sweep` stays normalized **by design**, commented. Startup banner prints resolved °/step + °/s. `max_deg_per_s_ceiling=90` **refuses to start** unless `--allow-fast` |
> | S5 | `require_interactive()` refuses to start the eval loop on a non-tty; mid-run `EOFError` aborts instead of fabricating a verdict. Gated on `run_episodes` only — `run_policy`/`replay`/`sweep` have no designed interlock and are intentionally unattended |
> | S7 | New **ABORTED** outcome, excluded from `pc_success`/`mean_ep_len`/`intervention_rate`, reported separately; arithmetic factored into a pure testable `_aggregate_records()`. Banner states there is no software e-stop |
> | C1 | `--save-frames DIR` — one frame per episode + manifest with shape, dtype, per-channel **mean pixel value** (the signal that would have caught the 78-vs-101 brightness drift) |
>
> Tests **61 passed / 3 skipped** (was 37/3), ruff clean.
>
> ⚠️ **NOT hardware-verified — no arm was attached.** Every change is fake/unit-tested only. Safety
> code that has never driven the thing it protects is a hypothesis: the first real session must treat
> these as untested paths and ladder up from 1.0 regardless.
>
> ⚠️ **Expect the ceiling to reject your first command.** The old `5.0` resolves to ~135–150 °/s under
> either unit system, so `robot-data-run --max-relative-target 5.0` now **refuses to start**. That is
> intended (it forces the S2 re-derivation) but will read as a bug to whoever hits it first.
>
> **Still open: S2** (re-derive the ladder — needs the arm) and **S6** (workspace clearance, operator
> position, supervisor, power-disconnect method, per-checkpoint dry-run — procedural, belongs in the
> session checklist, not in code).

**S1. The clamp ladder is not in degrees.** `--use-degrees` is never passed
(`config.py:89 use_degrees: bool = False`), so body joints normalize to `RANGE_M100_100`
(`so_follower.py:50`) and `--max-relative-target 5.0` means **2.5 % of each joint's calibrated range
per step**, not 5°. Measured against the live calibration at 30 Hz:

| joint | per step | per second |
|---|---|---|
| wrist_roll | 9.00° | **270 °/s** |
| wrist_flex | 5.08° | 152 °/s |
| shoulder_lift | 5.03° | 151 °/s |
| elbow_flex | 4.57° | 137 °/s |
| gripper (`RANGE_0_100`) | 6.49° | full jaw travel in **0.67 s** |

Every document in the chain mislabels this "deg" — `cli_eval.py:32-33`,
`docs/runbook/10-deploy-to-hardware.md:119-124`, and the source plan's "ramp 1→3→5°". The operator's
mental model of "5 is a small safe number" is wrong. **Fix:** pass `--use-degrees` and re-derive the
whole ladder in real degrees, and correct the mislabels.

⚠️ **`--use-degrees` alone does not make 5.0 safe — re-deriving the ladder is the load-bearing part.**
In DEGREES mode clamp 5.0 means 5°/step = **150 °/s on every joint**. Note also that lerobot 0.6.0's
own `config_so_follower.py:42` defaults `use_degrees = True`; only the runner path pins it False. So
anyone switching to lerobot-native `lerobot-record`/`lerobot-rollout` silently changes what the clamp
number means **again, in the opposite direction**.

**S2. Clamp 5.0 has no supporting evidence and contradicts the scoreboard.** The 5.0 lock rests on
episodes 1-6, which line 11 of the scoreboard **voids** ("CAMERA REALIGNED … all prior eps void") —
and those episodes ran while the camera was 62 px / −98 px off, so the policy was consuming OOD
images; how fast the arm had to slew to execute garbage commands says nothing. Worse, "≤3.0 cripples
reach/grasp" is **false in that same table**: ep 3 at clamp 3.0 is logged "carry/release — reach+grasp
OK". Only clamp 1.0 was demonstrably too slow. **Fix:** re-derive the ladder post-realignment from
1.0; do not exceed 3.0 without a post-realignment episode showing 3.0 is rate-limited.

**S3. No absolute joint-limit clamp exists in the runner.** `max_relative_target` bounds per-step
delta only; nothing bounds cumulative excursion, so a policy that persistently saturates the clamp
walks the arm into a joint limit or the table at full clamp rate. Verified: zero `np.clip` /
`joint_limit` hits across `src/robot-data-runner/src/`. The sibling package already implements a
two-layer limit (`lerobot_isaac_deploy/arm_motor_writer.py:25-38`, cal-derived ∩ a hardcoded floor
incl. `elbow_flex: [-10, 90]` "avoids table"). **Fix:** port it into `_build_robot` as a target clip
before `send_action`. Highest-value hardware-safety change available.

**S4. `--home-on-exit` does not home, and the arm goes limp after every episode.** It is one
`send_action` to normalized zero — itself clamped, so it travels at most `max_relative_target` units —
then `sleep(0.5)`, then `disconnect()`, which runs with lerobot's default
`disable_torque_on_disconnect: bool = True`. Torque off ⇒ **the arm drops under gravity from wherever
it stopped, still gripping the die** — ~60 times across the session. The runner docstrings claiming it
"ramps to zero/center" describe `ramped_home()` from a **different package** Track B does not use.
**Fix:** port `ramped_home()`, or state in the protocol that the arm goes limp at the end of every
episode and require the fall path to be clear.
⚠️ Related: the zero pose "often goes straight through the table on first calibration" (runbook
Layer 5). Today the clamp masks that — **raising the clamp turns `--home-on-exit` into a table strike.**

**S5. Running the eval non-interactively deletes the interlock AND fabricates a 0/N scoreboard.**
`episode_runner.py:55-56` catches `EOFError` and logs *"stdin closed; assuming continue"*;
`task_specs.py:77-78` catches it and returns `"n"` = FAIL. Under `nohup`/`setsid`/a pipe — **the exact
detach habit this project has institutionalised** (`[[detach-long-training-jobs]]`) — all N episodes
run back-to-back with a live torqued arm, no operator acknowledgement, and a clean-looking
`pc_success=0.0` JSON. C4.2 documents this foot-gun for the *recorder* and never carries it to the
*runner*. **Fix:** Track B must run in a real tty, always.

**S6. Missing baseline safety content** (all absent from rev 1/2, all present in
`docs/runbook/10-deploy-to-hardware.md` — cite it): workspace clearance; operator standing position
relative to the reach envelope; who supervises; the physical power-disconnect method
("Stand next to the e-stop / power strip", :270); and the **mandatory per-checkpoint dry-run**
(motors off, inspect the action stream for NaN / large jumps / oscillation) — which matters most for
vla_jepa, the one candidate whose real closed-loop behaviour is explicitly unverified.
Also restore the source plan's rule: **first live episode of each policy at clamp 1.0.** Under rev 2,
SmolVLA and vla_jepa — which have *never* driven this arm — would have their first-ever hardware
motion at the top clamp, scored.

**S7. There is no software e-stop.** `safety.py:52-60` only flips a boolean, and a Python signal
handler cannot interrupt a blocked serial `sync_read` (PEP 475 retries on EINTR) — **Ctrl-C is not an
e-stop.** The only real stop is physical power. Additionally, Ctrl-C during a reset prompt silently
records a spurious FAIL that counts toward the B2 stop-loss; the correct abort is typing `abort` at
the verdict prompt (`task_specs.py:79-80`). Mandate that; forbid Ctrl-C as the abort mechanism.

### B-CORRECTIONS — things rev 2 asserted that are false

**C1. Rollout recording does not exist.** Rev 2 said `physical-reset-agent`,
`binary_success_verifier`, and rollout recording were "all installed, none wired". The first two are
installed; **the third does not exist**. The runner's only capture is `--actions-out`, a JSONL of
actions with **no images and no state** (`runner.py:129-133`). This matters twice: the DAgger set
(B4.3/F5) has no capture mechanism, and **a 0/60 outcome cannot be diagnosed post-hoc** for framing,
exposure, or wrong-node problems. *Dumping `obs["overhead"]` once per episode is a few lines and is
the single cheapest diagnostic insurance available.*

**C2. `binary_success_verifier` cannot produce a Track-B verdict.** Every task predicate needs object
pose (`object_in_bin` → `state["object_position"]`, `object_lifted` → `object_height`); the rig has
one overhead RGB camera and **no pose estimator**. Wiring it either cannot run or gets fed a
fabricated pose that reads as a real verdict. The working verdict is `prompt_user_observer` (human
y/n) — what session 1 actually used.

**C3. `physical-reset-agent` should stay manual.** Its clamp contract references
`arm_motor_writer.py`, which is in a **different package**; the clamps it promises never to weaken
do not exist on the Track-B path. Its `at_home_pose` gate uses `eps=0.05` as a 6-D Euclidean
tolerance against normalized units spanning −100..100 — effectively never satisfiable. `dry_run`
defaults to **true**, so a naive wiring silently does nothing and the next episode starts unreset.
And the reset that actually matters — re-placing the die within ±2 cm and emptying the cup — **cannot
be automated at all**. The `input()` gate is the sole interlock keeping hands out of a torque-live
workspace. **Automate the verification (log final pose + a frame), never the motion.**

**C4. The framing gate as specified validates the wrong pipeline.** Deeper than rev 2's warning: the
**training data itself came through pyrealsense** (`recorder.py:19` → `d435.py`; the recorder has no
V4L2 path at all), while the runner reads V4L2 via `OpenCVCamera`. **A pyrealsense gate compared to
the dataset reference compares the training path to itself** and is structurally incapable of
detecting a deploy-path defect. Unpinned differences that each invalidate every episode equally and
are invisible in the scoreboard: `fourcc=None` (YUYV vs MJPG negotiated per session), `backend=ANY`,
two independent auto-exposure/AWB stacks (the 78-vs-101 brightness is a *symptom*), `warmup_s=1` vs
multi-second UVC AE convergence, cached `read_latest()` vs the recorder's blocking
`wait_for_frames()`, and — critically — **the D435 exposes IR nodes that also pass lerobot's 640×480
validation**, so a near-greyscale frame is accepted silently.
**Better gate:** re-run `scripts/_open_loop_eval.py` on frames captured **live through the runner's
own camera object**. That measures the deploy path by construction. Use a stable
`/dev/v4l/by-id/...` path, not `/dev/video4`, and assert a 3-channel colour frame with plausible
saturation.

**C5. The negative control tests the wrong half of the stack.** Human teleop drives follower-from-
leader — **the camera is not in the loop at all**, nor is state mapping, image normalization, action
decoding, the clamp, or the 30 Hz cadence. It cannot distinguish the most likely cause of a 0/60: a
broken observation path. **Use instead:** (a) `robot-data-run-replay` — replay a *recorded successful
training trajectory* through the same runner/clamp/rate; failure there isolates the action path
independent of perception; and (b) the live-through-the-runner open-loop eval from C4.

**C6. The three candidates will run at three different effective control rates** — an uncontrolled
confound in the headline metric. ACT (small conv+transformer) and SmolVLA (500 M VLM) will not
sustain 30 Hz equally, and `send_action` performs a *second* full bus `sync_read` whenever
`max_relative_target` is set. Because the clamp is a per-step rate limit, a slower candidate moves
the arm slower **and** time-stretches its action chunk relative to training — so the ranking partly
measures inference latency. The loop is `if slack > 0: sleep(slack)` with **no else branch and no
overrun log**, and the summary prints the *requested* duration. **Fix:** log achieved Hz per episode
and add a rate column to the scoreboard.

**3 candidates** (ACT-15k, SmolVLA-020k, vla_jepa-020k) — all `overhead` + `state[12]`.
trial_7/LoRA deferred to Track D.

### B0 — Prerequisites that do not exist yet
1. **`scripts/_framing_gate.py`** — ORB median feature shift + mean brightness vs a reference frame
   from `so101-pickplace-new`. Gate: shift < 10 px AND brightness within ~10 % of 101. Exit non-zero
   on fail. ⚠️ The gate captures via pyrealsense but the runner reads V4L2 — **verify both paths see
   the same geometry**, or the threshold measures the wrong camera.
2. **Episode-loop artifacts — see B-CORRECTIONS C1-C3, which supersede this item.**
   `physical-reset-agent` and `binary_success_verifier` are installed but must **stay unwired**
   (C2: the verifier needs object pose the rig cannot produce; C3: the reset that matters cannot be
   automated and the `input()` gate is the only interlock). **Rollout recording does not exist at
   all** (C1) — the runner's only capture is an action JSONL with no images.
   **Build instead:** dump `obs["overhead"]` once per episode. A few lines, and without it a 0/60
   outcome is undiagnosable.
3. **Negative control — use the mechanical ones, not human teleop (C5).** Teleop drives
   follower-from-leader with the camera *entirely out of the loop*, so it cannot detect the most
   likely cause of a 0/60. Use instead: (a) `robot-data-run-replay` on a recorded **successful
   training trajectory** through the same runner/clamp/rate — failure there isolates the action path
   independent of perception; and (b) `scripts/_open_loop_eval.py` re-run on frames captured **live
   through the runner's own camera object** (C4).

### B1 — Preconditions
1. Framing gate PASS. Re-run at session start, every 25 episodes, after every break.
2. **Fix the lighting.** Session 1 measured brightness 78 vs training 101 (~23 % dark) and *accepted*
   it. Post-realign shift was 7.3 px magnitude — the rig only just met the 10 px bar.
3. `pixi run -e record robot-data-check --connect`; confirm `so101_follower` calibration.
4. **Re-verify the camera index — do not assume `/dev/video4`.** Vault `05-Wiki/entities/SO-101.md`
   records "Camera Index Instability" as a known gotcha.
5. Runner from `robot_data_runner @ feat/deploy-runner-hardening` (Track E1). Its mapper fixes
   (joint-order permutation, 6→12 state pad) are documented in
   `04-Archive/2026-07-04-act-real-first-hw-success.md` — verify, don't re-derive.
6. **Clamp ladder: re-derive from 1.0 in real degrees (S1+S2). Do NOT carry 5.0 forward.**
   The "≤3.0 cripples reach/grasp" claim is **disproved by the scoreboard itself** — ep 3 at clamp
   3.0 is logged "reach+grasp OK". Only 1.0 was ever too slow, and every ladder episode ran
   pre-realignment on OOD images. Do not exceed 3.0 without a **post-realignment** episode showing
   3.0 is rate-limited. **First live episode of each policy at 1.0** (S6) — SmolVLA and vla_jepa have
   never driven this arm.

### B2 — Protocol: sequential with a stop-loss (CHANGED)
Rev 1 scheduled 60 blind episodes. The plan simultaneously predicts all three land low — so spend
evidence-first:
- Score candidate 1 to **10 episodes**. **If 0/10 at the same failure stage → stop the session** and
  go to Track C. That is 1/6 of the day for the conclusion the plan already expects.
- Otherwise continue to 20/candidate, paired spot order (1-8 center, 9-16 ±2 cm NSEW, 17-20 ±3 cm diag).
- Use the **5-stage progress rubric** from `03-Resources/Technical/act-so101-training-research.md`
  (reach → grasp → reach-container → release → in-container), not just the binary verdict — it is what
  makes a 0/N result diagnosable.
- Restart scoring from episode 1 if B1.2 changes the lighting; state the decision in the scoreboard
  before episode 1.

### B3 — Reading it honestly
At N=20 the 95 % CI is ±~22 pp; **gaps < 3/20 are ties.** Rev 1 contradicted itself by adding a
"vla_jepa ≥ SmolVLA → invest in the WM path" row immediately after declaring that unreadable.
**Corrected:** a tie is *not* evidence for the WM path. Only a gap ≥ ~6/20 warrants redirecting
investment; anything less means "undecided, get more data".

### B4 — Cheap levers to try BEFORE spending a hardware day
All three are cheaper than 60 arm episodes and could change the ranking:
1. **ACT temporal ensembling** — eval-time only, no retrain (the fields are read solely in
   `ACTPolicy.select_action`, never in `forward()`). Currently **0 hits** in `scripts/` and
   `src/*/src/`. Aimed exactly at the carry/release failures on the scoreboard.
   ⚠️ **Both settings are required together.** The coefficient alone raises `NotImplementedError` at
   config load (`configuration_act.py:138-142` enforces `n_action_steps == 1` whenever the coeff is
   set). Verified: the ACT-15k checkpoint has `n_action_steps: 100`, `temporal_ensemble_coeff: null`.
   🚧 **BLOCKER — there is no carrier for these flags on the deploy path.** `policy_loader.py:68`
   calls `PreTrainedConfig.from_pretrained(policy_path)` with **no `cli_overrides`**, and `cli_eval.py`
   exposes no `--policy.*` passthrough. So this lever requires *first* either (a) editing a **copy**
   of the checkpoint's `config.json`, or (b) adding an override flag to the runner. Pick one and do it
   before claiming the lever is "free". Every reviewer verified the lerobot constraint; none checked
   whether the flags could be delivered.
2. **`image_transforms.enable`** — off in all three runs. Rev 1 called it "a free lever never pulled"
   and then scheduled it *after* the bake-off. Pull it first. (Note: this one DOES require a retrain.)
3. A ~20-episode **DAgger correction set** — see Track F.

> **Deploy-time normalization is NOT a variable here.** `--dataset-root` does not supply normalization
> stats: `policy_loader.py:81-93` → `lerobot/policies/factory.py:304-352` takes the `if
> pretrained_path:` branch for ACT/SmolVLA/vla_jepa and loads the processor pipeline from the
> **checkpoint's own** saved JSON, ignoring the `dataset_stats` kwarg. This is correct behaviour — but
> it means a `--dataset-root` mismatch is **not** a viable explanation if a candidate underperforms.
> Its only real effect is supplying feature shapes when the checkpoint config is incomplete.

---

## Track C — Data-scale lever (150+ demos) — pilot first

### C0 — Baseline (verified)
`datasets/local/so101-pickplace-new`: **50 ep / 18 804 frames / 48 successes** (fails: ep 0, 34),
30 fps, one camera `observation.images.overhead` 480×640 PNG-in-parquet, `state[12]`, reward/done
present, 10.4 min total footage, **441 KB/frame**. All three candidates trained on the same 48.

### C1 — Target regime
`03-Resources/Technical/act-so101-training-research.md:35-61`: 50 ep failed; 72 ep → 60 % ID / 10 % OOD;
**150+ at 25/bin + diversity → 90 % ID / 75 % OOD**. **That rig used two cameras (top + front); ours is
overhead-only.** Unclosed gap — do not promise 90 % from data alone.
Lift from the same page: the **5-stage progress rubric** and the **held-out ID/OOD discipline**.

### C2 — PILOT FIRST (NEW — 30 episodes, not 205)
Rev 1 specified a full 205-episode designed experiment for a task that currently scores 0/4 on the arm
with **admittedly unmeasured** recording throughput. That is hypertrophy. Instead:

**Pilot: 30 episodes, one bin, 5 yaw levels.** It validates — before ~6 h of human time is committed —
(a) the real throughput, (b) the merge path, (c) `_rebuild_episode_labels.py`, (d) the condition
ledger, and (e) that added data moves the needle at all. Then one retrain, one re-eval, then decide.

### C3 — Full matrix (only if the pilot converts)
6 bins × 5 yaw (−45/−22/0/+22/+45) × 5 demos = 150; +15 recovery, +20 OOD holdout, +~20 natural
failures ⇒ ~205 recorded (~185 success) → with the existing 48 ≈ **213 BC successes**.
Object type (3) and container (2) rotate within each cell. Bin B2 = the existing 50-demo position.
⚠️ Time estimate **~4.5–6 h is derived, not measured** — the pilot replaces it with a real number.
**Track a success learning-curve across sessions** — that is what tells you when to stop recording.

### C4 — Recording gotchas
1. **Cannot append** — `LeRobotDataset.create()` → `mkdir(exist_ok=False)` → `FileExistsError`
   (`dual_writer.py:177`). One fresh `--repo-id` per session, then merge.
2. **Never under `nohup`/`setsid`/a pipe** — non-tty → `NullKeyListener` (`recorder.py:172-176`) →
   every episode runs to `max_steps` and auto-saves as **FAILURE**.
3. `--max-steps 1200` (40 s). Default 18000 = 10 min + ~5.5 GB per forgotten keypress.
4. `meta/episode_labels.json` is written only in `finalize()` under a bare `except: pass`
   (`dual_writer.py:354-371`). A crash loses the whole session's sidecar and `--successes_only` then
   silently trains on everything (`policy_lerobot.py:441-445`).
5. Schema locked: `--camera-name overhead`, `--resolution 640x480`.
6. `--successes_only` is silently ignored for multi-dataset training (`policy_lerobot.py:457-467`).
7. Disk: 150 eps ≈ 25 GB (`parquet`) / ~84 GB (`dual`). **`datasets/` is gitignored, single box, no
   backup** — add one before recording days of human effort.
8. **Plug in the leader arm** (`/dev/ttyACM1`).

### C5 — Two helpers that must be written
1. `scripts/_rebuild_episode_labels.py` — merged parquet `reward` → `meta/episode_labels.json`
   (last-frame reward > 0 ⇒ success, `dual_writer.py:279`). `aggregate_datasets` does **not** carry
   the sidecar, so `--successes_only` breaks after every merge without this.
2. A per-episode condition ledger (CSV): bin / yaw / object / container vs episode index. The recorder
   stores no per-episode metadata — which is exactly why the **existing 50 episodes' scene conditions
   are unknown today** and their coverage of the C3 matrix is unverifiable.

### C6 — Merge + retrain
```bash
lerobot-edit-dataset --operation.type merge \
  --new_repo_id local/so101-pickplace-v2 \
  --operation.repo_ids "['local/so101-pickplace-new','local/so101-pickplace-v2-s1']" \
  --operation.roots  "['datasets/local/so101-pickplace-new','datasets/local/so101-pickplace-v2-s1']"
```
⚠️ **`--new_repo_id` is mandatory AND top-level.** Omitting it raises
`ValueError: --new_repo_id is required for merge operation` before any I/O
(`lerobot_edit_dataset.py:812`). It is a field of `EditDatasetConfig`, **not** `MergeConfig` —
writing `--operation.new_repo_id` is rejected by draccus with `unrecognized arguments` (all four
merge examples in the script's own docstring, `:78-104`, use the bare form).

**Do not use `merge_utilities.py`** — v2.x-hardcoded, emits an empty dataset silently.

Merge **does** preserve `reward`/`done` (both datasets must declare identical features; enforced, not
incidental) and **does** recompute `meta/stats.json` — via an analytically-correct weighted combination
of each source's stats (`aggregate.py:701-724`), not a full rescan. It does **not** touch
`meta/episode_labels.json`, which is why C5.1 exists.

Retrain cost at **measured** rates for 60 k steps: ACT ≈ 16.3 h, SmolVLA ≈ 4.4 h, vla_jepa ≈ 3.9 h
(≈ 25 h total sweep).

**Train from scratch — but not for the reason rev 2 gave.** Rev 2 justified it by "normalization
shifts / `train_config.json` bakes the old episode list". Both are wrong for a lerobot 0.6.0
`--policy.path=<ckpt>` fine-tune: `lerobot_train.py:314-349` rebuilds the normalizer from the
**current** dataset's freshly computed stats, and the baked-episode-list problem belongs to the
separate `--resume` codepath. The real justification is simply a **clean, controlled 3-architecture
comparison** — fine-tuning would confound "more data" with "warm start".

> **Lever REMOVED — the "re-encode to 128–256 px" option in rev 2 was wrong.** It was added on the
> generalist critic's recommendation and refuted with code evidence by the domain review:
> (a) `features_equal_for_merge` (`lerobot/datasets/feature_utils.py:125`) requires **exact shape
> equality** across merged datasets, so down-resing only the old 50 while C4.5 locks new recordings to
> 640×480 guarantees a hard `ValueError` at merge — it **destroys** mergeability rather than
> preserving it; and (b) lerobot 0.6.0's `lerobot-edit-dataset` **has no resize operation at all**
> (`convert_image_to_video` and `reencode_videos` touch only codec/container, never pixel dimensions).
> Its original justification (av1 random-frame seeking) also no longer applies — this dataset is
> PNG-in-parquet. **If a smaller image size is wanted, it must be decided BEFORE recording and applied
> uniformly, accepting that the existing 50 episodes then cannot be merged in.**

---

## Track D — Make trial_7 / LoRA comparable (deferred)
Both are `d435_rgb` + `state[6]` + CHW. Their offline advantage is a dataset artifact. The only path:
retrain both recipes on `so101-pickplace-new` (or the merged set) as part of the Track-C sweep.
**Reuse** the `info.json` schema-compatibility check in
`05-Wiki/concepts/WM-Schema-Viewpoint-Incompatibility.md` rather than writing a new checker.

---

## Track E — Commit / PR debt
Verified via `git ls-remote` + `gh` (auth as `kvgork`). All 10 checkouts clean. **Zero open PRs.**

### E1 — Only-on-disk work (three items, do first)
1. **`scripts/launch_residual_rl.sh`** — uncommitted, load-bearing for Track A.
2. **`src/robot-data-runner @ feat/deploy-runner-hardening`** (`40d3198`, 885 ins / 11 files, 2 new
   CLIs). Unpushed, **no upstream**. Also a Track-B blocker.
   `git -C src/robot-data-runner push -u origin feat/deploy-runner-hardening`
3. **`src/lerobot-isaac-autoresearch feature/auto-lora`** @ `014f884` — 9 commits,
   `[origin/feature/auto-lora: gone]`, **no merge-base with origin/main**. Rev 1 missed this entirely
   while asserting item 2 was "the sole real risk of loss".

### E2 — Pushed but never PR'd (all three repos have zero PRs ever)
| Repo | Branch | Ahead | Note |
|---|---|---|---|
| `lerobot-isaac-env` | `feature/wm-isaac-env` | 24 | **Blocking Track A** — this is the branch §0 is about. `main` 0 behind → clean FF. |
| `lerobot-isaac-deploy` | `feature/sim-deploy` | 26 | `feature/world-model` is an ancestor; one PR covers both. No local `main`. |
| `lerobot-isaac-synthetic` | `feat/align-synthetic-dr-schema` | 4 | PR or explicitly abandon. |

### E3 — Config warts (on merged `main`, not just a branch)
1. `src/lerobot-isaac-configs/src/lerobot_isaac_configs/scenes/so101_workspace.usd:22` —
   `prepend references = @/home/koen/workspaces/.../so101.usd@`. Breaks any other clone, any wheel
   install, and the `frozen` env entirely.
2. `src/lerobot-isaac-configs/configs/task/so101_pickplace_with_cameras.yaml` — repo root, so invisible
   to `load_config()` and **not in the wheel**. Zero references anywhere → dead.

### E4 — Housekeeping
- `src/lerobot-isaac-autoresearch` local `main` has orphan history (no merge-base; diff is a strict
  subset). Blocks `pixi run sync-update`. Fix: `git reset --hard origin/main`
  — ⚠️ **do this on `main` only.** `feature/auto-lora` (E1.3) is one careless command from the same
  treatment.
- ~~Local merged-branch delete sweep~~ — **dropped.** Zero value, and it sits next to two branches
  with 9 and 18 commits of unmerged work.

---

## Track F — Open items rev 1 dropped entirely

The user asked to combine **all** open points. These were in the source plans, the backlogs, or the
repo, and rev 1 omitted them.

| # | Item | Where |
|---|---|---|
| F1 | **Failing test on `main`**: `test_cli_delegation.py::test_dr_replay_delegates_dry_run` — `TypeError: main() takes 0 positional arguments but 1 was given` at `lerobot_isaac_meta/cli.py:92`. The meta `dr-replay` subcommand is broken. 1 failed / 70 passed. | repo |
| F2 | **`docs/internals/system-improvements.md`** (172 lines) — eval-results JSON schema contract; `outputs/curriculum_stage.json` never written; no run/snapshot registry; `lerobot_world_model_bridge` broken on v3.0 array-columns; torchcodec/FFmpeg; autoresearch `_in_monorepo()` mis-gating (silently auto-skips the 6-arch e2e metric contract in CI). | backlog |
| F3 | **`src/lerobot-isaac-deploy/system-improvements.md`** (31 lines) — LeWM MPC/CEM planning shim; V-JEPA/Cosmos/GAIA real loaders; `wm_rollout` real dataset loader + decoder (currently emits zero arrays with `partial: True`); a test that "passes for the wrong reason". | backlog |
| F4 | **WM Route C** — the recorded-first WM baseline (`outputs/wm_offline_real/dreamerv3_data.hdf5`, 490 MB, exists; the run stopped at policy_step 1800) and DreamerV3 online on the real arm. Rev 1's B3 gate fired "invest in the WM path" **with no destination to invest in**. | `act-real-campaign-plan.md:80-85` |
| F5 | **DAgger** — the cheaper branch matching session 1's observed grasp-stage failures. Rev 1 routed *every* bake-off outcome to Track C, deleting it. | `phase3-…:74`, `wrap-plan:335` |
| F6 | **Phase 4b MimicGen bridge** + **insertion task** Stage-5 stub (`tasks/insertion.py`, `NotImplementedError`) — both unchecked in CLAUDE.md's build status. | CLAUDE.md |
| F7 | Dashboard `events.parquet` written empty (mixed-type `commits` column). | `wrap-plan:66-67` |
| F8 | **Vault write-back** — no note exists for v6's 9.5 %/handoff-survival result, the 2026-07-25 camera-drift lesson (the most expensive operational lesson of the last session), or this session's env-branch incident. | vault gap |

---

## Ordering

```
DONE   §0 incident: v7 killed, env branch restored, SHA pin added, v6 backed up
NOW    A1 smoke gate (running) ──┬── E1 commit launcher + push runner + rescue auto-lora
                                 ├── B0.1 build framing gate
                                 └── F1 fix the failing test
   A1 PASS ──> A2 full run (13h, A3 abort criteria armed)
        │
        └─ in parallel (CPU/human): B4 cheap levers → B0.2/B0.3 episode loop + negative control
                    │
              Track B: 10-episode stop-loss → 20/candidate if it converts
                    │
              Track C: 30-episode PILOT → retrain → re-eval → full matrix only if it converts
                    │
              Track D folded into C's retrain sweep
```

| Gate | Pass → | Fail → |
|---|---|---|
| A1 smoke (**rate bar**): `fixed_base=True` + **≥2 distinct attempts** with `oz>0.07` + CARRY on one of them + `min_ez≤0.110` | launch A2 | 1 attempt = INCONCLUSIVE → re-run; 2× inconclusive = **do not launch 13 h**, diagnose |
| A3 T+40 min behavioural check | let it run | kill immediately |
| Framing gate PASS | score episodes | realign — never eyeball it |
| B2 stop-loss: ≥1/10 on candidate 1 | continue to 20 | end session, go to Track C |
| C2 pilot converts | full matrix | stop; the bottleneck is not data volume |

---

## Open risks
- **Track B may be uninformative** even at N=20. Budget it as "confirm the bottleneck", not "pick a winner".
- **Overhead-only camera** may cap achievable success below the two-camera reference recipe. Untested.
- **Recording throughput is unmeasured.** The pilot exists to fix this.
- **Single GPU.** Only one of {residual extension, Track-C retrain sweep} runs at a time.
- **No owner or calendar slot** is assigned to the ~6 h hardware day or the ~6 h recording day.
- **v6's replay buffer is ~3–4 % polluted** and cannot be restored; the backup preserves the current
  state, not the pristine one.
- The **existing 50 episodes' scene conditions are unrecoverable** — no per-episode metadata was kept.

## Related
- Sources: the four plans named at the top · `outputs/bakeoff-20260725/scoreboard.md`
- Vault: `03-Resources/Technical/act-so101-training-research.md` ·
  `05-Wiki/concepts/Residual-RL-on-Scripted-Controllers.md` ·
  `05-Wiki/concepts/WM-Schema-Viewpoint-Incompatibility.md` ·
  `05-Wiki/sources/2026-07-19-gpu-campaign-3day-lessons.md` (reuse `scripts/gpu_campaign.sh`
  chaining — do not hand-roll a supervisor)
- Memory: `[[detach-long-training-jobs]]` · `[[act-real-campaign-result]]` ·
  `[[sheeprl-action-override-buffer-seam]]`
