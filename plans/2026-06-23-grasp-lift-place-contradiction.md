# Grasp lift+place CONTRADICTION — possibilities (2026-06-23)

## The contradiction
- **My headless probes concluded:** the SO-101 sim scripted grasp is MARGINAL — it lifts the 16 mm die only
  ~2 cm (max die-z **0.071**) then SLIPS, never holding for a sustained 4 cm (0.09) lift. I concluded
  carry-place is blocked on a sim-gripper limitation (the −0.175 rad closed grip is too loose), and the
  residual lever is blocked. (commits a969d79, 93c3783; memory [[scripted-grasp-infeasible]] → MARGINAL.)
- **The user reports:** while watching the Isaac sim GUI during a session, the robot **successfully LIFTED AND
  PLACED** the object.

One of these is wrong, OR they describe different conditions. This document enumerates the possibilities so
the resolution (workflow `grasp-contradiction-investigation` running) is checked against each.

## Possibilities (hypotheses) — ranked by prior likelihood

### H1 — My probe is FLAWED / replicates the WRONG controller (HIGH)
My probe `_probe_demo_grasp.py` copies `_gen_sim_demos.step_to`, NOT the 2026-06-13 SOLVED grasper
`_scripted_pickplace.py` that the memory says VISUALLY lifted the die. Sub-causes:
- **Missing 30-step pre-grasp SETTLE.** `_gen_sim_demos.rollout` runs 30 settle steps (grip open) BEFORE
  reading the die pose + grasping; my probe may read the UNSETTLED die (z=0.05) and target a slightly-off
  pose → misses/marginal. *Evidence if true:* probe-diff agent flags the missing settle.
- **Wrong grasp_z / sequence.** `_scripted_pickplace` uses grasp_z 0.108 + specific dwell/close_steps; my
  probe 0.106. 2 mm + sequence differences at the contact moment.
- **Measurement bug.** max die-z read from the wrong object, or after an auto-reset, under-reporting the true
  lift. *Evidence if true:* probe-diff agent flags it.
*Implication:* the grasp WORKS; my "marginal/infeasible" conclusion is a probe artifact → RETRACT.

### H2 — The grasp is STOCHASTIC (MEDIUM-HIGH)
Lift success depends on die jitter / exact contact. The user saw one of the successes; my few probe episodes
(≈6 single rollouts) happened to slip. *Evidence:* recorded demos show SOME episodes with max die-z >0.09 and
some <0.07 (a mix); or a training run shows ep_len_avg variance dipping <300 intermittently.
*Implication:* the grasp works at a non-trivial rate → carry-place is learnable; "infeasible" is wrong, but
the grip IS marginal (a reliability problem, not a hard block).

### H3 — The recorded demos ARE real lifts (the scripted grasp works) (decisive if confirmed)
If `datasets/local/so101-sim-pickplace-demos-op` episodes show observation.state[8] (die-z) rising to >0.09,
the scripted grasp genuinely lifts+carries (the demos are real, not slides). *Evidence:* demos-parquet agent.
*Implication:* RETRACT the marginal-grasp conclusion; my probe is the outlier.

### H4 — The RL agent actually LEARNED it in some run (decisive if confirmed)
If ANY training run's Game/ep_len_avg dropped <300 (a real termination fired), the agent achieved a
grasp/lift/place — the user watched that run's GUI. *Evidence:* TB-scan agent finds a run with ep_len<300 or a
place_success/task_success scalar firing. *Implication:* not an RL wall NOR infeasible; revisit what made that
run succeed (config, seed, curriculum stage) and reproduce it.

### H5 — UNCLAMPED script works; only the RESIDUAL clamp kills it (MEDIUM)
The user watched an UNCLAMPED scripted run (`_scripted_pickplace --gui` or `_gen_sim_demos`), where actions
can be |a|>1 and the grasp lifts; my residual run clamps to [−1,1] (actor-reproducibility) → too weak. The
probes (raw) still slipped at 0.071 though — so this alone doesn't explain the probe slip, only the residual
slip. *Implication:* the residual lever needs an un-clamped action path or a stronger grip; the underlying
scripted grasp may still work.

### H6 — Different CONFIG in the watched session (MEDIUM)
The watched session used a different object position / scale / gripper config (e.g. OBJECT_X=0.18 within
reach, bigger die, an earlier env state) where the grasp lifts. My probes used a mix; the bigger-die probe DID
reach 0.092-0.096 (crossed thresholds). *Evidence:* identify which config/run the user likely watched.
*Implication:* the grasp works in some configs → pick that config.

### H7 — GUI visual ≠ my threshold (LOW, partially already known)
The die lifting ~2 cm (0.071) LOOKS like a successful "pick up" in the GUI, and a subsequent slide into the
bin LOOKS like a "place". The user's "lift and place" may be the marginal 2 cm lift + a slide-place. This is
consistent with my probe (0.071) — NOT a contradiction, just a perception vs the 0.09 threshold.
*Implication:* my conclusion stands; the fix is still firmer grip OR a lower threshold. (Least exciting; the
user said "lift AND place" implying a real carry, so H7 is only partial.)

## Investigation underway (workflow `grasp-contradiction-investigation`, run wf_e145920b-9c4)
Four parallel read-only probes → synthesis:
- **[A] demos parquet** — per-episode max die-z in the recorded demos → real lifts (>0.09) vs slides? (tests H3)
- **[B] TB scan** — any training run with ep_len_avg <300 ever? (tests H4)
- **[C] probe-vs-real diff** — missing settle / wrong controller / measurement bug? (tests H1)
- **[D] SOLVED grasper** — `_scripted_pickplace.py` sequence vs `_gen_sim_demos` (tests H1/H5/H6)
- **Synthesis** — do real lifts happen? is my probe flawed? retract the conclusion? exact next GPU verification.

## Next (after synthesis)
Run the recommended GPU verification headless WITH proper die-z logging + the 30-step settle + MANY episodes,
on the SOLVED grasper config — settle definitively whether real lifts happen and at what rate. If they do:
RETRACT the marginal-grasp/gripper-infeasible conclusion, identify the working config, and re-enable the
carry-place line (the residual + P2E machinery is built and ready). Update [[scripted-grasp-infeasible]] +
[[carryplace-place-wall-plateau]] accordingly.

---

## RESOLVED 2026-06-23 PM — workflow `wbxrp730v` completed; "infeasible" RETRACTED

The investigation workflow (5 agents, 4 probes + synthesis) completed and its two load-bearing claims were
re-verified directly on disk. **Outcome: H1 (probe flawed) + H2 (stochastic) CONFIRMED; the "gripper-infeasible /
grasp regressed / carry-place BLOCKED" conclusion is RETRACTED.**

### Probe results (per hypothesis)
- **[A] demos parquet (tests H3):** the 25 `-op` demos ARE slides (max die-z **0.015**, 0/25 above 0.07). BUT
  they were recorded 2026-06-21, BEFORE the friction=3.0 fix (commit a969d79, 2026-06-23) — they reflect the
  OLD loose-grip env, so they are NOT evidence about the current build. (The 2 non-op datasets have a 12-dim
  joint-only state — no object pose — so die-z is unmeasurable there.)
- **[B] TB scan (tests H4):** 89 runs, NO success/place/lift scalar logged anywhere; rew_avg never positive
  (global max Test/cumulative_reward = −23.2). The 7 runs with ep_len<300 are early-FAILURE terminations
  (short episode ↔ least-negative reward = less accumulated penalty), NOT place successes. TB does NOT
  corroborate the GUI lift — but it also can't refute a transient lift (no place metric exists; matches
  [[success-termination-reach-bug]]).
- **[C] probe-vs-real diff (tests H1):** probe skips the 30-step settle AND never exports OBJECT_FRICTION=3.0;
  kinematics otherwise faithful. (Synth later showed the missing-settle is a code smell but NOT the slip cause.)
- **[D] SOLVED grasper (tests H1/H5/H6):** `_scripted_pickplace.py` and `_gen_sim_demos.py` run essentially
  IDENTICAL grasp→lift kinematics (settle30/approach50/descend90/dwell30/close80/seat25/lift60); only a 2mm
  grasp_z diff (0.108 vs 0.106) — no controller knob explains hold-vs-slip.

### The two artifacts that voided my prior conclusion (re-verified on disk)
1. **WRONG THRESHOLD.** `_probe_demo_grasp.py` print hardcoded `"lift thresh 0.09"`. Real `lift_termination`
   threshold = `rest_height 0.05 + lift_margin 0.02` = **0.07** (terminations.py:266-267; lift_margin hardcoded
   at pick_and_place.py:497). I called 0.07-crossings "below."
2. **FRICTION-FIXED LIFTS CROSS 0.07/0.09.** scratchpad logs: demo_grasp_firm **0.092**, demo_grasp_max
   **0.093**, demo_grasp_big **0.096** (friction=3.0) vs demo_grasp_x18/gs5 = 0.071 (NO friction fix). My
   "always 0.071/slips" anchor was the un-fixed runs.

### Corrected status
Grasp is **MARGINAL + TRANSIENT, not infeasible.** It lifts the die ~7-8cm (die-z 0.013→~0.09) crossing the
real 0.07, then slips back to ~0.013. **Hold-duration is UNKNOWN** — the probe logged only every 20 steps, so
the t277-295 window (where a hold would be counted) was never measured; and the "hold_steps=1 didn't trigger"
test was confounded by the residual [−1,1] clamp. The user's GUI "lift+place" was a REAL transient lift +
slide-into-bin via `_scripted_pickplace.py --gui` — contradiction RESOLVED. The RL ep_len=300 wall is
consistent with hold_steps=10 being unsatisfiable by a ~5-step transient lift (the wall may be the HOLD
requirement, not RL control).

### Open question → next action (the clean stats probe)
Write `scripts/_probe_lift_stats.py` fixing all 3 flaws: N=30 jittered rollouts (±2cm), 30-step open-grip
SETTLE before reading op0, identical SOLVED sequence, **log die-z EVERY step**, report per-rollout max_die_z +
max_consecutive_steps_above(0.07) + would-fire at hold_steps∈{1,3,10} as a RATE over 30. Env:
`FIX_BASE=1 OBJECT_SCALE=0.267 OBJECT_FRICTION=3.0 OBJECT_X=0.18 OBJECT_Y=0.05 STAGED_REWARD=1 GRASP_STAGE=1`,
headless. **DECISION RULE:** hold_steps=1 rate >0% → cheap unblock = `LIFT_HOLD_STEPS=1` + new
`LEROBOT_ISAAC_LIFT_MARGIN` env knob (0.02→0.015, threshold 0.065) so the transient lift COUNTS → RL/residual
get a success signal; firmer USD gripper close-range remains the durable fix. Run this (~6 min, no training)
BEFORE any multi-hour residual/grasp-stage run.

### Memory updates done this session
- `[[scripted-grasp-infeasible]]` — description + opening block rewritten to RETRACTED; old analysis kept below
  a superseded-marker for the raw numbers + SOLVED reference config.
- `[[carryplace-place-wall-plateau]]` — GRASP-KNOB "control failure" attribution flagged PARTIALLY RETRACTED.
- `[[so101-gripper-kinematic-floor]]` — "slips/blocked" caveated (shares the same artifact).
- MEMORY.md index lines updated for all three.

---

## ✅ CONFIRMED WORKING 2026-06-23 PM — GPU probes run; scripted grasp is a FULL ~80% PICK-PLACE

Ran the recommended clean probes (this session, autonomous). Result is far stronger than "retract infeasible":
**the scripted grasp does a fully-working, ~80%-reliable pick → lift → carry → place.**

### `scripts/_probe_lift_stats.py` (N=30, friction=3.0 SOLVED config, 30-step settle, every-step die-z log)
- **80% (24/30)** grip+hold the die above the REAL 0.07 threshold for **55–59 consecutive steps**.
- Binary: a grip either HOLDS ~56 steps (≫ hold_steps 10) or fails to grip (consec 0). hold_steps∈{1,3,10} ALL
  fire at **80%**. → `lift_termination` (current hold_steps=10, margin 0.02) already fires 80%. **No band-aid /
  LIFT_MARGIN change needed** (the proposed knob is moot).
- max_die_z mean 0.068, max 0.083, **frac>0.09 = 0%** — see "0.09 unreachable" below.

### `scripts/_probe_carry_mechanism.py` (N=5, logs ee_z + ee↔die distance + gripper joint, full place tail)
- **4/5 complete pick→lift→carry→place** — die placed at the bin, XY dist 0.004–0.019 < success_radius 0.06.
- **Mechanism = robust grip:** ee↔die distance stays **CONSTANT at 0.096 m through BOTH the lift AND the 30 cm
  horizontal carry** (gripJ −0.175). The die is genuinely gripped, not slipping or caught on geometry.
- 1/5 fail = a grip-fail at the start (same ~20% as the stats probe).

### Why every probe before this misled (3 artifacts, now fully understood)
1. **The 0.09 threshold is GEOMETRICALLY UNREACHABLE.** The grip hangs the die **9.6 cm below gripper_link**;
   at lift target z_high=0.17 the die maxes at **0.072**. It can never reach 0.09 when lifted correctly. The old
   probe compared against 0.09 (its hardcoded print) — a height the held die physically cannot reach. Real
   threshold 0.07; the die clears it and holds. (frac>0.09=0% is geometry, not failure.)
2. **No 30-step settle** → unsettled die → off-axis grasp → the 0.092-spike-then-slip transients were the
   NO-SETTLE runs. WITH the settle the grip is a stable 0.072 hold.
3. **n=1 + 20-step subsampling** → a single aliased sample read as "always slips."

### Corrected conclusion + next actions
The carry-place RL wall is **PURELY an RL exploration/learning failure** — the agent never learned the grip the
scripted controller performs reliably — NOT env grasp-infeasibility. The **residual-RL pivot** stands on a
genuinely working scripted base. Next:
1. **Regenerate the sim demos** with the working grasp — the existing `-op` 25 demos are slides (recorded
   2026-06-21, pre-friction-fix); `_gen_sim_demos` now yields real ~80% lift+carry+place demos → fixes BC
   seeding ([[demo-warmstart-pipeline]], which had state=12 slide demos).
2. **Resume residual-RL / DreamerFD warm-start** on the working scripted base.
3. (Optional) push grip reliability past ~80% via a firmer USD close-range — an optimization, not a blocker.

Memories `[[scripted-grasp-infeasible]]`, `[[carryplace-place-wall-plateau]]`, `[[so101-gripper-kinematic-floor]]`
all upgraded to CONFIRMED-WORKING. New probe scripts: `scripts/_probe_lift_stats.py`,
`scripts/_probe_carry_mechanism.py`.
