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
