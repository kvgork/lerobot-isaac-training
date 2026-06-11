# Autonomous fix ledger (2026-06-09 eve →)

Running log of things that broke + how I fixed them, while executing the 3-step autonomous
plan (`plans/2026-06-09-staged-reward-next-steps-plan.md`) unattended. Newest at bottom.

Mode: full autonomy — no input requests; fix breaks in place; record each here.

Format: `[time] SYMPTOM → ROOT CAUSE → FIX (commit/file)`.

---

- [19:2x] (baseline) run #3b healthy at −14.6 (3.9k steps), closure reward live. No breaks yet.
- [19:5x] STEP 1 done: `success_bonus` was a redundant proximity Gaussian + Isaac scales all
  reward terms by dt → negligible. Added dt-invariant `place_success_reward` (commit 892f5d5,
  lerobot-isaac-env). Not a break — a design gap fixed. 80 tests green.
- [19:5x] WATCH: run #3b GPU at 9.1 GB / 10 GB (desktop session :1 + replay buffer growth).
  Near OOM ceiling. Monitor greps for CUDA OOM; if it crashes → reduce batch_size or buffer.
  Branch trending A (reward climbing −18→−14.6). (No OOM occurred; ran fine to 13k.)
- [21:15] DECISION (not a break — risk-driven reorder): run #3b plateaued A-weak at −12.4
  (6.6k→13.2k flat). Cut it early to free GPU. **Reordered steps 2 before 3**: launch the
  place-chase run (step 2, the actual success-criterion science, num_envs=1) NOW overnight,
  attempt Fix 2 (step 3, risky throughput infra) AFTER — so Fix 2's risk can't eat the science
  run's overnight window. #3b checkpoint saved: logs/runs/dreamer_v3/isaac_so101/
  2026-06-09_17-52-56_.../version_0/checkpoint/ckpt_10000_0.ckpt.
- [23:5x] DIAGNOSIS forming: place-chase ALSO plateaus ~−12.4 (6.6k→13.8k flat) despite
  place_success active → blocker is physical grasp, not reward incentive. Wrote
  `scripts/_grip_physics_probe.py` (CPU-ready) to test if a closed jaw holds+lifts the cube.
  Fix ladder if it SLIPS: jaw friction ↑ → gripper effort_limit ↑ → cube mass/size ↓ →
  contact-based grasp. Will run probe when GPU frees (place-chase done ~05:30 or cut at ~20k).
- [00:0x] Cut place-chase at 13.8k (plateau confirmed 2nd run, place_success didn't break it).
  Freed GPU, running grip-physics probe to confirm the physical-grasp hypothesis. Verdict next.
- [00:0x] FIX (my own probe bug): grip probe v1 said "SLIPPED" but the data showed the object
  tracking the EE almost exactly (obj Δ ≈ ee Δ in x AND z) → it WAS gripped. Two bugs: (a) the
  lift action drove the EE DOWN (shoulder_lift sign wrong), (b) verdict required upward motion.
  Fixed: shoulder_lift=-1 raises EE; verdict now = "object displacement tracks EE" (direction-
  agnostic). Re-running. IMPLICATION: grip-physics likely FINE → the −12.4 plateau is the RL
  agent not discovering lift/carry, NOT a physics failure. Changes the fix direction.
- [00:0x] FIX (real, code): added `lift_shaping_reward` (grip × ee_height) — the missing
  gradient for raising the gripped object (lift_reward keys on object-z → no motion gradient).
  Opt-in LEROBOT_ISAAC_LIFT_SHAPING_WEIGHT. Commit f0500cd. 80 tests green. Relaunched overnight
  as `20260610-lift-chase` with lift_shaping=4 + full ladder.
- [00:0x] FIX (tooling): launch commands kept aborting "exit 1/144" — the Bash harness runs with
  `set -e`, so a no-match `pkill` (rc 1) or a pkill that hit the shell (rc 143→144) aborted the
  whole multi-line launch before the nohup. Fix: prefix launch scripts with `set +e` and guard
  `pkill ... || true`. Launch succeeded once guarded (lift-chase PID 188727).
- [02:5x] ROOT-CAUSE of the launch aborts (the real one): `pkill -f "_run_wm_isaac_overnight"`
  matched MY OWN shell — the launch command's text contains that string (the nohup arg), so
  pkill -f killed the parent script before it could nohup+write pid files. Fix: NEVER pkill -f a
  pattern that appears in the current command; kill by explicit PID, or put kills in a SEPARATE
  Bash call from the launch. Launched clean (no self-matching pkill) → lift-chase-v2 PID 214542.
- [02:5x] TUNE: lift_shaping@4 didn't break the −12.4 plateau (hovered −11.8..−12.4 to 13.8k).
  Relaunched lift-chase-v2 with lift_shaping=14 (per-step lift gradient ~0.4 vs grip ~0.13) so
  raising the gripped object clearly dominates. Target: reward climbing past −12.
- [03–06:00] lift-chase-v2 (lift_shaping=14): broke −12.4 → climbed to ~−10.6, plateaued
  (10.5k→16.8k). Grips + lifts partially, doesn't carry. Cut at 16.8k; ckpt_10000 saved.
- [06:10] Diagnosis: `place_reward` std=0.05 only pays within 5 cm of bin — too local to guide a
  carry from the 0.16 m pickup. Made std env-tunable (commit on feature/wm-isaac-env). Relaunched
  lift-chase-v4 with PLACE_STD=0.15 (wide carry gradient) — the lift→carry→place fix. PID 263591.
- [06:10] Research TODO written to vault inbox (lift_shaping / reward-shaping / curriculum /
  demo-bootstrapping / DreamerV3 knobs) per user request.
- [reminder] Self-kill pkill bit AGAIN (exit 144) — pkill -f patterns "lerobot_isaac_adapters.train"
  etc. match my own command line. Kills still landed (ran before self-kill) but STOP doing this:
  put kills in a separate Bash call from any command that echoes those patterns.
- [08:30] lift-chase-v4 (place_std=0.15 wide carry): plateaued −10.6 again (same as v2). Wide
  carry gradient did NOT help → hand-shaping exhausted. Cut, moved to scripted-demo plateau-break.
- [08:45] **ROOT CAUSE #2 — object too SMALL:** source_object = Isaac DexCube scaled
  (0.05,0.05,0.05) → a ~4 mm cube (rest z=0.0015). Intended object is a **16 mm die**. A 4 mm
  speck is ungraspable regardless of reward → why every run plateaued (closure/lift_shaping fired
  on proximity, jaw never held the speck). **FIX:** `LEROBOT_ISAAC_OBJECT_SCALE` default 0.267 →
  16 mm die (DexCube native edge ≈ 0.06 m; verified rest z=0.008). Commit on feature/wm-isaac-env.
  Re-running RL as `die16` with the existing shaping.
- [09:00] **CORRECTION (my error) — there is NO vertical-reach bug.** I first concluded the
  gripper "can't reach the floor (min z 0.062, 6 cm gap)". WRONG: `_reach_down_probe` measured
  robot LINK-FRAME origins (gripper_link/moving_jaw ≈0.06–0.07), NOT the geometric fingertips,
  which DO reach the table. The robot's owner confirmed the arm reaches its table fine. Corrected
  memory + wiki + lessons note. Lesson: validate a probe against ground truth — a link frame is
  not the contact point. (Horizontal reach 0.346 m IS a real limit; the vertical "limit" was bogus.)
- [20:00] **Fix 2 (num_envs>1) — partial: structurally works, env>0 state bug.** Built
  `IsaacSO101VectorEnv` (boots ONE N-parallel Isaac, batched obs/reward, episode+final_info
  stats) + a gym.vector patch in `_wm_isaac_entry.py` (intercepts SyncVectorEnv when len>1 +
  isaac → one vector env, avoids the N-instance singleton crash). NUM_ENVS=2 smoke: my patch
  FIRED, booted, stepped, **NO is_first crash** (the old bug is GONE). BUT `reward_env_1=-6.6e14`
  (garbage) vs env_0 −61 → env_1's object/robot world pos ~1e12 = env_1 physics uninitialised
  (Isaac scene-replication issue at num_envs=2, NOT a reward-frame offset — those are bounded).
  Needs Isaac scene/env-origin debugging (env_1 not built/reset). Per Fix 2 plan fallback, FELL
  BACK to num_envs=1 to keep runs moving. Fix 2 committed gated (4dae806); num_envs=1 untouched.
  **Bounded remaining work:** debug why env_1 doesn't initialise (scene replicate / reset_idx /
  env_origins) at num_envs>1; verify reward_env_1 sane; then num_envs=4.
- [22:00] Fix2 env>0 bug — CPU narrowing (no GPU): `so101_env_cfg.py:639` hardcodes
  `SO101SceneCfg(num_envs=1, env_spacing=2.5)`; make_env overrides scene.num_envs=N after, so
  env_1 exists at +2.5 m offset. Rewards use WORLD coords (root_pos_w/body_pos_w) + fixed world
  target_pos — offset terms are bounded (exp∈[0,1]), so the −6.6e14 means env_1 physics state is
  uninitialised (NaN/huge), a scene-replication issue (check replicate_physics + per-env clone).
  Next-GPU fix: (a) ensure env_1 replicates+resets; (b) make target-based reward terms env-local
  (subtract env.scene.env_origins) so place/carry/place_success are per-env correct. Then num_envs=4.
- [00:05] Fix2 env>0 ROOT CAUSE (num_envs=2 diag `scripts/_vec_diag.py`, `outputs`): at RESET both
  envs sane (robot |max|=4.4, env_1 properly offset). After 20 ZERO-action steps the ROBOT
  diverges to |max|=**1.7e12** (object stays 1.47). ⇒ NOT reward-frame, NOT uninitialised — the
  **SO-101 articulation physics EXPLODES at num_envs>1** (PhysX/replication instability). Deep
  Isaac bug (replicate_physics / solver iters / prim isolation / actuator), needs dedicated
  physics debugging — beyond a quick fix. Vectorization wrapper itself WORKS (no is_first crash).
  **VERDICT: Fix 2 plumbing done; throughput blocked by articulation instability. num_envs=1 stays
  the production path.** Carry→place (the actual goal) does NOT depend on Fix 2.
- [00:10] **Scripted-demo IK: floating-base jacobian fix → position control WORKS.** Diagnostic
  (`/tmp/_jac_diag`): SO-101 is_fixed_base=FALSE, jacobian (1,8,6,12) = 6 root + 6 joint DOFs. The
  controller used ee_idx-1 row + arm_ids cols (the ROOT cols) → garbage IK. Fixed: row=ee_idx,
  joint cols = arm_ids+6 (`scripts/_scripted_pickplace.py`). Now the gripper_link TRACKS Cartesian
  waypoints (above-obj, descend, move-to-bin) cleanly.
- [00:10] **Remaining scripted-grasp blocker: orientation.** command_type="position" controls only
  gripper_link POSITION, not wrist orientation → gripper isn't pointing DOWN → gripper_link bottoms
  at z≈0.10 at (0.22,0.05), fingertips ~0.06, can't reach the die at z=0.008. (The arm CAN reach
  the table — owner-confirmed — but needs the gripper oriented downward.) **Next: pose IK
  (command_type="pose") with a downward-pointing target quaternion** + tune; then the scripted
  controller grasps → generate demos → BC-pretrain/seed DreamerV3 (the carry→place unlock).
- [00:10] **Floating-base is ALSO the likely Fix2 lead:** an unanchored (floating) SO-101
  destabilises at num_envs>1 → the 1.7e12 explosion. Making the articulation fixed-base (anchor the
  root) may fix BOTH the multi-env explosion AND simplify IK. Worth trying first for Fix 2.
- [06:00 +1d] **FIX 2 COMPLETE — fixed-base was the key.** `fix_root_link=True` (env-gated
  LEROBOT_ISAAC_FIX_BASE). vec_diag num_envs=2: robot stable (|max|=1.25 after 20 steps, was
  1.7e12). num_envs=2 training smoke: BOTH rewards sane (env_0 −60.6, env_1 −61.1, was −6.6e14),
  patch fires, steps progress, no crash. vectorization (4dae806) + fix_root_link = working
  num_envs>1. Throughput unlock LANDED. Relaunching carry at num_envs=4 (watch VRAM; fall back to
  2 on OOM). num_envs=1 unaffected.
- [10:00] **die16 BREAKTHROUGH:** with the 16 mm die, reward −7.8 at 5.1k — first run EVER to
  break past the −10.6 ceiling all prior runs hit. Confirms root cause = object size: a graspable
  object lets the existing shaping (closure+lift_shaping+place) actually grasp+lift. Climbing
  toward 0 / place_success. Run continues.
- [16:30 +1d] DR/generalization run (stage 4, num_envs=4) CRASHED at boot: `object_pose term is
  not EventTermCfg (got ObjectPoseRandomizationCfg)`. The DR randomization (randomization.py +
  SO101EventsCfg stages 3/4) is UNIMPLEMENTED SCAFFOLD — the custom ObjectPoseRandomizationCfg
  was never wired to Isaac's EventManager (needs an EventTermCfg with e.g. mdp.reset_root_state_
  uniform). So stage 3/4 DR doesn't work without a wiring fix. DR also premature (would harden a
  grasp+lift policy that can't yet place). Reverted to stage 2 (working). LEROBOT_ISAAC_STAGE env
  var added (commit) but stage>2 needs the DR event wiring fixed first.
- [17:00 +1d] Research-backed knobs run (replay_ratio=4 + horizon=30 + seq_len=128) → CUDA OOM on
  the 10GB RTX 3080 (batch 16 AND batch-8 retry both OOM'd at ~900 steps). horizon=30 + seq_len=128
  blow VRAM (imagination rollouts × horizon × longer sequences). Backing off to the memory-neutral
  knob: replay_ratio=4 ALONE (4x WM grad steps/env step, ~no extra peak mem — research's "most
  impactful single change"), default horizon=15/seq=64. horizon/seq increases need a bigger GPU.
