# Fresh DreamerV3 training-run plan — carry→place (2026-06-16)

Goal: a **clean, from-scratch** DreamerV3 run that completes SO-101 carry→place in sim,
folding in everything the 5-day debugging campaign + the vault playbooks established.
Supersedes the messy v1–v5 / cur1–cur2b resume chain (those ran on a broken env and/or
explored knobs piecemeal). This is the run to trust.

---

## 1. What is now SOLID (the foundation — don't re-litigate)

- **Env is correct** (3 bugs fixed + committed, lerobot-isaac-env feature/wm-isaac-env):
  - `success` = **object-in-bin** (place), not EE-to-object (reach). `place_termination`.
  - **object reset** every episode (`reset_root_state_uniform`, ±3cm jitter).
  - **robot reset** every episode (`reset_joints_by_scale`).
  - Verified: EventManager has both resets; success fires only on real place.
- **Grasp physics works**: convexDecomposition fingers + solid `CuboidCfg` 16mm die. Scripted
  controller places open-loop; cur1 placed ~30% closed-loop.
- **Place IS learnable**: cur1 (die 6.6cm from bin, correct env) → **~30% place-success** — the
  first carry→place ever. The plateau was the bugs, not exploration.
- **Reach/feasibility**: straight-down grasp reach ee_x≈0.218; die graspable at r≲0.20; bin at
  (0.22,−0.13) reachable while carrying. Keep die within r≲0.20.
- **VRAM (10GB)**: batch 8 + horizon 25 + replay 16 + bf16 FITS (cur2b). batch 16 OOMs when the
  desktop session is up. Use batch 8.
- **Pitfalls**: kill stray GPU procs before launch; launch at batch 8 directly (watchdog vs
  OOM-retry conflict); `LEROBOT_TRAIN_TIMEOUT=40000` for full runs; torch.load weights_only=False
  patch for resume; reward sidecar for demo seeding (reward-0 poisons the reward model).

## 2. The OPEN levers this run resolves (in priority order)

1. **Actor observability (the #1 WM check, both playbooks).** Current actor obs =
   `joint_pos(6)` + wrist d435 cam — **no object_pose, no bin location**. The actor likely
   can't see *where to carry to* → learns place only via the reward gradient (cur1's 30% =
   partial). **Lever: `LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1`** (state 6→13 = joint_pos[6] +
   object_pose[7]). Gives the actor the object's location directly. HIGHEST-EV.
2. **Playbook knobs** (vault `wm-vla-training-playbook`): `algo.actor.ent_coef=1e-3` (vs
   default 3e-4 — anti entropy-collapse), `algo.horizon=25` (vs 15 — credit the 4-phase
   carry→place chain), `algo.replay_ratio=16` (vs 2 — more WM grad steps). cur2b is testing
   these at 13cm; fold the verdict in (if they helped, keep 16; if too slow, replay 8).
3. **Curriculum granularity.** The 6.6→13cm jump (cur2/cur2b) was too large (place skill
   didn't transfer). Use **fine steps**: 6→9→12→15→18 cm, resume each from the prior ckpt,
   advance only when place-success ≥ ~50% over the last 60 episodes.
4. **Reward clip [−10,+10]** (playbook — clean symlog) + keep tuned shaping (closure 4,
   lift_shaping 14, place_std 0.15). Consider PBRS-form shaping (playbook) over binary gates.

## 3. Recommended fresh-run config

| Knob | Value | Source |
|------|-------|--------|
| env | pick_and_place, FIX_BASE=1, OBJECT_SCALE=0.267 | fixed env |
| **obs** | `LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1` (state=13) | lever #1 |
| reward | STAGED=1, CLOSURE=4, LIFT_SHAPING=14, PLACE_STD=0.15 | tuned |
| batch_size | 8 | VRAM |
| precision | bf16-mixed | Ampere |
| algo.horizon | 25 | playbook |
| algo.actor.ent_coef | 1e-3 | playbook |
| algo.replay_ratio | 16 (→8 if wall-clock too slow per cur2b) | playbook |
| STEPS | 50000/stage | — |
| LEROBOT_TRAIN_TIMEOUT | 40000 | full run fits |
| CHECKPOINT_EVERY | 5000 | resume safety |
| die start | curriculum: (0.16,−0.10)→…→(0.18,0.05) | lever #3 |

## 4. Demos / seeding decision (needs a call)

Seeding the replay buffer with the scripted demos requires the demo obs to MATCH the run's
obs. With `INCLUDE_OBJECT_POSE=1` the env state is 13 (joint_pos[6]+object_pose[7]); the
current demos recorded state=12 (joint_pos+joint_vel) → **mismatch**. Options:
- **(A) Regenerate demos with object_pose** — modify `_gen_sim_demos.py` to record the env's
  actual obs `state` (the 13-dim incl object_pose) instead of pos⊕vel, re-run 40 demos +
  reward sidecar. Then seed. Cleanest; keeps the DreamerFD WM-dynamics benefit.
- **(B) No seeding** — rely on curriculum + knobs + object_pose obs alone. Simpler; tests
  whether seeding is even needed once the actor can see the object. cur1 placed 30% WITH
  seeding but mismatched obs; (B) isolates the obs lever.
- Recommendation: **(A)** — regen demos to match (the seeding helped the WM dynamics; keep it).
  Fallback (B) if regen is a blocker.

## 5. Execution ladder (autonomous)

0. **Pre-flight** (playbook Part 0): dry-run the config; confirm EventManager=2 resets,
   success=place, state=13 (object_pose in obs); **input-independent-baseline check** —
   zero the obs, confirm WM/actor loss worsens (proves the actor uses the object_pose).
1. **Regen demos** with object_pose state (option A) → datasets/local/so101-sim-pickplace-demos-op.
2. **Stage 1**: die (0.16,−0.10) (~6.6cm), config §3, seeded. Target ≥50% place over last 60.
3. **Stage 2–5**: resume prior ckpt, die → (0.165,−0.06)→(0.17,−0.02)→(0.175,0.02)→(0.18,0.05).
   Advance on ≥50% place; if a stage stalls, halve the step or add steps.
4. **Verify**: closed-loop eval on the FIXED place criterion (the env now terminates on
   object-in-bin, so `_sim_eval` pc_success is now meaningful — re-confirm it's not trivially
   firing). ≥50 episodes, ≥3 object positions.
5. If a stage genuinely stalls with object_pose + knobs + fine curriculum → **DreamerFD
   BC-loss** (the deepest lever; demo_buffer.behavior_cloning_loss scaffolded).

## 6. Diagnostics to run throughout (debugging playbook)

- Collapse watcher (launcher has it): kill on `Grads/actor→0` + flat `rew_avg` + step ≥15k.
- WM sanity (first 5k): observation_loss falls; KL > free-bits (no posterior collapse);
  reward-prediction tracks actual reward.
- **place-success rate** = fraction of `reward_env_0 > −10` (the +5 place bonus) per window —
  the primary signal (NOT raw reward, whose scale shifts with episode length).
- update-ratio ‖Δw‖/‖w‖ ≈ 1e-3 sanity if a stage misbehaves.

## 7. Success criteria
- Per stage: ≥50% place-success over the last 60 episodes → advance.
- Final (die 0.18,0.05, full task): ≥50% closed-loop pc_success on the place criterion, ≥3 seeds.
- Guardrail: reject any change regressing place-success ≥0.05 absolute (playbook).

## Related
- memory: `success-termination-reach-bug`, `demo-warmstart-pipeline`, `wm-vla-playbook-knobs`,
  `so101-sim-reach-envelope`, `scripted-grasp-infeasible`
- vault: `05-Wiki/synthesis/2026-06-16-wm-vla-training-playbook`, `concepts/Training-Debugging-Playbook`,
  `concepts/Hyperparameter-Effects`, `concepts/Curriculum-Learning-(Robot-Manipulation)`
- plans: `2026-06-13-pipeline-analysis.md`, `2026-06-11-demo-warmstart-plan.md`

## 8. Vet + fixes (2026-06-20) — BEFORE committing GPU-days

A smoke (cp-smoke-20260620, 2000 steps) confirmed the run TRAINS (env boots, 8 staged-reward
terms active, no abort — the cur2b abort was stale/dreamer-path-specific, env-boot is healthy).
An adversarial vet (6-agent workflow) then caught **3 blockers** the smoke alone hid — each
would waste the ~11-14h stage-1 run:

1. **`clip_rewards` is BOOLEAN, not magnitude.** §3's "reward clip [−10,+10]" CANNOT be done via
   `env.clip_rewards=10.0` — sheeprl applies `np.tanh()` to every reward when truthy (saturates
   the staged landscape). Leave `clip_rewards=false`; bound range by scaling weights instead.
2. **Lever #1 (object_pose) was a NO-OP for the WM** — `mlp_keys.encoder=[]` (only `cnn_keys=[rgb]`).
   Fix: add `algo.mlp_keys.encoder=[state]` (decoder auto-interpolates). The whole premise of this
   run depends on this one Hydra flag.
3. **Spawn-in-bin** — `success_radius=0.06` + ±3cm jitter spawned the die in-bin ~31% at 6.6cm,
   4.6% at 9cm (Monte-Carlo) → free 2-step "successes". **FIXED in `pick_and_place.py`:
   `success_radius` 0.06→0.04, reset jitter ±0.03→±0.015** → P(spawn-in-bin)≈0 across 6→18cm.

Also: throughput ~0.5 env-steps/s (replay 16, num_envs=1) → 50k/stage ≈ 27h; **use STEPS≈20k/stage**.
Seeding is forced OFF (option B) — the seeder only truncates (state 12↛13), can't grow obs.
Resume between stages is **manual** (`checkpoint.resume_from=<prior-ckpt>` per relaunch).

**curriculum_controller MISALIGNMENT (blocks autonomous Phase-2):** the shipped controller advances
an integer DR-intensity stage (2-4) via `LEROBOT_ISAAC_STAGE`; it does NOT emit `OBJECT_X/Y`, so it
cannot drive this plan's die-DISTANCE ladder. Multi-stage advance is MANUAL for now (see
`docs/internals/system-improvements.md` 2026-06-20 entry).

**Corrected stage-1 launch command (vetted, GO after a clean re-smoke):**
```
SESSION_ID=cp-stage1-<date> STEPS=20000 BATCH_SIZE=8 REPLAY_RATIO=16 PRECISION=bf16-mixed \
NUM_ENVS=1 CHECKPOINT_EVERY=5000 SECONDS_PER_EXP=50000 \
LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1 LEROBOT_ISAAC_STAGED_REWARD=1 LEROBOT_ISAAC_CLOSURE_WEIGHT=4 \
LEROBOT_ISAAC_LIFT_SHAPING_WEIGHT=14 LEROBOT_ISAAC_PLACE_STD=0.15 LEROBOT_ISAAC_FIX_BASE=1 \
LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_OBJECT_X=0.16 LEROBOT_ISAAC_OBJECT_Y=-0.10 \
LEROBOT_ISAAC_TARGET_X=0.22 LEROBOT_ISAAC_TARGET_Y=-0.13 \
EXTRA_HYDRA="algo.horizon=25 algo.actor.ent_coef=1e-3 metric.log_every=500 algo.mlp_keys.encoder=[state]" \
bash scripts/_run_wm_isaac_overnight.sh
```
Memory: `dreamerv3-carryplace-launch-gotchas`.

## 9. Stage-1 result (2026-06-21) — PLATEAU at the place wall

`cp-stage1-r8-20260620` (replay 8, object_pose obs, staged reward, fixed geometry, fine-curriculum
step-0 die at 6.6cm). Ran clean ~5.5h to step ~7500. **Outcome: confirmed place-breakthrough plateau.**

- Reward climbed −71 → −24 by ~step 2500 (learned reach+lift via dense shaping), then **dead flat ~−25 for 5000 steps** (2500→7500). Max-ever −24.1.
- `Game/ep_len_avg` = 300 the ENTIRE run, min-ever 300 → **not one episode ever placed** (no `place_termination`).
- Diagnostic combo: `Grads/actor` decayed 0.9→0.25 while `post_entropy` rose 9.6→12.0 and `State/kl` clamped to the 1.0 free-bits floor. = **over-explore / under-converge**: the agent never randomly carries the die to within 4cm of the bin from 6.6cm, so the place reward never fires, so there is no gradient toward placing — the actor freezes into a non-placing local optimum.
- NOT a collapse, NOT a config bug (geometry/obs/reward/throughput all verified healthy). It is the classic **sparse-reward exploration failure**: the success event is too rare to discover by chance from a cold 6.6cm start.
- `ckpt_5000` saved → any lever can resume the learned reach/lift instead of restarting cold.

**Lever options (ranked, NOT yet launched — strategy change, needs user sign-off):**
1. **(B) Easier curriculum step-0 — highest EV, cheapest, root-cause fix.** Restart (resume `ckpt_5000`) with the die VERY close to the bin (~3-4cm, `LEROBOT_ISAAC_OBJECT_X/Y` near TARGET) so a place happens by chance within the existing reach/lift policy → the place reward fires → credit locks on. Then the distance-curriculum (now built: `DISTANCE_LADDER`) walks the die back out 4→6→9→12→15→18cm. This is exactly what the curriculum was for; the 6.6cm cold-start was too hard. Pure env-var change.
2. **(A) DreamerFD BC-loss demo seeding.** Strongest if demos exist, BUT blocked: existing demos are state=12, this run needs state=13 (object_pose) and the seeder only truncates (can't grow) — needs a demo REGEN with object_pose state first (`_gen_sim_demos.py` change). Real work; do as a follow-up if (B) stalls.
3. **(C) `LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT` terminal bonus.** Makes a place dominate the return — but USELESS standalone (a bigger reward for an event that never happens = still no gradient). Only useful COMBINED with (B): once (B) makes places occur, (C) reinforces them strongly. Recommend B+C together.

**Recommendation:** kill the plateaued run (ckpt_5000 preserved, nothing lost), relaunch lever **B+C** resuming `ckpt_5000` with die at ~3.5cm + terminal place bonus on. Awaiting user sign-off (do not auto-launch).
