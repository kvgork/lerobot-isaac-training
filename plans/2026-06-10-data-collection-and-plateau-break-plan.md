# Data collection + plateau-break plan (2026-06-10)

> **UPDATE 2026-06-10 08:45 — ROOT CAUSE FOUND, supersedes the plateau-break ranking below.**
> The sim plateau is NOT exploration — it's a **second geometry bug**: the source_object is the
> DexCube scaled 0.05 → a ~4 mm cube resting on the floor (z=0.0015), and the gripper bottoms out
> at z=0.062, so it can't reach down to grasp (`scripts/_reach_down_probe.py`,
> `outputs/reach-down.json`). Reward shaping + scripted demos were chasing an impossible task.
> **Do this FIRST:** make a graspable-sized cube (~2.5–3 cm, fits the small jaw) sit on a static
> pedestal (~5–7 cm) so its grasp point is in the 0.06–0.12 reach band; verify reach+grip; THEN
> the existing shaping (closure+lift_shaping+place) should produce real picks. The scripted-demo /
> curriculum / Fix 2 items below remain valid AFTER geometry is graspable.

Context: sim DreamerV3 pick-place plateaus ~−10.6 (reach+grip+lift partial, no carry) after
reward shaping hit diminishing returns (see `2026-06-09-staged-reward-tuning-results.md`,
[[staged-reward-shaping-progression]]). Existing real data: **~50 clean episodes**
(`so101-pickplace-new`, dual-cam) with a couple recoveries + failures.

## Key strategic split (read first)
| Lever | Fixes | Data needed |
|-------|-------|-------------|
| Sim DreamerV3 plateau | in-sim exploration | curriculum OR **sim** demos (scripted/teleop-in-sim). Real demos DON'T transfer (sim2real obs gap, memory `sim-policy-eval`). |
| Deployable BC/VLA policy | real task performance | **real** hardware demos (successes-heavy) |
| World model (DreamerV3/LeWM) | dynamics + reward model | real + sim, INCLUDING failures (state coverage) |

So: record real demos to improve the **real policy + world model**; break the **sim plateau**
with curriculum or scripted-sim demos (no hardware needed).

---

## Part A — What still needs recording (real hardware)
The 50 clean demos are a solid BC base. Add ONLY if the goal is a robust real/VLA policy or
richer WM data — **not** required to test the sim-plateau break. Priority order:

1. **Position diversity (biggest gap, ~30–50 eps).** The 50 are likely near-fixed object/bin
   poses → policy won't generalize. Record across a spread of object start xy (a ~3×3 grid inside
   the SO-101 reach, r ≲ 0.30 m) and 2–3 bin positions. This is the #1 add for generalization +
   for matching sim domain randomization.
2. **More total successes for VLA (→ ~100–150 total).** SmolVLA/ACT fine-tune wants 100+. If
   staying BC-only on the fixed task, the 50 + diversity set suffices.
3. **Recovery demos → ~10% of the set (~10–15 eps).** Drift toward a miss, then recover. Proven
   to improve BC robustness (you have "a couple" — bump up). Tag them.
4. **Failures → keep recording naturally (~10–20%), diverse modes** (slip, miss-grasp, knock-over,
   drop-in-transit). Do NOT pad for BC (BC uses `--successes_only`), but they materially help the
   **world model** (HDF5 keeps reward/done) and offline/demo-RL value learning.
5. **Lighting/background variation** if real deployment robustness matters (cheap DR).

**Do NOT** record isolated stages (just-lift, just-carry). Record **combined full pick→place**
trajectories — BC/VLA must learn the chaining; MimicGen (if used later) auto-segments clean full
demos into 100s of synthetic ones. Record clean + consistent if MimicGen is on the roadmap.

Before any BC train: run `lerobot_dataset_quality` (SAL/TED), drop jerkiest ~10–20% (+16–20% succ).

**Target add:** ~40–60 new eps = position-diverse successes (~35) + recovery (~10) + let failures
accrue (~10–15). Brings the set to ~100, position-diverse, recovery-rich — good for VLA + WM.

---

## Part B — Best plateau-break (sim), ranked — runs AFTER the already-planned tests
Already queued: **lift-chase-v4** (place_std=0.15 carry gradient, RUNNING) → **curriculum** run.
After those, ranked by expected leverage:

1. **Scripted sim demos → warm-start (TOP PICK, no hardware, no sim2real gap).**
   We know everything needed: object (0.22,0.05,0.05), target (0.22,−0.13,0.01), gripper closes
   toward the upper limit, reach 0.346 m, grip physics works. Write a scripted/IK pick-place
   controller in sim → generate demo episodes → either BC-pretrain the DreamerV3 actor or seed
   sheeprl's replay buffer. This SHOWS the agent the lift→carry→place it can't discover. Highest
   EV; fully in our control. (Isaac Lab `DifferentialIKController`, or hand-tuned joint waypoints
   since poses are fixed.)
2. **Phased curriculum** (already next-planned): lock reach+grip+lift, ramp place. Cleaner than
   stacked shaping; decent shot without demos.
3. **Demo-augmented / residual RL**: seed replay with scripted-demo transitions (combines 1+RL).
4. **Exploration boost**: actor entropy ↑ / Plan2Explore — cheaper, lower-odds.

**Decision rule:** if v4 + curriculum both plateau ≲ −10, go straight to (1) scripted sim demos —
it's the principled break and stops the hand-shaping whack-a-mole.

---

## Execution order (single GPU, sequential)
1. lift-chase-v4 (RUNNING) — carry gradient.
2. Curriculum run — if v4 plateaus.
3. **Scripted-sim-demo warm-start** — the real plateau break (prep the controller on CPU now).
4. Fix 2 (num_envs>1) — throughput for the longer runs (3) needs.
5. Real recording (Part A) — for the deployable VLA policy + WM, in parallel with hardware time.
6. Research TODO (vault inbox) — demos/curriculum literature to refine the above.
