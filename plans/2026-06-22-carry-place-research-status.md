# Carry-place — research status + strategic fork (2026-06-22)

**TL;DR:** The sim carry-place task is a genuinely hard sparse multi-stage manipulation problem
(grasp→lift→carry→place) for DreamerV3 on a single RTX 3080. After **8 training runs (~14 GPU-h)**
the **cheap lever space is exhausted** — the agent reaches+pushes but won't discover the grasp→carry
chain. The next directions are **research-scale bets** (multi-hour-to-multi-day), so this is a fork
worth a human decision. The durable deliverable is the mechanistic understanding below.

## What was proven (the value, regardless of the unsolved task)
1. **Plateau "break" was a SLIDE ARTIFACT.** The easier-3.5cm start let the agent SLIDE the die into
   the bin; `place_termination` was XY-only so it fired on a slide (ep_len→43, rew −3). True carry-place
   was never learned. (memory: `carryplace-place-wall-plateau`.)
2. **Objective favors placing** (placing return −3 ≫ timeout −25) → regression was NOT a reward bug; it
   was WM-imagination instability + the slide shortcut.
3. **The lift-gate fix (committed `fd8e977`)** — `place_termination` now requires a lift (`LEROBOT_ISAAC_PLACE_REQUIRE_LIFT`,
   mirrors `place_success_reward`) → slides stop counting → ep_len back to 300, CONFIRMING the agent only slid.
4. **The real wall is EXPLORATION of grasp→lift→carry→place.** With sliding blocked, reward moved to ~−14
   (`lift_shaping` engaging = partial grasp+lift) but the agent never completes the chain; even strong BC
   (weight 1.5, bc_loss active) doesn't induce it.

## Ladder tried (all hit the same wall)
| Lever | Result |
|---|---|
| Easier start (lever B) | slide-place, then regressed |
| BC 0.3 / 0.6 (DreamerFD) | slide-place, regressed; bc_loss small |
| + PLACE_SUCCESS=10 | bonus dormant (needs lift; agent slid) |
| **lift-gate** (place=carry) | exposed the slide; ep_len→300, partial lift (rew −14), no carry |
| **strong BC 1.5 + lift-gate + PS** | bc_loss active but no carry-place emergence (this run) |

Shipped + committed this session: BC actor-loss wiring (GPU-validated), PLACE_SUCCESS lever, lift-gate,
demo-gen state=13 + terminated-as-success, distance-curriculum driver, the RL+WM dual-mode recommendations.

## Strategic fork (ranked by EV÷effort) — needs a direction
1. **Plan2Explore integration** (sheeprl ships `p2e_dv3`). Principled exploration cure for the sparse
   grasp→carry chain. EFFORT: real — adapter hardcodes `exp=dreamer_v3`, the BC/seed monkeypatches target
   dreamer_v3, and p2e is two-phase (reward-free explore → task finetune). ~1 day to integrate+validate.
2. **Grasp-first sub-curriculum.** Train grasp+lift as its own stage (reward = lift height) until reliable,
   THEN add carry+place. Decomposes the chain. EFFORT: medium (new reward stage + a 2-phase run). The
   chain breaks at GRASP (agent pushes, doesn't grip) — attack that first.
3. **More + better demos / stronger DreamerFD.** 25 demos may be too few / too narrow; regenerate more
   lift+carry demos across positions, or add an explicit demo-replay-prioritization. EFFORT: medium.
4. **Longer compute** at the current config. EFFORT: cheap but LOW-EV (8 runs say it won't spontaneously
   discover grasp+carry).

## Recommendation
Lead with **#2 (grasp-first sub-curriculum)** — cheapest principled attack on the actual break point
(grasp), and composes with the existing machinery (lift_shaping, demos, lift-gate). Then **#1 (P2E)** if
needed. Avoid #4. The distance-curriculum driver + lift-gate are already in place to harden outward once
a real grasp→carry→place policy exists.

## Current run
`cp-stage1-bcstrong-20260622` (strong BC) left running for option value; expected to confirm-stuck by ~step 16000.
