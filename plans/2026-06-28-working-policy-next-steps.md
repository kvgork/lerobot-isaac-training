# Next steps — toward a WORKING (world-model) SO-101 policy

**2026-06-28.** Reframed goal (user): a *working* policy on the real SO-101, world-model as the
intended technique — **never a sim-only model**. This plan captures what was just fixed + the
realistic routes, after a long campaign that established the sim carry-place wall is robust.

## What just got fixed (this session)
- **Recorder parquet reward bug** — reward/done now save natively. Root cause: lerobot 0.5.1 maps a
  shape-(1,) feature to a scalar `datasets.Value` but the HF `datasets` save did `float(np.array([x]))`,
  which raises under numpy≥2 (numpy<2 silently coerced). Fix: `DualWriter._patch_datasets_scalar_save`
  restores the size-1→scalar coercion in `Value.encode_example` (referencing the already-loaded module,
  never force-importing). reward/done re-enabled in `_write_lerobot` + declared in `lerobot_features_dict`.
- **Backfilled `so101-pickplace-new`** — `reward_parquet.inject_reward_done_into_parquet` reconstructs
  sparse-terminal reward/done from the sidecar into the parquet (48/50 success → total reward 48.0).
- **pyarrow `ArrowKeyError` (double-registration)** — was introduced by an earlier force-import of
  `datasets.features.features`; fixed by not re-importing. Recorder suite green (112 passed).
- **WM bridge reward** — `lerobot_world_model_bridge` now reads bare `reward`/`done` (not only
  `next.*`); HDF5 episodes carry correct rewards (fail 0.0 / success 1.0). The offline WM no longer
  sees reward≡0.
- **Uncommitted/needs-sync:** recorder changes (sibling `robot-data-recorder`, push to its `main`
  gated — commit locally / open PR); bridge edit is in the installed skill copy
  (`~/.claude/skills/lerobot_world_model_bridge/`) → sync to the `claude_code` source via `install.sh`.

## The campaign verdict (why we're here)
- BC (ACT ×3 variants + temporal-ensembling + diffusion) = **0/20** in sim — knife-edge grasp precision.
- RL (DreamerV3 warm-start ×N) = **0 places** — sparse-reward discovery wall.
- Residual-RL base (`compute_scripted_action`) was **stuck in APPROACH** under the blend (separate
  state-gated reimpl with too-tight gates; the proven `_gen_sim_demos` controller works ~67%).
- All sim; none transfers (sim2real gap, both directions).

## Routes to a WORKING policy (pick by hardware availability)
| route | working on real? | WM? | effort |
|------|------------------|-----|--------|
| **A — BC on real demos** (ACT/SmolVLA) → real SO-101 | yes, proven (HF blog 90%) | no | low |
| **B — WM in sim** (DreamerV3, warm-started) → sim2real | only after sim2real | yes | high (sim place wall + sim2real both open) |
| **C — WM on the real arm** (DreamerV3 online, real reward) | yes | yes | research (hardware-in-loop) |

## Recommended sequence
1. **Decide the deployment target.** Real SO-101 in the loop? → enables A & C. Sim-only? → drop (doesn't meet the goal).
2. **Route A first (fast working baseline):** `lerobot-train --policy.type=act` (or SmolVLA) on
   `so101-pickplace-new` (`--successes_only` drops ep 0,34). Deploy via `lerobot-rollout` on the arm.
   *Caveat:* 50 ep is the blog's *failed* size — collect **150+ diverse** demos (phospho/lerobot recorder;
   rotation/object/container variation) for the ~90% regime.
3. **Recorded-first WM baseline (now unblocked):** with reward flowing, re-run a recorded-only WM
   (DreamerV3 via the bridge HDF5) — but note offline+sparse+expert is weak for *control*; treat as a
   representation/dynamics baseline, gate on a *real* metric not recon_loss.
4. **For a genuine WM policy (Route C):** scope DreamerV3 online on the real arm (HIL-SERL-style),
   real reward, safety + reset automation (we have `rollout-executor` + `physical-reset-agent`).
   References just ingested: ggando SAC-lift (shaped reward + IK-frame/gripper-offset lessons),
   NVIDIA GR00T sim2real strategies (DR/co-train/Cosmos), HF ACT blog (data diversity + eval discipline).

## Open leads worth chasing
- **ggando IK-frame / asymmetric-gripper-offset bug** — likely explains our scripted-grasp ee-misalign;
  check our IK target frame (graspframe vs gripperframe) + gripper offset.
- More real data (150+ diverse) is the single highest-leverage item for any route.

## Related
- `[[2026-06-23-carryplace-cup-campaign]]` · `[[2026-06-23-world-model-training-plan]]`
- vault: `[[act-so101-training-research]]` · `[[nvidia-sim-to-real-so101-research]]` ·
  `[[so101-rl-lift-and-phospho-research]]`
- memory: `[[carryplace-cup0-warmstart-r4-result]]`
