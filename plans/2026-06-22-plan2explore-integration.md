# Plan2Explore (p2e_dv3) integration — status + roadmap (2026-06-22)

**Goal:** intrinsic-reward (ensemble-disagreement) exploration as a reusable basis for autonomous training,
and a principled cure for the sparse grasp→carry chain. Built via the orchestration pipeline (Understand→Plan→Implement→Verify).

## MVP — DONE (commit adapter `0b0a336`, GPU smoke DEFERRED)
sheeprl ships `p2e_dv3` (two phases, same `exp=` dispatch as dreamer_v3): `p2e_dv3_exploration` (reward-free
novelty: 8-MLP ensemble, intrinsic reward = next-state disagreement variance) + `p2e_dv3_finetuning`
(3-arg main; resumes the exploration WM/actor; **reuses dreamer_v3.train**). Both inherit dreamer_v3's
cnn/mlp keys + env wiring → the Isaac wiring carries over unchanged.
- **Adapter plumbing:** `--exp` / `LEROBOT_ISAAC_EXP` selects the variant (resolver: arg→env→`dreamer_v3`
  default UNCHANGED). Double-exp guard (suppress ours if remainder has `exp=`). Finetuning guard appends
  `checkpoint.exploration_ckpt_path` from `--exploration_ckpt`/`LEROBOT_ISAAC_EXPLORATION_CKPT`.
- **Entry + monkeypatches: NO change** (verified algo-agnostic). Seed-patch applies to both phases (gated
  OFF by empty `LEROBOT_ISAAC_DEMO_DATASET`); BC-patch is a no-op in exploration (own train fn), fires
  correctly in finetuning (calls dreamer_v3.train). torch.load + gym-compat patches algo-agnostic.
- 9 tests, default back-compat verified (dry-run emits `exp=dreamer_v3`). CPU-verified; GPU smoke pending.

## GPU smoke (DEFERRED behind grasp-gs3 — run when GPU frees)
```
LEROBOT_ISAAC_EXP=p2e_dv3_exploration STEPS=50000 BATCH_SIZE=16 NUM_ENVS=1 PRECISION=bf16-mixed REPLAY_RATIO=2 \
EXTRA_HYDRA='algo.cnn_keys.encoder=[d435_rgb] algo.mlp_keys.encoder=[state] algo.ensembles.n=5 algo.learning_starts=1024' \
SESSION_ID=p2e-explore-<ts> bash scripts/_run_wm_isaac_overnight.sh
```
First-run watch: **VRAM** (8-ensemble + exploration actor + 2 critics on top of dreamer_v3 — start `ensembles.n=5`,
the OOM ladder doesn't account for ensemble cost) + steps/s (heavier than dreamer_v3's ~0.5).

## Ranked follow-ups (the autonomous-training basis)
1. **Two-phase explore→finetune driver** (highest value): after exploration, find the ckpt at
   `logs/runs/p2e_dv3_exploration/isaac_so101/<run>/checkpoint/ckpt_*_0.ckpt` (sheeprl's logger root, NOT
   `output_dir`), then launch `exp=p2e_dv3_finetuning checkpoint.exploration_ckpt_path=<abs>`. sheeprl loads
   the sibling `config.yaml` (parent.parent), validates `env.id` match, force-copies WM/actor/cnn/mlp from
   exploration. The adapter's single-subprocess metric flow must chain two subprocesses.
2. **p2e metric extraction** (blocks any autoresearch-scored p2e sweep): `metric_name='recon_loss'` never
   matches p2e stdout → −9999 sentinel. Exploration emits `Loss/world_model_loss`/`Loss/ensemble_loss`/
   `Rewards/intrinsic`; finetuning emits dreamer_v3 keys. Add an exp/phase-keyed metric branch.
3. Buffer carry-over into finetune (`buffer.load_from_exploration=true`). 4. BC-into-finetune (existing
   patch fires unchanged). 5. Demo-seed into finetune (not explore — novelty-bias tension). 6. BC-into-explore
   (new patch on p2e_dv3_exploration.train, lowest priority). 7. Version-control SO-101 p2e knobs as a plugin exp yaml.

## Key risks
VRAM OOM (ensemble) → start n=5; hydra double-`exp=` (guarded); −9999 metric sentinel (don't score until #2);
num_envs MUST stay 1 (is_first bug — use bare p2e_dv3_exploration.yaml + explicit env.num_envs=1, NOT shipped
multi-env variants); finetuning ignores CLI cnn/mlp+model-shape (overwrites from exploration_cfg → sweep those
on EXPLORATION); finetuning default learning_starts=16384 (~9h prefill at 0.5 steps/s — lower for SO-101).
