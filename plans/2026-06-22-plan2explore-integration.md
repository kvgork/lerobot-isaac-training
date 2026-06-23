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

## GPU smoke — DONE / VALIDATED RUNNABLE (2026-06-23)
The MVP launch cmd above was WRONG on 3 counts; the **validated-fit** config is:
```
LEROBOT_ISAAC_EXP=p2e_dv3_exploration STEPS=50000 BATCH_SIZE=4 NUM_ENVS=1 PRECISION=32-true REPLAY_RATIO=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
EXTRA_HYDRA='algo.cnn_keys.encoder=[rgb] algo.cnn_keys.decoder=[rgb] algo.mlp_keys.encoder=[state] algo.mlp_keys.decoder=[state] \
  algo.dense_units=256 algo.world_model.encoder.cnn_channels_multiplier=24 \
  algo.world_model.recurrent_model.recurrent_state_size=256 algo.world_model.transition_model.hidden_size=256 \
  algo.world_model.representation_model.hidden_size=256 algo.ensembles.n=2 algo.learning_starts=1024' \
SESSION_ID=p2e-explore-<ts> bash scripts/_run_wm_isaac_overnight.sh
```
Result: 7.6 GB stable, runs past `learning_starts`, logs `Rewards/intrinsic` + `Loss/ensemble_loss` + `Loss/world_model_loss`.
**The 3 bugs fixed (see [[dreamerv3-carryplace-launch-gotchas]] for full notes):**
1. **Obs key is `rgb` NOT `d435_rgb`** — and BOTH encoder+decoder must be set for cnn AND mlp, else sheeprl errors
   "CNN keys of the decoder must be contained in the encoder ones".
2. **P2E inherits the XL world model** (`dreamer_v3.yaml` default: recurrent 4096 / dense 1024 / cnn_mult 96) → OOM at
   the first train step. Force the compact profile explicitly (the launcher does NOT set sizes; dreamer_v3 Isaac runs were
   compact via an inheritance path I couldn't locate — force it to be safe).
3. **`bf16-mixed` → `RuntimeError: requires fabric.backward(loss)`** — p2e_dv3_exploration's train has a raw
   `loss.backward()`. Worked around with `PRECISION=32-true`; but fp32 + Isaac's ~4.5 GB fixed CUDA footprint leaves only
   ~5 GB for PyTorch → had to shrink hard (bs4/n2/recurrent256). **To run FULL-size P2E (real grasp bet) needs a
   bf16+fabric.backward monkeypatch in `_wm_isaac_entry.py`** — promote follow-up #0 below.

## NEW follow-up #0 (blocks a real P2E grasp run): bf16 + fabric.backward monkeypatch
Patch `sheeprl.algos.p2e_dv3.p2e_dv3_exploration` (or wrap its train) so its `loss.backward()` calls route through
`fabric.backward(loss)` → unlocks `bf16-mixed` → halves memory → run the FULL recurrent-512 / n=5 model that an actual
grasp-discovery bet needs. Until then P2E only runs at the shrunk smoke config (pipeline-valid, but too small to expect a
grasp win). Follow the existing `_patch_bc_actor_loss` gating pattern (env-var, default OFF).

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
