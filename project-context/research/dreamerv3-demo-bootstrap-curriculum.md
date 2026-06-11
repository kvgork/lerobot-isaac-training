---
type: research-summary
topic: "DreamerV3 demo bootstrapping + curriculum for long-horizon pick-place"
date: 2026-06-11
tags: [dreamerv3, sheeprl, demonstrations, curriculum, pick-place, manipulation, offline-to-online, RFCL, MoDem, AWAC]
sources_count: 16
context: "SO-101 6-DOF arm in Isaac Lab, 16mm die pick-place, reward shaping plateaued at grasp+lift, carry→place transport never completes"
---

# Research Summary: Breaking the carry→place Plateau in DreamerV3

## TL;DR

Pure reward shaping has a structural ceiling for long-horizon manipulation: without ever seeing a successful carry→place, the actor has no gradient signal to improve beyond the last sub-skill it can reliably execute. The three escape routes, in order of implementation leverage for SO-101:

1. **Demo bootstrapping** — inject real SO-101 teleop episodes into the sheeprl replay buffer before / during online training; add a BC loss on demo transitions. The DreamerFD recipe (separate demo buffer, BC loss weight decaying over time) applies directly to DreamerV3 and has shipped as MoDem-V2 at the model-based level. **No native sheeprl hook; requires ~30-line patch.**
2. **Reverse curriculum (RFCL)** — reset the Isaac Lab environment to states sampled from demo trajectories that are close to the goal, then progressively move reset points earlier. Requires Isaac `env.reset_to(state)` which is available in Isaac Lab. Solves PegInsertionSide from 1-10 demos without dense rewards.
3. **DreamerV3 knob tuning** — raise `replay_ratio` (default 1 → try 4-8), raise imagination horizon (15 → 25-30), lower actor entropy after the grasp stage. Impact alone is incremental; combines well with (1).

---

## Key Findings

### 1. Demonstration Bootstrapping (HIGHEST PRIORITY)

**DreamerFD (Lin et al., IROS 2023, arXiv:2303.03675)**
- Built on DreamerV2; directly applicable to DreamerV3.
- Architecture: two replay buffers — standard online buffer B and a fixed demo buffer D.
- At each update, half the batch comes from B, half from D (50/50 mix, or annealed from high demo to low).
- Adds a BC loss on demo transitions: `L_BC = -E_{(o,a)~D} [log π_θ(a | h(o))]`
- The world model is *also* trained on demo transitions — both representation learning and policy benefit.
- Result: 80% success on needle picking with 8k demo timesteps + 140k online steps, where vanilla DreamerV2 fails entirely.
- "Virtual Clutch": masks the BC gradient when the latent prior/posterior KL is large, preventing the BC loss from destabilising early world model training.

**MoDem-V2 (Hansen et al., arXiv:2309.14236)**
- TD-MPC2 (not RSSM/DreamerV3) with three-phase demo injection:
  1. BC pretraining of actor on demos (~10 demos).
  2. Seeding: BC policy + Gaussian noise collects 5k-7.5k trajectories to fill the replay buffer.
  3. Online: demo sampling ratio anneals 75% → 25% over 100k steps.
- Prioritised experience replay across the unified buffer.
- Achieves manipulation tasks in ≤100k real-robot steps.
- Key transfer to DreamerV3: the 75%→25% annealing schedule is a clean default.

**Demo3 (Lopez-Escoriza et al., arXiv:2503.01837)**
- TD-MPC2 backbone + stage discriminators + 50% fixed demo ratio.
- ~5 demos sufficient for humanoid-scale tasks.
- Stage discriminators add a shaped reward: `r̂_t = r_t + β·tanh(δ_{stage}(z_t))` with β ≤ 1/3.
- Result: 90% on Meta-World Pick-Place, 75-80% on ManiSkill at 500k steps. 40% better data efficiency vs baselines.
- For DreamerV3: the discriminator idea is a future upgrade; the 50% demo ratio + BC loss is the portable part.

**AWAC (Nair et al., arXiv:2006.09359)**
- Model-free actor-critic. Solves dexterous manipulation 20× faster than DAPG.
- Uses advantage-weighted policy update: `π* = argmax E[exp(A/λ) log π(a|s)]`.
- No separate BC loss — demos and online data share one buffer; advantage weighting naturally de-emphasises low-advantage transitions.
- Not directly applicable to DreamerV3's imagination-based updates, but the concept of advantage-weighted BC is portable.

**HIL-SERL (Luo et al., arXiv:2410.21845, Science Robotics 2026)**
- Model-free (RLPD). 1:1 fixed demo/online sampling ratio. ~20-30 demos.
- 100% success rate on insertion and assembly tasks within 1-2.5 hours.
- Key lesson: 1:1 ratio is a robust default even as the online buffer grows. The demo buffer remains small and fixed; only the online buffer grows.

**sheeprl implementation gap**: sheeprl's `dreamer_v3.py` has NO native demo loading hook. The replay buffer uses `rb.add(step_data)` where `step_data` is a dict with keys `{obs_keys, actions, rewards, terminated, truncated, is_first}`. Injecting demos requires:
- Converting LeRobot parquet episodes into numpy arrays matching this schema.
- Calling `rb.add(step_data)` for each timestep before `learning_starts` triggers (default 1024 steps).
- OR maintaining a separate demo buffer and mixing batches at the sample call.
- Estimated effort: ~100 lines; no sheeprl internals need modification beyond the training loop.

**LeRobot → sheeprl mapping**:
- `observation.images.d435_rgb` → CNN obs key (resize to 64×64 uint8 → float32/255 before add)
- `observation.state` → MLP obs key (joint positions)
- `action` → actions array
- `done` / episode boundary → `terminated` + `is_first`
- `reward` → rewards (can use 0 for all demo steps, or hand-assign a shaped reward offline)

**Reward assignment for demos**: using reward=0 for all demo steps still helps the world model learn dynamics; the BC loss on the actor is what drives imitation. Alternatively, assign the staged reward terms post-hoc from state observations.

---

### 2. Curriculum Learning

**RFCL — Reverse Forward Curriculum Learning (Tao et al., ICLR 2024, arXiv:2405.03379)**
- Stage 1 (Reverse): for each demo trajectory τ_i, reset the env to state s_{i,t+k} near the goal; train until policy can reach goal from there; then shift reset point earlier (s_{i,t+k-1}, ...).
- Stage 2 (Forward): adapt Prioritised Level Replay (PLR) — score initial states by "signs of life" (score 3 if sometimes nonzero return, 2 if always zero, 1 if always nonzero) and oversample "signs of life" states.
- Requires: Isaac Lab `env.set_state(state_dict)` capability, which is available via `env.scene.rigid_objects[...].write_root_state_to_sim()`.
- Results: solves PegInsertionSide (harder than pick-place) from 5 demos in <2M steps model-free. PickCube: <1M steps from 1 demo.
- **Limitation**: state reset requires sim. Cannot directly transfer to real hardware — need sim2real. For our use case (Isaac Lab only) this is not a blocker.
- Demo format: any sequence of (obs, action, state) tuples. Real SO-101 teleop data provides actions and observations; states (joint positions + object pose) need to be logged during collection or reconstructed. The object state in Isaac Lab must be a full rigid body state (pos + quat + vel).

**Isaac Lab curriculum API** (native, no external library):
- `CurriculumTermCfg` with `modify_reward_weight` to ramp term weights.
- `modify_env_param` to adjust spawn ranges (e.g., narrow initial object position range, then widen).
- Pattern for SO-101: define a curriculum that sets `lift_shaping_weight=0, place_weight=0` for the first N steps, then ramps them in — reducing multi-objective confusion during carry learning.

**Phased curriculum (manual)**:
Phase 1 (steps 0-5k): progress + grasp + closure only. Object spawns near-hand.
Phase 2 (steps 5k-15k): add lift + lift_shaping. Object spawns at normal position.
Phase 3 (steps 15k+): add carry + place. Target randomisation.
Implemented via `CurriculumTermCfg` in `pick_and_place_env_cfg.py` — modify_reward_weight callbacks at step thresholds.

**ALP-GMM (Portelas et al.)**: fits a GMM on task parameters (object position, target position) weighted by absolute learning progress. More principled than manual phasing but requires ~10× more infrastructure. Deferred — manual phasing is the right first step.

---

### 3. DreamerV3 Knobs for Sparse / Long-Horizon Manipulation

**Confirmed defaults from sheeprl config** (`dreamer_v3.yaml`):
- `replay_ratio: 1` — very low; each env step → 1 gradient step on the world model.
- `horizon: 15` — imagination horizon (steps).
- `ent_coef: 3e-4` — actor entropy scale.
- `kl_free_nats: 1.0` — free bits threshold.
- `kl_dynamic: 0.5, kl_representation: 0.1` — KL balance.

**From DreamerV3 paper** (Hafner et al., 2023):
- Batch size B=16, sequence length T=64.
- Discount horizon: 333 steps (γ = 1 - 1/333 ≈ 0.997).
- Return λ = 0.95.
- Imagination horizon H=15 is a global default; paper notes it is sufficient for Minecraft (long horizon) — increasing to 25-30 may help for pick-place but is not a published ablation.

**Training ratio / replay_ratio** (most impactful single knob):
- Paper and Ray RLlib both note: higher training ratio → better data efficiency, at cost of wall-clock time per env step.
- Tested values: 1 (default), 4, 8, 16. At num_envs=1 and ~83 steps/min, replay_ratio=4 increases WM gradient steps 4× at ~same env throughput.
- Recommendation: try `replay_ratio=4` first; `replay_ratio=8` if still plateaued.

**Actor entropy** (`ent_coef`):
- Default 3e-4 is deliberately low (DreamerV3 does not rely on entropy for exploration — the stochastic latent space provides implicit exploration).
- If the actor collapses to a deterministic lift-but-not-carry policy, try raising to 1e-3 for a run.
- Plan2Explore / DreamerV3-XP (arXiv:2510.21418): adds ensemble-disagreement intrinsic reward. Prioritised replay on reconstruction + value error also helps sparse-reward settings. Not in sheeprl natively; requires adding an ensemble of reward heads.

**Imagination horizon for carry**:
- The carry→place sub-task requires ≥ ~30 timesteps of coordinated motion (lift → translate → lower → release).
- At H=15 and dt~0.02s, the actor imagines ≤ 0.3s ahead — shorter than a carry stroke.
- Try `horizon=30` or `horizon=50` to cover the full carry motion. This increases actor/critic training cost linearly but not dramatically at batch 16.

**Free bits** (`kl_free_nats=1.0`):
- Prevents KL collapse in early training. Default is fine; no manipulation-specific tuning needed.

**Sequence length T=64**:
- At dt=0.02s, T=64 covers 1.28s. A full pick-place takes ~3-5s. Consider T=128 if memory allows (doubles sequence VRAM).

---

### 4. Carry Reward — the Missing Gradient

The current shaping stack (progress → grasp_closure → lift_shaping) has no gradient for horizontal transport toward the target. The actor's only signal after lifting is `place_reward` which fires only when the object is *already* at the target. This creates a "lifted and hovering" local optimum.

**Carry shaping term** (not in existing code):
```python
carry_reward = lifted_gate(obj_z) * (1 / (1 + dist(obj_xy, target_xy)))
```
Where `lifted_gate = sigmoid((obj_z - z_threshold) / 0.02)`. This gives a dense gradient toward the target XY plane while the object is lifted. Analogous to how `lift_shaping` added EE-height gradient that `lift_reward` lacked.

From the Text2Reward / hlfshell pick-place analysis: the standard robot arm reward decomposition adds "distance to goal" for the *object* (not the EE) during the carry phase — this is the natural next term in our stack.

**Ordering**: add `carry_shaping` at the same weight as `lift_shaping` (14), gated by `obj_z > 0.05`. Preserve `place_reward` as the terminal signal.

---

## Sources

| Type | Title | Key Takeaway | URL |
|------|-------|--------------|-----|
| Paper | DreamerFD (Lin et al., 2023) | Dual-buffer + BC loss for DreamerV2; 80% success needle pick with 8k demos + 140k online | https://arxiv.org/abs/2303.03675 |
| Paper | MoDem-V2 (Hansen et al., 2023) | TD-MPC2 3-phase: BC pretrain → seed buffer → online with 75%→25% annealing | https://arxiv.org/abs/2309.14236 |
| Paper | Demo3 (Lopez-Escoriza et al., 2025) | TD-MPC2 + stage discriminators + 50% demo ratio; 90% Meta-World pick-place | https://arxiv.org/abs/2503.01837 |
| Paper | RFCL (Tao et al., ICLR 2024) | Reverse curriculum via state resets; solves PegInsertionSide from 5 demos | https://arxiv.org/abs/2405.03379 |
| Paper | AWAC (Nair et al., 2020) | Advantage-weighted BC; 20× faster than DAPG; model-free but principle portable | https://arxiv.org/abs/2006.09359 |
| Paper | HIL-SERL (Luo et al., 2024) | 1:1 demo/online ratio; 100% success insertion tasks; ~20-30 demos | https://arxiv.org/abs/2410.21845 |
| Paper | DreamerV3 (Hafner et al., 2023) | Canonical hyperparameter defaults; H=15, B=16, T=64, η=3e-4 | https://arxiv.org/abs/2301.04104 |
| Paper | DayDreamer (Wu et al., 2022) | Dreamer on real robots; pick-place from scratch, no demos; H=15, B=32, T=32 | https://arxiv.org/abs/2206.14176 |
| Paper | DreamerV3-XP (2024) | Ensemble-disagreement intrinsic reward for sparse-reward exploration | https://arxiv.org/abs/2510.21418 |
| Impl | sheeprl dreamer_v3.py | replay_ratio=1, horizon=15, no demo hook; rb.add(step_data dict) is the entry point | https://github.com/Eclectic-Sheep/sheeprl |
| Docs | Isaac Lab curriculum API | CurriculumTermCfg / modify_reward_weight / modify_env_param for phased training | https://isaac-sim.github.io/IsaacLab/main/source/how-to/curriculums.html |
| Paper | Offline vs Online MBRL (2025) | Offline data alone insufficient; exploration data + demos beats demos alone | https://arxiv.org/abs/2509.05735 |

---

## Implementation Roadmap (ranked by expected lift × effort)

### Option A — Carry shaping (effort: 1h, expected lift: medium)
Add `carry_shaping_reward = lifted_gate * (1 - dist_obj_to_target_xy / max_dist)` to `rewards.py`. Gated by `obj_z > 0.06`. Weight start: 14. This alone may break the "lift and hover" plateau. Zero risk of reward hacking (object must stay lifted AND approach target).

### Option B — Demo buffer injection for sheeprl DreamerV3 (effort: 4-8h, expected lift: HIGH)
1. Write a `load_lerobot_episodes_to_sheeprl_buffer(dataset_path, rb, obs_keys)` function:
   - Load parquet → iterate episodes → resize images 64×64 → normalize → construct step_data dicts.
   - Assign reward=0 for all demo steps (world model learns dynamics; BC loss handles policy).
   - Set `is_first=True` on episode start, `terminated=True` on episode end.
2. Call before `learning_starts` in `dreamer_v3.py` (monkey-patch or subclass).
3. Add demo buffer `rb_demo` (same schema, fixed size); at each gradient step, mix 50% from `rb_demo` + 50% from `rb` in the world model + actor updates.
4. Add BC loss on actor: `L_BC = -E_{(o,a)~rb_demo} [log π(a|h(o))]`; scale weight from 1.0 → 0.0 over 50k steps.
5. Use `so101-pickplace-new` (48 successful episodes, 18804 frames) as the demo set.

### Option C — RFCL reverse curriculum in Isaac Lab (effort: 1-2 days, expected lift: HIGH)
1. Log full rigid-body state (arm joint positions, gripper, object pos+quat+vel) during demo collection. Existing recorder already logs joint states; add object state logging via `env.scene.source_object.data.root_state_w`.
2. Implement `ReverseCurriculumWrapper` around Isaac env that at episode reset samples a state from a demo trajectory at progress p ∈ [0,1], calls `write_root_state_to_sim()`.
3. Start with p=0.9 (near goal), train until success > 70%, then decrease p by 0.1.
4. After reverse phase converges, switch to PLR forward curriculum on initial state diversity.
5. This bypasses the carry→place exploration problem entirely — the agent starts near success.

### Option D — Replay ratio + horizon tuning (effort: 30min, expected lift: small-medium)
In sheeprl config override: `replay_ratio=4`, `horizon=30`. Run 25k steps. Compare plateau level to current −7.5. If replay_ratio=4 doesn't help, do not increase further (diminishing returns at num_envs=1).

### Recommended execution order:
1. Option A (carry shaping) — low-risk, quick win, always worth trying.
2. Option D (replay_ratio + horizon) — zero code change, just config.
3. Option B (demo bootstrap) — highest expected payoff; run alongside A if A stalls.
4. Option C (RFCL) — most principled for "exploration is the blocker" diagnosis; implement if B also stalls.

---

## Open Questions

1. **sheeprl demo buffer**: does `is_first` get fed through to the RSSM reset gate correctly when injecting demo episodes? Verify by checking `dreamer_v3.py` handling of `is_first` in batch construction.
2. **Object state logging**: the existing recorder does NOT log object pose (only robot state). Need to add `episode_labels` or sidecar with object state per timestep for RFCL state resets.
3. **BC loss stability with DreamerV3**: the "Virtual Clutch" in DreamerFD (mask BC grad when KL is large) may be needed to avoid the BC loss pulling the policy away from good latent states early in training. Monitor `kl_loss` during demo-seeded runs.
4. **replay_ratio scaling**: at num_envs=1, ~83 steps/min. replay_ratio=4 → 4 gradient steps/env step. With batch 16, seq 64 → each step processes 1024 tokens. Verify GPU stays busy and doesn't OOM with higher ratios.
5. **Carry shaping height threshold**: the current `lift_shaping` fires when EE is above `z_threshold`. The carry term needs to fire when the *object* is above a "clear of table" height (≈0.07m for the 16mm die). Validate with the probe script.
