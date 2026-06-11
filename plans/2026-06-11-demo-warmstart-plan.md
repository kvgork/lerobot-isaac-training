# Carry→place via scripted-demo warm-start — plan (2026-06-11)

**Why:** RL + reward-shaping (single AND num_envs=4) reliably reaches grasp+lift on the 16 mm die
(reward ~−8.5) but NEVER cracks the long-horizon **carry→place** — a structural exploration
limit, not data quantity (confirmed across die16, carry, carry2/3, carry-ne4). The only remaining
lever is **demonstrations**: show the agent the full reach→grasp→lift→carry→place, then warm-start.

## Research-backed execution order (2026-06-11, research-agent → project-context/research/dreamerv3-demo-bootstrap-curriculum.md)
The plateau = no gradient/imagination for horizontal transport. Ordered cheapest→biggest:
1. **DreamerV3 knobs (RUNNING — `20260611-knobs`):** replay_ratio=4, horizon 15→30, seq_len 128.
   Diagnosis: imagination horizon (15) < carry stroke → WM can't credit transport. Cheap, high-EV.
2. **Carry shaping** (≈ have it; research uses plain `lifted × 1/(1+dist)`, no grip gate — loosen
   mine if knobs don't break it).
3. **Demo-buffer injection (DreamerFD recipe, ~100 lines, HIGHEST payoff):** dual buffer, 50% demo
   sampling, BC loss decaying 1→0, "virtual clutch" (suppress BC when latent KL high). sheeprl has
   NO native demo hook — add a LeRobot-parquet→`rb.add` loader + demo buffer + BC term. **Use SIM
   demos** (real so101-pickplace frames mismatch sim obs) → the scripted-IK controller (Stage 1
   below) or sim teleop. ~5–30 demos suffice (Demo3/AWAC).
4. **RFCL reverse curriculum:** reset env near goal (object-in-bin via Isaac `write_root_state_to_sim`),
   train backward. Needs object-state logging + state-reset wrapper. Most principled if #3 stalls.
Refs: DreamerFD 2303.03675, MoDem-V2 2309.14236, Demo3 2503.01837, RFCL 2405.03379.

## Stage 1 — working scripted pick+place (sim demo source)
Status: IK **position** control works (gripper_link tracks Cartesian waypoints); adaptive
jacobian indexing handles fixed-base (commit). Object/target/gripper conventions all known.
**Remaining:** the grasp. Position-only IK leaves the gripper not pointing down → gripper_link
bottoms ~0.10 at the pickup, fingertips can't reach the table die (z=0.008).
- **Fix: pose IK** (`command_type="pose"`) commanding a **downward gripper orientation** so the
  fingers point at the table. The target quaternion depends on gripper_link's frame — determine
  empirically: drive the arm to a low reach, read gripper_link quat there, command that.
- Then tune descend depth + close timing until `obj` rises with the gripper to the bin (SUCCESS).
- Effort: a few GPU iterations (quaternion + waypoint tuning).

## Stage 2 — generate demos
Run the working controller N times (with small object-pose jitter for diversity) →
`isaac_data_recorder.record_episodes()` → LeRobotDataset Parquet (sim demos, 64² d435 frames +
joint state + action). ~30–50 successful demos.

## Stage 3 — warm-start the policy  ← KEY DESIGN DECISION (needs a call)
DreamerV3 (sheeprl) does NOT trivially accept external demos. Options:
1. **Seed sheeprl's replay buffer** with the demo transitions before training (modify the entry
   to pre-fill the buffer). Most aligned with the WM approach; needs sheeprl buffer-API work.
2. **BC-pretrain a separate policy** (lerobot ACT/SmolVLA on the sim demos) — different policy,
   not the DreamerV3 WM; but a fast path to a working sim pick-place policy.
3. **Residual / demo-guided RL** — init from demos, fine-tune with the existing shaped reward.
Recommendation: try (1) for the WM track; (2) is the quickest path to *a* working policy.

**USER DECISION (2026-06-11): warm-start order = 1 → 3 → 2.** Seed sheeprl's DreamerV3 replay
buffer first; if that doesn't work, demo-guided/residual RL; finally BC separate policy. Stages 1
(scripted pick+place) + 2 (demo gen) are prerequisites, done first.

## Stage 4 — verify
Closed-loop eval (`scripts/_sim_eval.py`) → pc_success on the full pick-and-place.

## Also available now (byproducts of this session)
- **num_envs>1 works** (fix_root_link + IsaacSO101VectorEnv) — useful for DR/diverse data even
  though throughput is camera-bound (~no speed gain on this RTX 3080).
- Real-hardware recording plan (`2026-06-10-data-collection-and-plateau-break-plan.md`) for the
  deployable VLA policy + world model — independent of the sim carry→place.
