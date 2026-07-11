# Track B — DR100 schema decision + next steps (2026-06-06)

**Status:** BLOCKED on a schema decision (see "Open decision"). Pipeline code is
fixed + GPU-verified; merge cannot complete until the real-data schema is settled.

Owner: lerobot-isaac workspace. Session context: continued from the harness
"work on C" request — Isaac Lab v2.3.2 + RTX 3080 + SO-101 on `/dev/ttyACM0` all
verified present this session.

---

## What landed this session (committed + pushed)

### WIP cleanup — 20 commits across 5 repos (all pushed)
The 2026-05-30 GPU session's deliverables were uncommitted across 5 repos. Reviewed
+ committed into conventional commits, then pushed:
- `lerobot-isaac-env` (feature/wm-isaac-env): 3 commits — d435 camera obs wiring (A.1), env smoke/warmup runners (A.2).
- `lerobot-isaac-adapters` (feature/wm-isaac-env): 4 commits — W&B loggers (Bundle E), multi-dataset training flag (B.2), deploy robustness.
- `lerobot-isaac-synthetic` (**feat/align-synthetic-dr-schema** — NEW branch; main untouched): DR100 pipeline + MimicGen un-stub + this session's schema fixes.
- `lerobot-isaac-deploy` (feature/sim-deploy): 3 commits — A.3 sim runtime boot fixes, calibration-id persistence, HF-offline perf.
- workspace `main`: 7 commits — `env smoke` CLI delegation, feetech dep default, dashboards, GPU+HW checklist, NEXT_STEPS.

### Track A — sim verify: GREEN (GPU-verified this session)
- A.1: env obs dict includes `d435_rgb` ✅
- A.2: `lerobot-isaac env smoke` state-only `(1,18)` ✅; `--cameras=d435` dict obs w/ `d435_rgb (3,480,640)` ✅; 100 steps finite, exit 0.
- A.3: `IsaacSimRuntime._boot()` returns, `num_joints=6`, no raise ✅. **Caveat:** `rt.close()` hangs (Isaac `SimulationApp.close()` teardown quirk) → process needs timeout/kill. Worth an `os._exit()` teardown fallback. 100-step assertion not re-run.

**Operational gotcha:** Isaac Sim hard-exits via `os._exit()` which drops buffered
stdout — run all smokes with `PYTHONUNBUFFERED=1` or the verdict line is lost
(exit 0 with no output looks like a crash but isn't). The PhysX
`Failed to get a valid attached USD stage id ... kinematic bodies` line is
non-fatal noise.

### Track B — DR100 12-dim/overhead writer/replay: FIXED + GPU-verified
Commit `f3737d9` on `feat/align-synthetic-dr-schema` (pushed). Three real bugs that
blocked usable dataset production:
1. `replay_runner.main()` forced the **source** dataset's 6-dim/d435_rgb schema onto
   the writer → 12-dim rows hit "Feature mismatch (12,) vs (6,)". Fix: `features=None`
   → writer derives schema from generated episodes.
2. `next.reward` emitted as shape-(1,) array → lerobot 0.5 maps shape-(1,) to a scalar
   HF `Value` whose `encode_example` calls `float()` → raises under **numpy 2.x**.
3. `next.reward`/`next.done` don't exist in any real producer's schema → removed the
   injection in `_derive_features_from_episode` (a merged real+sim union can't carry them).

Probe verified on GPU: 2 src eps × 1 variant → `observation.state (12,)` +
`observation.images.overhead` + `action` + lerobot bookkeeping, 600 frames, parquet
intact, exit 0. Synthetic unit tests: 18 pass, 1 pre-existing fail
(`test_full_run_raises_import_error_not_not_implemented` — a deps-ABSENT contract test
that mis-fires in the deps-PRESENT `sim` env; worth a `skipif(lerobot importable)`).

---

## Schema reality — TWO recorders, TWO schemas (corrected 2026-06-06)

There are **two** distinct recorders/schemas, each legitimate. Earlier framing
("no producer emits 12-dim/overhead") was WRONG — it conflated the two.

| Producer | observation.state | camera key(s) | role |
|----------|-------------------|---------------|------|
| `robot-data-recorder` (`dual_writer.py:114,185`) | **[6]** joint_pos | **d435_rgb** | REAL hardware teleop |
| Real dataset `so101-pickplace1`, old dr100, merged (2026-05-30) | [6] | d435_rgb | on disk (hardware track) |
| **`lerobot-isaac-adapters/isaac_data_recorder.py`** (`:29,199,203`) | **[12]** joint_pos+joint_vel | **overhead** (+wrist) | Isaac **SIM** rollout recorder |
| `isaac_dr` dr-replay (synthetic, this session's fix) | **[12]** | **overhead** | Isaac **SIM** DR replay |

So **12-dim/overhead is the canonical SIM schema** (real joint_vel from sim;
`isaac_data_recorder` + `dr-replay` both emit it; `source="dr"`; merged via
`merge_utilities.merge_datasets`). **6-dim/d435_rgb is the hardware schema**
(`joint_vel` unavailable — `so101_teleop.py` zero-fills it).

`merge_datasets` merges a `"real"`-tagged leg + `"sim_dr"`-tagged leg, and **both
legs must share one schema** (it builds features from the inputs; 6-dim + 12-dim
would mismatch on frame combine).

dr-replay uses the real dataset only as an **action source** — it regenerates
12-dim/overhead sim obs from the real 6-dim action trajectories (proven by this
session's GPU probe). So the real data does NOT need re-recording to be a dr-replay
source.

---

## Open decision (BLOCKING) — what is the 12-dim/overhead set FOR?

The re-record question only arises if the training set must include HARDWARE real data.

1. **Sim-only training set (world-model / RL) — NO re-record, doable now.**
   `isaac_data_recorder` (sim rollouts) + `dr-replay` (DR, both 12-dim/overhead) →
   `merge_datasets`. Everything already speaks 12-dim/overhead. Remaining work is
   purely running the pipeline (full dr100 + record sim rollouts + merge). Nothing
   blocks this; the synthetic fix (`f3737d9`) made it work.

2. **Training set must include HARDWARE real data (sim-to-real / deployable policy).**
   Two sub-paths:
   - **2a. Keep two tracks:** 6-dim/d435_rgb for the real-deployable policy (existing
     100-ep merged set already valid), 12-dim/overhead for sim world-model. dr-replay's
     target schema then depends on which track it feeds. No re-record; pick per use.
   - **2b. Unify on 12-dim/overhead incl. hardware:** migrate `robot-data-recorder`
     (`state_dim` 6→12, `d435_rgb`→`overhead`) + re-record on the arm. Caveat: hardware
     `joint_vel` is zeros (no velocity sensor) → 6 zero columns on the real side.

USER LEANING (pre-correction): chose "12-dim migration" + "re-record". After the
two-recorder correction, "re-record" is only needed under 2b; if the goal is the sim
world-model set (option 1), no re-record is needed. **Confirm which use-case before
acting.**

---

## Next steps by option

- **Option 1 (sim-only 12-dim/overhead — no re-record):**
  1. Full dr100 (12-dim/overhead): `PYTHONUNBUFFERED=1 lerobot-isaac dr-replay --source_dataset datasets/kvgork/so101-pickplace1 --output_path datasets/kvgork/so101-pickplace1-dr100-12d --n_variants 4 --camera_key overhead --source_tag sim_dr` (GPU ~1hr).
  2. Optionally record fresh sim rollouts: `python -m lerobot_isaac_adapters.isaac_data_recorder --env_id Isaac-SO101-PickPlace-v0 --output_dir datasets/.../sim_rollouts --num_episodes 50` (GPU).
  3. Merge (both legs 12-dim/overhead): `python -m lerobot_isaac_synthetic.merge --real <12d-leg> --sim <dr100-12d> --out <merged-12d>`.
  4. NOTE: `merge_datasets` expects a `"real"` leg — for a sim-only set the "real" leg is whichever 12-dim/overhead dataset is the anchor (e.g. isaac_data_recorder rollouts). Verify the merge's source-tagging handles a sim anchor.
- **Option 2a (two tracks):** leave the existing 6-dim/d435_rgb merged set as the
  deployable-policy track (already valid); use 12-dim/overhead for the sim world-model
  track. Decide per-run which schema dr-replay targets (`--camera_key` + revert/keep the
  writer derive change). Possibly parameterize dr-replay to support BOTH schemas.
- **Option 2b (unify incl. hardware):** migrate `robot-data-recorder`
  (`dual_writer.py` `state_dim` 6→12, `d435_rgb`→`overhead`; `so101_teleop.py` 12-dim
  state assembly) → decide `joint_vel` policy (zeros vs finite-diff) → operator re-record
  → dr100 + merge, full 12-dim/overhead union.

Track B mechanics are unblocked (synthetic fix `f3737d9`); only the use-case choice gates which to run.

---

## Branch housekeeping (separate follow-up)
- `lerobot-isaac-synthetic` work is on **`feat/align-synthetic-dr-schema`** (created off
  `main` by the WIP-commit step; `main` untouched at `origin/main`). Decide: PR→main,
  rename, or keep. Other siblings used their existing feature branches.

## Key files
- `src/lerobot-isaac-synthetic/src/lerobot_isaac_synthetic/isaac_dr/parquet_writer.py` — writer + `_DEFAULT_SO101_FEATURES` + `_derive_features_from_episode`
- `src/lerobot-isaac-synthetic/src/lerobot_isaac_synthetic/isaac_dr/replay_runner.py` — replay + `main()` feature resolution
- `src/robot-data-recorder/src/robot_data_recorder/dual_writer.py` — recorder schema (`state_dim=6`, `d435_rgb`)
- `src/robot-data-recorder/src/robot_data_recorder/so101_teleop.py` — `joint_vel = zeros`
- `plans/2026-05-30-gpu-hw-execution-checklist.md` — the live Track A–D checklist (stale on schema)
