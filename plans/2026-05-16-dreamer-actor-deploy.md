# DreamerV3 Actor Deploy — Deferred Plan

**Date:** 2026-05-16
**Owner:** TBD
**Status:** **deferred** — waiting on first real sheeprl ckpt to land.

## Why deferred

The deploy package (`lerobot_isaac_deploy`) currently routes hardware
deploy through `robot-data-runner`'s subprocess CLI which loads policies
via lerobot's `make_policy` factory. DreamerV3 (sheeprl) checkpoints
are NOT lerobot policies — they have a different file layout
(`ckpt_*.ckpt` + `.hydra/config.yaml`) and a different forward-pass API
(`encoder + RSSM + actor`, recurrent state across steps).

Implementing dreamer actor deploy requires:

1. **In-process inference loop** in the deploy session (no subprocess) so
   we can hold the recurrent RSSM state across steps. The current
   subprocess design is stateless per-call.
2. **A laptop-side dreamer env**. sheeprl's deps (lightning, pettingzoo)
   are heavy. Pixi feature `dreamerv3` already enumerated, just not
   wired into the active environment yet.
3. **Validation against a real ckpt**. As of 2026-05-16 no DreamerV3
   run has produced a real `ckpt_*.ckpt` file in this workspace
   (`outputs/long-train-*/wm-dreamerv3/` contains only hydra logs +
   cli.log). The first real ckpt will reveal the exact sheeprl version's
   actor API.

## What's already in place

* `lerobot_isaac_deploy.policy_kind.detect_policy_kind` recognises
  dreamerv3 checkpoints by `.hydra/config.yaml + ckpt_*.ckpt`.
* `lerobot_isaac_deploy.wm_loader.load_dreamerv3` has the import +
  config-yaml + state-dict skeleton ready. Body raises actionable
  `ImportError` if sheeprl missing; otherwise calls `build_agent` +
  `load_state_dict`.
* `LoadedWMActor` exposes `select_action(obs)` + `reset()`.
* `DeploySession._validate_inputs` already refuses dreamerv3 with a
  one-paragraph hint pointing to this plan.

## When to execute

Triggers (any one is enough):
* A sheeprl DreamerV3 run completes ≥ 1 epoch and writes a real
  `ckpt_*.ckpt`.
* A user explicitly asks to deploy a dreamer-actor on the SO-101.

## Implementation outline (~4-6 h)

1. **Resolve sheeprl actor API drift.** Open the saved ckpt, dump its
   key set, confirm `build_agent` signature matches.
2. **In-process session path.** Refactor `DeploySession` so that the
   inference loop is overridable. For LeRobot path, keep the
   `robot-data-runner` subprocess. For dreamerv3 path, load the actor
   in-process and call `robot_data_runner.run_policy(cfg, loaded=...)`
   with our `LoadedWMActor` (the public Python API already accepts a
   pre-loaded policy).
3. **Reset hook.** Call `LoadedWMActor.reset()` between episodes.
4. **Smoke test on hardware** with `--max-relative-target 1.0` and the
   confirm-gated ladder.
5. **Update the runbook** with a DreamerV3-specific recipe in
   `docs/runbook/10-deploy-to-hardware.md`.

## Acceptance criteria

- [ ] `lerobot_isaac_deploy.wm_loader.load_dreamerv3` returns a working
      `LoadedWMActor` on a real sheeprl ckpt.
- [ ] `LoadedWMActor.select_action(obs)` returns a tensor of the SO-101
      action shape `(1, 6)` and updates the recurrent state.
- [ ] A dry-run ladder step (no `--execute`) prints sensible action
      values on real-camera observations.
- [ ] The confirm-gated ladder advances through `--execute --max-relative-target 1.0`
      without OOM on the laptop GPU (6 GB).
- [ ] A short closed-loop eval (3 episodes × 15 s) writes a JSON with
      finite `pc_success`.

## Out of scope (still)

* DreamerV3 *training* on the laptop. The actor is heavy enough that
  training would not fit; this plan covers deploy only.
* MPC planning on top of LeWM. Separate plan; see existing LeWM
  documentation in `docs/research/leworldmodel-reference.md`.
