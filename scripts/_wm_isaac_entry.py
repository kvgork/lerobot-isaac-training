"""AppLauncher-first sheeprl entry for the Isaac Lab WM training.

Isaac Lab requires SimulationApp to boot BEFORE any module-load that
pulls in `omni.kit.*` or shared libs that link against libgobject.
`python -m sheeprl` imports a chain of hydra + lightning + protobuf
packages that claim libgobject FIRST → Isaac Sim's `libgpu.foundation.plugin.so`
later fails to load with `undefined symbol: g_string_copy`.

This entry script:
  1. Boots SimulationApp via AppLauncher (libgobject is now bound to
     Isaac Sim's expected version).
  2. Forwards argv to sheeprl's hydra-decorated `run()`.

Use:
    python scripts/_wm_isaac_entry.py --config-dir <plugin_configs> \
        exp=dreamer_v3 env=isaac_so101 ...

Same args as `python -m sheeprl`, just with AppLauncher pre-booted.
"""
from __future__ import annotations

import sys


def _patch_gym_transform_observation() -> None:
    """Compat shim: sheeprl 0.5.8 calls ``TransformObservation(env, func)``
    (2-arg), but gymnasium 1.2.1 (hard-pinned by Isaac Lab) made
    ``observation_space`` a required positional with no default →
    ``TypeError: ... missing 1 required positional argument: 'observation_space'``.

    Default it to ``None``. Old gymnasium inferred the env's own (unchanged)
    observation_space when omitted, and ``observation_space=None`` in gymnasium
    1.2.1 yields the same unchanged-space behaviour — faithful, not a semantics
    change. Lets the WM-Isaac DreamerV3 sweep run without downgrading gymnasium
    (which would break Isaac Lab's ``gymnasium==1.2.1`` pin).
    """
    import inspect

    import gymnasium.wrappers as gw

    cls = gw.TransformObservation
    osp = inspect.signature(cls.__init__).parameters.get("observation_space")
    if osp is not None and osp.default is inspect.Parameter.empty:
        _orig_init = cls.__init__

        def _init(self, env, func, observation_space=None, *args, **kwargs):
            return _orig_init(self, env, func, observation_space, *args, **kwargs)

        cls.__init__ = _init


def _patch_gym_vector_final_info() -> None:
    """Compat shim: sheeprl 0.5.8 reads per-episode stats from
    ``infos["final_info"]`` (a per-env list — gymnasium <1.0 vector API):

        for i, ep in enumerate(infos["final_info"]):
            if ep is not None:
                aggregator.update("Rewards/rew_avg", ep["episode"]["r"])

    gymnasium 1.2.1 (hard-pinned by Isaac Lab) **removed** ``final_info``. Its
    vector ``RecordEpisodeStatistics`` now reports finished-episode stats as
    ``infos["episode"] = {"r": array, "l": array, ...}`` plus an
    ``infos["_episode"]`` boolean mask. So ``"final_info" in infos`` is always
    False and ``Rewards/rew_avg`` / ``Game/ep_len_avg`` NEVER log — the WM-Isaac
    sweep then ratchets every trial at the ``-9999`` sentinel even though
    training is healthy (losses, which don't read final_info, log fine).

    Rebuild the ``final_info`` list from the new keys on every vector step, so
    sheeprl's existing reader works unchanged. Same incompatibility class as
    ``_patch_gym_transform_observation`` — apply BEFORE importing sheeprl.
    """
    import numpy as np

    try:
        import gymnasium.vector as gv
    except Exception:  # noqa: BLE001
        return

    def _augment(infos: object, n: int) -> None:
        if not (isinstance(infos, dict) and "final_info" not in infos):
            return
        ep = infos.get("episode")
        mask = infos.get("_episode")
        if not (isinstance(ep, dict) and ep.get("r") is not None and mask is not None):
            return
        r = ep["r"]
        length = ep.get("l")
        final_info = [None] * n
        for i in range(n):
            if bool(mask[i]):
                final_info[i] = {
                    "episode": {
                        "r": np.array([float(r[i])]),
                        "l": np.array([float(length[i])]) if length is not None else np.array([0.0]),
                    }
                }
        infos["final_info"] = final_info

    for cls_name in ("SyncVectorEnv", "AsyncVectorEnv"):
        cls = getattr(gv, cls_name, None)
        if cls is None or getattr(cls, "_lerobot_final_info_patched", False):
            continue
        _orig_step = cls.step

        def _step(self, actions, _orig=_orig_step):  # noqa: ANN001
            obs, rew, term, trunc, infos = _orig(self, actions)
            _augment(infos, getattr(self, "num_envs", 1))
            return obs, rew, term, trunc, infos

        cls.step = _step
        cls._lerobot_final_info_patched = True


def _patch_gym_vector_isaac() -> None:
    """Fix 2: true num_envs>1 vectorization for the Isaac env.

    sheeprl's dreamer_v3 builds ``gym.vector.SyncVectorEnv([make_env... for i in
    range(num_envs)])`` — N separate IsaacSO101Env wrappers all fighting over the
    Isaac SimulationContext singleton → crash at num_envs>1. Intercept the vector
    constructors: when given >1 env_fns whose env unwraps to IsaacSO101Env, build
    ONE ``IsaacSO101VectorEnv`` over the (already N-parallel) backing env instead.
    num_envs=1 (len==1) falls through to the original constructor untouched.
    """
    try:
        import gymnasium.vector as gv
        from lerobot_isaac_adapters.sheeprl_plugin.isaac_env import IsaacSO101Env
        from lerobot_isaac_adapters.sheeprl_plugin.isaac_vector_env import IsaacSO101VectorEnv
    except Exception as exc:  # noqa: BLE001
        print(f"[wm-isaac-entry] vector patch skipped: {exc}", flush=True)
        return

    for cls_name in ("SyncVectorEnv", "AsyncVectorEnv"):
        orig = getattr(gv, cls_name, None)
        if orig is None or getattr(orig, "_lerobot_isaac_vec_patched", False):
            continue

        def _factory(env_fns, *args, _orig=orig, **kwargs):
            fns = list(env_fns)
            if len(fns) > 1:
                first = fns[0]()
                base = getattr(first, "unwrapped", first)
                if isinstance(base, IsaacSO101Env):
                    print(f"[wm-isaac-entry] Fix2: {len(fns)} isaac env_fns → "
                          f"IsaacSO101VectorEnv(num_envs={len(fns)})", flush=True)
                    return IsaacSO101VectorEnv(existing_env=base, num_envs=len(fns))
                try:
                    first.close()
                except Exception:  # noqa: BLE001
                    pass
            return _orig(env_fns, *args, **kwargs)

        _factory._lerobot_isaac_vec_patched = True
        setattr(gv, cls_name, _factory)


def _patch_seed_demo_buffer() -> None:
    """Stage 3 warm-start: pre-seed sheeprl's replay buffer with SIM pick-place demos.

    Gated by env var LEROBOT_ISAAC_DEMO_DATASET (path to a LeRobotDataset). The demos
    give the DreamerV3 world model the carry->place DYNAMICS it can't discover online
    (plans/2026-06-11-demo-warmstart-plan.md, user order 1->3->2: seed buffer first).

    Mechanism: monkeypatch EnvIndependentReplayBuffer.add to LAZY-seed on its first
    call — at that point the online step_data reveals the exact obs schema (keys, state
    dim, image size, dtypes), so the demo arrays are adapted to match before insertion.
    Demos are added to env-0's sub-buffer as full (T,1,...) sequences. Rewards are 0 (the
    WM learns dynamics; the actor learns reward from online episodes / a later BC pass).
    """
    import os
    demo_root = os.environ.get("LEROBOT_ISAAC_DEMO_DATASET", "").strip()
    if not demo_root:
        return
    try:
        import numpy as np
        from sheeprl.data.buffers import EnvIndependentReplayBuffer
        from lerobot_isaac_adapters.sheeprl_plugin.demo_buffer import load_sim_demos
    except Exception as exc:  # noqa: BLE001
        print(f"[wm-isaac-entry] demo-seed patch skipped (import): {exc}", flush=True)
        return
    if getattr(EnvIndependentReplayBuffer.add, "_lerobot_demo_seed_patched", False):
        return

    _orig_add = EnvIndependentReplayBuffer.add
    max_demos = int(os.environ.get("LEROBOT_ISAAC_DEMO_MAX", "0")) or None

    def _seed(self, sample):
        keys = list(sample.keys())
        state_dim = int(sample["state"].shape[-1]) if "state" in sample else None
        img = int(sample["rgb"].shape[-1]) if "rgb" in sample else 64
        eps = load_sim_demos(demo_root, image_size=img, max_episodes=max_demos)
        n_steps = 0
        for ep in eps:
            data = {}
            for k in keys:
                if k not in ep:
                    raise KeyError(f"demo episode missing online key {k!r} (have {list(ep)})")
                arr = np.asarray(ep[k])
                if k == "state" and state_dim and arr.shape[-1] != state_dim:
                    arr = arr[..., :state_dim]   # 12-dim (pos+vel) demo -> 6-dim env (joint pos)
                data[k] = arr[:, None]           # (T, ...) -> (T, 1, ...) for one env
            _orig_add(self, data, indices=[0], validate_args=False)
            n_steps += data[keys[0]].shape[0]
        print(f"[wm-isaac-entry] SEEDED {len(eps)} demo episodes ({n_steps} transitions) "
              f"into replay buffer from {demo_root}", flush=True)

    def patched_add(self, data, indices=None, validate_args=False):
        if not getattr(self, "_lerobot_demos_seeded", False):
            self._lerobot_demos_seeded = True
            try:
                _seed(self, data)
            except Exception as exc:  # noqa: BLE001 — never block training on a seed error
                import traceback
                traceback.print_exc()
                print(f"[wm-isaac-entry] demo seeding FAILED: {exc}", flush=True)
        return _orig_add(self, data, indices=indices, validate_args=validate_args)

    patched_add._lerobot_demo_seed_patched = True
    EnvIndependentReplayBuffer.add = patched_add
    print(f"[wm-isaac-entry] demo-seed patch armed (dataset={demo_root}, "
          f"max={max_demos or 'all'})", flush=True)


def _patch_bc_actor_loss() -> None:
    """DreamerFD BC actor loss: inject an explicit behavior-cloning gradient into
    DreamerV3's actor update, so the actor *converts* demo dynamics (already loaded
    into the replay buffer by _patch_seed_demo_buffer) into demo BEHAVIOUR.

    Without this patch the actor gets zero imitation gradient — it can only
    discover the carry→place policy through online reward, which plateaus because
    exploration never reaches the bin. This patch closes the loop.

    Gating (env vars):
        LEROBOT_ISAAC_BC_WEIGHT    (float, default 0.0 = OFF)
            Initial BC weight.  When 0.0 the patch is a no-op and existing runs
            are completely unaffected (no sheeprl module touched on import).
        LEROBOT_ISAAC_BC_DECAY_STEPS  (int, default 20000)
            Number of actor-update steps over which bc_weight decays linearly
            toward 0.0 (DreamerFD "virtual clutch").
        LEROBOT_ISAAC_DEMO_DATASET
            Path to the demo LeRobotDataset.  Required for BC to work;
            if absent the patch exits early with a warning.

    Mechanism — monkeypatch ``sheeprl.algos.dreamer_v3.dreamer_v3.train``:
        After the standard DreamerV3 train() call (WM update + imagined actor
        update + critic update), a SEPARATE BC actor step fires:
          1. Sample a demo mini-batch from DemoBuffer (RLPD-style: same
             batch_size and sequence_length as the online batch).
          2. Encode demo obs (rgb + state) through world_model.encoder and
             world_model.rssm to produce latent_states.
          3. Call actor(latent_states) → distributions.
          4. BC loss = -bc_weight(step) * mean(log pi(a_demo | latent)).
          5. actor_optimizer.zero_grad → fabric.backward(bc_loss) → step.

    The latent encoding path mirrors sheeprl's dreamer_v3.train() exactly
    (same RSSM dynamic call, same latent concatenation) so the actor sees the
    same latent geometry as during imagination — no distribution shift.

    GPU validation required (see tests/test_bc_loss.py for CPU-only unit tests).
    """
    import os

    bc_w = float(os.environ.get("LEROBOT_ISAAC_BC_WEIGHT", "0.0"))
    if bc_w <= 0.0:
        return  # OFF — zero behaviour change, nothing patched

    demo_root = os.environ.get("LEROBOT_ISAAC_DEMO_DATASET", "").strip()
    if not demo_root:
        print(
            "[wm-isaac-entry] BC patch: LEROBOT_ISAAC_BC_WEIGHT > 0 but "
            "LEROBOT_ISAAC_DEMO_DATASET not set — BC actor loss DISABLED.",
            flush=True,
        )
        return

    decay_steps = int(os.environ.get("LEROBOT_ISAAC_BC_DECAY_STEPS", "20000"))

    try:
        import sheeprl.algos.dreamer_v3.dreamer_v3 as _dv3_mod
        from lerobot_isaac_adapters.sheeprl_plugin.demo_buffer import (
            DemoBuffer,
            bc_weight,
            load_sim_demos,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[wm-isaac-entry] BC patch skipped (import): {exc}", flush=True)
        return

    if getattr(_dv3_mod.train, "_lerobot_bc_patched", False):
        return

    _orig_train = _dv3_mod.train

    # State shared across train() calls (closed over by the wrapper).
    # Populated lazily on the first call so we know the obs schema from cfg.
    _state: dict = {
        "demo_buf": None,       # DemoBuffer | None
        "step": 0,              # actor-update step counter
        "warned_once": False,   # suppress repeated error spam
    }

    def _ensure_demo_buf(cfg) -> "DemoBuffer | None":
        """Load DemoBuffer once; return None on failure."""
        if _state["demo_buf"] is not None:
            return _state["demo_buf"]
        # Infer image size from cfg: cfg.env.screen_size or fall back to 64
        img = 64
        try:
            img = int(cfg.env.screen_size)
        except Exception:  # noqa: BLE001
            pass
        try:
            max_ep = int(os.environ.get("LEROBOT_ISAAC_DEMO_MAX", "0")) or None
            episodes = load_sim_demos(demo_root, image_size=img, max_episodes=max_ep)
            buf = DemoBuffer(episodes=episodes)
            _state["demo_buf"] = buf
            print(
                f"[wm-isaac-entry] BC patch: loaded {buf.n_episodes} demo episodes "
                f"({buf.n_transitions} transitions) into DemoBuffer.",
                flush=True,
            )
            return buf
        except Exception as exc:  # noqa: BLE001
            if not _state["warned_once"]:
                print(f"[wm-isaac-entry] BC patch: DemoBuffer load FAILED: {exc}", flush=True)
                _state["warned_once"] = True
            return None

    def _bc_step(fabric, world_model, actor, actor_optimizer, cfg, aggregator):
        """One BC actor update on a demo mini-batch.

        Encodes demo obs through world_model → latents → actor log_prob(demo_actions)
        → BC loss → fabric.backward → actor_optimizer step.
        """
        import torch

        step = _state["step"]
        w = bc_weight(step, start=bc_w, decay_steps=decay_steps)
        if w <= 0.0:
            return  # weight has fully decayed

        buf = _ensure_demo_buf(cfg)
        if buf is None:
            return

        batch_size = cfg.algo.per_rank_batch_size
        seq_len = cfg.algo.per_rank_sequence_length
        stochastic_size = cfg.algo.world_model.stochastic_size
        discrete_size = cfg.algo.world_model.discrete_size
        recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
        device = fabric.device

        try:
            demo = buf.sample(batch_size=batch_size, seq_len=seq_len)
        except Exception as exc:  # noqa: BLE001
            if not _state["warned_once"]:
                print(f"[wm-isaac-entry] BC patch: demo sample FAILED: {exc}", flush=True)
                _state["warned_once"] = True
            return

        # Build obs dict matching what sheeprl's train() uses
        cnn_keys = list(cfg.algo.cnn_keys.encoder)  # e.g. ["rgb"]
        mlp_keys = list(cfg.algo.mlp_keys.encoder)  # e.g. ["state"]

        demo_obs: dict = {}
        for k in cnn_keys:
            if k in demo:
                arr = demo[k]  # (seq, batch, 3, H, W) uint8
                demo_obs[k] = torch.from_numpy(arr).float().to(device) / 255.0 - 0.5
            else:
                # sheeprl key may be "rgb" but DemoBuffer stores "d435_rgb" — try both
                for dk in ("rgb", "d435_rgb"):
                    if dk in demo:
                        arr = demo[dk]
                        demo_obs[k] = torch.from_numpy(arr).float().to(device) / 255.0 - 0.5
                        break
        for k in mlp_keys:
            if k in demo:
                arr = demo[k]  # (seq, batch, state_dim)
                demo_obs[k] = torch.from_numpy(arr).float().to(device)

        # Demo actions: (seq, batch, action_dim)
        demo_actions = torch.from_numpy(demo["actions"]).float().to(device)

        # is_first: (seq, batch, 1) — force first step = True as train() does
        is_first = torch.from_numpy(demo["is_first"]).float().to(device)
        is_first[0] = torch.ones_like(is_first[0])

        # Build batch_actions with leading zero (mirrors sheeprl train())
        batch_actions_demo = torch.cat(
            (torch.zeros_like(demo_actions[:1]), demo_actions[:-1]), dim=0
        )

        # Encode demo obs → latent states (same RSSM path as sheeprl train())
        with torch.no_grad():
            embedded = world_model.encoder(demo_obs)

        recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)

        decoupled = getattr(cfg.algo.world_model, "decoupled_rssm", False)
        if decoupled:
            with torch.no_grad():
                _posteriors_logits, posteriors_d = world_model.rssm._representation(embedded)
            recurrent_states = torch.empty(seq_len, batch_size, recurrent_state_size, device=device)
            posteriors = posteriors_d
            for i in range(seq_len):
                prior_post = torch.zeros_like(posteriors_d[:1]) if i == 0 else posteriors_d[i - 1:i]
                with torch.no_grad():
                    recurrent_state, _, _ = world_model.rssm.dynamic(
                        prior_post, recurrent_state,
                        batch_actions_demo[i:i + 1], is_first[i:i + 1],
                    )
                recurrent_states[i] = recurrent_state
        else:
            posterior = torch.zeros(1, batch_size, stochastic_size, discrete_size, device=device)
            posteriors = torch.empty(seq_len, batch_size, stochastic_size, discrete_size, device=device)
            recurrent_states = torch.empty(seq_len, batch_size, recurrent_state_size, device=device)
            for i in range(seq_len):
                with torch.no_grad():
                    recurrent_state, posterior, _, _, _ = world_model.rssm.dynamic(
                        posterior, recurrent_state,
                        batch_actions_demo[i:i + 1],
                        embedded[i:i + 1],
                        is_first[i:i + 1],
                    )
                recurrent_states[i] = recurrent_state
                posteriors[i] = posterior

        # latent_states: (seq, batch, stoch+recurrent)
        latent_states = torch.cat(
            (posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1
        )

        # BC loss: -w * E[log pi(a_demo | latent)]
        # Split demo_actions across actor heads (continuous: 1 head; discrete: N heads).
        # Uniform split: if action_dim divisible by n_heads; otherwise all to head 0.
        actor_optimizer.zero_grad(set_to_none=True)
        _, policies = actor(latent_states)
        n_heads = len(policies)
        act_dim = demo_actions.shape[-1]
        if n_heads > 0 and act_dim % n_heads == 0:
            action_splits = torch.split(demo_actions, act_dim // n_heads, dim=-1)
        else:
            # SO-101 continuous actor is single-head (n_heads==1) so this never fires.
            # If a future multi-head actor doesn't divide the action dim evenly, BC would
            # silently supervise only head 0 — warn loudly rather than degrade in silence.
            import logging as _lg
            _lg.getLogger(__name__).warning(
                "[wm-isaac-entry] BC: actor has %d heads but action_dim=%d is not divisible "
                "— BC supervises head 0 only; multi-head BC split is unimplemented.",
                n_heads, act_dim,
            )
            action_splits = [demo_actions] + [demo_actions[:, :, :0]] * (n_heads - 1)

        lp_list = []
        for pol, act_chunk in zip(policies, action_splits):
            try:
                lp = pol.log_prob(act_chunk)  # (seq, batch) or (seq, batch, 1)
                lp_list.append(lp.reshape(seq_len, batch_size))
            except Exception:  # noqa: BLE001
                pass

        if not lp_list:
            actor_optimizer.zero_grad(set_to_none=True)
            return

        log_prob = torch.stack(lp_list, dim=-1).sum(dim=-1)  # (seq, batch)
        bc_loss = -w * log_prob.mean()

        fabric.backward(bc_loss)
        actor_optimizer.step()

        if aggregator and not aggregator.disabled:
            try:
                # Register our keys once (lazily) — sheeprl builds the aggregator from
                # cfg.metric.aggregator, which does NOT include our BC keys, so an
                # unregistered update() is silently dropped with a UserWarning. add()
                # itself warns if the key exists, so guard on membership to avoid
                # per-step warn-spam. MeanMetric mirrors sheeprl's Loss/* metrics;
                # sync_on_compute=False is correct for the single-process (devices=1) run.
                from torchmetrics import MeanMetric

                _dev = bc_loss.device
                if "Loss/bc_loss" not in aggregator.metrics:
                    aggregator.add("Loss/bc_loss", MeanMetric(sync_on_compute=False).to(_dev))
                if "Params/bc_weight" not in aggregator.metrics:
                    aggregator.add("Params/bc_weight", MeanMetric(sync_on_compute=False).to(_dev))
                aggregator.update("Loss/bc_loss", bc_loss.detach())
                aggregator.update("Params/bc_weight", torch.tensor(w, device=_dev))
            except Exception:  # noqa: BLE001
                pass

        _state["step"] += 1

    def patched_train(
        fabric, world_model, actor, critic, target_critic,
        world_optimizer, actor_optimizer, critic_optimizer,
        data, aggregator, cfg, is_continuous, actions_dim, moments,
    ):
        # 1. Standard DreamerV3 update (WM update + imagined actor + critic)
        _orig_train(
            fabric, world_model, actor, critic, target_critic,
            world_optimizer, actor_optimizer, critic_optimizer,
            data, aggregator, cfg, is_continuous, actions_dim, moments,
        )
        # 2. Extra BC actor step on a demo mini-batch (DreamerFD)
        try:
            _bc_step(fabric, world_model, actor, actor_optimizer, cfg, aggregator)
        except Exception as exc:  # noqa: BLE001 — never crash training on BC error
            import traceback
            traceback.print_exc()
            print(f"[wm-isaac-entry] BC actor step FAILED (non-fatal): {exc}", flush=True)

    patched_train._lerobot_bc_patched = True
    _dv3_mod.train = patched_train
    print(
        f"[wm-isaac-entry] BC actor-loss patch armed "
        f"(bc_weight={bc_w}, decay_steps={decay_steps}, demo={demo_root})",
        flush=True,
    )


def _patch_torch_load_weights_only() -> None:
    """PyTorch 2.6 defaults torch.load(weights_only=True), which rejects sheeprl
    checkpoints on resume (checkpoint.resume_from) — they pickle the replay buffer +
    cfg, not just tensors → UnpicklingError "Weights only load failed". Our own ckpts
    are trusted, so force weights_only=False. Needed for the curriculum resume chain."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"[wm-isaac-entry] torch.load patch skipped: {exc}", flush=True)
        return
    if getattr(torch.load, "_lerobot_wo_patched", False):
        return
    _orig = torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)

    _load._lerobot_wo_patched = True
    torch.load = _load
    print("[wm-isaac-entry] torch.load weights_only=False patch armed (for resume)", flush=True)


def _patch_residual_rl_action() -> None:
    """Residual RL on the scripted grasp: blend a scripted base action with the policy
    action so the agent learns a residual on a WORKING grasp primitive instead of
    re-discovering the (unlearnable) grasp from scratch.

    Env vars:
        LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT       (float, default 0.0 = OFF) — w0, the
                                                SCRIPT fraction at step 0; w0=1.0 ⇒ pure
                                                script initially. The POLICY fraction is
                                                (1 - script_frac) and rises 0→1.
        LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS  (int, default 50000) — env-steps over
                                                which script_frac decays w0→0 (DAgger /
                                                residual handoff: script-dominant early,
                                                policy-solo late). Counts non-eval
                                                get_actions calls (= env steps at
                                                num_envs=1, post-prefill).

    CRITICAL (model-based RL): the blend MUST happen at the action-SELECTION seam
    (PlayerDV3.get_actions, called at dreamer_v3.py:577) — BEFORE rb.add (:587) — so the
    SAME action is recorded to the replay buffer AND executed by the env. Blending inside
    IsaacSO101Env.step would record the policy action but execute the blended one, teaching
    the world model wrong dynamics (silent failure). We also overwrite `player.actions`
    with the blended action so the player's online recurrent latent stays consistent with
    what the env executed. See memory `sheeprl-action-override-buffer-seam`.

    Eval: sheeprl calls `test(..., greedy=False)` (dreamer_v3.py:767) — so the `greedy`
    flag does NOT mark eval. We therefore wrap dreamer_v3's `test` to set an `in_eval`
    flag and skip the blend during eval, so eval measures the PURE policy.

    Caveats (GPU-validation pending — see plans/2026-06-22-grasp-learning-wall-CONVERGED.md):
      * Actions are clipped to [-1,1] so the tanh actor can reproduce them; the scripted
        controller is REACTIVE (re-solves each step), so the clip rate-limits rather than
        breaks it (multi-step convergence vs the open-loop demo's |a|>1 single moves).
      * Run with LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1 so the actor observes the object
        location the script uses — else it cannot learn to reproduce the grasp.
      * num_envs MUST be 1 (also the is_first constraint); the patch guards >1.
      * The scripted action is sim-only (Isaac IK); on hardware compute_scripted_action()
        returns None → blend skipped → pure policy (residual effectively OFF).
    """
    import os

    w0 = float(os.environ.get("LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT", "0.0"))
    if w0 <= 0.0:
        return  # OFF — no patch (default); returns before importing sheeprl
    decay_steps = int(os.environ.get("LEROBOT_ISAAC_RESIDUAL_RL_DECAY_STEPS", "50000"))

    try:
        from sheeprl.algos.dreamer_v3.agent import PlayerDV3
    except Exception as exc:  # noqa: BLE001
        print(f"[wm-isaac-entry] residual RL patch skipped: {exc}", flush=True)
        return
    if getattr(PlayerDV3.get_actions, "_lerobot_residual_patched", False):
        return

    _orig_get_actions = PlayerDV3.get_actions
    _state = {"step": 0, "in_eval": False, "warned_layout": False, "warned_nenvs": False}

    # Wrap dreamer_v3's `test` so the blend is skipped during eval (eval uses
    # greedy=False, so the greedy flag alone cannot detect it).
    try:
        import sheeprl.algos.dreamer_v3.dreamer_v3 as _dv3_main

        if hasattr(_dv3_main, "test") and not getattr(_dv3_main.test, "_lerobot_eval_wrap", False):
            _orig_test = _dv3_main.test

            def _test_wrap(*a, **k):
                _state["in_eval"] = True
                try:
                    return _orig_test(*a, **k)
                finally:
                    _state["in_eval"] = False

            _test_wrap._lerobot_eval_wrap = True
            _dv3_main.test = _test_wrap
    except Exception as exc:  # noqa: BLE001 — non-fatal; falls back to greedy-only guard
        print(f"[wm-isaac-entry] residual RL: eval-guard wrap skipped ({exc})", flush=True)

    def patched_get_actions(self, obs, greedy=False, mask=None):
        actions = _orig_get_actions(self, obs, greedy=greedy, mask=mask)
        # Skip blend during eval (greedy OR the test() phase) — eval measures pure policy.
        if greedy or _state["in_eval"]:
            return actions
        # Only the continuous single-block action layout is supported (multi-categorical
        # → no-op). Warn once so an operator can't mistake "armed" for "active".
        if not isinstance(actions, (list, tuple)) or len(actions) != 1:
            if not _state["warned_layout"]:
                print(
                    "[residual-rl] action layout is not single-block (len != 1) — residual "
                    "is a NO-OP for this actor; pure policy used.",
                    flush=True,
                )
                _state["warned_layout"] = True
            return actions
        # Advance the decay clock on EVERY non-eval training step (even ones we end up
        # not blending), so the schedule never stalls on a transient skip.
        step = _state["step"]
        _state["step"] = step + 1
        script_frac = w0 * max(0.0, 1.0 - step / max(1, decay_steps))
        if script_frac <= 1e-4:
            return actions  # handed off to the policy — skip the IK cost entirely
        try:
            import torch

            from lerobot_isaac_adapters.sheeprl_plugin import isaac_env as _ienv

            wrapper = getattr(_ienv, "_LAST_WRAPPER", None)
            if wrapper is None:
                return actions
            a_pol = actions[0]
            # num_envs>1 is unsupported (one scripted action can't be broadcast across
            # envs correctly) — and num_envs must be 1 anyway (is_first bug). Guard + warn.
            # Action shape is (1, num_envs, action_dim) → shape[-2] is num_envs.
            n_envs = a_pol.shape[-2] if a_pol.dim() >= 2 else 1
            if n_envs != 1:
                if not _state["warned_nenvs"]:
                    print(
                        "[residual-rl] num_envs>1 detected — residual unsupported, pure "
                        "policy used.",
                        flush=True,
                    )
                    _state["warned_nenvs"] = True
                return actions
            a_script = wrapper.compute_scripted_action()
            if a_script is None:
                return actions  # hardware / scene unavailable → pure policy
            a_scr = torch.as_tensor(a_script, dtype=a_pol.dtype, device=a_pol.device)
            if a_scr.numel() != a_pol.numel():
                return actions
            # Match a_pol's exact shape (e.g. (1,1,6)) — no broadcast luck.
            a_scr = a_scr.reshape(a_pol.shape)
            # Clip to [-1,1]: the tanh actor can only reproduce in-range actions, so the
            # recorded/executed action must stay in range for the handoff to converge.
            a_blend = torch.clamp(script_frac * a_scr + (1.0 - script_frac) * a_pol, -1.0, 1.0)
            # Keep the player's internal "last action" consistent with the executed action.
            self.actions = torch.cat([a_blend], -1)
            if step % 500 == 0:
                print(
                    f"[residual-rl] step={step} script_frac={script_frac:.3f} "
                    f"(w0={w0}, decay={decay_steps})",
                    flush=True,
                )
            return [a_blend]
        except Exception as exc:  # noqa: BLE001 — never crash the rollout on a residual error
            print(f"[residual-rl] blend FAILED (non-fatal, using policy action): {exc}", flush=True)
            return actions

    patched_get_actions._lerobot_residual_patched = True
    PlayerDV3.get_actions = patched_get_actions
    print(
        f"[wm-isaac-entry] residual RL patch armed (w0={w0}, decay_steps={decay_steps}) — "
        "scripted-grasp base action blended at the get_actions seam (sim-only, eval-guarded)",
        flush=True,
    )


def main() -> None:
    # 1. Boot SimulationApp FIRST — claims libgobject + omni.kit.app.
    from isaaclab.app import AppLauncher

    headless = "--no-headless" not in sys.argv
    launcher = AppLauncher(headless=headless, enable_cameras=True)
    # Two update ticks to let extensions settle before sheeprl imports.
    for _ in range(2):
        launcher.app.update()

    # 2. Strip our own flags from sys.argv before handing off to hydra.
    for tok in ("--no-headless",):
        while tok in sys.argv:
            sys.argv.remove(tok)
    # Hydra resolves config_path relative to this file's dir by default.
    # The sheeprl CLI uses `--config-dir <path>` to override → pass through.

    # 2b. Patch gymnasium for sheeprl 0.5.8 compat (BEFORE importing sheeprl,
    #     which builds the env + vector wrappers at run() time).
    _patch_gym_transform_observation()
    _patch_gym_vector_final_info()
    _patch_gym_vector_isaac()  # Fix 2: num_envs>1 → one batched IsaacSO101VectorEnv
    _patch_seed_demo_buffer()  # Stage 3: seed replay buffer with sim demos (env-gated)
    _patch_bc_actor_loss()     # DreamerFD: explicit BC actor gradient (env-gated by LEROBOT_ISAAC_BC_WEIGHT)
    _patch_residual_rl_action()  # Residual RL: blend scripted-grasp base action (env-gated by LEROBOT_ISAAC_RESIDUAL_RL_WEIGHT)
    _patch_torch_load_weights_only()  # allow checkpoint.resume_from on torch 2.6 (curriculum)

    # 3. Call sheeprl's hydra-decorated `run()` with the remaining argv.
    #    sys.argv[0] is expected to be the program name; rewrite to mimic
    #    `python -m sheeprl`.
    sys.argv[0] = "sheeprl"
    from sheeprl.cli import run

    # Run sheeprl, then FORCE-exit with os._exit to bypass Isaac Sim's atexit
    # SimulationApp.close(), which hangs forever in render() on shutdown. That
    # hang is the WM-Isaac "stall": a finished OR crashed trial keeps the
    # process alive holding GPU/VRAM, never emits a final metric, and looks
    # like training froze. os._exit skips all atexit handlers, so a trial dies
    # the instant run() returns or raises — freeing the GPU for the next trial.
    import os

    exit_code = 0
    try:
        run()
    except SystemExit as exc:  # hydra may sys.exit on a job error
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:  # noqa: BLE001 — log then hard-exit, never hang
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
