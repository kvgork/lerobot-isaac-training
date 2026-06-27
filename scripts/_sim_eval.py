"""Closed-loop sim eval: roll a LeRobot policy out in the Isaac SO-101 task env.

Unlike `_open_loop_eval.py` (action-MSE on recorded frames, no environment), this
actually steps the policy through the Isaac Lab `ManagerBasedRLEnv` and scores the
env's own termination manager's "success" term (place_termination -> object_in_bin).
Reports `pc_success` = fraction of episodes that hit a (non-timeout) terminal state.

Also reports `task_success` = fraction of episodes where the env's "success"
termination term fired — read from the termination manager's pre-reset cache, so it
is immune to Isaac Lab's auto-reset that clobbers `root_pos_w` before step() returns.

The "success" term for pick_and_place wires to place_termination (object_in_bin),
verified at pick_and_place.py:461:
  terminations.success = TermCfg(func=place_termination,
                                  params={target_pos, success_radius:0.06,
                                          object_name:"source_object"})

AppLauncher MUST boot before any `isaaclab.*` import — same recipe as
`scripts/_wm_isaac_entry.py`.

TODO: This fix (sourcing task_success from termination_manager pre-reset cache) is
NOT yet GPU-verified and must be confirmed on the next GPU run.

Caveats (sim2real): the policy was trained on REAL D435 frames + real joint-state
units. The Isaac `d435_rgb`/joint obs differ in appearance and unit scaling, so a
real-trained policy will likely score low here — this harness measures *that the
closed-loop eval runs + scores*, not that the policy transfers. Mapping:
  observation.images.overhead <- d435_rgb (3,480,640)
  observation.state           <- cat(joint_pos[6], joint_vel[6])  (12,)

Sim-env prereqs (one-time — the `sim` env ships isaaclab but the LeRobot policy
stack needs aligning; install_train_deps.sh does NOT target the `sim` env):
  .pixi/envs/sim/bin/python -m pip install "transformers==5.3.0" num2words
  # transformers 5.3.0 accepts huggingface-hub 1.x (4.57.x demands hub<1.0 → import
  # error); num2words is required by transformers' SmolVLM processor. lerobot is
  # already present in the sim env (feature: dev+lerobot+isaaclab+editable-siblings).

Usage:
  .pixi/envs/sim/bin/python scripts/_sim_eval.py \
    --policy_path outputs/.../checkpoints/last/pretrained_model \
    --dataset_root datasets/local/so101-pickplace-new \
    --task pick_and_place --n_episodes 10 --max_steps 300 \
    --output_json outputs/smolvla-sim-eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


def _read_success_term(env) -> "bool | None":
    """Whether the env's 'success' termination term fired on the last step.

    Reads the termination manager's cached per-term done (computed pre-reset),
    so it is immune to the auto-reset that clobbers root_pos_w. None if unavailable.

    For pick_and_place the 'success' term is place_termination -> object_in_bin,
    wired at pick_and_place.py:461. This is the canonical predicate captured at
    the correct (pre-reset) time.
    """
    try:
        tm = env.unwrapped.termination_manager
        done = tm.get_term("success") if hasattr(tm, "get_term") else tm._term_dones["success"]
        return bool(done[0])
    except Exception:  # noqa: BLE001
        return None


def _boot_app(headless: bool):
    """Boot SimulationApp via AppLauncher (cameras on) before isaaclab imports."""
    from isaaclab.app import AppLauncher

    # Strip our argv so Kit's arg parser doesn't choke on our flags.
    sys.argv = [sys.argv[0]]
    launcher = AppLauncher(headless=headless, enable_cameras=True)
    return launcher.app


def _load_policy(policy_path: Path, dataset_root: Path, seed: int):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    cfg.pretrained_path = Path(policy_path)

    parts = Path(dataset_root).resolve().parts
    repo_id = "/".join(parts[-2:]) if len(parts) >= 2 else Path(dataset_root).name
    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(dataset_root))

    policy = make_policy(cfg, ds_meta=ds_meta)
    policy.to(device)
    policy.eval()
    # Same pre/post pipeline training used. For SmolVLA the preprocessor's
    # tokenizer step reads obs["task"] -> observation.language.tokens (+ mask)
    # and the normalizer scales state/images; without it select_action KeyErrors
    # on the language tokens. postprocessor un-normalizes the action.
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(policy_path),
        dataset_stats=getattr(ds_meta, "stats", None),
    )
    return policy, preprocessor, postprocessor, device


def _build_policy_input(obs_group: dict, device: str, task: str):
    """Isaac obs['policy'] dict -> LeRobot policy input dict."""
    import torch

    def _t(x):
        return x if torch.is_tensor(x) else torch.as_tensor(x)

    jp = _t(obs_group["joint_pos"]).float().to(device)      # (1,6) = joint_pos_rel
    # State layout must MATCH the policy's training data:
    #   - real so101-pickplace-new policy: 12-dim = joint_pos_rel[6] + joint_vel[6]
    #   - sim demo policy (object_pose obs, LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1):
    #     13-dim = joint_pos_rel[6] + object_pose[7] (pos3+quat4), matching
    #     _gen_sim_demos.py's state assembly. Auto-detect via the obs group:
    #     INCLUDE_OBJECT_POSE=1 adds "object_pose" to the policy obs group
    #     (so101_env_cfg.py:375). Picking the wrong layout -> normalize-stats
    #     size mismatch (12 vs 13) at the first step.
    if "object_pose" in obs_group:
        objp = _t(obs_group["object_pose"]).float().to(device)   # (1,7)
        state = torch.cat([jp, objp], dim=-1)                # (1,13)
    else:
        jv = _t(obs_group["joint_vel"]).float().to(device)   # (1,6)
        state = torch.cat([jp, jv], dim=-1)                  # (1,12)

    import torch.nn.functional as F
    rgb = _t(obs_group["d435_rgb"]).to(device).float()       # (1,3,480,640) uint8->float
    if rgb.max() > 1.5:                                       # uint8 [0,255] -> [0,1]
        rgb = rgb / 255.0
    # Policy was trained on the demo dataset's `observation.images.d435_rgb` at 64x64
    # (sim demos). Resize the env's 480x640 frame to match the trained input shape +
    # use the SAME key, else the policy preprocessor KeyErrors / shape-mismatches.
    rgb = F.interpolate(rgb, size=(64, 64), mode="bilinear", align_corners=False)

    # Raw batch (batch dim already present, B=1). The preprocessor tokenizes
    # `task` and normalizes state/image — do NOT pre-normalize beyond uint8->[0,1].
    return {
        "observation.state": state,
        "observation.images.d435_rgb": rgb,
        "task": task,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--task", default="pick_and_place")
    ap.add_argument("--task_prompt", default="pick and place cube")
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args(argv)

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Boot Isaac FIRST.  Keep reference alive — SimulationApp must not be GC'd.
    _app = _boot_app(args.headless)  # noqa: F841

    import torch

    # 2. Build the task env (cameras on → dict obs incl d435_rgb).
    import lerobot_isaac_env  # noqa: F401 (registration side-effect)
    from lerobot_isaac_env import make_env
    from lerobot_isaac_env.tasks import _register_envs

    _register_envs()
    env = make_env(task=args.task, num_envs=1, headless=args.headless, enable_cameras=True)

    # 3. Load policy + pre/post processors.
    policy, preprocessor, postprocessor, device = _load_policy(
        Path(args.policy_path), Path(args.dataset_root), args.seed
    )

    # 4. Rollout.
    successes = 0
    task_successes = 0
    ep_lens: list[int] = []
    _verifier_available = True  # flip False if termination_manager is unavailable

    for ep in range(args.n_episodes):
        obs_dict, _ = env.reset()
        obs_group = obs_dict["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        success = False
        steps = 0
        for steps in range(1, args.max_steps + 1):
            with torch.no_grad():
                pin = _build_policy_input(obs_group, device, args.task_prompt)
                processed = preprocessor(pin)               # tokenize + normalize
                action = postprocessor(policy.select_action(processed))  # (1,6)
            action = action.to(env.device) if hasattr(env, "device") else action
            obs_dict, _reward, terminated, truncated, _info = env.step(action)
            obs_group = obs_dict["policy"]
            term = bool(terminated[0].item()) if torch.is_tensor(terminated) else bool(terminated)
            trunc = bool(truncated[0].item()) if torch.is_tensor(truncated) else bool(truncated)
            if term and not trunc:
                success = True
                break
            if term or trunc:
                break
        successes += int(success)
        ep_lens.append(steps)

        # Source task_success from the termination manager's pre-reset "success" term
        # verdict. Isaac Lab auto-resets terminated sub-envs INSIDE step() before
        # returning, clobbering root_pos_w — the termination manager caches its
        # per-term done flags before the reset, making this the only correct read.
        if _verifier_available:
            ts_verdict = _read_success_term(env)
            if ts_verdict is None:
                # termination_manager unavailable — fall back to env-termination outcome
                # and disable further reads.
                _verifier_available = False
                task_successes += int(success)
            else:
                task_successes += int(ts_verdict)
        else:
            task_successes += int(success)

        print(f"[sim-eval] ep={ep} success={success} steps={steps}", flush=True)

    pc_success = successes / max(1, args.n_episodes)
    task_success = task_successes / max(1, args.n_episodes)

    if _verifier_available:
        _success_criterion = (
            "env \"success\" termination-term verdict (place_termination -> object_in_bin),"
            " captured pre-reset via termination_manager —"
            " geom comes from the term params (pick_and_place.py:461:"
            " target_pos, success_radius=0.06, object_name='source_object')"
        )
    else:
        _success_criterion = "env termination fallback (termination_manager unavailable)"

    payload = {
        "run_id": "sim-eval-" + uuid.uuid4().hex[:8],
        "task": f"{Path(args.dataset_root).name}-sim-{args.task}",
        "pc_success": pc_success,
        "task_success": task_success,
        "n_episodes": args.n_episodes,
        "mean_ep_len": sum(ep_lens) / max(1, len(ep_lens)),
        "_metadata": {
            "source": "closed_loop_isaac_sim",
            "successes": successes,
            "task_successes": task_successes,
            "max_steps": args.max_steps,
            "policy_path": str(Path(args.policy_path).resolve()),
            "success_criterion": _success_criterion,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[sim-eval] result: pc_success={pc_success:.4f} task_success={task_success:.4f}"
        f" ({successes}/{args.n_episodes}) -> {out_path}",
        flush=True,
    )

    # Hard exit to bypass Isaac's hanging atexit SimulationApp.close().
    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass
    import os
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
