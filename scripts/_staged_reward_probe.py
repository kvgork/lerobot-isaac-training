"""Step-0 instrument probe for the staged pick-place reward (plan 2026-06-08).

Boots the Isaac SO-101 pick_and_place env at num_envs=1 and reads the staged
reward TERMS DIRECTLY (grasp / lift / place / progress) each step — bypassing
sheeprl, which does not forward Isaac per-term rewards to TB. Validates:

  1. The 6 reward terms register in the RewardManager at runtime.
  2. Under random actions: grasp ~0 (EE far from object), lift == 0 (object at
     rest), place == 0 (gate closed because not lifted). progress is the dense
     reach gradient (negative, scales with EE->object distance).
  3. Manual-lift sanity: teleport the object up -> lift SPIKES; teleport it over
     the target xy while lifted -> place FIRES; drop it back -> place gate closes.

No cameras (reward funcs read body/object poses only) -> fast boot. Headless.

Usage:
  LEROBOT_ISAAC_STAGED_REWARD=1 .pixi/envs/sim/bin/python \
      scripts/_staged_reward_probe.py --steps 200 --out outputs/staged-probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _boot_app(headless: bool):
    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]]
    launcher = AppLauncher(headless=headless, enable_cameras=False)
    return launcher.app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--task", default="pick_and_place")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--out", default="outputs/staged-probe.json")
    ap.add_argument("--print_every", type=int, default=20)
    args = ap.parse_args()

    app = _boot_app(args.headless)

    import torch
    from lerobot_isaac_env import make_env
    from lerobot_isaac_env import rewards as R

    env = make_env(task=args.task, num_envs=1, headless=args.headless, enable_cameras=False)
    device = env.device if hasattr(env, "device") else "cuda"

    # ---- term registration check -----------------------------------------
    rm = env.reward_manager
    active_terms = list(getattr(rm, "active_terms", []))
    print(f"[probe] RewardManager active terms ({len(active_terms)}): {active_terms}", flush=True)

    def read_terms():
        """Call the staged reward funcs directly -> raw (unweighted) per-term values."""
        out = {}
        try:
            out["grasp"] = float(R.grasp_reward(env).mean().item())
        except Exception as e:  # noqa: BLE001
            out["grasp"] = f"err:{e}"
        try:
            out["lift"] = float(R.lift_reward(env).mean().item())
        except Exception as e:  # noqa: BLE001
            out["lift"] = f"err:{e}"
        try:
            out["place"] = float(R.place_reward(env).mean().item())
        except Exception as e:  # noqa: BLE001
            out["place"] = f"err:{e}"
        try:
            # progress_reward default weight handled by manager; raw distance proxy here.
            out["progress"] = float(R.progress_reward(env).mean().item())
        except Exception as e:  # noqa: BLE001
            out["progress"] = f"err:{e}"
        return out

    def ee_obj_dist():
        try:
            from isaaclab.managers import SceneEntityCfg
            _, _, d = R._ee_object_distance(
                env, SceneEntityCfg("robot"), SceneEntityCfg("source_object"), "gripper_link"
            )
            return float(d.mean().item())
        except Exception as e:  # noqa: BLE001
            return f"err:{e}"

    obs, _ = env.reset()
    action_dim = env.action_space.shape[-1] if env.action_space.shape else 6

    # ---- Phase A: random actions, log per-term decomposition --------------
    history = []
    agg = {k: [] for k in ("grasp", "lift", "place", "progress", "dist")}
    for step in range(args.steps):
        action = (torch.rand((1, action_dim), device=device) * 2.0 - 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        terms = read_terms()
        dist = ee_obj_dist()
        for k in ("grasp", "lift", "place", "progress"):
            if isinstance(terms[k], float):
                agg[k].append(terms[k])
        if isinstance(dist, float):
            agg["dist"].append(dist)
        if step % args.print_every == 0:
            print(
                f"[probe] A step={step:4d} dist={dist} "
                f"grasp={terms['grasp']} lift={terms['lift']} "
                f"place={terms['place']} progress={terms['progress']} "
                f"env_rew={float(reward.mean().item()):.4f}",
                flush=True,
            )
        history.append({"phase": "A", "step": step, "dist": dist, **terms})
        if bool(terminated.any().item()) or bool(truncated.any().item()):
            obs, _ = env.reset()

    def summ(vals):
        if not vals:
            return None
        return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals), "n": len(vals)}

    phase_a_summary = {k: summ(v) for k, v in agg.items()}
    print(f"[probe] Phase A summary: {json.dumps(phase_a_summary, indent=2)}", flush=True)

    # ---- Phase B: manual-lift sanity (teleport object) --------------------
    manual = {}
    try:
        obj = env.scene["source_object"]
        sim_dt = env.physics_dt if hasattr(env, "physics_dt") else 1.0 / 120.0

        def set_obj_xyz(x, y, z):
            pose = obj.data.root_pose_w.clone()  # (N,7) pos+quat
            pose[:, 0] = x
            pose[:, 1] = y
            pose[:, 2] = z
            obj.write_root_pose_to_sim(pose)
            obj.write_data_to_sim()
            env.sim.step(render=False)
            obj.update(sim_dt)

        # B1: object lifted high, over the target xy (0.5, -0.2).
        set_obj_xyz(0.5, -0.2, 0.18)
        manual["lifted_over_target"] = read_terms()
        print(f"[probe] B1 lifted_over_target: {manual['lifted_over_target']}", flush=True)

        # B2: object lifted high, but far from target xy.
        set_obj_xyz(0.5, 0.3, 0.18)
        manual["lifted_off_target"] = read_terms()
        print(f"[probe] B2 lifted_off_target: {manual['lifted_off_target']}", flush=True)

        # B3: object back at rest height over target (place gate must close).
        set_obj_xyz(0.5, -0.2, 0.05)
        manual["rest_over_target"] = read_terms()
        print(f"[probe] B3 rest_over_target: {manual['rest_over_target']}", flush=True)
    except Exception as e:  # noqa: BLE001
        manual["error"] = str(e)
        print(f"[probe] Phase B manual-lift FAILED: {e}", flush=True)

    result = {
        "active_terms": active_terms,
        "phase_a_summary": phase_a_summary,
        "manual_lift": manual,
        "steps": args.steps,
        "history_tail": history[-10:],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[probe] wrote {out}", flush=True)

    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
