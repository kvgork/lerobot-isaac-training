"""Map the SO-101 gripper reach envelope (commandable workspace).

The staged-reward run plateaued because the object at (0.5, 0.1) = 0.51 m is
beyond the arm. This probe applies a grid of joint-target actions (the SAME
JointPositionAction interface the policy uses), lets each settle, and records
the gripper_link world (x,y,z). Reports the reachable (x,y) set + max planar
radius so the object/target can be placed INSIDE it.

Usage:
  .pixi/envs/sim/bin/python scripts/_reach_probe.py --out outputs/reach-probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path


def _boot_app(headless: bool):
    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]]
    return AppLauncher(headless=headless, enable_cameras=False).app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_and_place")
    ap.add_argument("--settle", type=int, default=15)
    ap.add_argument("--out", default="outputs/reach-probe.json")
    args = ap.parse_args()

    app = _boot_app(True)

    import torch
    from lerobot_isaac_env import make_env

    env = make_env(task=args.task, num_envs=1, headless=True, enable_cameras=False)
    device = env.device if hasattr(env, "device") else "cuda"
    robot = env.scene["robot"]
    obj = env.scene["source_object"]

    try:
        ee_list, _ = robot.find_bodies("gripper_link")
        ee_idx = int(ee_list[0]) if len(ee_list) else 0
    except Exception:
        ee_idx = 0

    base_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
    obj_pos = obj.data.root_pos_w[0].detach().cpu().numpy()

    env.reset()
    action_dim = env.action_space.shape[-1] if env.action_space.shape else 6

    levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    # Sweep joints 0,1,2 (pan, shoulder, elbow) — the macro reach drivers.
    # Hold joints 3..5 at 0. This samples the forward-reach envelope.
    samples = []
    for a0, a1, a2 in product(levels, repeat=3):
        action = torch.zeros((1, action_dim), device=device)
        action[0, 0] = a0
        if action_dim > 1:
            action[0, 1] = a1
        if action_dim > 2:
            action[0, 2] = a2
        for _ in range(args.settle):
            env.step(action)
        ee = robot.data.body_pos_w[0, ee_idx, :].detach().cpu().numpy()
        planar_r = float((ee[0] ** 2 + ee[1] ** 2) ** 0.5)
        samples.append(
            {"a": [a0, a1, a2], "ee": [float(ee[0]), float(ee[1]), float(ee[2])], "r": planar_r}
        )

    rs = [s["r"] for s in samples]
    xs = [s["ee"][0] for s in samples]
    ys = [s["ee"][1] for s in samples]
    zs = [s["ee"][2] for s in samples]
    max_r = max(rs)
    best = max(samples, key=lambda s: s["r"])

    # Suggest an object at ~65% of max planar radius, in the +x direction the
    # arm most extends toward (use the best-reach sample's xy direction).
    bx, by = best["ee"][0], best["ee"][1]
    bnorm = (bx**2 + by**2) ** 0.5 or 1.0
    sugg_r = 0.65 * max_r
    sugg_obj = [round(bx / bnorm * sugg_r, 3), round(by / bnorm * sugg_r, 3), 0.05]

    result = {
        "base_pos": [float(x) for x in base_pos],
        "current_object_pos": [float(x) for x in obj_pos],
        "current_object_planar_r": float((obj_pos[0] ** 2 + obj_pos[1] ** 2) ** 0.5),
        "max_planar_reach": max_r,
        "ee_x_range": [min(xs), max(xs)],
        "ee_y_range": [min(ys), max(ys)],
        "ee_z_range": [min(zs), max(zs)],
        "best_reach_sample": best,
        "suggested_object_xy": sugg_obj,
        "n_samples": len(samples),
    }
    print(json.dumps(result, indent=2), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**result, "samples": samples}, indent=2))
    print(f"[reach] wrote {out}", flush=True)

    try:
        env.close()
    except Exception:
        pass
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
