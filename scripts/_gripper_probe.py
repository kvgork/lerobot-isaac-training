"""Determine the SO-101 gripper open/closed convention + validate grasp_closure_reward.

The new grasp_closure_reward needs to know which joint-limit extreme is "closed".
This probe commands the gripper action to each extreme, settles, and measures the
moving-jaw body distance to gripper_link (smaller = jaws together = closed). Then
it teleports the object to the EE and checks the closure reward is high when the
jaw is closed (and object near) and low when open.

Usage:
  LEROBOT_ISAAC_STAGED_REWARD=1 .pixi/envs/sim/bin/python scripts/_gripper_probe.py \
      --out outputs/gripper-probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _boot_app(headless: bool):
    from isaaclab.app import AppLauncher

    sys.argv = [sys.argv[0]]
    return AppLauncher(headless=headless, enable_cameras=False).app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_and_place")
    ap.add_argument("--settle", type=int, default=30)
    ap.add_argument("--out", default="outputs/gripper-probe.json")
    args = ap.parse_args()

    app = _boot_app(True)

    import torch
    from lerobot_isaac_env import make_env
    from lerobot_isaac_env import rewards as R

    env = make_env(task=args.task, num_envs=1, headless=True, enable_cameras=False)
    device = env.device if hasattr(env, "device") else "cuda"
    robot = env.scene["robot"]
    obj = env.scene["source_object"]

    def body_idx(name):
        try:
            ids, _ = robot.find_bodies(name)
            return int(ids[0]) if len(ids) else 0
        except Exception:
            return 0

    ee_i = body_idx("gripper_link")
    jaw_i = body_idx("moving_jaw_so101_v1_link")
    try:
        jids, _ = robot.find_joints("gripper")
        jaw_joint = int(jids[0]) if len(jids) else -1
    except Exception:
        jaw_joint = -1

    env.reset()
    action_dim = env.action_space.shape[-1] if env.action_space.shape else 6

    def drive_gripper(val):
        action = torch.zeros((1, action_dim), device=device)
        action[0, -1] = val  # gripper is the last action dim
        for _ in range(args.settle):
            env.step(action)
        jaw_pos = float(robot.data.joint_pos[0, jaw_joint].item())
        ee = robot.data.body_pos_w[0, ee_i, :]
        jaw = robot.data.body_pos_w[0, jaw_i, :]
        jaw_gap = float(torch.norm(ee - jaw).item())
        return {"action": val, "jaw_joint_pos": jaw_pos, "jaw_to_eelink_dist": jaw_gap}

    open_cmd = drive_gripper(-1.0)
    close_cmd = drive_gripper(1.0)

    limits = robot.data.joint_pos_limits[0, jaw_joint, :].detach().cpu().tolist()

    # "closed" = the command whose moving-jaw sits CLOSER to gripper_link.
    closed_is_pos1 = close_cmd["jaw_to_eelink_dist"] < open_cmd["jaw_to_eelink_dist"]
    # action +1 drives the joint toward which limit?
    closed_joint_pos = close_cmd["jaw_joint_pos"] if closed_is_pos1 else open_cmd["jaw_joint_pos"]
    # closed_high = closed joint pos is nearer the UPPER limit
    closed_high = abs(closed_joint_pos - limits[1]) < abs(closed_joint_pos - limits[0])

    # Validate closure reward: object at the EE, jaw open vs closed.
    def set_obj_to_ee():
        ee = robot.data.body_pos_w[0, ee_i, :].clone()
        pose = obj.data.root_pose_w.clone()
        pose[0, :3] = ee
        obj.write_root_pose_to_sim(pose)
        obj.write_data_to_sim()
        env.sim.step(render=False)
        obj.update(1.0 / 120.0)

    def closure_val(closed_high_flag):
        return float(
            R.grasp_closure_reward(env, closed_high=closed_high_flag).mean().item()
        )

    val = {}
    try:
        # closed jaw + object near
        drive_gripper(1.0 if closed_is_pos1 else -1.0)
        set_obj_to_ee()
        val["closed_jaw_object_near"] = closure_val(closed_high)
        # open jaw + object near
        drive_gripper(-1.0 if closed_is_pos1 else 1.0)
        set_obj_to_ee()
        val["open_jaw_object_near"] = closure_val(closed_high)
    except Exception as e:  # noqa: BLE001
        val["error"] = str(e)

    result = {
        "gripper_joint_index": jaw_joint,
        "gripper_joint_limits": limits,
        "open_cmd_-1": open_cmd,
        "close_cmd_+1": close_cmd,
        "closed_at_action_+1": closed_is_pos1,
        "closed_high": closed_high,
        "closure_reward_check": val,
    }
    print(json.dumps(result, indent=2), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[gripper] wrote {out}", flush=True)

    try:
        env.close()
    except Exception:
        pass
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
