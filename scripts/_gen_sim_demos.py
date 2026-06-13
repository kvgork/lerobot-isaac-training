"""Generate SIM pick-place demos with the working scripted controller -> LeRobotDataset.

Stage 2 of plans/2026-06-11-demo-warmstart-plan.md. Runs the (now working) scripted
straight-down grasp N times with small object-pose jitter, records per-step
(observation.state, observation.images.d435_rgb @ 64x64, action), and writes ONLY the
SUCCESS episodes to a LeRobotDataset for the DreamerV3 warm-start.

Per-episode jitter is done by teleporting the die with write_root_pose_to_sim AFTER
reset (the spawn xy is fixed at env build; OBJECT_X/Y env vars only set the default).
The controller reads the LIVE die pose so it auto-aims at the jittered position.

  LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_STAGED_REWARD=1 \
    .pixi/envs/sim/bin/python scripts/_gen_sim_demos.py \
      --episodes 40 --out datasets/local/so101-sim-pickplace-demos
"""
from __future__ import annotations
import argparse, sys, math, random
from pathlib import Path


def _boot(headless: bool):
    from isaaclab.app import AppLauncher
    sys.argv = [sys.argv[0]]
    return AppLauncher(headless=headless, enable_cameras=True).app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/local/so101-sim-pickplace-demos")
    ap.add_argument("--episodes", type=int, default=40, help="number of SUCCESSFUL demos to collect")
    ap.add_argument("--max_attempts", type=int, default=80, help="cap on total rollouts (success or not)")
    ap.add_argument("--img", type=int, default=64, help="square d435 frame size")
    ap.add_argument("--grasp_z", type=float, default=0.106)
    ap.add_argument("--obj_x", type=float, default=0.18)
    ap.add_argument("--obj_y", type=float, default=0.05)
    ap.add_argument("--jitter", type=float, default=0.03, help="+/- uniform xy jitter (m) around obj_x/y")
    ap.add_argument("--tgt_x", type=float, default=0.22)
    ap.add_argument("--tgt_y", type=float, default=-0.13)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    app = _boot(not args.gui)
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F
    from lerobot_isaac_env import make_env
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import subtract_frame_transforms
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    env = make_env(task="pick_and_place", num_envs=1, headless=not args.gui, enable_cameras=True)
    device = env.device
    robot = env.scene["robot"]
    obj = env.scene["source_object"]
    cam = env.scene["d435_camera"]
    ee_idx = int(robot.find_bodies("gripper_link")[0][0])
    arm_ids = list(robot.find_joints(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])[0])
    grip_idx = int(robot.find_joints("gripper")[0][0])
    _fixed = bool(getattr(robot, "is_fixed_base", True))
    ee_jac = (ee_idx - 1) if _fixed else ee_idx
    _OFF = 0 if _fixed else 6
    q_default = robot.data.default_joint_pos.clone()
    action_dim = env.action_space.shape[-1]
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=1, device=device)
    GRASP_QUAT = [1.0, 0.0, 0.0, 0.0]
    GRIP_OPEN, GRIP_CLOSE = 1.0, -1.0
    z_high = 0.17

    # LeRobotDataset features — match the env d435 obs (3,H,W) + 12-dim state + 6 action.
    feats = {
        "observation.state": {"dtype": "float32", "shape": (12,), "names": None},
        "observation.images.d435_rgb": {"dtype": "image", "shape": (3, args.img, args.img),
                                        "names": ["channels", "height", "width"]},
        "action": {"dtype": "float32", "shape": (6,), "names": None},
    }
    out_dir = Path(args.out)
    if out_dir.exists():
        print(f"[demos] ERROR: {out_dir} exists — remove it first (LeRobotDataset.create won't overwrite).", flush=True)
        os._exit(1)
    ds = LeRobotDataset.create(repo_id=f"local/{out_dir.name}", root=str(out_dir), fps=30, features=feats)

    frames: list[dict] = []

    def grab_frame(action_vec):
        st = torch.cat([robot.data.joint_pos[0], robot.data.joint_vel[0]]).detach().cpu().numpy().astype("float32")
        rgb = cam.data.output["rgb"][0]                      # (480,640,3) uint8
        rgb = rgb[..., :3].permute(2, 0, 1).float().unsqueeze(0)  # (1,3,480,640)
        rgb = F.interpolate(rgb, size=(args.img, args.img), mode="bilinear", align_corners=False)
        rgb = rgb[0].clamp(0, 255).to(torch.uint8).cpu().numpy()  # (3,img,img)
        frames.append({"observation.state": st,
                       "observation.images.d435_rgb": rgb,
                       "action": np.asarray(action_vec, dtype="float32"),
                       "task": "pick and place the die in the bin"})

    def step_to(target_b, grip, n, quat, grip_end=None, record=True):
        ik.reset()
        cmd = torch.tensor([list(target_b) + list(quat)], device=device, dtype=torch.float32)
        for s in range(n):
            g = grip if grip_end is None else grip + (grip_end - grip) * (s / max(1, n - 1))
            rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
            pos_b, quat_b = subtract_frame_transforms(rp, rq, robot.data.body_pos_w[:, ee_idx, :], robot.data.body_quat_w[:, ee_idx, :])
            ik.set_command(cmd, ee_pos=pos_b, ee_quat=quat_b)
            jac = robot.root_physx_view.get_jacobians()[:, ee_jac, :6, [_OFF + j for j in arm_ids]]
            q_des = ik.compute(pos_b, quat_b, jac, robot.data.joint_pos[:, arm_ids])
            action = torch.zeros((1, action_dim), device=device)
            for k, jid in enumerate(arm_ids):
                action[0, jid] = (q_des[0, k] - q_default[0, jid]) / 0.5
            action[0, grip_idx] = g
            if record:
                grab_frame(action[0].detach().cpu().numpy())
            env.step(action)

    def rollout(ox, oy):
        """Reset, jitter die to (ox,oy), settle, full pick->place. Returns SUCCESS."""
        frames.clear()
        env.reset()
        # teleport die to jittered xy (keep spawn z + identity rot)
        root = obj.data.root_state_w.clone()
        root[0, 0], root[0, 1] = ox, oy
        obj.write_root_state_to_sim(root)
        s = torch.zeros((1, action_dim), device=device); s[0, grip_idx] = GRIP_OPEN
        for _ in range(30):
            env.step(s)                                       # settle (not recorded)
        op0 = obj.data.root_pos_w[0].detach().cpu().numpy()
        gx, gy = float(op0[0]), float(op0[1])
        q = GRASP_QUAT
        step_to([gx, gy, z_high], GRIP_OPEN, 50, q)
        step_to([gx, gy, args.grasp_z], GRIP_OPEN, 90, q)
        step_to([gx, gy, args.grasp_z], GRIP_OPEN, 30, q)
        step_to([gx, gy, args.grasp_z], GRIP_OPEN, 80, q, grip_end=GRIP_CLOSE)
        step_to([gx, gy, args.grasp_z], GRIP_CLOSE, 25, q)
        step_to([gx, gy, z_high], GRIP_CLOSE, 60, q)
        step_to([args.tgt_x, args.tgt_y, z_high], GRIP_CLOSE, 60, q)
        step_to([args.tgt_x, args.tgt_y, 0.06], GRIP_CLOSE, 40, q)
        step_to([args.tgt_x, args.tgt_y, 0.06], GRIP_OPEN, 20, q)
        op = obj.data.root_pos_w[0].detach().cpu().numpy()
        return bool(((op[0] - args.tgt_x) ** 2 + (op[1] - args.tgt_y) ** 2) ** 0.5 < 0.06)

    saved, attempts = 0, 0
    while saved < args.episodes and attempts < args.max_attempts:
        attempts += 1
        ox = args.obj_x + random.uniform(-args.jitter, args.jitter)
        oy = args.obj_y + random.uniform(-args.jitter, args.jitter)
        ok = rollout(ox, oy)
        if ok:
            for fr in frames:
                ds.add_frame(fr)
            ds.save_episode()
            saved += 1
            print(f"[demos] SAVED {saved}/{args.episodes} (attempt {attempts}, obj=({ox:.3f},{oy:.3f}), {len(frames)} frames)", flush=True)
        else:
            print(f"[demos] skip fail (attempt {attempts}, obj=({ox:.3f},{oy:.3f}))", flush=True)

    if saved > 0:
        ds.finalize()  # flush episode metadata (episodes.parquet) + final stats
        print(f"[demos] finalized dataset metadata", flush=True)
    print(f"[demos] DONE: {saved} demos in {out_dir} ({attempts} attempts)", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
