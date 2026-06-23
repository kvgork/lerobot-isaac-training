"""Instrument the WORKING demo grasp (_gen_sim_demos.step_to open-loop sequence) and log
the gripper JOINT trajectory + die/ee geometry at close+lift. Reference for diffing
against compute_scripted_action (scripts/_probe_scripted_grasp.py), which does NOT capture.

Logs per step: phase, commanded grip ACTION, actual gripper JOINT pos, ee z, die z,
ee↔die 3D dist. Reports whether the die lifts (>0.09 = the lift threshold).
"""
import os
import sys

sys.argv = [sys.argv[0]]
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app

import numpy as np
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms
from lerobot_isaac_env import make_env

env = make_env(task="pick_and_place", num_envs=1, headless=True, enable_cameras=True)
device = env.device
robot = env.scene["robot"]
obj = env.scene["source_object"]
ee_idx = int(robot.find_bodies("gripper_link")[0][0])
arm_ids = list(robot.find_joints(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])[0])
grip_idx = int(robot.find_joints("gripper")[0][0])
_fixed = bool(getattr(robot, "is_fixed_base", True))
ee_jac = (ee_idx - 1) if _fixed else ee_idx
_OFF = 0 if _fixed else 6
q_default = robot.data.default_joint_pos.clone()
adim = env.action_space.shape[-1]
ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=1, device=device)
GRASP_QUAT = [1.0, 0.0, 0.0, 0.0]
GRIP_OPEN, GRIP_CLOSE = 1.0, -1.0
z_high, grasp_z = 0.17, 0.106
print(f"[demo] body names: {robot.body_names}", flush=True)
print(f"[demo] joint names: {robot.joint_names}  grip_idx={grip_idx}", flush=True)

max_obj_z = [-9.0]
_g = {"t": 0}


def step_to(target_b, grip, n, quat, grip_end=None, label=""):
    ik.reset()
    cmd = torch.tensor([list(target_b) + list(quat)], device=device, dtype=torch.float32)
    for s in range(n):
        g = grip if grip_end is None else grip + (grip_end - grip) * (s / max(1, n - 1))
        rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
        pos_b, quat_b = subtract_frame_transforms(rp, rq, robot.data.body_pos_w[:, ee_idx, :], robot.data.body_quat_w[:, ee_idx, :])
        ik.set_command(cmd, ee_pos=pos_b, ee_quat=quat_b)
        jac = robot.root_physx_view.get_jacobians()[:, ee_jac, :6, [_OFF + j for j in arm_ids]]
        q_des = ik.compute(pos_b, quat_b, jac, robot.data.joint_pos[:, arm_ids])
        action = torch.zeros((1, adim), device=device)
        for k, jid in enumerate(arm_ids):
            action[0, jid] = (q_des[0, k] - q_default[0, jid]) / 0.5
        action[0, grip_idx] = g
        out = env.step(action)
        _g["t"] += 1
        oz = float(obj.data.root_pos_w[0, 2])
        max_obj_z[0] = max(max_obj_z[0], oz)
        if s % 20 == 0 or s == n - 1:
            gj = float(robot.data.joint_pos[0, grip_idx])
            ez = float(robot.data.body_pos_w[0, ee_idx, 2])
            d = float(((robot.data.body_pos_w[0, ee_idx, :] - obj.data.root_pos_w[0]) ** 2).sum() ** 0.5)
            print(f"[demo] t={_g['t']:3d} {label:9s} gripA={g:+.2f} gripJOINT={gj:+.3f} ee_z={ez:.3f} die_z={oz:.3f} ee-die={d:.3f} term={bool(out[2])}", flush=True)


env.reset()
op0 = obj.data.root_pos_w[0].detach().cpu().numpy()
gx, gy = float(op0[0]), float(op0[1])
print(f"[demo] die start=({gx:.3f},{gy:.3f},{op0[2]:.3f})", flush=True)
q = GRASP_QUAT
step_to([gx, gy, z_high], GRIP_OPEN, 50, q, label="approach")
step_to([gx, gy, grasp_z], GRIP_OPEN, 90, q, label="descend")
step_to([gx, gy, grasp_z], GRIP_OPEN, 30, q, label="stabilize")
step_to([gx, gy, grasp_z], GRIP_OPEN, 80, q, grip_end=GRIP_CLOSE, label="close")
step_to([gx, gy, grasp_z], GRIP_CLOSE, 25, q, label="hold")
step_to([gx, gy, z_high], GRIP_CLOSE, 60, q, label="lift")
print(f"[demo] RESULT max_die_z={max_obj_z[0]:.3f} (lift thresh 0.09); final_die_z={float(obj.data.root_pos_w[0,2]):.3f}", flush=True)
sys.stdout.flush()
os._exit(0)
