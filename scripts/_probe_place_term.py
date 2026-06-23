"""Diagnostic: does the LATCH-fixed place_termination fire on a full scripted carry?
Runs ONE scripted pick->lift->carry->place with time_out DISABLED, GRASP_STAGE unset (so
place_termination is the success term), and prints per step: die_z, in_bin(XY), currently_lifted,
the latch (env._place_ever_lifted), episode_length_buf, and place_termination() output.

Run:
  LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 LEROBOT_ISAAC_OBJECT_FRICTION=3.0 \
  LEROBOT_ISAAC_STAGED_REWARD=1 \
  pixi run -e sim python scripts/_probe_place_term.py
"""
import os
import sys

sys.argv = [sys.argv[0]]
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app

import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms
from lerobot_isaac_env import make_env
from lerobot_isaac_env.terminations import place_termination

env = make_env(task="pick_and_place", num_envs=1, headless=True, enable_cameras=True)
try:
    env.cfg.episode_length_s = 1.0e6  # disable time_out so the full 485-step sequence runs
except Exception as e:
    print(f"[pt] WARN episode_length_s: {e}", flush=True)
print(f"[pt] max_episode_length -> {getattr(env, 'max_episode_length', None)}", flush=True)

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
ik = DifferentialIKController(
    DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=1, device=device
)
GRASP_QUAT = [1.0, 0.0, 0.0, 0.0]
GRIP_OPEN, GRIP_CLOSE = 1.0, -1.0
z_high, grasp_z = 0.17, 0.106
TGT_X, TGT_Y = 0.22, -0.13
fired = {"at": None}


def step_to(target_b, grip, n, quat, phase, grip_end=None):
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
        # die pose BEFORE step (the state the env's termination manager evaluates this step)
        pre = obj.data.root_pos_w[0].detach().cpu().numpy()
        out = env.step(action)
        term = bool(out[2].reshape(-1)[0].item())
        trunc = bool(out[3].reshape(-1)[0].item()) if len(out) > 3 else False
        diez = float(obj.data.root_pos_w[0, 2])  # POST-step (post-reset if a reset fired)
        elb = getattr(env, "episode_length_buf", None)
        elb_v = int(elb.reshape(-1)[0].item()) if elb is not None else None
        if (term or trunc) and fired["at"] is None:
            fired["at"] = f"{phase} s={s}"
            print(f"[pt] *** DONE term={term} trunc={trunc} at {phase} s={s}: "
                  f"die_PRE=({pre[0]:.3f},{pre[1]:.3f},{pre[2]:.3f}) "
                  f"dist_to_bin_XY={((pre[0]-TGT_X)**2+(pre[1]-TGT_Y)**2)**0.5:.4f}", flush=True)
        if s % 15 == 0 or s == n - 1 or term or trunc:
            print(f"[pt] {phase:8s} s={s:3d} elb={elb_v} die_PRE=({pre[0]:.3f},{pre[1]:.3f},{pre[2]:.3f}) "
                  f"diePOSTz={diez:.4f} term={int(term)} trunc={int(trunc)}", flush=True)


env.reset()
ss = torch.zeros((1, adim), device=device)
ss[0, grip_idx] = GRIP_OPEN
for _ in range(30):
    env.step(ss)
op0 = obj.data.root_pos_w[0].detach().cpu().numpy()
gx, gy = float(op0[0]), float(op0[1])
print(f"[pt] die start=({gx:.3f},{gy:.3f}); PLACE_REQUIRE_LIFT={os.environ.get('LEROBOT_ISAAC_PLACE_REQUIRE_LIFT','(default 1)')}", flush=True)
q = GRASP_QUAT
step_to([gx, gy, z_high], GRIP_OPEN, 50, q, "approach")
step_to([gx, gy, grasp_z], GRIP_OPEN, 90, q, "descend")
step_to([gx, gy, grasp_z], GRIP_OPEN, 30, q, "stabilize")
step_to([gx, gy, grasp_z], GRIP_OPEN, 80, q, "close", grip_end=GRIP_CLOSE)
step_to([gx, gy, grasp_z], GRIP_CLOSE, 25, q, "seat")
step_to([gx, gy, z_high], GRIP_CLOSE, 60, q, "lift")
step_to([TGT_X, TGT_Y, z_high], GRIP_CLOSE, 60, q, "carry")
step_to([TGT_X, TGT_Y, 0.06], GRIP_CLOSE, 40, q, "descend2")
step_to([TGT_X, TGT_Y, 0.06], GRIP_OPEN, 20, q, "release")
print(f"[pt] RESULT place_termination first fired at phase: {fired['at']}", flush=True)
sys.stdout.flush()
os._exit(0)
