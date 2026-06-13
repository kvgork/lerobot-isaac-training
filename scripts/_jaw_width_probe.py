"""Measure the SO-101 gripper: does the jaw actuate, how wide is the real gap, and
along which gripper_link-local axis does the jaw span (= which way a die escapes).

The single moving jaw (moving_jaw_so101_v1_link) closes against the fixed
gripper_frame_link. moving_jaw's link origin is at the HINGE (doesn't translate), so
we measure (a) the gripper JOINT position open vs closed (real actuation), and
(b) the moving_jaw -> gripper_frame_link vector expressed in gripper_link LOCAL frame
(orientation-invariant => at grasp with GRASP_QUAT=identity this IS the world gap axis).
Fast (os._exit).
"""
import os, sys, numpy as np
from isaaclab.app import AppLauncher
sys.argv = [sys.argv[0]]
app = AppLauncher(headless=True, enable_cameras=False).app
import torch
from lerobot_isaac_env import make_env
from isaaclab.utils.math import subtract_frame_transforms, quat_apply, quat_inv

env = make_env(task="pick_and_place", num_envs=1, headless=True, enable_cameras=False)
r = env.scene["robot"]
ee = int(r.find_bodies("gripper_link")[0][0])
mj = int(r.find_bodies("moving_jaw_so101_v1_link")[0][0])
gf = int(r.find_bodies("gripper_frame_link")[0][0])
gid = int(r.find_joints("gripper")[0][0])
ad = env.action_space.shape[-1]


def snap(grip, label):
    env.reset()
    a = torch.zeros((1, ad), device=env.device)
    a[0, gid] = grip
    for _ in range(40):
        env.step(a)
    jpos = float(r.data.joint_pos[0, gid].item())
    mj_w = r.data.body_pos_w[0, mj, :]
    gf_w = r.data.body_pos_w[0, gf, :]
    ee_w = r.data.body_pos_w[0, ee, :]
    ee_q = r.data.body_quat_w[0, ee, :]
    gap_w = mj_w - gf_w
    # express gap in gripper_link LOCAL frame (rotate world vec by inverse ee quat)
    gap_local = quat_apply(quat_inv(ee_q.unsqueeze(0)), gap_w.unsqueeze(0))[0]
    print(f"[jaw] --- {label} grip={grip} | gripper joint_pos={jpos:+.4f} rad ---", flush=True)
    print(f"[jaw]   |moving_jaw - gripper_frame| = {float(torch.linalg.norm(gap_w))*1000:.1f} mm", flush=True)
    print(f"[jaw]   gap vec (gripper_link LOCAL) = {np.array2string(gap_local.cpu().numpy(), precision=4)} "
          f"(dominant axis = jaw span direction)", flush=True)
    return jpos


jo = snap(1.0, "OPEN")
jc = snap(-1.0, "CLOSED")
print(f"[jaw] joint travel open->closed = {abs(jo - jc):.4f} rad "
      f"({'ACTUATES' if abs(jo - jc) > 0.05 else 'NOT MOVING'})", flush=True)
print(f"[jaw] die: OBJECT_SCALE=0.267 -> DexCube edge ~0.06*0.267 = 16 mm", flush=True)
sys.stdout.flush()
os._exit(0)
