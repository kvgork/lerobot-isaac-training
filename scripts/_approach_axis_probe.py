"""Find the gripper_link local 'approach/finger' axis and the joint config that
points it straight DOWN (world -Z) near the die xy.

The old _graspose_probe scored jaw-near-object only -> picked a SIDE reach (eyeball
confirmed the gripper grabs from the side). Here we score APPROACH-AXIS alignment
with world -Z so we recover a true top-down grasp quaternion.

Method:
  1. Read gripper_link world quat + moving_jaw world pos at several pitch configs.
     The finger/approach direction in world = unit(jaw_pos - ee_pos).
  2. For the rest pose, rotate each local basis axis (+/-x,y,z) into world via the
     quat; whichever matches the measured finger direction IS the local approach axis.
  3. Sweep pitch joints; report the config whose finger direction is closest to
     (0,0,-1) AND whose ee xy is near the die, plus its gripper_link world quat
     (-> drop into GRASP_QUAT).
"""
import sys, numpy as np
from itertools import product
from isaaclab.app import AppLauncher
sys.argv = [sys.argv[0]]
app = AppLauncher(headless=True, enable_cameras=False).app
import torch
from lerobot_isaac_env import make_env
from isaaclab.utils.math import quat_apply

env = make_env(task="pick_and_place", num_envs=1, headless=True, enable_cameras=False)
r = env.scene["robot"]; obj = env.scene["source_object"]
ee = r.find_bodies("gripper_link")[0][0]
jaw = r.find_bodies("moving_jaw_so101_v1_link")[0][0]
nm = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
idx = {n: int(r.find_joints(n)[0][0]) for n in nm}
ad = env.action_space.shape[-1]
env.reset()
op = obj.data.root_pos_w[0].cpu().numpy()
print(f"[axis] object rest={np.array2string(op, precision=3)}", flush=True)


def drive(sp, sh, el, wf, wr=0.0, n=25):
    a = torch.zeros((1, ad), device=env.device)
    a[0, idx["shoulder_pan"]] = sp; a[0, idx["shoulder_lift"]] = sh
    a[0, idx["elbow_flex"]] = el; a[0, idx["wrist_flex"]] = wf; a[0, idx["wrist_roll"]] = wr
    for _ in range(n):
        env.step(a)
    q = r.data.body_quat_w[0, ee, :]
    eep = r.data.body_pos_w[0, ee, :]
    jp = r.data.body_pos_w[0, jaw, :]
    return q, eep, jp


# --- step 1+2: identify the local approach axis at rest ---
q, eep, jp = drive(0, 0, 0, 0)
finger_w = (jp - eep)
finger_w = finger_w / torch.linalg.norm(finger_w)
print(f"[axis] rest finger dir (world, jaw-ee)={np.array2string(finger_w.cpu().numpy(), precision=3)}", flush=True)
axes = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1]}
best_axis, best_dot = None, -2
for name, v in axes.items():
    vw = quat_apply(q.unsqueeze(0), torch.tensor([v], device=env.device, dtype=torch.float32))[0]
    vw = vw / torch.linalg.norm(vw)
    d = float(torch.dot(vw, finger_w))
    print(f"[axis]   local {name} -> world {np.array2string(vw.cpu().numpy(), precision=2)} dot(finger)={d:+.3f}", flush=True)
    if d > best_dot:
        best_dot, best_axis = d, name
print(f"[axis] APPROACH AXIS (local) = {best_axis} (dot={best_dot:.3f})", flush=True)
approach_local = torch.tensor([axes[best_axis]], device=env.device, dtype=torch.float32)

# --- step 3: sweep pitch joints, score approach-down alignment + xy-near-die ---
down = torch.tensor([0.0, 0.0, -1.0], device=env.device)
best = None
for sp, sh, el, wf in product([-0.3, 0, 0.3], [-1, -0.5, 0, 0.5, 1], [-1, -0.5, 0, 0.5, 1], [-1, -0.5, 0, 0.5, 1]):
    q, eep, jp = drive(sp, sh, el, wf, n=20)
    aw = quat_apply(q.unsqueeze(0), approach_local)[0]
    aw = aw / torch.linalg.norm(aw)
    down_dot = float(torch.dot(aw, down))            # +1 = straight down
    eepn = eep.cpu().numpy()
    d_xy = float(((eepn[0] - op[0]) ** 2 + (eepn[1] - op[1]) ** 2) ** 0.5)
    # score: maximise downness, penalise xy distance from die
    score = down_dot - 2.0 * d_xy
    if best is None or score > best["score"]:
        best = {"score": score, "a": [sp, sh, el, wf], "down_dot": down_dot,
                "d_xy": d_xy, "ee": eepn.tolist(), "quat": q.cpu().numpy().tolist()}
print(f"[axis] BEST top-down: a(sp,sh,el,wf)={best['a']} down_dot={best['down_dot']:+.3f} "
      f"d_xy={best['d_xy']:.3f} ee_z={best['ee'][2]:.3f}", flush=True)
print(f"[axis] GRASP_QUAT (top-down, w,x,y,z)={np.array2string(np.array(best['quat']), precision=4)}", flush=True)
app.close()
