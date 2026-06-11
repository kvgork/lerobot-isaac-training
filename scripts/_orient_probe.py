import sys, numpy as np
from isaaclab.app import AppLauncher
sys.argv=[sys.argv[0]]
app=AppLauncher(headless=True, enable_cameras=False).app
import torch
from lerobot_isaac_env import make_env
env=make_env(task="pick_and_place", num_envs=1, headless=True, enable_cameras=False)
r=env.scene["robot"]
ee=r.find_bodies("gripper_link")[0][0]
jaw=r.find_bodies("moving_jaw_so101_v1_link")[0][0]
names=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
idx={n:int(r.find_joints(n)[0][0]) for n in names}
ad=env.action_space.shape[-1]
env.reset()
def go(sh,el,wr,n=40):
    a=torch.zeros((1,ad),device=env.device)
    a[0,idx["shoulder_lift"]]=sh; a[0,idx["elbow_flex"]]=el; a[0,idx["wrist_flex"]]=wr
    for _ in range(n): env.step(a)
    q=r.data.body_quat_w[0,ee,:].cpu().numpy()
    eez=r.data.body_pos_w[0,ee,2].item(); jz=r.data.body_pos_w[0,jaw,2].item()
    print(f"[orient] sh={sh} el={el} wr={wr} ee_quat={np.array2string(q,precision=3)} ee_z={eez:.3f} jaw_z={jz:.3f}",flush=True)
for wr in [-1.0,-0.5,0.0,0.5,1.0]:
    go(1.0,-1.0,wr)
app.close()
