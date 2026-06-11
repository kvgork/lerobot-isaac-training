import sys, numpy as np
from itertools import product
from isaaclab.app import AppLauncher
sys.argv=[sys.argv[0]]
app=AppLauncher(headless=True, enable_cameras=False).app
import torch
from lerobot_isaac_env import make_env
env=make_env(task="pick_and_place", num_envs=1, headless=True, enable_cameras=False)
r=env.scene["robot"]; obj=env.scene["source_object"]
ee=r.find_bodies("gripper_link")[0][0]; jaw=r.find_bodies("moving_jaw_so101_v1_link")[0][0]
nm=["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
idx={n:int(r.find_joints(n)[0][0]) for n in nm}
ad=env.action_space.shape[-1]
env.reset()
op=obj.data.root_pos_w[0].cpu().numpy()
print(f"[grasppose] object rest={np.array2string(op,precision=3)}",flush=True)
best=None
for sp,sh,el,wr in product([-0.3,0,0.3],[-1,-0.5,0,0.5,1],[-1,-0.5,0,0.5,1],[-1,0,1]):
    a=torch.zeros((1,ad),device=env.device)
    a[0,idx["shoulder_pan"]]=sp; a[0,idx["shoulder_lift"]]=sh; a[0,idx["elbow_flex"]]=el; a[0,idx["wrist_flex"]]=wr
    for _ in range(20): env.step(a)
    eep=r.data.body_pos_w[0,ee,:].cpu().numpy(); jp=r.data.body_pos_w[0,jaw,:].cpu().numpy()
    q=r.data.body_quat_w[0,ee,:].cpu().numpy()
    # score: jaw close to object (xy) AND low z (near object)
    d_xy=((jp[0]-op[0])**2+(jp[1]-op[1])**2)**0.5
    score=d_xy + abs(jp[2]-op[2])  # want jaw at object xy + object z
    if best is None or score<best["score"]:
        best={"score":float(score),"a":[sp,sh,el,wr],"ee":eep.tolist(),"jaw":jp.tolist(),"quat":q.tolist(),"d_xy":float(d_xy)}
print(f"[grasppose] BEST jaw-to-object: a={best['a']} jaw={np.array2string(np.array(best['jaw']),precision=3)} ee_z={best['ee'][2]:.3f} d_xy={best['d_xy']:.3f}",flush=True)
print(f"[grasppose] grasp quat (ee, base~world)={np.array2string(np.array(best['quat']),precision=4)}",flush=True)
app.close()
