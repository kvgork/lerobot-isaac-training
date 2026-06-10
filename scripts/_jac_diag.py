import sys
from isaaclab.app import AppLauncher
sys.argv=[sys.argv[0]]
app=AppLauncher(headless=True, enable_cameras=False).app
from lerobot_isaac_env import make_env
env=make_env(task="pick_and_place", num_envs=1, headless=True, enable_cameras=False)
r=env.scene["robot"]
env.reset()
ee=r.find_bodies("gripper_link")[0][0]
jac=r.root_physx_view.get_jacobians()
print("[jac] is_fixed_base", r.is_fixed_base)
print("[jac] num_bodies", r.num_bodies, "num_joints", r.num_joints)
print("[jac] jacobian shape", tuple(jac.shape))
print("[jac] gripper_link body_idx", ee)
arm=r.find_joints(["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"])[0]
print("[jac] arm joint ids", list(arm))
app.close()
