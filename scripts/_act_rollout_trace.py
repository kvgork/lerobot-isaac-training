"""Trace one ACT-policy rollout: log object + EE trajectory to SEE why it fails to
place (grasp then drift? never grasp? carry then drop?). Reuses _sim_eval's loader.

  PATH=$PWD/.pixi/envs/sim/bin:$PATH LEROBOT_ISAAC_FIX_BASE=1 LEROBOT_ISAAC_OBJECT_SCALE=0.267 \
    LEROBOT_ISAAC_OBJECT_X=0.18 LEROBOT_ISAAC_OBJECT_Y=0.05 LEROBOT_ISAAC_STAGED_REWARD=1 \
    .pixi/envs/sim/bin/python scripts/_act_rollout_trace.py \
      --policy_path outputs/act-sim-demos-v1/checkpoints/020000/pretrained_model \
      --dataset_root datasets/local/so101-sim-pickplace-demos-tuned
"""
import os, sys, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--task", default="pick_and_place")
    ap.add_argument("--task_prompt", default="pick and place the die in the bin")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import _sim_eval as SE
    app = SE._boot_app(True)
    import torch
    import lerobot_isaac_env  # noqa: F401
    from lerobot_isaac_env import make_env
    from lerobot_isaac_env.tasks import _register_envs
    _register_envs()
    env = make_env(task=args.task, num_envs=1, headless=True, enable_cameras=True)
    policy, preprocessor, postprocessor, device = SE._load_policy(
        Path(args.policy_path), Path(args.dataset_root), args.seed)
    robot = env.scene["robot"]; obj = env.scene["source_object"]
    ee_idx = int(robot.find_bodies("gripper_link")[0][0])
    grip_idx = int(robot.find_joints("gripper")[0][0])
    tgt = (0.22, -0.13)

    for ep in range(args.episodes):
        obs_dict, _ = env.reset()
        obs_group = obs_dict["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        op0 = obj.data.root_pos_w[0].detach().cpu().numpy()
        print(f"\n[trace] === ep {ep} === die spawn={np.array2string(op0, precision=3)} target={tgt}", flush=True)
        max_z = float(op0[2]); min_xy_to_tgt = 9.9
        for step in range(1, args.max_steps + 1):
            with torch.no_grad():
                pin = SE._build_policy_input(obs_group, device, args.task_prompt)
                action = postprocessor(policy.select_action(preprocessor(pin)))
            act_np = action.detach().cpu().numpy().reshape(-1)
            obs_dict, _r, term, trunc, _i = env.step(action.to(env.device) if hasattr(env, "device") else action)
            obs_group = obs_dict["policy"]
            op = obj.data.root_pos_w[0].detach().cpu().numpy()
            ee = robot.data.body_pos_w[0, ee_idx, :].detach().cpu().numpy()
            gq = float(robot.data.joint_pos[0, grip_idx].item())
            max_z = max(max_z, float(op[2]))
            xy_to_tgt = float(((op[0]-tgt[0])**2 + (op[1]-tgt[1])**2) ** 0.5)
            min_xy_to_tgt = min(min_xy_to_tgt, xy_to_tgt)
            if step % args.every == 0 or step == 1:
                print(f"[trace] s{step:3d} obj=({op[0]:.3f},{op[1]:.3f},{op[2]:.3f}) "
                      f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) grip={gq:+.2f} "
                      f"obj_xy->bin={xy_to_tgt:.3f} act[grip]={act_np[-1]:+.2f}", flush=True)
            if bool(term[0].item()) or bool(trunc[0].item()):
                print(f"[trace]  terminated s{step} term={bool(term[0].item())} trunc={bool(trunc[0].item())}", flush=True)
                break
        opf = obj.data.root_pos_w[0].detach().cpu().numpy()
        print(f"[trace] ep {ep} SUMMARY: obj_final=({opf[0]:.3f},{opf[1]:.3f},{opf[2]:.3f}) "
              f"max_obj_z={max_z:.3f} (spawn {op0[2]:.3f}) min_obj_xy->bin={min_xy_to_tgt:.3f} "
              f"=> {'LIFTED' if max_z>op0[2]+0.03 else 'NEVER-LIFTED'}, "
              f"{'REACHED-BIN' if min_xy_to_tgt<0.06 else 'never-near-bin'}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
