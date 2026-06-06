"""Scripted-arm audit for the SO-101 Isaac Lab env.

Bypasses RL — drives the arm with hand-crafted action sequences and logs:
  - EE pos vs object pos at each step
  - reward signal per step (success_bonus + progress + action_penalty)
  - whether `success` termination fires
  - whether actions actually move the arm

Run:
    LEROBOT_ISAAC_OBJECT_X=0.30 LEROBOT_ISAAC_OBJECT_Y=0.05 LEROBOT_ISAAC_OBJECT_Z=0.05 \\
    LEROBOT_ISAAC_PROGRESS_WEIGHT=1.0 \\
    pixi run -e sim python scripts/_scripted_arm_audit.py 2>&1 | tee scripts/_scripted_arm_audit.log

Phases:
  Phase 1: action=zeros for 60 steps. Arm should stay at default pose.
            If arm drifts → physics / gravity issue.
  Phase 2: per-joint probe — action = +1.0 on one joint at a time, 30 steps each.
            Records EE pos delta per joint.
  Phase 3: per-joint negative probe — action = -1.0 on one joint, 30 steps each.
  Phase 4: gross-search — sample 32 random actions, hold 30 steps each, pick
           the one that minimizes EE-to-object distance. Run that on repeat
           to see if reward decreases monotonically.
  Phase 5: success-probe — directly set joint targets to push EE toward object,
           verify success termination fires when within 5cm.

Output: a structured report on stdout.
"""
from __future__ import annotations

import sys


def main() -> None:
    # 1. AppLauncher FIRST.
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=False)
    for _ in range(2):
        launcher.app.update()

    # 2. NOW it's safe to import lerobot_isaac_env (which transitively imports isaaclab managers).
    import numpy as np
    import torch

    from lerobot_isaac_env import make_env

    print("[audit] booting env task=pick_and_place num_envs=1 headless=True")
    env = make_env(task="pick_and_place", num_envs=1, headless=True)
    print(f"[audit] env created: {type(env).__name__}")

    # Warm-up
    sim = getattr(env, "sim", None)
    if sim is not None:
        for _ in range(30):
            try:
                sim.step(render=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[audit] sim.step warmup error: {exc}")
                break

    # Helpers
    def get_ee_obj():
        robot = env.scene["robot"]
        obj = env.scene["source_object"]
        ee = robot.data.body_pos_w[:, -1, :].clone()
        op = obj.data.root_pos_w.clone()
        return ee, op

    def reset_and_get():
        obs, _ = env.reset()
        ee, op = get_ee_obj()
        return ee[0].cpu().numpy(), op[0].cpu().numpy()

    # ------------------------------------------------------------------ #
    # Phase 1: action=zeros baseline
    # ------------------------------------------------------------------ #
    print("\n=== Phase 1: action=zeros for 60 steps ===")
    ee0, op0 = reset_and_get()
    print(f"[audit] initial EE = {ee0.round(4).tolist()}  object = {op0.round(4).tolist()}")
    print(f"[audit] initial dist = {np.linalg.norm(ee0 - op0):.4f} m")

    zero_action = torch.zeros((1, 6), device=env.device if hasattr(env, "device") else "cuda")
    rewards = []
    for i in range(60):
        out = env.step(zero_action)
        obs, r, term, trunc, info = out
        rewards.append(float(r[0].item()) if hasattr(r, "item") else float(r))
    ee_f, op_f = get_ee_obj()
    ee_f = ee_f[0].cpu().numpy()
    op_f = op_f[0].cpu().numpy()
    print(f"[audit] after 60 zero-steps EE = {ee_f.round(4).tolist()}")
    print(f"[audit] EE drift = {np.linalg.norm(ee_f - ee0):.4f} m")
    print(f"[audit] reward range: min={min(rewards):.4f} max={max(rewards):.4f} mean={np.mean(rewards):.4f}")

    # ------------------------------------------------------------------ #
    # Phase 2/3: per-joint probe (positive then negative) — log joint pos too
    # ------------------------------------------------------------------ #
    def get_joint_pos():
        robot = env.scene["robot"]
        return robot.data.joint_pos[0].cpu().numpy()

    for sign, label in [(+1.0, "Phase 2: per-joint +1.0 probe"), (-1.0, "Phase 3: per-joint -1.0 probe")]:
        print(f"\n=== {label} ===")
        for j in range(6):
            reset_and_get()
            jp0 = get_joint_pos()
            ee0, _ = get_ee_obj()
            ee0 = ee0[0].cpu().numpy()
            act = torch.zeros((1, 6), device=zero_action.device)
            act[0, j] = sign
            for _ in range(30):
                env.step(act)
            jp_f = get_joint_pos()
            ee_f, _ = get_ee_obj()
            ee_f = ee_f[0].cpu().numpy()
            jdelta = jp_f - jp0
            edelta = ee_f - ee0
            print(f"[audit]   joint[{j}] action={sign:+.1f}: "
                  f"jp_delta={jdelta.round(4).tolist()}  "
                  f"ee_delta={edelta.round(4).tolist()}  "
                  f"|ee|={np.linalg.norm(edelta):.4f}")

    # Phase 2.5: direct joint target test (bypass action manager).
    print("\n=== Phase 2.5: direct joint_position_target write (bypass action manager) ===")
    reset_and_get()
    jp0 = get_joint_pos()
    ee0, _ = get_ee_obj()
    ee0 = ee0[0].cpu().numpy()
    robot = env.scene["robot"]
    target = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], device=zero_action.device)
    try:
        robot.set_joint_position_target(target)
        for _ in range(60):
            sim = getattr(env, "sim", None)
            if sim is not None:
                sim.step(render=False)
            robot.update(dt=1.0/120.0)
        jp_f = get_joint_pos()
        ee_f, _ = get_ee_obj()
        ee_f = ee_f[0].cpu().numpy()
        print(f"[audit]   set target=[0.5×6]: jp_delta={(jp_f-jp0).round(4).tolist()}  ee_delta={(ee_f-ee0).round(4).tolist()}")
    except Exception as exc:  # noqa: BLE001
        print(f"[audit]   direct target write FAILED: {exc}")

    # Phase 2.6: check actuator stiffness/damping reported by sim
    try:
        robot = env.scene["robot"]
        print(f"[audit]   actuators: {list(robot.actuators.keys()) if hasattr(robot, 'actuators') else 'n/a'}")
        for name, act_obj in (robot.actuators.items() if hasattr(robot, "actuators") else []):
            print(f"[audit]     {name}: stiffness={act_obj.stiffness} damping={act_obj.damping} "
                  f"effort_limit={getattr(act_obj, 'effort_limit', 'n/a')}")
    except Exception as exc:  # noqa: BLE001
        print(f"[audit]   actuator introspection FAILED: {exc}")

    # ------------------------------------------------------------------ #
    # Phase 4: gross random search for shortest dist
    # ------------------------------------------------------------------ #
    print("\n=== Phase 4: random-action search (32 samples × 30 steps) ===")
    best_dist = float("inf")
    best_act = None
    for trial in range(32):
        reset_and_get()
        rng = np.random.default_rng(trial)
        act_np = rng.uniform(-1.0, 1.0, size=(1, 6)).astype(np.float32)
        act = torch.from_numpy(act_np).to(zero_action.device)
        last_r = None
        for _ in range(30):
            out = env.step(act)
            _, r, _, _, _ = out
            last_r = float(r[0].item()) if hasattr(r, "item") else float(r)
        ee_f, op_f = get_ee_obj()
        d = float(torch.norm(ee_f[0] - op_f[0]).item())
        if d < best_dist:
            best_dist = d
            best_act = act_np.copy()
        if trial < 5 or trial == 31:
            print(f"[audit]   trial {trial:2d}: dist={d:.4f}  reward[last]={last_r:.4f}  action={act_np[0].round(2).tolist()}")
    print(f"[audit] best random-action dist = {best_dist:.4f} m  (success threshold = 0.05 m)")
    if best_act is not None:
        print(f"[audit] best action = {best_act[0].round(3).tolist()}")

    # ------------------------------------------------------------------ #
    # Phase 5: hold best action longer + watch reward/termination
    # ------------------------------------------------------------------ #
    if best_act is not None:
        print("\n=== Phase 5: hold best random action for 200 steps ===")
        reset_and_get()
        act = torch.from_numpy(best_act).to(zero_action.device)
        rewards = []
        terminated_any = False
        for step in range(200):
            out = env.step(act)
            _, r, term, trunc, info = out
            rewards.append(float(r[0].item()) if hasattr(r, "item") else float(r))
            term_val = bool(term[0].item()) if hasattr(term, "item") else bool(term)
            if term_val:
                terminated_any = True
                print(f"[audit] TERMINATION fired at step {step}: reward={rewards[-1]:.4f}")
                # Inspect which term fired
                try:
                    tm = env.termination_manager
                    print(f"[audit]   term_keys: {tm.active_terms}")
                    print(f"[audit]   term values: {tm.compute().tolist() if hasattr(tm, 'compute') else 'n/a'}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[audit]   could not read termination_manager: {exc}")
                break
        ee_f, op_f = get_ee_obj()
        d_final = float(torch.norm(ee_f[0] - op_f[0]).item())
        print(f"[audit] final dist = {d_final:.4f} m  terminated={terminated_any}")
        print(f"[audit] reward trajectory: first={rewards[0]:.4f} last={rewards[-1]:.4f} min={min(rewards):.4f} max={max(rewards):.4f}")

    # ------------------------------------------------------------------ #
    # Phase 6: action manager params introspection
    # ------------------------------------------------------------------ #
    print("\n=== Phase 6: action / scene introspection ===")
    try:
        am = env.action_manager
        print(f"[audit] action_manager total dim: {am.total_action_dim}")
        for name, term in am._terms.items():
            scale = getattr(term, "_scale", None)
            offset = getattr(term, "_offset", None)
            print(f"[audit]   term '{name}': scale={scale} offset={offset}")
    except Exception as exc:  # noqa: BLE001
        print(f"[audit] could not introspect action_manager: {exc}")

    try:
        for n in ["robot", "source_object", "target_bin"]:
            asset = env.scene[n]
            cls = type(asset).__name__
            print(f"[audit] scene['{n}']: {cls}")
    except Exception as exc:  # noqa: BLE001
        print(f"[audit] scene introspection error: {exc}")

    print("\n[audit] done — closing app")
    sys.exit(0)


if __name__ == "__main__":
    main()
