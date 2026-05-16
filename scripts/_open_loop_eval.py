"""Open-loop action-MSE eval on a held-out slice of a LeRobotDataset.

SO-101 has no registered gym env, so the standard `lerobot-eval` rollout path
does not apply. We instead measure how well a trained policy predicts the
human-teleoperated actions on a held-out fraction of the same dataset:

    action_pred = policy(obs)
    mse        = mean((action_pred - action_true) ** 2)
    pc_success = 1 / (1 + mse)        # monotone in -MSE, [0,1] range

This is NOT a closed-loop success rate. It is a proxy useful for tracking
policy quality across runs without an environment. The output JSON is marked
with `task: <repo_id>-open-loop-mse` so anyone reading the dashboard can see
it's not rollout-based.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy_path", required=True, help="pretrained_model dir")
    ap.add_argument(
        "--dataset_root", required=True, help="LeRobotDataset root with meta/, data/"
    )
    ap.add_argument(
        "--n_episodes", type=int, default=4, help="Number of held-out episodes to eval"
    )
    ap.add_argument(
        "--output_json", required=True, help="Where to write the eval JSON"
    )
    ap.add_argument(
        "--task_label",
        default=None,
        help="task field for the JSON (default: derive from dataset_root)",
    )
    ap.add_argument(
        "--run_id",
        default=None,
        help="run_id field for the JSON (default: UUID4 hex prefix)",
    )
    args = ap.parse_args(argv)

    policy_path = Path(args.policy_path).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    repo_id = "/".join(dataset_root.parts[-2:])
    task = args.task_label or f"{repo_id}-open-loop-mse"
    run_id = args.run_id or f"open-loop-{uuid.uuid4().hex[:8]}"

    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total_episodes = int(info["total_episodes"])
    n_eval = max(1, min(args.n_episodes, total_episodes))
    # Held-out slice: take the last n_eval episodes.
    eval_episode_ids = list(range(total_episodes - n_eval, total_episodes))
    print(
        f"[open-loop-eval] dataset={repo_id} total_eps={total_episodes} "
        f"eval_eps={eval_episode_ids}"
    )

    # Lazy-import lerobot to avoid hard dep at module load.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    ds = LeRobotDataset(
        repo_id=repo_id,
        root=str(dataset_root),
        episodes=eval_episode_ids,
        video_backend="pyav",
    )
    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(dataset_root))
    print(f"[open-loop-eval] loaded dataset, num_frames={len(ds)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # `make_policy` resolves the right concrete subclass from cfg.type and
    # loads pretrained weights from cfg.pretrained_path.
    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    policy_cfg.pretrained_path = Path(policy_path)
    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.to(device)
    policy.eval()

    # Load the SAME preprocessor pipeline that training used. For SmolVLA
    # this includes `tokenizer_processor` which reads `obs['task']` and
    # writes `observation.language.tokens`. Without it, select_action
    # crashes on the missing key for every frame.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=str(policy_path),
        dataset_stats=getattr(ds_meta, "stats", None),
    )
    print(
        f"[open-loop-eval] loaded policy from {policy_path} "
        f"device={device} class={type(policy).__name__} "
        f"preprocessor_steps={len(getattr(preprocessor, 'steps', []) or [])}"
    )

    squared_errs: list[float] = []
    ep_frame_counts: dict[int, int] = {}
    t0 = time.monotonic()
    with torch.no_grad():
        for i, sample in enumerate(ds):
            ep_id = int(sample["episode_index"].item())
            ep_frame_counts[ep_id] = ep_frame_counts.get(ep_id, 0) + 1
            # Move tensors to device.
            batch = {
                k: v.unsqueeze(0).to(device)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in sample.items()
            }
            action_true = batch["action"].squeeze(0).cpu().numpy()
            try:
                # Apply preprocessor (rename → batch → newline → tokenize →
                # device → normalize). Required for SmolVLA's language path.
                processed = preprocessor(batch)
                action_pred = policy.select_action(processed)
                action_pred = postprocessor(action_pred)
                action_pred = action_pred.squeeze(0).cpu().numpy()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[open-loop-eval] select_action failed on frame {i}: {exc}",
                    file=sys.stderr,
                )
                continue
            squared_errs.append(float(np.mean((action_pred - action_true) ** 2)))
            if i % 50 == 0:
                print(
                    f"[open-loop-eval] frame={i} ep={ep_id} mse={squared_errs[-1]:.6f}"
                )
            # Bound wall-clock — open-loop eval should be fast.
            if time.monotonic() - t0 > 600:
                print("[open-loop-eval] 10-min cap hit; stopping early")
                break

    if not squared_errs:
        print(
            "[open-loop-eval] no frames evaluated; "
            "writing empty eval JSON",
            file=sys.stderr,
        )
        mse = float("nan")
        pc_success = float("nan")
    else:
        mse = float(np.mean(squared_errs))
        pc_success = float(1.0 / (1.0 + mse))

    mean_ep_len = (
        float(np.mean(list(ep_frame_counts.values()))) if ep_frame_counts else 0.0
    )
    payload = {
        "run_id": run_id,
        "task": task,
        "ts": datetime.now(UTC).isoformat(),
        "pc_success": pc_success,
        "n_episodes": len(eval_episode_ids),
        "intervention_rate": None,
        "mean_ep_len": mean_ep_len,
        "_metadata": {
            "source": "open_loop_action_mse",
            "mse": mse,
            "n_frames_evaluated": len(squared_errs),
            "policy_path": str(policy_path),
            "dataset_root": str(dataset_root),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[open-loop-eval] wrote {out_path}")
    print(
        f"[open-loop-eval] result: mse={mse:.6f} pc_success={pc_success:.6f} "
        f"n_eps={len(eval_episode_ids)} n_frames={len(squared_errs)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
