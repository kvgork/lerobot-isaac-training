"""Merge a peft-wrapped SmolVLA checkpoint into a plain SmolVLA checkpoint.

The training pipeline wraps the policy with `peft.get_peft_model(...)` before
the lerobot trainer captures parameters. The saved `model.safetensors` therefore
contains peft-prefixed keys (`model.base_model.model.*`) plus `lora_A/lora_B`
matrices. `lerobot.policies.factory.make_policy()` builds a fresh, *unwrapped*
SmolVLAPolicy and tries to load the state dict — keys don't match → "missing
keys" warning → random init → eval pc_success ≈ 0.

This script restores a loadable checkpoint by:

  1. Loading the anchor (or any base SmolVLA) — the *pretrained_path* the trial
     was launched from.
  2. Re-applying the same LoRA wrap with the trial's rank/alpha/target_modules.
  3. Loading the peft state dict into the wrapped policy.
  4. Calling `merge_and_unload()` to fold the LoRA delta back into the base
     weights. The result is a plain SmolVLAPolicy.
  5. Saving via `policy.save_pretrained(<out>)` → a normal-shaped ckpt dir
     identical to the anchor format.

Usage::

    python scripts/_merge_lora_ckpt.py \
        --anchor outputs/overnight-smolvla-.../checkpoints/last/pretrained_model \
        --trial_ckpt outputs/autoresearch-.../trial_2/checkpoints/last/pretrained_model \
        --lora_rank 32 --lora_alpha 32 --lora_dropout 0.05 \
        --lora_target_modules attn_qv \
        --out outputs/autoresearch-.../trial_2/checkpoints/merged/pretrained_model

The merged dir is eval-ready: pass `--policy_path <out>` to
`scripts/_open_loop_eval.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--anchor",
        required=True,
        help="Base SmolVLA pretrained_model dir the trial was finetuned from.",
    )
    ap.add_argument(
        "--trial_ckpt",
        required=True,
        help="The peft-wrapped trial pretrained_model dir to merge.",
    )
    ap.add_argument("--lora_rank", type=int, required=True)
    ap.add_argument("--lora_alpha", type=int, required=True)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument(
        "--lora_target_modules",
        required=True,
        help="Preset (attn_qv/attn_qkvo/expert_only) or comma-separated layer suffixes.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output dir for the merged plain-SmolVLA pretrained_model.",
    )
    ap.add_argument(
        "--dataset_root",
        required=True,
        help="LeRobotDataset root — make_policy needs dataset metadata to "
        "construct the policy (input/output feature shapes).",
    )
    args = ap.parse_args(argv)

    anchor = Path(args.anchor).resolve()
    trial_ckpt = Path(args.trial_ckpt).resolve()
    out_dir = Path(args.out).resolve()

    if not anchor.is_dir():
        print(f"[merge] ERROR: anchor not found: {anchor}", file=sys.stderr)
        return 2
    if not trial_ckpt.is_dir():
        print(f"[merge] ERROR: trial_ckpt not found: {trial_ckpt}", file=sys.stderr)
        return 2

    # Soft imports — heavy deps only loaded when actually merging.
    import torch
    from safetensors.torch import load_file as load_safetensors

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy

    # Lazy import LoRA helpers from the adapter package.
    sys.path.insert(
        0,
        str(Path(__file__).parent.parent / "src/lerobot-isaac-adapters/src"),
    )
    from lerobot_isaac_adapters.targets._lora import LoraSpec, wrap_smolvla_policy

    print(f"[merge] anchor={anchor}")
    print(f"[merge] trial_ckpt={trial_ckpt}")
    print(
        f"[merge] lora r={args.lora_rank} alpha={args.lora_alpha} "
        f"drop={args.lora_dropout} target={args.lora_target_modules}"
    )

    # 1. Load anchor as a fresh SmolVLA.
    dataset_root = Path(args.dataset_root).resolve()
    repo_id = "/".join(dataset_root.parts[-2:])
    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(dataset_root))
    policy_cfg = PreTrainedConfig.from_pretrained(str(anchor))
    policy_cfg.pretrained_path = str(anchor)
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()

    # 2. Wrap with the same LoRA config used during training.
    spec = LoraSpec.from_args(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules_spec=args.lora_target_modules,
    )
    policy = wrap_smolvla_policy(policy, spec)

    # 3. Load the peft state dict into the wrapped policy.
    state_path = trial_ckpt / "model.safetensors"
    if not state_path.is_file():
        print(f"[merge] ERROR: {state_path} missing", file=sys.stderr)
        return 3
    raw_state = load_safetensors(str(state_path), device="cpu")

    # Trial keys are `model.base_model.model.*`. Strip the leading `model.`
    # because the policy's state_dict already lives under `model`.
    # Actually the policy itself has `.model` attribute — so when calling
    # `policy.load_state_dict`, keys should look like `model.base_model.model.*`.
    # That matches the saved format → pass through verbatim.
    missing, unexpected = policy.load_state_dict(raw_state, strict=False)
    if missing:
        # `target_modules` mismatch would leave LoRA matrices missing.
        print(
            f"[merge] WARNING: {len(missing)} missing keys after load. "
            f"First few: {missing[:5]}",
            file=sys.stderr,
        )
    if unexpected:
        print(
            f"[merge] WARNING: {len(unexpected)} unexpected keys. "
            f"First few: {unexpected[:5]}",
            file=sys.stderr,
        )

    # 4. Merge LoRA delta into base + drop adapter modules.
    print("[merge] calling merge_and_unload()…")
    merged_model = policy.model.merge_and_unload()
    policy.model = merged_model

    # 5. Save the merged policy. Use the same dir layout the anchor uses so
    #    `_open_loop_eval.py` can pick it up without further changes.
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(policy, "save_pretrained"):
        policy.save_pretrained(str(out_dir))
        print(f"[merge] wrote merged ckpt → {out_dir}")
    else:
        # Fallback: dump state dict alone.
        torch.save(policy.state_dict(), out_dir / "model.pt")
        print(f"[merge] fallback wrote {out_dir / 'model.pt'}")

    # Also copy preprocessor / postprocessor JSONs from the trial dir so the
    # merged checkpoint mirrors the anchor layout (eval script needs them).
    for fname in (
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "train_config.json",
    ):
        src = trial_ckpt / fname
        if src.is_file():
            dst = out_dir / fname
            if not dst.exists():
                dst.write_bytes(src.read_bytes())

    return 0


if __name__ == "__main__":
    sys.exit(main())
