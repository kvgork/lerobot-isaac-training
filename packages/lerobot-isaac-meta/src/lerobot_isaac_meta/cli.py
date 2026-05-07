"""lerobot-isaac CLI — top-level argparse entrypoint.

Subcommands:
    train           — train a policy or world model (Phase 2)
    record          — record SO-101 teleop data (Phase 2)
    dr-replay       — replay with Isaac Lab domain randomization (Phase 4)
    mimicgen-augment — augment dataset via MimicGen (Phase 4b, deferred)

Each subcommand prints a "not yet wired" message until its phase is complete.
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Subcommand handlers — stubs until their phases are implemented
# ---------------------------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> int:
    print(
        "lerobot-isaac train: not yet wired — see Phase 2 "
        "(packages/lerobot-isaac-adapters).\n"
        "When implemented: python -m lerobot_isaac_adapters.train --target_arch <arch> ..."
    )
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    print(
        "lerobot-isaac record: not yet wired — see Phase 2 "
        "(packages/lerobot-isaac-adapters, lerobot-data-collection-agent).\n"
        "When implemented: invokes lerobot-data-collection-agent for SO-101 teleop."
    )
    return 0


def _cmd_dr_replay(args: argparse.Namespace) -> int:
    print(
        "lerobot-isaac dr-replay: not yet wired — see Phase 4 "
        "(packages/lerobot-isaac-synthetic).\n"
        "When implemented: python -m lerobot_isaac_synthetic.isaac_dr.replay_runner ..."
    )
    return 0


def _cmd_mimicgen_augment(args: argparse.Namespace) -> int:
    print(
        "lerobot-isaac mimicgen-augment: not yet wired — see Phase 4b (deferred).\n"
        "This is the MimicGen bridge path. It requires MuJoCo/robosuite "
        "and the lerobot_mimicgen_bridge skill.\n"
        "Skill location: /home/koen/tools/claude_code/skills/lerobot_mimicgen_bridge/"
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand registry
# ---------------------------------------------------------------------------

_SUBCOMMANDS: dict[str, tuple[callable, str]] = {
    "train": (_cmd_train, "train a policy or world model (Phase 2+)"),
    "record": (_cmd_record, "record SO-101 teleop data (Phase 2+)"),
    "dr-replay": (_cmd_dr_replay, "replay with Isaac Lab domain randomization (Phase 4+)"),
    "mimicgen-augment": (
        _cmd_mimicgen_augment,
        "augment dataset via MimicGen (Phase 4b, deferred)",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-isaac",
        description=(
            "LeRobot + Isaac Lab training workspace CLI.\n"
            "Workspace: ~/workspaces/lerobot-isaac-training/\n"
            "Status: Phase 0 (scaffolding). Subcommands are stubs until Phase 2+."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="lerobot-isaac 0.1.0 (Phase 0 scaffold)",
    )

    subparsers = parser.add_subparsers(dest="subcommand", title="subcommands")
    for name, (handler, help_text) in _SUBCOMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text)
        sub.set_defaults(func=handler)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
