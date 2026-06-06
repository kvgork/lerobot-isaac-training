"""
test_cli_delegation.py
======================
`lerobot-isaac train` and `lerobot-isaac dr-replay` delegate to their sibling
backends (adapters.train / synthetic.replay_runner) with forwarded args.

Dry-run tests — no GPU, lerobot, or Isaac Lab needed.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout


def test_train_delegates_dry_run():
    from lerobot_isaac_meta.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["train", "--target_arch", "smolvla", "--dataset", "x", "--dry_run"])

    assert rc == 0
    out = buf.getvalue()
    assert "dry_run" in out
    assert "target_arch=smolvla" in out


def test_train_forwards_after_double_dash():
    from lerobot_isaac_meta.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["train", "--", "--target_arch", "act", "--dry_run"])

    assert rc == 0
    assert "target_arch=act" in buf.getvalue()


def test_train_invalid_arch_propagates_exit_code():
    from lerobot_isaac_meta.cli import main

    # argparse in the adapters parser exits 2 on bad choice; meta must surface it.
    rc = main(["train", "--target_arch", "banana"])
    assert rc == 2


def test_dr_replay_delegates_dry_run():
    from lerobot_isaac_meta.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "dr-replay",
                "--source_dataset",
                "datasets/x",
                "--camera_key",
                "d435_rgb",
                "--dry_run",
            ]
        )

    assert rc == 0
    out = buf.getvalue()
    assert "replay_runner dry-run" in out
    assert "d435_rgb" in out
