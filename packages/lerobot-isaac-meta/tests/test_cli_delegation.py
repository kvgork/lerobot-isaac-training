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
    """dr-replay forwards argv to replay_runner and its dry-run reports the values.

    Two stacked bugs were fixed here on 2026-09-12. The visible one was
    ``TypeError: main() takes 0 positional arguments`` — ``replay_runner.main`` did not
    accept ``argv`` while the other two delegation targets
    (``lerobot_isaac_adapters.train.main``, ``robot_data_recorder.cli.main``) both do.

    Behind it, this test asserted passthrough of ``--camera_key d435_rgb``, a flag
    ``replay_runner`` has NEVER accepted — ``git log -S camera_key`` finds it only in
    this test, added by 3dce29d. The arity error masked the stale flag: the call blew
    up before argparse could reject it. Passthrough is now proven with ``--n_variants``,
    which the target really does accept, so the assertion tests the forwarding rather
    than a contract that never existed.
    """
    from lerobot_isaac_meta.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "dr-replay",
                "--source_dataset",
                "datasets/x",
                "--n_variants",
                "7",
                "--dry_run",
            ]
        )

    assert rc == 0
    out = buf.getvalue()
    assert "replay_runner dry-run" in out
    assert "datasets/x" in out, "source_dataset was not forwarded"
    assert "7" in out, "n_variants was not forwarded"
