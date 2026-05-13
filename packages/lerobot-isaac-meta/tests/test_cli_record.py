"""test_cli_record.py — Tests for the `record` subcommand wiring (Addendum 4 §14).

Tests are organized into two tiers:

  1. **Tier 1 — parser-only** tests run anywhere, no sibling pkg needed. They
     verify that ``lerobot-isaac record`` is registered, has the expected
     help text, and respects the ``--`` separator convention.

  2. **Tier 2 — delegation** tests need the ``robot_data_recorder`` package
     to be importable. They auto-skip when ``importlib.util.find_spec``
     reports the package missing, so the test file works both in monorepo
     mode (where ``robot_data_recorder`` is installed as an editable workspace
     dep) and in standalone mode (where it may or may not be installed).
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_RECORDER = importlib.util.find_spec("robot_data_recorder") is not None
_recorder_required = pytest.mark.skipif(
    not _HAS_RECORDER,
    reason="robot_data_recorder is not installed; pip install robot-data-recorder to run",
)


class TestRecordSubcommand:
    def test_record_in_subcommands(self):
        from lerobot_isaac_meta.cli import build_parser

        parser = build_parser()
        subparsers_action = next(
            a for a in parser._actions if hasattr(a, "_name_parser_map")
        )
        assert "record" in subparsers_action._name_parser_map

    def test_record_help_mentions_recorder(self, capsys):
        from lerobot_isaac_meta.cli import main

        with pytest.raises(SystemExit):
            main(["record", "--help"])
        out = capsys.readouterr().out
        assert "recorder_args" in out or "recorder" in out.lower()

    @_recorder_required
    def test_record_dry_run_via_meta(self, capsys):
        """Forwarding `record -- --dry-run ...` must hit recorder CLI and exit 0."""
        from lerobot_isaac_meta.cli import main

        rc = main(
            [
                "record",
                "--",
                "--repo-id=test",
                "--num-episodes=1",
                "--dry-run",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "test" in out  # repo_id appears in resolved config

    def test_record_dash_dash_separator_required_for_flag_args(self, capsys):
        """argparse REMAINDER does not consume args that look like top-level flags.
        Document the required `--` separator pattern."""
        from lerobot_isaac_meta.cli import main

        # Without `--`, top-level parser rejects --repo-id (correct argparse behavior)
        with pytest.raises(SystemExit):
            main(["record", "--repo-id=test2", "--num-episodes=2", "--dry-run"])

    def test_record_import_error_when_recorder_missing(self, monkeypatch, capsys):
        """If robot_data_recorder is not importable, record subcommand returns 1."""
        import sys as _sys

        class _BlockingFinder:
            def find_spec(self, name, path=None, target=None):
                if name.startswith("robot_data_recorder"):
                    raise ImportError("forced fail for test")
                return None

        # Pop cached modules
        for k in list(_sys.modules):
            if k.startswith("robot_data_recorder"):
                _sys.modules.pop(k, None)

        finder = _BlockingFinder()
        _sys.meta_path.insert(0, finder)
        try:
            from lerobot_isaac_meta.cli import main

            rc = main(
                ["record", "--", "--repo-id=test", "--num-episodes=1", "--dry-run"]
            )
            assert rc == 1
            err = capsys.readouterr().err
            assert "robot_data_recorder" in err or "forced fail" in err
        finally:
            _sys.meta_path.remove(finder)
