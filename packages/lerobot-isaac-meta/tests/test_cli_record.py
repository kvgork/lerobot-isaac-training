"""test_cli_record.py — Tests for the `record` subcommand wiring (Addendum 4 §14)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _add_recorder_src_to_path():
    """Insert robot-data-recorder src on sys.path so meta CLI delegation resolves."""
    recorder_src = (
        Path(__file__).resolve().parents[3] / "lerobot-isaac-recorder" / "src"
    )
    if str(recorder_src) not in sys.path:
        sys.path.insert(0, str(recorder_src))
    yield


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

    @pytest.mark.requires_workspace_root
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
        # Block import by injecting a fake finder that raises on robot_data_recorder
        # Strategy: pop any cached module + insert a sentinel that fails import via meta_path

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
