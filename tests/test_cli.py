"""Tests for the openai-cost-guard CLI.

The CLI routes output to stdout via its own logging handler and disables propagation,
so these tests assert on captured stdout (capsys) rather than caplog.
"""
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from openai_cost_guard.cli import main
from openai_cost_guard.reporters.json import write_json
from openai_cost_guard.tracker import CostTracker


@pytest.fixture(autouse=True)
def _restore_package_logger() -> Iterator[None]:
    """The CLI reconfigures the openai_cost_guard logger (handlers, level, propagate).

    Snapshot and restore it so that mutation does not leak into caplog-based tests
    elsewhere in the suite, which depend on propagation to the root logger.
    """
    pkg_logger = logging.getLogger("openai_cost_guard")
    saved_handlers = pkg_logger.handlers[:]
    saved_level = pkg_logger.level
    saved_propagate = pkg_logger.propagate
    try:
        yield
    finally:
        pkg_logger.handlers = saved_handlers
        pkg_logger.level = saved_level
        pkg_logger.propagate = saved_propagate


@pytest.fixture
def report_file(tmp_path: Path) -> Path:
    tracker = CostTracker()
    tracker.record("gpt-4o", prompt_tokens=1000, completion_tokens=500, endpoint="chat")
    tracker.record("gpt-4o-mini", prompt_tokens=200, completion_tokens=100, endpoint="cheap")
    target = tmp_path / "report.json"
    write_json(tracker.report(), target)
    return target


def test_show_prints_table(report_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["show", str(report_file)])
    out = capsys.readouterr().out

    assert code == 0
    assert "gpt-4o" in out
    assert "TOTAL" in out


def test_summary_prints_valid_json(
    report_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["summary", str(report_file)])
    out = capsys.readouterr().out

    assert code == 0
    payload = json.loads(out)
    assert payload["call_count"] == 2
    assert payload["total_tokens"] == 1800
    assert "gpt-4o" in payload["by_model"]


def test_missing_file_returns_error_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["show", str(tmp_path / "nope.json")])
    out = capsys.readouterr().out

    assert code == 1
    assert "not found" in out


def test_invalid_json_returns_error_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not a valid report }", encoding="utf-8")

    code = main(["show", str(bad)])
    out = capsys.readouterr().out

    assert code == 1
    assert "valid cost report" in out


def test_no_command_exits_with_error() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])
    # argparse exits with code 2 on missing required subcommand
    assert exc_info.value.code == 2


def test_roundtrip_show_after_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A report written by write_json must load and render through the CLI."""
    tracker = CostTracker()
    tracker.record("gpt-4o", prompt_tokens=500, completion_tokens=250, endpoint="x")
    target = tmp_path / "r.json"
    write_json(tracker.report(), target)

    code = main(["show", str(target)])
    out = capsys.readouterr().out

    assert code == 0
    assert "gpt-4o" in out
