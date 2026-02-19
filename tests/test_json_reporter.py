import json
from pathlib import Path

import pytest

from openai_cost_guard.reporters.json import to_json, to_summary_dict, write_json
from openai_cost_guard.tracker import CostTracker


def _populated_tracker() -> CostTracker:
    tracker = CostTracker()
    tracker.record("gpt-4o", prompt_tokens=1000, completion_tokens=500, endpoint="chat")
    tracker.record("gpt-4o-mini", prompt_tokens=200, completion_tokens=100, endpoint="cheap")
    tracker.record("gpt-4o", prompt_tokens=300, completion_tokens=150, endpoint="chat")
    return tracker


def test_to_json_is_valid_json() -> None:
    report = _populated_tracker().report()
    parsed = json.loads(to_json(report))

    assert parsed["total_tokens"] == 2250
    assert len(parsed["records"]) == 3


def test_to_json_serialises_datetime_as_iso_string() -> None:
    report = _populated_tracker().report()
    parsed = json.loads(to_json(report))

    # Each record timestamp must be a parseable ISO string, not an object
    ts = parsed["records"][0]["timestamp"]
    assert isinstance(ts, str)
    assert "T" in ts


def test_to_json_compact_mode() -> None:
    report = _populated_tracker().report()
    compact = to_json(report, indent=None)

    assert "\n" not in compact


def test_to_json_empty_report() -> None:
    parsed = json.loads(to_json(CostTracker().report()))
    assert parsed["records"] == []
    assert parsed["total_cost"] == 0.0


def test_write_json_creates_file(tmp_path: Path) -> None:
    report = _populated_tracker().report()
    target = tmp_path / "report.json"

    resolved = write_json(report, target)

    assert resolved.exists()
    parsed = json.loads(resolved.read_text())
    assert len(parsed["records"]) == 3


def test_write_json_creates_parent_dirs(tmp_path: Path) -> None:
    report = _populated_tracker().report()
    target = tmp_path / "nested" / "dir" / "report.json"

    write_json(report, target)

    assert target.exists()


def test_write_json_accepts_string_path(tmp_path: Path) -> None:
    report = _populated_tracker().report()
    target = str(tmp_path / "report.json")

    resolved = write_json(report, target)
    assert resolved.exists()


def test_to_summary_dict_aggregates_by_model() -> None:
    report = _populated_tracker().report()
    summary = to_summary_dict(report)

    assert summary["call_count"] == 3
    assert summary["total_tokens"] == 2250

    by_model = summary["by_model"]
    assert isinstance(by_model, dict)
    # gpt-4o was called twice
    assert by_model["gpt-4o"]["calls"] == 2
    assert by_model["gpt-4o-mini"]["calls"] == 1


def test_to_summary_dict_total_cost_matches_report() -> None:
    report = _populated_tracker().report()
    summary = to_summary_dict(report)

    assert summary["total_cost"] == pytest.approx(report.total_cost)


def test_to_summary_dict_empty() -> None:
    summary = to_summary_dict(CostTracker().report())
    assert summary["call_count"] == 0
    assert summary["by_model"] == {}
