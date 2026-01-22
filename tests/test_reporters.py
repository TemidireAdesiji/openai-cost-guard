import logging

import pytest

from openai_cost_guard.models import CostReport
from openai_cost_guard.reporters.console import print_report
from openai_cost_guard.tracker import CostTracker


def test_print_report_empty(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        print_report(CostReport())

    assert any("no usage recorded" in msg for msg in caplog.messages)


def test_print_report_contains_model_name(caplog: pytest.LogCaptureFixture) -> None:
    tracker = CostTracker()
    tracker.record("gpt-4o", prompt_tokens=1000, completion_tokens=500)

    with caplog.at_level(logging.INFO):
        print_report(tracker.report())

    combined = "\n".join(caplog.messages)
    assert "gpt-4o" in combined


def test_print_report_contains_total_row(caplog: pytest.LogCaptureFixture) -> None:
    tracker = CostTracker()
    tracker.record("gpt-4o", prompt_tokens=100, completion_tokens=50)
    tracker.record("gpt-4o-mini", prompt_tokens=200, completion_tokens=100)

    with caplog.at_level(logging.INFO):
        print_report(tracker.report())

    combined = "\n".join(caplog.messages)
    assert "TOTAL" in combined


def test_print_report_custom_logger(caplog: pytest.LogCaptureFixture) -> None:
    tracker = CostTracker()
    tracker.record("gpt-4o", prompt_tokens=500, completion_tokens=250)

    with caplog.at_level(logging.INFO, logger="my.app.logger"):
        print_report(tracker.report(), logger_name="my.app.logger")

    combined = "\n".join(caplog.messages)
    assert "gpt-4o" in combined
