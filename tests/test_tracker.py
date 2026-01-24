import pytest

from openai_cost_guard.models import BudgetConfig, ModelPricing
from openai_cost_guard.tracker import BudgetExceededError, CostTracker, UnknownModelError


def test_record_known_model_calculates_cost() -> None:
    tracker = CostTracker()
    record = tracker.record("gpt-4o", prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert record.input_cost == pytest.approx(2.50)
    assert record.output_cost == pytest.approx(10.00)
    assert record.total_cost == pytest.approx(12.50)
    assert record.total_tokens == 2_000_000


def test_record_mini_model() -> None:
    tracker = CostTracker()
    record = tracker.record("gpt-4o-mini", prompt_tokens=500_000, completion_tokens=0)

    assert record.input_cost == pytest.approx(0.075)
    assert record.output_cost == pytest.approx(0.0)


def test_record_embedding_model_has_no_output_cost() -> None:
    tracker = CostTracker()
    record = tracker.record("text-embedding-3-small", prompt_tokens=100_000)

    assert record.output_cost == pytest.approx(0.0)
    # text-embedding-3-small = $0.022 / 1M tokens -> 100k tokens = $0.0022
    assert record.input_cost == pytest.approx(0.0022)


def test_report_aggregates_multiple_records() -> None:
    tracker = CostTracker()
    tracker.record("gpt-4o", prompt_tokens=100, completion_tokens=50)
    tracker.record("gpt-4o-mini", prompt_tokens=200, completion_tokens=100)

    report = tracker.report()
    assert len(report.records) == 2
    assert report.total_tokens == 450
    assert report.total_cost > 0


def test_reset_clears_records() -> None:
    tracker = CostTracker()
    tracker.record("gpt-4o", prompt_tokens=100)
    tracker.reset()

    report = tracker.report()
    assert len(report.records) == 0
    assert report.total_cost == pytest.approx(0.0)


def test_unknown_model_returns_zero_cost_by_default() -> None:
    tracker = CostTracker()
    record = tracker.record("some-custom-deployment", prompt_tokens=1000)

    assert record.total_cost == pytest.approx(0.0)


def test_unknown_model_raises_in_strict_mode() -> None:
    tracker = CostTracker(strict=True)

    with pytest.raises(UnknownModelError):
        tracker.record("some-custom-deployment", prompt_tokens=1000)


def test_prefix_match_handles_versioned_deployment_names() -> None:
    tracker = CostTracker()
    # Deployment named "gpt-4o-20240513" should match the "gpt-4o" pricing entry
    record = tracker.record("gpt-4o-20240513", prompt_tokens=1_000_000, completion_tokens=0)

    assert record.input_cost == pytest.approx(2.50)


def test_gpt_41_mini_pricing() -> None:
    tracker = CostTracker()
    record = tracker.record("gpt-4.1-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert record.input_cost == pytest.approx(0.40)
    assert record.output_cost == pytest.approx(1.60)


def test_versioned_mini_not_mispriced_as_parent() -> None:
    """A versioned mini deployment must match the mini entry, not the cheaper-prefix parent.

    'gpt-4.1-mini-2025-04-14' starts with both 'gpt-4.1' and 'gpt-4.1-mini'; the longest
    (most specific) prefix must win, or the mini call gets the 5x more expensive gpt-4.1 rate.
    """
    tracker = CostTracker()
    record = tracker.record(
        "gpt-4.1-mini-2025-04-14", prompt_tokens=1_000_000, completion_tokens=0
    )
    assert record.input_cost == pytest.approx(0.40)  # gpt-4.1-mini, NOT gpt-4.1 ($2.00)


def test_versioned_4o_mini_not_mispriced_as_4o() -> None:
    tracker = CostTracker()
    record = tracker.record(
        "gpt-4o-mini-2024-07-18", prompt_tokens=1_000_000, completion_tokens=0
    )
    assert record.input_cost == pytest.approx(0.15)  # gpt-4o-mini, NOT gpt-4o ($2.50)


def test_custom_pricing_overrides_default() -> None:
    custom = {"gpt-4o": ModelPricing(model="gpt-4o", input_per_million=1.0, output_per_million=2.0)}
    tracker = CostTracker(pricing=custom)
    record = tracker.record("gpt-4o", prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert record.input_cost == pytest.approx(1.0)
    assert record.output_cost == pytest.approx(2.0)


def test_add_pricing_at_runtime() -> None:
    tracker = CostTracker(strict=True)
    tracker.add_pricing(
        ModelPricing(model="my-model", input_per_million=5.0, output_per_million=10.0)
    )
    record = tracker.record("my-model", prompt_tokens=1_000_000, completion_tokens=0)

    assert record.input_cost == pytest.approx(5.0)


def test_budget_exceeded_raises() -> None:
    tracker = CostTracker(budget=BudgetConfig(limit_usd=0.001))

    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.record("gpt-4o", prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert exc_info.value.limit == pytest.approx(0.001)


def test_budget_warning_does_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    tracker = CostTracker(budget=BudgetConfig(limit_usd=100.0, warn_at_percent=0.0))

    with caplog.at_level(logging.WARNING, logger="openai_cost_guard.tracker"):
        tracker.record("gpt-4o", prompt_tokens=100, completion_tokens=50)

    assert any("Budget warning" in msg for msg in caplog.messages)


def test_report_is_snapshot_not_live_reference() -> None:
    tracker = CostTracker()
    tracker.record("gpt-4o", prompt_tokens=100)
    report = tracker.report()

    tracker.record("gpt-4o", prompt_tokens=100)
    assert len(report.records) == 1


def test_endpoint_stored_on_record() -> None:
    tracker = CostTracker()
    record = tracker.record("gpt-4o", prompt_tokens=100, endpoint="summarise")

    assert record.endpoint == "summarise"


def test_metadata_stored_on_record() -> None:
    tracker = CostTracker()
    record = tracker.record("gpt-4o", prompt_tokens=100, metadata={"user": "alice"})

    assert record.metadata == {"user": "alice"}


def test_on_record_callback_fires_per_call() -> None:
    seen: list[str] = []
    tracker = CostTracker(on_record=lambda rec: seen.append(rec.model))

    tracker.record("gpt-4o", prompt_tokens=100)
    tracker.record("gpt-4o-mini", prompt_tokens=50)

    assert seen == ["gpt-4o", "gpt-4o-mini"]


def test_on_record_callback_receives_full_record() -> None:
    captured: list[float] = []
    tracker = CostTracker(on_record=lambda rec: captured.append(rec.total_cost))

    tracker.record("gpt-4o", prompt_tokens=1_000_000, completion_tokens=0)

    assert captured[0] == pytest.approx(2.50)


def test_on_record_not_fired_when_budget_exceeded() -> None:
    seen: list[str] = []
    tracker = CostTracker(
        budget=BudgetConfig(limit_usd=0.001),
        on_record=lambda rec: seen.append(rec.model),
    )

    with pytest.raises(BudgetExceededError):
        tracker.record("gpt-4o", prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert seen == []
