"""Tests for the Azure Monitor reporter.

Uses an in-memory OpenTelemetry MeterProvider so emission can be asserted without
a real Application Insights connection or the Azure exporter network path.
"""
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from openai_cost_guard.models import UsageRecord
from openai_cost_guard.reporters.azure_monitor import (
    METRIC_CALLS,
    METRIC_COST,
    METRIC_TOKENS,
    AzureMonitorReporter,
    configure_azure_monitor,
)
from openai_cost_guard.tracker import CostTracker


@pytest.fixture
def reader_and_reporter() -> tuple[InMemoryMetricReader, AzureMonitorReporter]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("test")
    return reader, AzureMonitorReporter(meter=meter)


def _collect_metrics(reader: InMemoryMetricReader) -> dict[str, list[object]]:
    """Return a mapping of metric name -> list of data points."""
    data = reader.get_metrics_data()
    points: dict[str, list[object]] = {}
    if data is None:
        return points
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                points.setdefault(metric.name, [])
                points[metric.name].extend(metric.data.data_points)
    return points


def test_emit_records_cost_metric(
    reader_and_reporter: tuple[InMemoryMetricReader, AzureMonitorReporter],
) -> None:
    reader, reporter = reader_and_reporter
    record = UsageRecord(
        model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        input_cost=0.0025,
        output_cost=0.005,
        total_cost=0.0075,
        endpoint="chat",
    )
    reporter.emit(record)

    metrics = _collect_metrics(reader)
    assert METRIC_COST in metrics
    cost_points = metrics[METRIC_COST]
    assert len(cost_points) == 1
    assert cost_points[0].value == pytest.approx(0.0075)  # type: ignore[attr-defined]


def test_emit_records_token_and_call_metrics(
    reader_and_reporter: tuple[InMemoryMetricReader, AzureMonitorReporter],
) -> None:
    reader, reporter = reader_and_reporter
    record = UsageRecord(
        model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        input_cost=0.0025,
        output_cost=0.005,
        total_cost=0.0075,
        endpoint="chat",
    )
    reporter.emit(record)

    metrics = _collect_metrics(reader)
    assert metrics[METRIC_TOKENS][0].value == 1500  # type: ignore[attr-defined]
    assert metrics[METRIC_CALLS][0].value == 1  # type: ignore[attr-defined]


def test_emit_attaches_model_and_endpoint_attributes(
    reader_and_reporter: tuple[InMemoryMetricReader, AzureMonitorReporter],
) -> None:
    reader, reporter = reader_and_reporter
    record = UsageRecord(
        model="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        input_cost=0.0,
        output_cost=0.0,
        total_cost=0.0001,
        endpoint="summarise",
    )
    reporter.emit(record)

    point = _collect_metrics(reader)[METRIC_COST][0]
    attrs = dict(point.attributes)  # type: ignore[attr-defined]
    assert attrs["model"] == "gpt-4o-mini"
    assert attrs["endpoint"] == "summarise"


def test_emit_uses_unknown_for_missing_endpoint(
    reader_and_reporter: tuple[InMemoryMetricReader, AzureMonitorReporter],
) -> None:
    reader, reporter = reader_and_reporter
    record = UsageRecord(
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=0,
        total_tokens=100,
        input_cost=0.0,
        output_cost=0.0,
        total_cost=0.00025,
        endpoint=None,
    )
    reporter.emit(record)

    point = _collect_metrics(reader)[METRIC_COST][0]
    attrs = dict(point.attributes)  # type: ignore[attr-defined]
    assert attrs["endpoint"] == "unknown"


def test_emit_report_emits_all_records(
    reader_and_reporter: tuple[InMemoryMetricReader, AzureMonitorReporter],
) -> None:
    reader, reporter = reader_and_reporter
    tracker = CostTracker()
    tracker.record("gpt-4o", prompt_tokens=100, completion_tokens=50, endpoint="a")
    tracker.record("gpt-4o-mini", prompt_tokens=200, completion_tokens=100, endpoint="b")

    reporter.emit_report(tracker.report())

    metrics = _collect_metrics(reader)
    # Two distinct attribute sets -> two data points on the call counter
    assert sum(p.value for p in metrics[METRIC_CALLS]) == 2  # type: ignore[attr-defined]


def test_reporter_as_tracker_on_record_callback(
    reader_and_reporter: tuple[InMemoryMetricReader, AzureMonitorReporter],
) -> None:
    """The streaming pattern: wire reporter.emit as the tracker's on_record hook."""
    reader, reporter = reader_and_reporter
    tracker = CostTracker(on_record=reporter.emit)

    tracker.record("gpt-4o", prompt_tokens=1000, completion_tokens=500, endpoint="chat")
    tracker.record("gpt-4o", prompt_tokens=1000, completion_tokens=500, endpoint="chat")

    metrics = _collect_metrics(reader)
    # Same attributes -> aggregated into one data point with summed values
    assert sum(p.value for p in metrics[METRIC_CALLS]) == 2  # type: ignore[attr-defined]
    assert sum(p.value for p in metrics[METRIC_TOKENS]) == 3000  # type: ignore[attr-defined]


def test_default_meter_does_not_raise() -> None:
    """Constructing without a meter uses the global (possibly no-op) meter."""
    reporter = AzureMonitorReporter()
    record = UsageRecord(
        model="gpt-4o",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        input_cost=0.0,
        output_cost=0.0,
        total_cost=0.0001,
    )
    # Must not raise even when no real MeterProvider is configured
    reporter.emit(record)


class TestConfigureAzureMonitor:
    def test_raises_without_connection_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("APPLICATIONINSIGHTS_CONNECTION_STRING", raising=False)
        with pytest.raises(RuntimeError, match="connection string"):
            configure_azure_monitor()

    def test_reads_connection_string_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A syntactically valid (fake) connection string - the exporter accepts it
        # without contacting Azure until the first export.
        conn = (
            "InstrumentationKey=00000000-0000-0000-0000-000000000000;"
            "IngestionEndpoint=https://example.in.applicationinsights.azure.com/"
        )
        monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", conn)
        provider = configure_azure_monitor(export_interval_millis=60_000)
        assert provider is not None
        provider.shutdown()
