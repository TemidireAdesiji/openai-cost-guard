"""Azure Monitor / Application Insights reporting via OpenTelemetry.

This reporter is vendor-neutral at its core: it records cost and token usage to
OpenTelemetry metric instruments. To ship those metrics to Application Insights,
configure the Azure Monitor exporter once at startup with ``configure_azure_monitor``
(or wire your own OTel MeterProvider).

Two emission patterns:

1. Streaming (recommended) - pass ``reporter.emit`` as the tracker's ``on_record``
   callback so each API call is emitted as it happens::

       reporter = AzureMonitorReporter()
       tracker = CostTracker(on_record=reporter.emit)

2. Batch - call ``reporter.emit_report(tracker.report())`` once. Note this re-emits
   every record in the report, so do not mix it with streaming on the same records.

Metrics emitted (all with ``model`` and ``endpoint`` attributes):

- ``openai.cost.usd``   (counter, USD)   - cost per call
- ``openai.tokens``     (counter, token) - total tokens per call
- ``openai.calls``      (counter, call)  - call count

Requires the ``[azure]`` extra: ``pip install "openai-cost-guard[azure]"``.
"""
import logging
import os
from typing import TYPE_CHECKING

from opentelemetry.metrics import Counter, Meter, get_meter

from ..models import CostReport, UsageRecord

if TYPE_CHECKING:
    from opentelemetry.sdk.metrics import MeterProvider

logger = logging.getLogger(__name__)

METRIC_COST = "openai.cost.usd"
METRIC_TOKENS = "openai.tokens"
METRIC_CALLS = "openai.calls"


class AzureMonitorReporter:
    """Emits cost/token usage to OpenTelemetry metric instruments.

    :param meter: an OpenTelemetry Meter. If omitted, the global meter is used -
        configure a MeterProvider (e.g. via ``configure_azure_monitor``) first,
        otherwise metrics go to a no-op provider and are discarded.
    """

    def __init__(self, meter: Meter | None = None) -> None:
        active_meter = meter or get_meter("openai_cost_guard")
        self._cost_counter: Counter = active_meter.create_counter(
            METRIC_COST,
            unit="USD",
            description="Azure OpenAI cost per call in US dollars",
        )
        self._token_counter: Counter = active_meter.create_counter(
            METRIC_TOKENS,
            unit="token",
            description="Total tokens (prompt + completion) per call",
        )
        self._call_counter: Counter = active_meter.create_counter(
            METRIC_CALLS,
            unit="call",
            description="Number of Azure OpenAI calls",
        )

    def emit(self, record: UsageRecord) -> None:
        """Emit a single usage record. Matches the ``on_record`` callback signature."""
        attributes = {
            "model": record.model,
            "endpoint": record.endpoint or "unknown",
        }
        self._cost_counter.add(record.total_cost, attributes)
        self._token_counter.add(record.total_tokens, attributes)
        self._call_counter.add(1, attributes)

    def emit_report(self, report: CostReport) -> None:
        """Emit every record in a report. Do not combine with streaming ``emit``."""
        for record in report.records:
            self.emit(record)


def configure_azure_monitor(
    connection_string: str | None = None,
    export_interval_millis: int = 60_000,
) -> "MeterProvider":
    """Set up a global OpenTelemetry MeterProvider that exports to Application Insights.

    Call this once at application startup, before creating an AzureMonitorReporter.

    :param connection_string: Application Insights connection string. Falls back to the
        ``APPLICATIONINSIGHTS_CONNECTION_STRING`` environment variable.
    :param export_interval_millis: how often metrics are pushed to Azure Monitor.
    :raises RuntimeError: if no connection string is available.
    :raises ImportError: if the ``[azure]`` extra is not installed.
    """
    conn = connection_string or os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not conn:
        raise RuntimeError(
            "No Application Insights connection string. Pass connection_string or set "
            "APPLICATIONINSIGHTS_CONNECTION_STRING."
        )

    try:
        from azure.monitor.opentelemetry.exporter import AzureMonitorMetricExporter
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Azure Monitor export requires the [azure] extra. "
            'Install with: pip install "openai-cost-guard[azure]"'
        ) from exc

    from opentelemetry.metrics import set_meter_provider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    exporter = AzureMonitorMetricExporter(connection_string=conn)
    reader = PeriodicExportingMetricReader(
        exporter, export_interval_millis=export_interval_millis
    )
    provider = MeterProvider(metric_readers=[reader])
    set_meter_provider(provider)
    logger.info("Azure Monitor metric export configured (interval=%dms)", export_interval_millis)
    return provider
