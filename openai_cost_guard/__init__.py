"""openai-cost-guard: cost tracking and budget enforcement for Azure OpenAI API calls."""

from .context import get_current_tracker, set_current_tracker
from .decorators import (
    get_default_tracker,
    reset_default_tracker,
    track_cost,
    track_cost_async,
    track_cost_method,
    track_cost_method_async,
)
from .models import BudgetConfig, CostReport, ModelPricing, UsageRecord
from .streaming import track_cost_stream, track_cost_stream_async
from .tracker import BudgetExceededError, CostTracker, UnknownModelError

# Note: CostGuardMiddleware (openai_cost_guard.middleware) requires the [fastapi]
# extra, and AzureMonitorReporter (openai_cost_guard.reporters.azure_monitor) requires
# the [azure] extra. Both are intentionally not imported here so the core package has
# no web-framework or telemetry dependency.

__version__ = "0.4.0"

__all__ = [
    "BudgetConfig",
    "BudgetExceededError",
    "CostReport",
    "CostTracker",
    "ModelPricing",
    "UnknownModelError",
    "UsageRecord",
    "get_current_tracker",
    "get_default_tracker",
    "reset_default_tracker",
    "set_current_tracker",
    "track_cost",
    "track_cost_async",
    "track_cost_method",
    "track_cost_method_async",
    "track_cost_stream",
    "track_cost_stream_async",
]
