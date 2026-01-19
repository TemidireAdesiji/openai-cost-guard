import logging
from collections.abc import Callable
from threading import Lock

from .models import DEFAULT_PRICING, BudgetConfig, CostReport, ModelPricing, UsageRecord

logger = logging.getLogger(__name__)

OnRecord = Callable[[UsageRecord], None]


class BudgetExceededError(Exception):
    """Raised when a recorded call would push spending past the configured budget limit."""

    def __init__(self, spent: float, limit: float) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(f"Budget exceeded: ${spent:.6f} spent, limit is ${limit:.2f}")


class UnknownModelError(Exception):
    """Raised when a model name has no pricing entry and strict mode is enabled."""


class CostTracker:
    """Thread-safe recorder for Azure OpenAI API call costs.

    Usage::

        tracker = CostTracker()
        record = tracker.record("gpt-4o", prompt_tokens=100, completion_tokens=50)
        report = tracker.report()
    """

    def __init__(
        self,
        pricing: dict[str, ModelPricing] | None = None,
        budget: BudgetConfig | None = None,
        strict: bool = False,
        on_record: OnRecord | None = None,
    ) -> None:
        # Merge caller-supplied pricing on top of defaults so only overrides are needed
        self._pricing: dict[str, ModelPricing] = {**DEFAULT_PRICING, **(pricing or {})}
        self._budget = budget
        self._strict = strict
        # Called once per successfully recorded call, after the budget check passes.
        # Used to stream usage to external sinks (e.g. AzureMonitorReporter.emit).
        self._on_record = on_record
        self._records: list[UsageRecord] = []
        # Running total so the budget check is O(1) per call rather than re-summing every
        # stored record (which would make record() O(n) and a budgeted run O(n^2)).
        self._total_cost = 0.0
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int = 0,
        endpoint: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> UsageRecord:
        """Calculate cost for one API call and store it.

        Raises BudgetExceededError if a budget is configured and this call
        pushes total spend past the limit.
        """
        pricing = self._resolve_pricing(model)
        input_cost = (prompt_tokens / 1_000_000) * pricing.input_per_million
        output_cost = (completion_tokens / 1_000_000) * pricing.output_per_million
        total_cost = input_cost + output_cost

        record = UsageRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            endpoint=endpoint,
            metadata=metadata or {},
        )

        with self._lock:
            self._records.append(record)
            self._total_cost += total_cost
            self._check_budget()

        # Fired outside the lock so a slow sink cannot stall other recorders.
        # Reached only when the budget check above did not raise.
        if self._on_record is not None:
            self._on_record(record)

        logger.debug(
            "Recorded cost for %s: $%.6f (%d prompt + %d completion tokens)",
            model,
            total_cost,
            prompt_tokens,
            completion_tokens,
        )
        return record

    def report(self) -> CostReport:
        """Return a snapshot of all recorded usage."""
        with self._lock:
            return CostReport.from_records(list(self._records))

    def reset(self) -> None:
        """Clear all recorded usage. Does not reset the budget config."""
        with self._lock:
            self._records.clear()
            self._total_cost = 0.0
        logger.debug("CostTracker reset")

    def add_pricing(self, pricing: ModelPricing) -> None:
        """Register or override pricing for a model at runtime."""
        self._pricing[pricing.model] = pricing

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_pricing(self, model: str) -> ModelPricing:
        if model in self._pricing:
            return self._pricing[model]

        # Prefix match - handles versioned deployment names like "gpt-4o-mini-2024-07-18".
        # Try the LONGEST matching key first, so a specific entry ("gpt-4o-mini") wins over
        # a shorter one that is also a prefix ("gpt-4o"); otherwise a versioned mini model
        # could be mispriced as its more expensive parent.
        matches = [key for key in self._pricing if model.startswith(key)]
        if matches:
            key = max(matches, key=len)
            logger.debug("Model %r matched pricing entry %r by prefix", model, key)
            return self._pricing[key]

        if self._strict:
            raise UnknownModelError(
                f"No pricing found for model {model!r}. "
                "Add it via CostTracker(pricing={...}) or tracker.add_pricing(...)."
            )

        logger.warning(
            "No pricing found for model %r - cost recorded as $0.00. "
            "Pass strict=True to raise instead.",
            model,
        )
        return ModelPricing(model=model, input_per_million=0.0, output_per_million=0.0)

    def _check_budget(self) -> None:
        if self._budget is None:
            return

        total = self._total_cost  # running total, kept current in record()
        limit = self._budget.limit_usd
        warn_threshold = limit * (self._budget.warn_at_percent / 100)

        if total > limit:
            raise BudgetExceededError(spent=total, limit=limit)

        if total >= warn_threshold:
            logger.warning(
                "Budget warning: $%.4f of $%.2f limit consumed (%.1f%%)",
                total,
                limit,
                (total / limit) * 100,
            )
