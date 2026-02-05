"""Context-local tracker resolution.

The FastAPI middleware sets a fresh CostTracker for the duration of each
request via a contextvar. The ``@track_cost`` decorator, when given no explicit
tracker, resolves in this order:

    1. the tracker bound to the current context (set by the middleware), else
    2. the module-level default tracker.

This lets the same decorated function record into a per-request tracker when
called inside a request, and into the global tracker otherwise - no plumbing
required at the call site.
"""
from contextvars import ContextVar, Token

from .tracker import CostTracker

# None means "no request-scoped tracker active - use the default".
_current_tracker: ContextVar[CostTracker | None] = ContextVar(
    "openai_cost_guard_current_tracker", default=None
)


def get_current_tracker() -> CostTracker | None:
    """Return the context-local tracker, or None if none is set."""
    return _current_tracker.get()


def set_current_tracker(tracker: CostTracker) -> Token[CostTracker | None]:
    """Bind a tracker to the current context. Returns a token for reset_current_tracker."""
    return _current_tracker.set(tracker)


def reset_current_tracker(token: Token[CostTracker | None]) -> None:
    """Restore the context-local tracker to its previous value."""
    _current_tracker.reset(token)
