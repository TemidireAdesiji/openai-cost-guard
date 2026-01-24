import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from .tracker import CostTracker

logger = logging.getLogger(__name__)

# Singleton tracker used when no tracker is passed to @track_cost
_default_tracker = CostTracker()


def get_default_tracker() -> CostTracker:
    """Return the module-level default tracker."""
    return _default_tracker


def reset_default_tracker() -> None:
    """Reset the module-level default tracker. Useful in tests."""
    _default_tracker.reset()


@runtime_checkable
class HasUsage(Protocol):
    """Structural protocol for any OpenAI-compatible response that carries usage data."""

    class usage:  # noqa: N801
        prompt_tokens: int
        completion_tokens: int

    model: str


def _extract_usage(response: Any) -> tuple[str, int, int] | None:
    """Pull (model, prompt_tokens, completion_tokens) from a response object.

    Returns None if the response does not carry usage info (e.g. streaming).
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    model: str = getattr(response, "model", "unknown")
    prompt_tokens: int = getattr(usage, "prompt_tokens", 0)
    completion_tokens: int = getattr(usage, "completion_tokens", 0)
    return model, prompt_tokens, completion_tokens


def _resolve_tracker(explicit: CostTracker | None) -> CostTracker:
    """Resolve which tracker a call should record into.

    Order: explicit argument > context-local tracker (set by middleware) > default.
    The context lookup is imported lazily to avoid a circular import at module load.
    """
    if explicit is not None:
        return explicit

    from .context import get_current_tracker

    return get_current_tracker() or _default_tracker


def track_cost(
    tracker: CostTracker | None = None,
    endpoint: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Callable[..., Any]:
    """Decorator that records the Azure OpenAI cost for any call returning a response
    with a ``.usage`` attribute.

    The tracker is resolved at call time: an explicit ``tracker`` wins, otherwise
    the request-scoped tracker set by ``CostGuardMiddleware`` is used, otherwise
    the module-level default tracker.

    Usage::

        @track_cost()
        def call_openai(client, prompt):
            return client.chat.completions.create(...)

        # With an explicit tracker and endpoint label:
        @track_cost(tracker=my_tracker, endpoint="summarise")
        def summarise(text):
            ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            response = fn(*args, **kwargs)
            usage = _extract_usage(response)
            if usage is None:
                logger.debug(
                    "%s returned a response with no usage data - skipping cost record",
                    fn.__qualname__,
                )
                return response

            model, prompt_tokens, completion_tokens = usage
            _resolve_tracker(tracker).record(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                endpoint=endpoint or fn.__qualname__,
                metadata=metadata,
            )
            return response

        return wrapper

    return decorator


def track_cost_method(
    tracker_attr: str = "cost_tracker",
    endpoint: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Callable[..., Any]:
    """Like ``track_cost`` but reads the tracker from an instance attribute.

    Useful when the tracker lives on a class::

        class MyService:
            def __init__(self):
                self.cost_tracker = CostTracker()

            @track_cost_method()
            def summarise(self, text):
                return self.client.chat.completions.create(...)
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            active_tracker: CostTracker = getattr(self, tracker_attr)
            response = fn(self, *args, **kwargs)
            usage = _extract_usage(response)
            if usage is None:
                return response

            model, prompt_tokens, completion_tokens = usage
            active_tracker.record(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                endpoint=endpoint or fn.__qualname__,
                metadata=metadata,
            )
            return response

        return wrapper

    return decorator


def track_cost_async(
    tracker: CostTracker | None = None,
    endpoint: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Async counterpart of ``track_cost`` for coroutine functions.

    Tracker resolution is identical (explicit > request scope > default). Recording
    is a fast, non-blocking, thread-safe operation, so it is safe to call from the
    event loop without awaiting.

    Usage::

        @track_cost_async(endpoint="chat")
        async def call_openai(client, messages):
            return await client.chat.completions.create(...)
    """

    def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            response = await fn(*args, **kwargs)
            usage = _extract_usage(response)
            if usage is None:
                logger.debug(
                    "%s returned a response with no usage data - skipping cost record",
                    fn.__qualname__,
                )
                return response

            model, prompt_tokens, completion_tokens = usage
            _resolve_tracker(tracker).record(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                endpoint=endpoint or fn.__qualname__,
                metadata=metadata,
            )
            return response

        return wrapper

    return decorator


def track_cost_method_async(
    tracker_attr: str = "cost_tracker",
    endpoint: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Async counterpart of ``track_cost_method`` for coroutine methods.

    Reads the tracker from ``self.<tracker_attr>``::

        class MyService:
            def __init__(self):
                self.cost_tracker = CostTracker()

            @track_cost_method_async()
            async def summarise(self, text):
                return await self.client.chat.completions.create(...)
    """

    def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(fn)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            active_tracker: CostTracker = getattr(self, tracker_attr)
            response = await fn(self, *args, **kwargs)
            usage = _extract_usage(response)
            if usage is None:
                return response

            model, prompt_tokens, completion_tokens = usage
            active_tracker.record(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                endpoint=endpoint or fn.__qualname__,
                metadata=metadata,
            )
            return response

        return wrapper

    return decorator
