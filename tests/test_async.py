"""Tests for the async decorators.

asyncio_mode = "auto" (set in pyproject) runs ``async def test_*`` without markers.
"""
import asyncio
from types import SimpleNamespace

from openai_cost_guard.decorators import (
    get_default_tracker,
    reset_default_tracker,
    track_cost_async,
    track_cost_method_async,
)
from openai_cost_guard.tracker import CostTracker


def _fake_response(model: str, prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def setup_function() -> None:
    reset_default_tracker()


async def test_async_decorator_records_to_explicit_tracker() -> None:
    tracker = CostTracker()

    @track_cost_async(tracker=tracker, endpoint="chat")
    async def call() -> SimpleNamespace:
        await asyncio.sleep(0)
        return _fake_response("gpt-4o", 1000, 500)

    await call()
    report = tracker.report()
    assert len(report.records) == 1
    assert report.records[0].endpoint == "chat"
    assert report.records[0].total_tokens == 1500


async def test_async_decorator_records_to_default_tracker() -> None:
    @track_cost_async()
    async def call() -> SimpleNamespace:
        return _fake_response("gpt-4o-mini", 200, 100)

    await call()
    assert len(get_default_tracker().report().records) == 1


async def test_async_decorator_preserves_return_value() -> None:
    expected = _fake_response("gpt-4o", 100, 50)

    @track_cost_async()
    async def call() -> SimpleNamespace:
        return expected

    result = await call()
    assert result is expected


async def test_async_decorator_skips_response_without_usage() -> None:
    @track_cost_async()
    async def call() -> SimpleNamespace:
        return SimpleNamespace(no_usage=True)

    await call()
    assert len(get_default_tracker().report().records) == 0


async def test_async_decorator_preserves_function_name() -> None:
    @track_cost_async()
    async def my_async_fn() -> SimpleNamespace:
        return _fake_response("gpt-4o", 10, 5)

    assert my_async_fn.__name__ == "my_async_fn"


async def test_async_method_decorator() -> None:
    class Service:
        def __init__(self) -> None:
            self.cost_tracker = CostTracker()

        @track_cost_method_async()
        async def call(self) -> SimpleNamespace:
            await asyncio.sleep(0)
            return _fake_response("gpt-4o", 500, 250)

    svc = Service()
    await svc.call()
    assert len(svc.cost_tracker.report().records) == 1


async def test_async_method_custom_attr() -> None:
    class Service:
        def __init__(self) -> None:
            self.my_tracker = CostTracker()

        @track_cost_method_async(tracker_attr="my_tracker")
        async def call(self) -> SimpleNamespace:
            return _fake_response("gpt-4o", 100, 50)

    svc = Service()
    await svc.call()
    assert len(svc.my_tracker.report().records) == 1


async def test_concurrent_async_calls_all_recorded() -> None:
    tracker = CostTracker()

    @track_cost_async(tracker=tracker)
    async def call() -> SimpleNamespace:
        await asyncio.sleep(0)
        return _fake_response("gpt-4o-mini", 100, 50)

    await asyncio.gather(*[call() for _ in range(20)])
    assert len(tracker.report().records) == 20


async def test_async_decorator_uses_context_tracker() -> None:
    """When no explicit tracker is given, a context-bound tracker is used."""
    from openai_cost_guard.context import reset_current_tracker, set_current_tracker

    context_tracker = CostTracker()
    token = set_current_tracker(context_tracker)
    try:

        @track_cost_async()
        async def call() -> SimpleNamespace:
            return _fake_response("gpt-4o", 100, 50)

        await call()
    finally:
        reset_current_tracker(token)

    assert len(context_tracker.report().records) == 1
    # Default tracker untouched
    assert len(get_default_tracker().report().records) == 0
