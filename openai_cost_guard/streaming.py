"""Cost capture for streamed Azure OpenAI responses.

A streamed completion (``stream=True``) returns an iterator of chunks rather than a
single response with a ``.usage`` field, so the plain ``@track_cost`` decorator records
nothing. Usage for a stream is delivered on the *final* chunk, and only when the caller
requests it::

    client.chat.completions.create(..., stream=True, stream_options={"include_usage": True})

The decorators here wrap the returned iterator. Every chunk is passed through to the
caller unchanged; when a chunk carrying usage arrives, its cost is recorded. Recording is
therefore lazy - it happens as the caller consumes the stream, not when the call returns.

If the stream yields no usage chunk (the caller did not set ``include_usage``), nothing is
recorded and a debug message explains why.
"""
import functools
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

from .decorators import _extract_usage, _resolve_tracker
from .tracker import CostTracker

logger = logging.getLogger(__name__)


def track_cost_stream(
    tracker: CostTracker | None = None,
    endpoint: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Callable[[Callable[..., Iterator[Any]]], Callable[..., Iterator[Any]]]:
    """Wrap a function returning a sync stream of chunks, recording cost on consumption.

    Tracker resolution matches ``track_cost`` (explicit > request scope > default).
    The decorated function still returns an iterator - iterate it exactly as before;
    the only change is that consuming it records cost.

    Usage::

        @track_cost_stream(endpoint="chat")
        def stream_chat(client, messages):
            return client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
            )

        for chunk in stream_chat(client, [...]):
            ...
    """

    def decorator(fn: Callable[..., Iterator[Any]]) -> Callable[..., Iterator[Any]]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Iterator[Any]:
            stream = fn(*args, **kwargs)
            resolved = _resolve_tracker(tracker)
            label = endpoint or fn.__qualname__
            return _consume_sync(stream, resolved, label, metadata)

        return wrapper

    return decorator


def track_cost_stream_async(
    tracker: CostTracker | None = None,
    endpoint: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Callable[[Callable[..., Awaitable[AsyncIterator[Any]]]], Callable[..., AsyncIterator[Any]]]:
    """Async counterpart of ``track_cost_stream``.

    The wrapped function is consumed directly with ``async for`` - the decorator awaits
    the underlying coroutine to obtain the async stream, then proxies its chunks::

        @track_cost_stream_async(endpoint="chat")
        async def stream_chat(client, messages):
            return await client.chat.completions.create(
                model="gpt-4o", messages=messages,
                stream=True, stream_options={"include_usage": True},
            )

        async for chunk in stream_chat(client, [...]):
            ...
    """

    def decorator(
        fn: Callable[..., Awaitable[AsyncIterator[Any]]],
    ) -> Callable[..., AsyncIterator[Any]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            stream = await fn(*args, **kwargs)
            resolved = _resolve_tracker(tracker)
            label = endpoint or fn.__qualname__
            recorded = False
            async for chunk in stream:
                if _record_chunk(chunk, resolved, label, metadata):
                    recorded = True
                yield chunk
            if not recorded:
                _warn_no_usage(label)

        return wrapper

    return decorator


def _consume_sync(
    stream: Iterator[Any],
    tracker: CostTracker,
    endpoint: str,
    metadata: dict[str, str] | None,
) -> Iterator[Any]:
    recorded = False
    for chunk in stream:
        if _record_chunk(chunk, tracker, endpoint, metadata):
            recorded = True
        yield chunk
    if not recorded:
        _warn_no_usage(endpoint)


def _record_chunk(
    chunk: Any,
    tracker: CostTracker,
    endpoint: str,
    metadata: dict[str, str] | None,
) -> bool:
    """Record the chunk if it carries usage. Returns True when a record was made."""
    usage = _extract_usage(chunk)
    if usage is None:
        return False
    model, prompt_tokens, completion_tokens = usage
    tracker.record(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        endpoint=endpoint,
        metadata=metadata,
    )
    return True


def _warn_no_usage(endpoint: str) -> None:
    logger.debug(
        "Stream %r produced no usage chunk - pass stream_options={'include_usage': True} "
        "to capture cost from streamed responses.",
        endpoint,
    )
