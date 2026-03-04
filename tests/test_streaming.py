"""Tests for streamed-response cost capture.

Uses real ChatCompletionChunk SDK objects: content chunks carry usage=None and the
final chunk carries a populated CompletionUsage with empty choices - exactly the shape
the Azure OpenAI SDK produces when stream_options={"include_usage": True}.
"""
import logging
from collections.abc import AsyncIterator, Iterator

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.completion_usage import CompletionUsage

from openai_cost_guard.decorators import get_default_tracker, reset_default_tracker
from openai_cost_guard.streaming import track_cost_stream, track_cost_stream_async
from openai_cost_guard.tracker import CostTracker


def _content_chunk(text: str, model: str = "gpt-4o") -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chunk",
        object="chat.completion.chunk",
        created=1_700_000_000,
        model=model,
        choices=[Choice(index=0, delta=ChoiceDelta(content=text), finish_reason=None)],
    )


def _usage_chunk(
    model: str = "gpt-4o", prompt_tokens: int = 100, completion_tokens: int = 50
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chunk",
        object="chat.completion.chunk",
        created=1_700_000_000,
        model=model,
        choices=[],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _sync_stream(*chunks: ChatCompletionChunk) -> Iterator[ChatCompletionChunk]:
    yield from chunks


async def _async_stream(*chunks: ChatCompletionChunk) -> AsyncIterator[ChatCompletionChunk]:
    for chunk in chunks:
        yield chunk


def setup_function() -> None:
    reset_default_tracker()


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

def test_sync_stream_records_usage_chunk() -> None:
    tracker = CostTracker()

    @track_cost_stream(tracker=tracker, endpoint="chat")
    def stream():  # type: ignore[no-untyped-def]
        return _sync_stream(
            _content_chunk("Hel"),
            _content_chunk("lo"),
            _usage_chunk(prompt_tokens=200, completion_tokens=100),
        )

    chunks = list(stream())

    assert len(chunks) == 3  # all chunks passed through to the caller
    report = tracker.report()
    assert len(report.records) == 1
    assert report.records[0].total_tokens == 300
    assert report.records[0].endpoint == "chat"


def test_sync_stream_yields_all_chunks_in_order() -> None:
    tracker = CostTracker()

    @track_cost_stream(tracker=tracker)
    def stream():  # type: ignore[no-untyped-def]
        return _sync_stream(_content_chunk("a"), _content_chunk("b"), _usage_chunk())

    contents = [c.choices[0].delta.content for c in stream() if c.choices]
    assert contents == ["a", "b"]


def test_sync_stream_records_lazily_during_iteration() -> None:
    """Nothing is recorded until the stream is actually consumed."""
    tracker = CostTracker()

    @track_cost_stream(tracker=tracker)
    def stream():  # type: ignore[no-untyped-def]
        return _sync_stream(_content_chunk("a"), _usage_chunk())

    gen = stream()
    # Decorator has run, but no iteration yet
    assert len(tracker.report().records) == 0

    list(gen)  # consume
    assert len(tracker.report().records) == 1


def test_sync_stream_without_usage_records_nothing(caplog) -> None:  # type: ignore[no-untyped-def]
    tracker = CostTracker()

    @track_cost_stream(tracker=tracker, endpoint="no-usage")
    def stream():  # type: ignore[no-untyped-def]
        return _sync_stream(_content_chunk("a"), _content_chunk("b"))

    with caplog.at_level(logging.DEBUG, logger="openai_cost_guard.streaming"):
        list(stream())

    assert len(tracker.report().records) == 0
    assert any("no usage chunk" in m for m in caplog.messages)


def test_sync_stream_uses_default_tracker() -> None:
    @track_cost_stream()
    def stream():  # type: ignore[no-untyped-def]
        return _sync_stream(_usage_chunk())

    list(stream())
    assert len(get_default_tracker().report().records) == 1


def test_sync_stream_uses_context_tracker() -> None:
    from openai_cost_guard.context import reset_current_tracker, set_current_tracker

    ctx_tracker = CostTracker()
    token = set_current_tracker(ctx_tracker)
    try:

        @track_cost_stream()
        def stream():  # type: ignore[no-untyped-def]
            return _sync_stream(_usage_chunk())

        list(stream())
    finally:
        reset_current_tracker(token)

    assert len(ctx_tracker.report().records) == 1
    assert len(get_default_tracker().report().records) == 0


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------

async def test_async_stream_records_usage_chunk() -> None:
    tracker = CostTracker()

    @track_cost_stream_async(tracker=tracker, endpoint="chat")
    async def stream():  # type: ignore[no-untyped-def]
        return _async_stream(
            _content_chunk("Hel"),
            _content_chunk("lo"),
            _usage_chunk(prompt_tokens=200, completion_tokens=100),
        )

    chunks = [c async for c in stream()]

    assert len(chunks) == 3
    report = tracker.report()
    assert len(report.records) == 1
    assert report.records[0].total_tokens == 300


async def test_async_stream_yields_all_content() -> None:
    tracker = CostTracker()

    @track_cost_stream_async(tracker=tracker)
    async def stream():  # type: ignore[no-untyped-def]
        return _async_stream(_content_chunk("a"), _content_chunk("b"), _usage_chunk())

    contents = [c.choices[0].delta.content async for c in stream() if c.choices]
    assert contents == ["a", "b"]


async def test_async_stream_without_usage_records_nothing() -> None:
    tracker = CostTracker()

    @track_cost_stream_async(tracker=tracker)
    async def stream():  # type: ignore[no-untyped-def]
        return _async_stream(_content_chunk("a"))

    _ = [c async for c in stream()]
    assert len(tracker.report().records) == 0


async def test_async_stream_uses_default_tracker() -> None:
    @track_cost_stream_async()
    async def stream():  # type: ignore[no-untyped-def]
        return _async_stream(_usage_chunk())

    _ = [c async for c in stream()]
    assert len(get_default_tracker().report().records) == 1
