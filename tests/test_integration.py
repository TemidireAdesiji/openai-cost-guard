"""Integration tests.

These tests exercise the full stack using real openai SDK types and a
respx-mocked HTTP layer - no real Azure credentials needed.

Coverage goals:
- _extract_usage works with actual openai SDK objects (not SimpleNamespace fakes)
- @track_cost integrates correctly with a live AzureOpenAI client call
- The full pipeline (HTTP mock -> client -> decorator -> tracker -> report -> reporter) works
- CostTracker is safe under concurrent writes from multiple threads
"""
import json
import logging
import threading
import time

import pytest
import respx
from httpx import Response
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage
from openai.types.embedding import Embedding

from openai_cost_guard import CostTracker, track_cost
from openai_cost_guard.reporters.console import print_report

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AZURE_ENDPOINT = "https://test-resource.openai.azure.com"
API_VERSION = "2024-02-01"
DEPLOYMENT = "gpt-4o"


def _make_chat_completion(
    model: str = "gpt-4o",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    content: str = "Hello.",
) -> ChatCompletion:
    """Construct a real ChatCompletion SDK object without an HTTP call."""
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1_700_000_000,
        model=model,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _make_embedding_response(
    model: str = "text-embedding-3-small",
    prompt_tokens: int = 200,
) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        object="list",
        model=model,
        data=[Embedding(object="embedding", index=0, embedding=[0.1, 0.2, 0.3])],
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )


def _chat_json(
    model: str = "gpt-4o",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> str:
    return json.dumps({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello from mock."},
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "system_fingerprint": None,
    })


# ---------------------------------------------------------------------------
# 1. Real SDK types flow through _extract_usage correctly
# ---------------------------------------------------------------------------

class TestRealSDKTypes:
    def test_chat_completion_cost_recorded(self) -> None:
        tracker = CostTracker()

        @track_cost(tracker=tracker)
        def call() -> ChatCompletion:
            return _make_chat_completion(
                model="gpt-4o", prompt_tokens=1_000_000, completion_tokens=1_000_000
            )

        call()
        report = tracker.report()
        assert len(report.records) == 1
        assert report.records[0].input_cost == pytest.approx(2.50)
        assert report.records[0].output_cost == pytest.approx(10.00)
        assert report.records[0].total_cost == pytest.approx(12.50)

    def test_embedding_response_cost_recorded(self) -> None:
        tracker = CostTracker()

        @track_cost(tracker=tracker, endpoint="embed")
        def embed() -> CreateEmbeddingResponse:
            return _make_embedding_response(
                model="text-embedding-3-small", prompt_tokens=1_000_000
            )

        embed()
        report = tracker.report()
        assert len(report.records) == 1
        # text-embedding-3-small = $0.022 / 1M tokens
        assert report.records[0].input_cost == pytest.approx(0.022)
        assert report.records[0].output_cost == pytest.approx(0.0)
        assert report.records[0].endpoint == "embed"

    def test_model_name_from_sdk_object_overrides_deployment_name(self) -> None:
        # The SDK response carries the actual model name, which may differ from
        # the Azure deployment name used in the request.
        tracker = CostTracker()

        @track_cost(tracker=tracker)
        def call() -> ChatCompletion:
            return _make_chat_completion(model="gpt-4o-mini", prompt_tokens=500_000)

        call()
        assert tracker.report().records[0].model == "gpt-4o-mini"

    def test_multiple_sdk_calls_aggregate_correctly(self) -> None:
        tracker = CostTracker()

        @track_cost(tracker=tracker, endpoint="chat")
        def chat() -> ChatCompletion:
            return _make_chat_completion(model="gpt-4o", prompt_tokens=100, completion_tokens=50)

        @track_cost(tracker=tracker, endpoint="embed")
        def embed() -> CreateEmbeddingResponse:
            return _make_embedding_response(model="text-embedding-3-small", prompt_tokens=200)

        chat()
        chat()
        embed()

        report = tracker.report()
        assert len(report.records) == 3
        assert report.total_prompt_tokens == 400
        assert report.total_completion_tokens == 100


# ---------------------------------------------------------------------------
# 2. Full HTTP mock - real AzureOpenAI client call
# ---------------------------------------------------------------------------

class TestLiveClientWithMockedHTTP:
    """Uses respx to intercept httpx at the transport layer.

    The AzureOpenAI client constructs the real request, respx returns a
    realistic JSON payload, and the SDK parses it into a real ChatCompletion -
    exactly what happens in production, minus the network hop.
    """

    @respx.mock
    def test_decorator_captures_real_client_response(self) -> None:
        respx.post(url__regex=r".*chat/completions.*").mock(
            return_value=Response(200, text=_chat_json(prompt_tokens=512, completion_tokens=128))
        )

        client = AzureOpenAI(
            api_key="test-key",
            api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
        )
        tracker = CostTracker()

        @track_cost(tracker=tracker, endpoint="test-call")
        def call() -> ChatCompletion:
            return client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[{"role": "user", "content": "hi"}],
            )

        call()

        report = tracker.report()
        assert len(report.records) == 1
        rec = report.records[0]
        assert rec.prompt_tokens == 512
        assert rec.completion_tokens == 128
        assert rec.total_tokens == 640
        assert rec.endpoint == "test-call"
        assert rec.total_cost > 0

    @respx.mock
    def test_multiple_client_calls_accumulate_in_tracker(self) -> None:
        respx.post(url__regex=r".*chat/completions.*").mock(
            return_value=Response(200, text=_chat_json(prompt_tokens=100, completion_tokens=50))
        )

        client = AzureOpenAI(
            api_key="test-key",
            api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
        )
        tracker = CostTracker()

        @track_cost(tracker=tracker)
        def call() -> ChatCompletion:
            return client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[{"role": "user", "content": "hi"}],
            )

        for _ in range(5):
            call()

        report = tracker.report()
        assert len(report.records) == 5
        assert report.total_prompt_tokens == 500
        assert report.total_completion_tokens == 250

    @respx.mock
    def test_report_total_cost_matches_sum_of_records(self) -> None:
        respx.post(url__regex=r".*chat/completions.*").mock(
            return_value=Response(
                200, text=_chat_json(model="gpt-4o", prompt_tokens=1_000, completion_tokens=500)
            )
        )

        client = AzureOpenAI(
            api_key="test-key",
            api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
        )
        tracker = CostTracker()

        @track_cost(tracker=tracker)
        def call() -> ChatCompletion:
            return client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[{"role": "user", "content": "test"}],
            )

        call()
        call()

        report = tracker.report()
        expected = sum(r.total_cost for r in report.records)
        assert report.total_cost == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 3. Full pipeline - HTTP mock -> client -> decorator -> tracker -> reporter
# ---------------------------------------------------------------------------

class TestFullPipeline:
    @respx.mock
    def test_end_to_end_pipeline_produces_report_output(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        respx.post(url__regex=r".*chat/completions.*").mock(
            return_value=Response(
                200, text=_chat_json(prompt_tokens=1_000, completion_tokens=500)
            )
        )

        client = AzureOpenAI(
            api_key="test-key",
            api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
        )
        tracker = CostTracker()

        @track_cost(tracker=tracker, endpoint="pipeline-test")
        def call() -> ChatCompletion:
            return client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[{"role": "user", "content": "test"}],
            )

        call()
        call()

        with caplog.at_level(logging.INFO):
            print_report(tracker.report())

        combined = "\n".join(caplog.messages)
        assert "gpt-4o" in combined
        assert "TOTAL" in combined
        assert "2" in combined  # 2 calls


# ---------------------------------------------------------------------------
# 4. Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_record_calls_all_stored(self) -> None:
        tracker = CostTracker()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                tracker.record("gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(tracker.report().records) == 50

    def test_concurrent_record_and_report_never_raises(self) -> None:
        tracker = CostTracker()
        errors: list[Exception] = []

        def recorder() -> None:
            for _ in range(20):
                try:
                    tracker.record("gpt-4o", prompt_tokens=50, completion_tokens=25)
                except Exception as exc:
                    errors.append(exc)

        def reporter() -> None:
            for _ in range(20):
                try:
                    tracker.report()
                except Exception as exc:
                    errors.append(exc)

        threads = [
            *[threading.Thread(target=recorder) for _ in range(5)],
            *[threading.Thread(target=reporter) for _ in range(3)],
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_reset_does_not_corrupt_state(self) -> None:
        tracker = CostTracker()
        errors: list[Exception] = []

        def recorder() -> None:
            for _ in range(10):
                try:
                    tracker.record("gpt-4o-mini", prompt_tokens=10)
                    time.sleep(0)  # yield to other threads
                except Exception as exc:
                    errors.append(exc)

        def resetter() -> None:
            for _ in range(5):
                try:
                    tracker.reset()
                    time.sleep(0)
                except Exception as exc:
                    errors.append(exc)

        threads = [
            *[threading.Thread(target=recorder) for _ in range(4)],
            threading.Thread(target=resetter),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        # State must be internally consistent regardless of interleaving
        report = tracker.report()
        assert report.total_cost == pytest.approx(
            sum(r.total_cost for r in report.records)
        )
