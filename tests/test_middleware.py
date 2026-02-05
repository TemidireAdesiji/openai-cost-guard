"""Tests for CostGuardMiddleware against a real FastAPI app via TestClient.

The endpoint calls a @track_cost-decorated function with no explicit tracker, so
the recording must flow through the request-scoped contextvar set by the middleware.
This is the core integration the middleware exists to provide.
"""
import logging
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.types import Scope

from openai_cost_guard import track_cost
from openai_cost_guard.middleware import HEADER_COST, HEADER_TOKENS, CostGuardMiddleware
from openai_cost_guard.models import BudgetConfig, CostReport
from openai_cost_guard.tracker import BudgetExceededError


def _fake_response(model: str, prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


@track_cost(endpoint="model-call")
def _call_model(model: str, prompt: int, completion: int) -> SimpleNamespace:
    """No explicit tracker - records into whatever the context provides."""
    return _fake_response(model, prompt, completion)


def _make_app(
    budget_per_request: BudgetConfig | None = None,
    add_headers: bool = True,
    strict: bool = False,
    on_complete: Callable[[Scope, CostReport], None] | None = None,
) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CostGuardMiddleware,
        budget_per_request=budget_per_request,
        add_headers=add_headers,
        strict=strict,
        on_complete=on_complete,
    )

    @app.post("/chat")
    def chat() -> dict[str, str]:
        _call_model("gpt-4o", 1000, 500)
        return {"status": "ok"}

    @app.post("/double")
    def double() -> dict[str, str]:
        _call_model("gpt-4o", 1000, 500)
        _call_model("gpt-4o-mini", 200, 100)
        return {"status": "ok"}

    @app.post("/state")
    def state(request: Request) -> dict[str, float]:
        _call_model("gpt-4o", 1000, 500)
        report: CostReport = request.state.cost_tracker.report()
        return {"cost": report.total_cost, "tokens": float(report.total_tokens)}

    @app.get("/noop")
    def noop() -> dict[str, str]:
        return {"status": "ok"}

    return app


def test_cost_headers_present_on_response() -> None:
    client = TestClient(_make_app())
    resp = client.post("/chat")

    assert resp.status_code == 200
    assert HEADER_COST in resp.headers
    assert HEADER_TOKENS in resp.headers
    assert resp.headers[HEADER_TOKENS] == "1500"
    assert float(resp.headers[HEADER_COST]) > 0


def test_cost_reflects_single_call() -> None:
    client = TestClient(_make_app())
    resp = client.post("/chat")

    # gpt-4o: 1000 prompt @ $2.50/M + 500 completion @ $10/M
    expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
    assert float(resp.headers[HEADER_COST]) == pytest.approx(expected, rel=1e-6)


def test_cost_accumulates_across_multiple_calls() -> None:
    client = TestClient(_make_app())
    resp = client.post("/double")

    assert resp.headers[HEADER_TOKENS] == "1800"  # 1500 + 300


def test_request_state_tracker_accessible_in_handler() -> None:
    client = TestClient(_make_app())
    resp = client.post("/state")

    body = resp.json()
    assert body["tokens"] == 1500.0
    assert body["cost"] > 0


def test_requests_are_isolated() -> None:
    """Each request gets a fresh tracker - costs must not leak between requests."""
    client = TestClient(_make_app())

    r1 = client.post("/chat")
    r2 = client.post("/chat")

    # Both report the same single-call total, proving no accumulation across requests
    assert r1.headers[HEADER_TOKENS] == "1500"
    assert r2.headers[HEADER_TOKENS] == "1500"


def test_non_tracked_endpoint_reports_zero() -> None:
    client = TestClient(_make_app())
    resp = client.get("/noop")

    assert resp.headers[HEADER_TOKENS] == "0"
    assert float(resp.headers[HEADER_COST]) == pytest.approx(0.0)


def test_headers_disabled() -> None:
    client = TestClient(_make_app(add_headers=False))
    resp = client.post("/chat")

    assert HEADER_COST not in resp.headers
    assert HEADER_TOKENS not in resp.headers


def test_per_request_budget_exceeded_raises() -> None:
    # Budget of $0.001 - a 1000+500 token gpt-4o call costs ~$0.0075, over the limit
    app = _make_app(budget_per_request=BudgetConfig(limit_usd=0.001))
    client = TestClient(app, raise_server_exceptions=True)

    with pytest.raises(BudgetExceededError):
        client.post("/chat")


def test_per_request_budget_not_exceeded_succeeds() -> None:
    app = _make_app(budget_per_request=BudgetConfig(limit_usd=100.0))
    client = TestClient(app)
    resp = client.post("/chat")

    assert resp.status_code == 200


def test_on_complete_callback_invoked() -> None:
    captured: list[CostReport] = []

    def on_complete(scope: Scope, report: CostReport) -> None:
        captured.append(report)

    client = TestClient(_make_app(on_complete=on_complete))
    client.post("/chat")

    assert len(captured) == 1
    assert captured[0].total_tokens == 1500


def test_request_cost_logged(caplog: pytest.LogCaptureFixture) -> None:
    client = TestClient(_make_app())

    with caplog.at_level(logging.INFO, logger="openai_cost_guard.middleware"):
        client.post("/chat")

    assert any("Request cost" in msg for msg in caplog.messages)
