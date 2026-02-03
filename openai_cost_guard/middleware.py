"""FastAPI / Starlette middleware that scopes a CostTracker to each request.

Each incoming HTTP request gets a fresh CostTracker. Any ``@track_cost``-decorated
call made while handling the request records into that per-request tracker (via a
contextvar - no need to pass the tracker around). When the response starts, the
middleware attaches cost headers and, after completion, logs the request total and
invokes an optional callback.

Because this is a pure ASGI middleware (not ``BaseHTTPMiddleware``), the contextvar
set here is visible to the endpoint - Starlette's BaseHTTPMiddleware runs the
endpoint in a separate task where the contextvar would not propagate.

Example::

    from fastapi import FastAPI, Request
    from openai_cost_guard import track_cost
    from openai_cost_guard.middleware import CostGuardMiddleware

    app = FastAPI()
    app.add_middleware(CostGuardMiddleware)

    @track_cost(endpoint="chat")          # no explicit tracker - uses request scope
    def call_model(client, messages): ...

    @app.post("/chat")
    async def chat(request: Request):
        call_model(client, [...])
        # request.state.cost_tracker.report() is available here
        return {"ok": True}
"""
import logging
from collections.abc import Callable

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .context import reset_current_tracker, set_current_tracker
from .models import BudgetConfig, CostReport
from .tracker import CostTracker

logger = logging.getLogger(__name__)

# Header names for per-request cost reporting
HEADER_COST = "X-OpenAI-Cost-USD"
HEADER_TOKENS = "X-OpenAI-Total-Tokens"

OnComplete = Callable[[Scope, CostReport], None]


class CostGuardMiddleware:
    """Bind a per-request CostTracker and surface its totals as response headers.

    :param app: the wrapped ASGI application (supplied by Starlette/FastAPI).
    :param budget_per_request: optional budget enforced per request. A call that
        exceeds it raises BudgetExceededError inside the handler.
    :param add_headers: when True (default), attach cost/token headers to responses.
    :param strict: forwarded to each request's CostTracker (raise on unknown model).
    :param on_complete: optional callback invoked after each request with
        ``(scope, report)`` - use it to push metrics or persist usage.
    """

    def __init__(
        self,
        app: ASGIApp,
        budget_per_request: BudgetConfig | None = None,
        add_headers: bool = True,
        strict: bool = False,
        on_complete: OnComplete | None = None,
    ) -> None:
        self.app = app
        self._budget = budget_per_request
        self._add_headers = add_headers
        self._strict = strict
        self._on_complete = on_complete

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        tracker = CostTracker(budget=self._budget, strict=self._strict)
        # Expose on request.state for handlers that want the tracker explicitly.
        scope.setdefault("state", {})["cost_tracker"] = tracker
        token = set_current_tracker(tracker)

        send_with_headers = self._wrap_send(send, tracker) if self._add_headers else send

        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            report = tracker.report()
            logger.info(
                "Request cost: $%.6f across %d call(s), %d tokens",
                report.total_cost,
                len(report.records),
                report.total_tokens,
            )
            if self._on_complete is not None:
                self._on_complete(scope, report)
            reset_current_tracker(token)

    def _wrap_send(self, send: Send, tracker: CostTracker) -> Send:
        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                report = tracker.report()
                headers = MutableHeaders(scope=message)
                headers[HEADER_COST] = f"{report.total_cost:.6f}"
                headers[HEADER_TOKENS] = str(report.total_tokens)
            await send(message)

        return send_wrapper
