from types import SimpleNamespace

from openai_cost_guard.decorators import (
    get_default_tracker,
    reset_default_tracker,
    track_cost,
    track_cost_method,
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


def test_track_cost_records_to_default_tracker() -> None:
    @track_cost()
    def call() -> SimpleNamespace:
        return _fake_response("gpt-4o", prompt_tokens=1000, completion_tokens=500)

    call()
    report = get_default_tracker().report()
    assert len(report.records) == 1
    assert report.records[0].model == "gpt-4o"
    assert report.records[0].prompt_tokens == 1000


def test_track_cost_records_to_explicit_tracker() -> None:
    tracker = CostTracker()

    @track_cost(tracker=tracker)
    def call() -> SimpleNamespace:
        return _fake_response("gpt-4o-mini", prompt_tokens=200, completion_tokens=100)

    call()
    report = tracker.report()
    assert len(report.records) == 1
    # Default tracker should be unaffected
    assert len(get_default_tracker().report().records) == 0


def test_track_cost_preserves_return_value() -> None:
    expected = _fake_response("gpt-4o", prompt_tokens=100, completion_tokens=50)

    @track_cost()
    def call() -> SimpleNamespace:
        return expected

    result = call()
    assert result is expected


def test_track_cost_skips_response_without_usage() -> None:
    @track_cost()
    def call() -> SimpleNamespace:
        return SimpleNamespace(no_usage_here=True)

    call()
    assert len(get_default_tracker().report().records) == 0


def test_track_cost_sets_endpoint_label() -> None:
    tracker = CostTracker()

    @track_cost(tracker=tracker, endpoint="my-endpoint")
    def call() -> SimpleNamespace:
        return _fake_response("gpt-4o", 100, 50)

    call()
    assert tracker.report().records[0].endpoint == "my-endpoint"


def test_track_cost_defaults_endpoint_to_qualname() -> None:
    tracker = CostTracker()

    @track_cost(tracker=tracker)
    def my_specific_function() -> SimpleNamespace:
        return _fake_response("gpt-4o", 100, 50)

    my_specific_function()
    assert "my_specific_function" in (tracker.report().records[0].endpoint or "")


def test_track_cost_metadata_forwarded() -> None:
    tracker = CostTracker()

    @track_cost(tracker=tracker, metadata={"env": "test"})
    def call() -> SimpleNamespace:
        return _fake_response("gpt-4o", 100, 50)

    call()
    assert tracker.report().records[0].metadata == {"env": "test"}


def test_track_cost_method_reads_tracker_from_instance() -> None:
    class Service:
        def __init__(self) -> None:
            self.cost_tracker = CostTracker()

        @track_cost_method()
        def call(self) -> SimpleNamespace:
            return _fake_response("gpt-4o", 500, 250)

    svc = Service()
    svc.call()
    report = svc.cost_tracker.report()
    assert len(report.records) == 1


def test_track_cost_method_custom_attr_name() -> None:
    class Service:
        def __init__(self) -> None:
            self.my_tracker = CostTracker()

        @track_cost_method(tracker_attr="my_tracker")
        def call(self) -> SimpleNamespace:
            return _fake_response("gpt-4o", 100, 50)

    svc = Service()
    svc.call()
    assert len(svc.my_tracker.report().records) == 1


def test_track_cost_wrapper_preserves_function_name() -> None:
    @track_cost()
    def my_named_function() -> SimpleNamespace:
        return _fake_response("gpt-4o", 10, 5)

    assert my_named_function.__name__ == "my_named_function"
