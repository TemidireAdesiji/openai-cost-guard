"""Benchmark the openai-cost-guard hot path.

openai-cost-guard is a library, so the meaningful benchmark is the throughput of the
code that runs on every API call: CostTracker.record() and the @track_cost decorator
wrapper. This script uses fake response objects, so it needs no live Azure and makes no
network calls. It prints a Markdown table to STDOUT only (it does not write any file).

Pipe the table into the README benchmarks section with the injector:

    python scripts/benchmark.py | python scripts/inject_readme_section.py --section benchmarks

Run from the project root (the package must be importable - `pip install -e .`):

    python scripts/benchmark.py
    python scripts/benchmark.py --iterations 200000
"""
import argparse
import sys
import timeit
from pathlib import Path
from typing import Any

# Make the package importable when run from the project root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai_cost_guard import CostTracker, track_cost  # noqa: E402


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    """Mimics the shape @track_cost reads: .model and .usage.{prompt,completion}_tokens."""

    def __init__(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.model = model
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


def _bench(label: str, fn: Any, iterations: int) -> tuple[str, int, float, float]:
    # Warm up first so the earliest-measured case does not pay one-time costs (import
    # caches, CPU ramp-up) that the others avoid - otherwise the table looks inconsistent.
    for _ in range(min(2_000, iterations)):
        fn()
    elapsed = timeit.timeit(fn, number=iterations)
    per_sec = iterations / elapsed if elapsed > 0 else float("inf")
    us_per_call = (elapsed / iterations) * 1_000_000
    return label, iterations, per_sec, us_per_call


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the openai-cost-guard hot path.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100_000,
        help="Number of iterations per benchmark (default: 100000).",
    )
    args = parser.parse_args(argv)
    n = args.iterations

    # 1. Raw record() throughput.
    record_tracker = CostTracker()

    def do_record() -> None:
        record_tracker.record("gpt-4o", prompt_tokens=512, completion_tokens=128)

    # 2. record() with a per-call budget check enabled.
    from openai_cost_guard import BudgetConfig

    budget_tracker = CostTracker(budget=BudgetConfig(limit_usd=10_000_000.0))

    def do_record_budget() -> None:
        budget_tracker.record("gpt-4o", prompt_tokens=512, completion_tokens=128)

    # 3. @track_cost decorator overhead over a fake API call.
    deco_tracker = CostTracker()
    response = _FakeResponse("gpt-4o", 512, 128)

    @track_cost(tracker=deco_tracker, endpoint="bench")
    def wrapped_call() -> _FakeResponse:
        return response

    def do_wrapped() -> None:
        wrapped_call()

    # 4. Prefix-match path (versioned deployment name not in the table directly).
    prefix_tracker = CostTracker()

    def do_record_prefix() -> None:
        prefix_tracker.record(
            "gpt-4o-mini-2024-07-18", prompt_tokens=512, completion_tokens=128
        )

    results = [
        _bench("CostTracker.record()", do_record, n),
        _bench("CostTracker.record() + budget check", do_record_budget, n),
        _bench("@track_cost-wrapped call (fake response)", do_wrapped, n),
        _bench("record() via prefix-match pricing", do_record_prefix, n),
    ]

    lines = [
        f"_Measured on this machine with {n:,} iterations per case, fake responses, no network._",
        "",
        "| Operation | Iterations | Calls/sec | us/call |",
        "|---|---:|---:|---:|",
    ]
    for label, iterations, per_sec, us_per_call in results:
        lines.append(f"| {label} | {iterations:,} | {per_sec:,.0f} | {us_per_call:.2f} |")

    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
