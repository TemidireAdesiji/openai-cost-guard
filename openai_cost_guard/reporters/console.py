import logging
from collections import defaultdict

from ..models import CostReport

logger = logging.getLogger(__name__)

# Column widths
_W_MODEL = 28
_W_CALLS = 7
_W_PROMPT = 10
_W_COMPL = 10
_W_TOTAL = 12
_W_COST = 12

_BORDER = (
    "+"
    + "-" * (_W_MODEL + 2)
    + "+"
    + "-" * (_W_CALLS + 2)
    + "+"
    + "-" * (_W_PROMPT + 2)
    + "+"
    + "-" * (_W_COMPL + 2)
    + "+"
    + "-" * (_W_TOTAL + 2)
    + "+"
    + "-" * (_W_COST + 2)
    + "+"
)


def _row(model: str, calls: int, prompt: int, compl: int, total: int, cost: float) -> str:
    return (
        f"| {model:<{_W_MODEL}} "
        f"| {calls:>{_W_CALLS}} "
        f"| {prompt:>{_W_PROMPT},} "
        f"| {compl:>{_W_COMPL},} "
        f"| {total:>{_W_TOTAL},} "
        f"| {'$' + f'{cost:.4f}':>{_W_COST}} |"
    )


def print_report(report: CostReport, logger_name: str | None = None) -> None:
    """Log a formatted cost report table, grouped by model.

    Output goes to the logger ``openai_cost_guard.reporters.console`` by default,
    or to the logger named by ``logger_name`` if provided.
    """
    out = logging.getLogger(logger_name or __name__)

    if not report.records:
        out.info("openai-cost-guard: no usage recorded.")
        return

    # Aggregate by model
    by_model: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"calls": 0, "prompt": 0, "compl": 0, "total": 0, "cost": 0.0}
    )
    for rec in report.records:
        row = by_model[rec.model]
        row["calls"] = int(row["calls"]) + 1
        row["prompt"] = int(row["prompt"]) + rec.prompt_tokens
        row["compl"] = int(row["compl"]) + rec.completion_tokens
        row["total"] = int(row["total"]) + rec.total_tokens
        row["cost"] = float(row["cost"]) + rec.total_cost

    header = _row("Model", 0, 0, 0, 0, 0.0)
    header = (
        f"| {'Model':<{_W_MODEL}} "
        f"| {'Calls':>{_W_CALLS}} "
        f"| {'Prompt':>{_W_PROMPT}} "
        f"| {'Completion':>{_W_COMPL}} "
        f"| {'Total Tokens':>{_W_TOTAL}} "
        f"| {'Cost (USD)':>{_W_COST}} |"
    )

    lines = [
        "",
        "openai-cost-guard report",
        _BORDER,
        header,
        _BORDER,
    ]

    for model, stats in sorted(by_model.items()):
        lines.append(
            _row(
                model,
                int(stats["calls"]),
                int(stats["prompt"]),
                int(stats["compl"]),
                int(stats["total"]),
                float(stats["cost"]),
            )
        )

    lines += [
        _BORDER,
        _row(
            "TOTAL",
            len(report.records),
            report.total_prompt_tokens,
            report.total_completion_tokens,
            report.total_tokens,
            report.total_cost,
        ),
        _BORDER,
    ]

    out.info("\n".join(lines))
