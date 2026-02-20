from pathlib import Path

from ..models import CostReport


def to_json(report: CostReport, indent: int | None = 2) -> str:
    """Serialise a CostReport to a JSON string.

    Uses Pydantic's serialisation so datetimes are ISO-formatted and all
    fields are included. Pass ``indent=None`` for compact output.
    """
    return report.model_dump_json(indent=indent)


def write_json(report: CostReport, path: str | Path, indent: int | None = 2) -> Path:
    """Write a CostReport to a JSON file and return the resolved path.

    Parent directories are created if they do not exist.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_json(report, indent=indent), encoding="utf-8")
    return target.resolve()


def to_summary_dict(report: CostReport) -> dict[str, object]:
    """Return a compact summary suitable for embedding in a JSON API response.

    Excludes the per-record detail - just the aggregate figures, grouped by model.
    Useful as a response header payload or a logging field.
    """
    by_model: dict[str, dict[str, float | int]] = {}
    for rec in report.records:
        row = by_model.setdefault(
            rec.model, {"calls": 0, "total_tokens": 0, "total_cost": 0.0}
        )
        row["calls"] = int(row["calls"]) + 1
        row["total_tokens"] = int(row["total_tokens"]) + rec.total_tokens
        row["total_cost"] = float(row["total_cost"]) + rec.total_cost

    return {
        "total_cost": report.total_cost,
        "total_tokens": report.total_tokens,
        "total_prompt_tokens": report.total_prompt_tokens,
        "total_completion_tokens": report.total_completion_tokens,
        "call_count": len(report.records),
        "by_model": by_model,
    }
