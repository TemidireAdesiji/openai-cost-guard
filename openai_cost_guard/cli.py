"""Command-line interface for inspecting saved cost reports.

A report is the JSON produced by ``openai_cost_guard.reporters.json.write_json``.

Commands::

    openai-cost-guard show <report.json>      # formatted per-model table
    openai-cost-guard summary <report.json>   # aggregate totals as JSON

Output is written to stdout via a logging handler (the package never uses print()).
"""
import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from .models import CostReport
from .reporters.console import print_report
from .reporters.json import to_summary_dict

logger = logging.getLogger("openai_cost_guard.cli")


def _configure_stdout_logging() -> None:
    """Route openai_cost_guard logs to stdout with a bare message format."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    pkg_logger = logging.getLogger("openai_cost_guard")
    pkg_logger.handlers.clear()
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.INFO)
    pkg_logger.propagate = False


def _load_report(path: Path) -> CostReport:
    return CostReport.model_validate_json(path.read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openai-cost-guard",
        description="Inspect saved openai-cost-guard JSON reports.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="Print a formatted per-model cost table.")
    show.add_argument("report", type=Path, help="Path to a JSON report file.")

    summary = sub.add_parser("summary", help="Print aggregate totals as JSON.")
    summary.add_argument("report", type=Path, help="Path to a JSON report file.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_stdout_logging()

    try:
        report = _load_report(args.report)
    except FileNotFoundError:
        logger.error("Report file not found: %s", args.report)
        return 1
    except ValidationError as exc:
        logger.error("Not a valid cost report (%s): %s", args.report, exc)
        return 1

    if args.command == "show":
        print_report(report)
    elif args.command == "summary":
        logger.info(json.dumps(to_summary_dict(report), indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
