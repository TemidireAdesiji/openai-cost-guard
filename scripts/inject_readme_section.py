"""Replace a sentinel-delimited section of README.md with content read from stdin.

The README marks injectable sections with HTML comment sentinels:

    <!-- BEGIN:benchmarks -->
    ...content the script manages...
    <!-- END:benchmarks -->

This tool reads replacement content from stdin and rewrites the text between the
matching BEGIN/END sentinels in place. The sentinels themselves are preserved.

Run from the project root:

    python scripts/benchmark.py | python scripts/inject_readme_section.py --section benchmarks
    cat new_text.md | python scripts/inject_readme_section.py --section benchmarks

Exits non-zero with a clear message if the sentinels are missing or malformed.
This script does not need the application running.
"""
import argparse
import sys
from pathlib import Path

DEFAULT_README = Path(__file__).resolve().parent.parent / "README.md"


def inject(readme_text: str, section: str, new_content: str) -> str:
    begin = f"<!-- BEGIN:{section} -->"
    end = f"<!-- END:{section} -->"

    begin_idx = readme_text.find(begin)
    if begin_idx == -1:
        raise ValueError(f"Sentinel {begin!r} not found in README.")
    end_idx = readme_text.find(end)
    if end_idx == -1:
        raise ValueError(f"Sentinel {end!r} not found in README.")
    if end_idx < begin_idx:
        raise ValueError(f"Sentinel {end!r} appears before {begin!r} in README.")

    after_begin = begin_idx + len(begin)
    body = new_content.strip("\n")
    return readme_text[:after_begin] + "\n" + body + "\n" + readme_text[end_idx:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inject stdin into a sentinel-delimited README section."
    )
    parser.add_argument(
        "--section",
        required=True,
        help="Section name between the BEGIN/END sentinels (e.g. 'benchmarks').",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=DEFAULT_README,
        help="Path to the README file (default: project README.md).",
    )
    args = parser.parse_args(argv)

    new_content = sys.stdin.read()
    if not new_content.strip():
        sys.stderr.write("error: no content received on stdin.\n")
        return 1

    try:
        original = args.readme.read_text(encoding="utf-8")
    except FileNotFoundError:
        sys.stderr.write(f"error: README not found: {args.readme}\n")
        return 1

    try:
        updated = inject(original, args.section, new_content)
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    args.readme.write_text(updated, encoding="utf-8")
    sys.stderr.write(f"Updated section {args.section!r} in {args.readme}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
