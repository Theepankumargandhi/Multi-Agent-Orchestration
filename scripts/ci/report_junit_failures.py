"""Publish pytest JUnit failures as GitHub Actions annotations."""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _command_escape(value: str) -> str:
    return (
        value.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


def _source_path(testcase: ET.Element) -> str:
    explicit = testcase.get("file")
    if explicit:
        return explicit.replace("\\", "/")
    module = testcase.get("classname", "").split(".")
    if module and module[0] in {"tests", "service", "schema"}:
        return "/".join(module) + ".py"
    return "tests"


def _failure_text(node: ET.Element) -> str:
    message = node.get("message", "").strip()
    details = " ".join((node.text or "").split())
    combined = f"{message} | {details}" if message and details else message or details
    return combined[:6000] or "pytest reported a failure without details"


def report_failures(report_path: Path) -> int:
    root = ET.parse(report_path).getroot()
    failures: list[tuple[str, str, str]] = []
    for testcase in root.iter("testcase"):
        problem = testcase.find("failure")
        if problem is None:
            problem = testcase.find("error")
        if problem is None:
            continue
        name = testcase.get("name", "unknown test")
        failures.append((_source_path(testcase), name, _failure_text(problem)))

    for path, name, details in failures:
        properties = f"file={_command_escape(path)},title={_command_escape(name)}"
        print(f"::error {properties}::{_command_escape(details)}")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(f"## Pytest failures ({len(failures)})\n\n")
            for path, name, details in failures:
                safe_details = details.replace("`", "'")
                summary.write(f"- `{path}` — **{name}**: {safe_details}\n")

    print(f"Published {len(failures)} pytest failure annotation(s) from {report_path}.")
    return len(failures)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: report_junit_failures.py <junit-xml>", file=sys.stderr)
        return 2
    report_path = Path(sys.argv[1])
    if not report_path.is_file():
        print(f"JUnit report not found: {report_path}", file=sys.stderr)
        return 2
    report_failures(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
