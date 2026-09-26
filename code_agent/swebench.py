"""Adapters for importing official SWE-bench records without leaking gold patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SWEBenchRecord(BaseModel):
    instance_id: str = Field(min_length=1, max_length=300)
    repo: str = Field(min_length=3, max_length=300)
    base_commit: str = Field(min_length=7, max_length=100)
    problem_statement: str = Field(min_length=10, max_length=100_000)
    version: str = ""
    FAIL_TO_PASS: str | list[str] = Field(default_factory=list)
    PASS_TO_PASS: str | list[str] = Field(default_factory=list)
    environment_setup_commit: str = ""
    language: str = Field(default="", max_length=100)


def _test_ids(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [value]
    return [str(item) for item in parsed] if isinstance(parsed, list) else [str(parsed)]


def load_swebench_records(path: Path) -> list[SWEBenchRecord]:
    """Load either the JSON array or JSONL form distributed by SWE-bench tooling."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    raw: list[dict[str, Any]]
    if stripped.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("SWE-bench JSON input must be an array")
        raw = parsed
    else:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    records = [SWEBenchRecord.model_validate(item) for item in raw]
    if not records:
        raise ValueError("SWE-bench input contains no records")
    return records


def convert_swebench_records(
    records: list[SWEBenchRecord],
    *,
    test_command: list[str] | None = None,
    benchmark_name: str = "swe-bench",
) -> list[dict[str, Any]]:
    """Convert official rows to local tasks while intentionally excluding solution patches."""
    command = test_command or ["python", "-m", "pytest", "-q"]
    cases = []
    for record in records:
        repository = record.repo.replace("/", "__")
        language = record.language.strip().lower()
        tags = ["swe-bench", f"benchmark:{benchmark_name}", f"repository:{record.repo}"]
        if language:
            tags.append(f"language:{language}")
        cases.append(
            {
                "id": record.instance_id,
                "repository": repository,
                "issue": record.problem_statement,
                "test_command": command,
                "tags": tags,
                "source": "swe-bench",
                "split": "test",
                "metadata": {
                    "instance_id": record.instance_id,
                    "repo": record.repo,
                    "base_commit": record.base_commit,
                    "version": record.version,
                    "fail_to_pass": _test_ids(record.FAIL_TO_PASS),
                    "pass_to_pass": _test_ids(record.PASS_TO_PASS),
                    "environment_setup_commit": record.environment_setup_commit,
                    "language": record.language,
                    "benchmark": benchmark_name,
                },
            }
        )
    return cases


def write_cases(cases: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(case, sort_keys=True, separators=(",", ":")) for case in cases]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official SWE-bench JSON/JSONL to AgentForge tasks")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--test-command",
        nargs="+",
        default=["python", "-m", "pytest", "-q"],
        help="Fixed argv run by the local sandbox for every imported case",
    )
    parser.add_argument(
        "--benchmark-name",
        default="swe-bench",
        help="Dataset identity retained in tags and metadata, such as swe-bench-multilingual",
    )
    args = parser.parse_args()
    cases = convert_swebench_records(
        load_swebench_records(args.input),
        test_command=args.test_command,
        benchmark_name=args.benchmark_name,
    )
    write_cases(cases, args.output)
    print(json.dumps({"cases": len(cases), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
