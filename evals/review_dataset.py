"""Auditable export/import workflow for human review of evaluation labels."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from evals.platform import EvalCase, load_cases

FIELDS = (
    "id",
    "input",
    "split",
    "tags",
    "expected_json",
    "decision",
    "reviewer",
    "review_notes",
)


def export_review_sheet(dataset: Path, output: Path) -> int:
    cases = load_cases(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "id": case.id,
                    "input": case.input,
                    "split": case.split,
                    "tags": ",".join(case.tags),
                    "expected_json": case.expected.model_dump_json(),
                    "decision": "",
                    "reviewer": "",
                    "review_notes": "",
                }
            )
    return len(cases)


def import_review_sheet(dataset: Path, review_file: Path, output: Path) -> dict[str, int]:
    cases = {case.id: case for case in load_cases(dataset)}
    approved = 0
    rejected = 0
    reviewed_at = datetime.now(UTC).isoformat()
    with review_file.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        case_id = str(row.get("id") or "").strip()
        if case_id not in cases:
            raise ValueError(f"review contains unknown case id: {case_id}")
        decision = str(row.get("decision") or "").strip().lower()
        if decision not in {"approve", "reject", ""}:
            raise ValueError(f"invalid decision for {case_id}: {decision}")
        if not decision:
            continue
        reviewer = str(row.get("reviewer") or "").strip()
        if not reviewer:
            raise ValueError(f"reviewer is required for {case_id}")
        if decision == "reject":
            rejected += 1
            continue
        expected_raw = str(row.get("expected_json") or "").strip()
        updated = cases[case_id].model_copy(
            update={
                "expected": cases[case_id].expected.model_validate_json(expected_raw),
                "review_status": "human_reviewed",
                "metadata": {
                    **cases[case_id].metadata,
                    "reviewer": reviewer,
                    "reviewed_at": reviewed_at,
                    "review_notes": str(row.get("review_notes") or "").strip(),
                },
            }
        )
        cases[case_id] = EvalCase.model_validate(updated)
        approved += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(case.model_dump_json() for case in cases.values()) + "\n",
        encoding="utf-8",
    )
    return {"approved": approved, "rejected": rejected, "unreviewed": len(cases) - approved - rejected}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export or import human evaluation-label reviews.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--dataset", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--dataset", type=Path, required=True)
    import_parser.add_argument("--review-file", type=Path, required=True)
    import_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        print(json.dumps({"exported": export_review_sheet(args.dataset, args.output)}))
    else:
        print(json.dumps(import_review_sheet(args.dataset, args.review_file, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
