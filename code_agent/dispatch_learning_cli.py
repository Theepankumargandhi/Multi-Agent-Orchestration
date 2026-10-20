"""Immutable export/review/train steps for a non-serving dispatch advisor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_agent.dispatch_learning import (
    AdvisorThresholds,
    DecisionSnapshot,
    DispatchAdvisorArtifact,
    ReviewedDecision,
    evaluate_advisor,
    fit_advisor,
    snapshots_from_report,
    validate_protocol,
)
from code_agent.models import RepairTournamentReport


def load_records(path: Path) -> list[ReviewedDecision]:
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not paths or len(paths) > 10000:
        raise ValueError("bounded reviewed inputs required")
    records = []
    total_bytes = 0
    for source in paths:
        total_bytes += source.stat().st_size
        if source.is_symlink() or source.stat().st_size > 2_000_000 or total_bytes > 20_000_000:
            raise ValueError("review input exceeds bounded regular-file limits")
        raw = source.read_text(encoding="utf-8")
        if source.suffix == ".jsonl":
            values = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            parsed = json.loads(raw)
            values = parsed if isinstance(parsed, list) else [parsed]
        records.extend(ReviewedDecision.model_validate(value) for value in values)
        if len(records) > 10000:
            raise ValueError("review record limit exceeded")
    return records


def fresh_output(path: Path, inputs: list[Path] = ()) -> Path:
    target = path.resolve()
    root = Path(__file__).resolve().parents[1]
    if (target.exists() or any(target == source.resolve() or target.is_relative_to(source.resolve()) for source in inputs)
            or (target.is_relative_to(root) and target.relative_to(root).parts[:2] != ("data", "evaluations"))):
        raise ValueError("fresh output outside source/input paths required")
    return target


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def review_card(report: dict) -> str:
    lines = ["# Reviewed dispatch advisor", "", f"Decision: {report['decision']}; serving activation: false.",
             "", "| Split | Families | Coverage | Accepted review agreement | Family error upper bound |",
             "|---|---:|---:|---|---:|"]
    for split, values in report["metrics"].items():
        agreement = values["accepted_review_agreement"]
        label = "unavailable" if agreement is None else f"{agreement:.1%}"
        lines.append(f"| {split} | {values['families']} | {values['coverage']:.1%} | {label} | "
                     f"{values['family_disagreement_upper_bound']:.1%} |")
    return "\n".join(lines + ["", "## Holds", ""] + [f"- {item}" for item in report["blocking_reasons"]]
                     + ["", "## Limits", ""] + [f"- {item}" for item in report["limitations"]] + [""])


def save_bundle(destination: Path, artifact: DispatchAdvisorArtifact, report: dict) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "advisor.json", artifact.model_dump(mode="json"))
    write_json(destination / "evaluation.json", report)
    with (destination / "review.md").open("x", encoding="utf-8") as stream:
        stream.write(review_card(report))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Export pending, label-free decision snapshots")
    export.add_argument("--report", type=Path, required=True)
    export.add_argument("--family-id", required=True)
    export.add_argument("--scope-sha256", required=True)
    export.add_argument("--observed-at", required=True, help="Declared observation time, not externally attested")
    export.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("review", help="Create a separate explicit review; never mutate a snapshot")
    review.add_argument("--snapshot", type=Path, required=True)
    review.add_argument("--record-id", help="Select one record from an exported snapshot array")
    review.add_argument("--preferred-role", choices=["boundary", "compatibility", "defensive"], required=True)
    review.add_argument("--reviewer-id", required=True)
    review.add_argument("--label-source", choices=["human", "synthetic"], required=True)
    review.add_argument("--independent-review", action="store_true", required=True)
    review.add_argument("--reviewed-at", required=True)
    review.add_argument("--split", choices=["train", "validation", "test"], required=True)
    review.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="Fit training families and evaluate frozen validation/test families")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--thresholds", type=Path)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--require-gate", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            output = fresh_output(args.output, [args.report])
            if args.report.stat().st_size > 2_000_000:
                raise ValueError("bounded receipt required")
            report = RepairTournamentReport.model_validate_json(args.report.read_text(encoding="utf-8"))
            snapshots = snapshots_from_report(report, family_id=args.family_id, scope_sha256=args.scope_sha256,
                                              observed_at=args.observed_at)
            write_json(output, [value.model_dump(mode="json") for value in snapshots])
            print(json.dumps({"pending_snapshots": len(snapshots), "labels_generated": False}))
        elif args.command == "review":
            output = fresh_output(args.output, [args.snapshot])
            if args.snapshot.stat().st_size > 100_000:
                raise ValueError("bounded single snapshot required")
            raw = json.loads(args.snapshot.read_text(encoding="utf-8"))
            values = raw if isinstance(raw, list) else [raw]
            if any(not isinstance(value, dict) for value in values):
                raise ValueError("snapshot objects required")
            selected = [value for value in values if not args.record_id or value.get("record_id") == args.record_id]
            if len(selected) != 1:
                raise ValueError("select exactly one pending decision snapshot")
            snapshot = DecisionSnapshot.model_validate(selected[0])
            record = ReviewedDecision(snapshot=snapshot, preferred_role=args.preferred_role, reviewer_id=args.reviewer_id,
                independent_review=args.independent_review, label_source=args.label_source,
                reviewed_at=args.reviewed_at, split=args.split)
            write_json(output, record.model_dump(mode="json"))
            print(json.dumps({"review_saved": True, "snapshot_unchanged": True}))
        else:
            output = fresh_output(args.output_dir, [args.dataset] + ([args.thresholds] if args.thresholds else []))
            records = load_records(args.dataset)
            validate_protocol(records)
            if args.thresholds and (args.thresholds.is_symlink() or args.thresholds.stat().st_size > 100_000):
                raise ValueError("bounded regular threshold file required")
            thresholds = AdvisorThresholds.model_validate_json(args.thresholds.read_text(encoding="utf-8")) if args.thresholds else None
            artifact = fit_advisor([row for row in records if row.split == "train"], thresholds)
            report = evaluate_advisor(records, artifact)
            save_bundle(output, artifact, report)
            print(json.dumps({"decision": report["decision"], "artifact_fingerprint": artifact.fingerprint,
                              "serving_activation": False}))
            return int(args.require_gate and report["decision"] != "ready_for_owner_review")
        return 0
    except (OSError, ValueError, TypeError):
        print("Dispatch learning failed: invalid reviewed inputs, lineage, protocol, or protected output path.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
