"""Queue blinded dispatch reviews, consolidate supervision and audit coarse shift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_agent.dispatch_learning import DecisionSnapshot, load_advisor
from code_agent.dispatch_learning_cli import fresh_output, load_records, write_json
from code_agent.dispatch_supervision import (
    DispatchAdjudication,
    IndependentDispatchReview,
    ReviewBatch,
    ReviewQueuePolicy,
    ShiftAuditPolicy,
    audit_distribution_shift,
    build_review_batch,
    consolidate_reviews,
)


def read_json(path: Path, *, max_bytes: int = 2_000_000):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("bounded regular JSON input required")
    return json.loads(path.read_text(encoding="utf-8"))


def read_many(path: Path, model) -> list:
    if path.is_symlink() or not path.exists():
        raise ValueError("existing regular input or directory required")
    sources = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not sources or len(sources) > 10000 or sum(source.stat().st_size for source in sources) > 20_000_000:
        raise ValueError("bounded nonempty input set required")
    records = []
    for source in sources:
        raw = read_json(source)
        values = raw if isinstance(raw, list) else [raw]
        records.extend(model.model_validate(value) for value in values)
        if len(records) > 10000:
            raise ValueError("record bound exceeded")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    queue = commands.add_parser("queue", help="Create an immutable private selection and blinded reviewer packet")
    queue.add_argument("--pool", type=Path, required=True)
    queue.add_argument("--artifact", type=Path, required=True)
    queue.add_argument("--artifact-fingerprint", required=True)
    queue.add_argument("--policy", type=Path)
    queue.add_argument("--created-at", required=True)
    queue.add_argument("--output-dir", type=Path, required=True)
    annotate = commands.add_parser("annotate", help="Save a separate explicit, blinded human or synthetic label")
    adjudicate = commands.add_parser("adjudicate", help="Save an explicit conflict resolution or evidence abstention")
    for command in (annotate, adjudicate):
        command.add_argument("--packet", type=Path, required=True)
        command.add_argument("--record-id", required=True)
        command.add_argument("--reviewer-id", required=True)
        command.add_argument("--preferred-role", choices=["boundary", "compatibility", "defensive"], required=command is annotate)
        command.add_argument("--independent-review", action="store_true", required=True)
        command.add_argument("--label-source", choices=["human", "synthetic"], required=True)
        command.add_argument("--reviewed-at", required=True)
        command.add_argument("--output", type=Path, required=True)
    adjudicate.add_argument("--reason", choices=["execution_evidence", "localization_evidence", "contract_evidence", "insufficient_evidence"], required=True)
    consensus = commands.add_parser("consensus", help="Require all assigned reviewers or independent adjudication")
    consensus.add_argument("--batch", type=Path, required=True)
    consensus.add_argument("--reviews", type=Path, required=True)
    consensus.add_argument("--adjudications", type=Path)
    consensus.add_argument("--output-dir", type=Path, required=True)
    shift = commands.add_parser("shift", help="Audit family-balanced marginals without changing dispatch")
    shift.add_argument("--reference", type=Path, required=True)
    shift.add_argument("--pool", type=Path, required=True)
    shift.add_argument("--artifact", type=Path, required=True)
    shift.add_argument("--artifact-fingerprint", required=True)
    shift.add_argument("--policy", type=Path)
    shift.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "queue":
            output = fresh_output(args.output_dir, [args.pool, args.artifact] + ([args.policy] if args.policy else []))
            artifact = load_advisor(args.artifact, args.artifact_fingerprint).artifact
            policy = ReviewQueuePolicy.model_validate(read_json(args.policy)) if args.policy else None
            batch = build_review_batch(read_many(args.pool, DecisionSnapshot), artifact, created_at=args.created_at, policy=policy)
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "private-batch.json", batch.model_dump(mode="json"))
            write_json(output / "reviewer-packet.json", batch.reviewer_packet())
            print(json.dumps({"selected": len(batch.snapshots), "planned_review_slots": batch.planned_review_slots,
                              "training_only": True, "serving_activation": False}))
        elif args.command in {"annotate", "adjudicate"}:
            output = fresh_output(args.output, [args.packet])
            packet = read_json(args.packet)
            if set(packet) != {"schema_version", "batch_fingerprint", "snapshots"} or packet["schema_version"] != "1.0":
                raise ValueError("blinded packet required")
            snapshots = [DecisionSnapshot.model_validate(value) for value in packet["snapshots"]]
            selected = [row for row in snapshots if row.record_id == args.record_id]
            if len(selected) != 1 or not selected[0].verify() or args.preferred_role in selected[0].context.used_roles:
                raise ValueError("valid selected snapshot and unused review role required")
            values = dict(record_id=selected[0].record_id, snapshot_fingerprint=selected[0].fingerprint,
                batch_fingerprint=packet["batch_fingerprint"], reviewer_id=args.reviewer_id, preferred_role=args.preferred_role,
                label_source=args.label_source, independent_review=args.independent_review, reviewed_at=args.reviewed_at)
            record = DispatchAdjudication(**values, reason=args.reason) if args.command == "adjudicate" else IndependentDispatchReview(**values)
            if record.reviewed_at < selected[0].observed_at:
                raise ValueError("review predates observation")
            write_json(output, record.seal().model_dump(mode="json"))
            print(json.dumps({"review_saved": True, "packet_unchanged": True}))
        elif args.command == "consensus":
            output = fresh_output(args.output_dir, [args.batch, args.reviews] + ([args.adjudications] if args.adjudications else []))
            batch = ReviewBatch.model_validate(read_json(args.batch))
            reviews = read_many(args.reviews, IndependentDispatchReview)
            adjudications = read_many(args.adjudications, DispatchAdjudication) if args.adjudications else []
            report = consolidate_reviews(batch, reviews, adjudications)
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "training-reviews.json", report["records"])
            write_json(output / "consensus-audit.json", report)
            print(json.dumps({"training_records": len(report["records"]), "unresolved_records": report["unresolved_records"], "serving_activation": False}))
        else:
            output = fresh_output(args.output, [args.reference, args.pool, args.artifact] + ([args.policy] if args.policy else []))
            artifact = load_advisor(args.artifact, args.artifact_fingerprint).artifact
            policy = ShiftAuditPolicy.model_validate(read_json(args.policy)) if args.policy else None
            report = audit_distribution_shift(load_records(args.reference), read_many(args.pool, DecisionSnapshot), artifact, policy)
            write_json(output, report)
            print(json.dumps({"status": report["status"], "serving_activation": False}))
        return 0
    except (OSError, ValueError, TypeError, KeyError):
        print("Dispatch supervision failed: invalid inputs, lineage, budget, review binding, or protected output path.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
