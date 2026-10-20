"""Authored learning controls for a non-serving specialist advisor."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from code_agent.dispatch_learning import (
    DecisionSnapshot,
    DispatchContext,
    ReviewedDecision,
    evaluate_advisor,
    fingerprint,
    fit_advisor,
    policy_fingerprint,
)
from code_agent.dispatch_learning_cli import fresh_output, save_bundle, write_json
from code_agent.models import RepairTournamentPolicy


def authored_records() -> list[ReviewedDecision]:
    policy = RepairTournamentPolicy(max_parallel=1, adaptive_dispatch={})
    context = DispatchContext(category="incomplete_fix", used_roles=["minimal"], changed_files="few",
                              patch_size="small", call_headroom="high")
    records = []
    for split, day, count in (("train", 1, 6), ("validation", 10, 8), ("test", 20, 8)):
        for index in range(count):
            receipt = fingerprint([split, index, "authored-receipt"])
            value = context if split == "train" or index < 6 else context.model_copy(update={"patch_size": "large"})
            observed = datetime(2026, 9, day, tzinfo=UTC) + timedelta(minutes=index)
            snapshot = DecisionSnapshot(record_id=fingerprint([receipt, "candidate-1"]), family_id=f"authored-{split}-{index}",
                scope_sha256="a" * 64, policy_sha256=policy_fingerprint(policy), source_receipt_sha256=receipt,
                candidate_id="candidate-1", observed_at=observed, context=value.model_copy(deep=True)).seal()
            records.append(ReviewedDecision(snapshot=snapshot, split=split, preferred_role="defensive",
                reviewer_id="authored-control", independent_review=True, label_source="synthetic",
                reviewed_at=observed + timedelta(hours=1)))
    return records


def run_controls() -> tuple:
    records = authored_records()
    artifact = fit_advisor([row for row in records if row.split == "train"])
    report = evaluate_advisor(records, artifact)
    controls_passed = (artifact.verify() and report["decision"] == "held" and not report["serving_activation"]
        and "synthetic_or_mixed_review_labels" in report["blocking_reasons"]
        and all(value["coverage"] == 0.75 and value["accepted_review_agreement"] == 1
                and value["dispositions"] == {"supported": 6, "unseen_context": 2} for value in report["metrics"].values()))
    return artifact, report, controls_passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-gate", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = fresh_output(args.output_dir)
        artifact, report, controls_passed = run_controls()
        save_bundle(output, artifact, report)
        write_json(output / "controls.json", {"kind": "authored_dispatch_learning_controls",
            "controls_passed": controls_passed, "decision": "held", "serving_activation": False,
            "provider_calls": 0, "artifact_fingerprint": artifact.fingerprint})
        print(f"Authored dispatch learning controls passed: {controls_passed}; decision: held; serving activation: false")
        return int(args.require_gate or not controls_passed)
    except (OSError, ValueError, TypeError):
        print("Dispatch learning controls failed: invalid inputs or protected output path.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
