"""Authored active-review, consensus and shift controls; no provider or activation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from code_agent.dispatch_learning import DecisionSnapshot, fingerprint, fit_advisor
from code_agent.dispatch_learning_cli import fresh_output, write_json
from code_agent.dispatch_supervision import (
    DispatchAdjudication,
    IndependentDispatchReview,
    ReviewQueuePolicy,
    audit_distribution_shift,
    build_review_batch,
    consolidate_reviews,
)
from evals.dispatch_learning_evaluation import authored_records


def authored_pool(*, count: int = 20, unseen: int = 4) -> list[DecisionSnapshot]:
    template = authored_records()[0].snapshot
    pool = []
    for index in range(count):
        row = template.model_copy(deep=True)
        row.family_id = f"new-authored-family-{index:03}"
        row.source_receipt_sha256 = fingerprint(["authored-supervision-receipt", index])
        row.record_id = fingerprint([row.source_receipt_sha256, row.candidate_id])
        row.observed_at = datetime(2026, 10, 1, tzinfo=UTC) + timedelta(minutes=index)
        row.context.patch_size = "large" if index < unseen else "small"
        pool.append(row.seal())
    return pool


def authored_reviews(batch, *, conflict: bool = True) -> list[IndependentDispatchReview]:
    reviews = []
    for index, row in enumerate(batch.snapshots):
        for reviewer in range(batch.policy.reviewers_per_record):
            reviews.append(IndependentDispatchReview(record_id=row.record_id, snapshot_fingerprint=row.fingerprint,
                batch_fingerprint=batch.fingerprint, reviewer_id=f"scripted-reviewer-{reviewer}",
                preferred_role="boundary" if conflict and index == 0 and reviewer == 0 else "defensive",
                label_source="synthetic", independent_review=True,
                reviewed_at=batch.created_at + timedelta(hours=1, minutes=reviewer)).seal())
    return reviews


def run_controls() -> dict:
    reference = [row for row in authored_records() if row.split == "train"]
    artifact = fit_advisor(reference)
    pool = authored_pool()
    batch = build_review_batch(pool, artifact, created_at="2026-10-02T00:00:00Z",
        policy=ReviewQueuePolicy(review_budget_slots=10, adjudication_reserve_slots=2, max_per_context=2))
    reviews = authored_reviews(batch)
    conflict = consolidate_reviews(batch, reviews)
    first = batch.snapshots[0]
    adjudication = DispatchAdjudication(record_id=first.record_id, snapshot_fingerprint=first.fingerprint,
        batch_fingerprint=batch.fingerprint, reviewer_id="scripted-third-reviewer", preferred_role="defensive",
        label_source="synthetic", independent_review=True, reason="execution_evidence",
        reviewed_at=batch.created_at + timedelta(hours=2)).seal()
    resolved = consolidate_reviews(batch, reviews, [adjudication])
    stable = audit_distribution_shift(reference, authored_pool(unseen=0), artifact)
    shifted = audit_distribution_shift(reference, authored_pool(unseen=20), artifact)
    small = audit_distribution_shift(reference, pool[:3], artifact)
    packet = batch.reviewer_packet()
    checks = {
        "selection_budget": batch.planned_review_slots == 8 and len(batch.snapshots) == 4,
        "blinded_packet": set(packet) == {"schema_version", "batch_fingerprint", "snapshots"},
        "conflict_held": conflict["unresolved_records"] == 1 and len(conflict["records"]) == 3,
        "independent_adjudication": resolved["unresolved_records"] == 0 and len(resolved["records"]) == 4,
        "synthetic_not_laundered": all(row["label_source"] == "synthetic" and row["split"] == "train" for row in resolved["records"]),
        "stable_control": stable["status"] == "stable",
        "shift_control": shifted["status"] == "shifted" and shifted["unseen_context_rate"] == 1,
        "small_sample_hold": small["status"] == "insufficient_data",
        "reproducible_selection": batch == build_review_batch(list(reversed(pool)), artifact,
                                                              created_at=batch.created_at, policy=batch.policy),
    }
    return {"schema_version": "1.0", "kind": "authored_dispatch_supervision_controls", "checks": checks,
            "controls_passed": all(checks.values()), "decision": "held", "serving_activation": False, "provider_calls": 0,
            "artifact": artifact.model_dump(mode="json"), "batch": batch.model_dump(mode="json"),
            "consensus": resolved, "shift": shifted,
            "limitations": ["Scripted labels and authored snapshots, not observed live-agent quality or reviewer-efficiency gains.",
                            "No code, reference repository, model provider or Docker execution is needed for these controls."]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-gate", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = fresh_output(args.output_dir)
        report = run_controls()
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "controls.json", report)
        for key in ("artifact", "batch", "consensus", "shift"):
            write_json(output / f"{key}.json", report[key])
        with (output / "review.md").open("x", encoding="utf-8") as stream:
            stream.write("# Dispatch supervision controls\n\nDecision: held; serving activation: false; provider calls: zero.\n\n"
                         + "\n".join(f"- {name}: {'passed' if passed else 'failed'}" for name, passed in report["checks"].items())
                         + "\n\nThese authored controls exercise contracts, not coding-quality or annotation-efficiency gains.\n")
        print(f"Dispatch supervision controls passed: {report['controls_passed']}; decision: held; serving activation: false")
        return int(args.require_gate or not report["controls_passed"])
    except (OSError, ValueError, TypeError):
        print("Dispatch supervision controls failed: invalid inputs or protected output path.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
