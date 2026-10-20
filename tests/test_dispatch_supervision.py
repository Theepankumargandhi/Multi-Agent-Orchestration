from __future__ import annotations

from datetime import timedelta

import pytest

from code_agent.dispatch_learning import ReviewedDecision, fingerprint, fit_advisor
from code_agent.dispatch_supervision import (
    DispatchAdjudication,
    ReviewQueuePolicy,
    ShiftAuditPolicy,
    audit_distribution_shift,
    build_review_batch,
    consolidate_reviews,
)
from evals.dispatch_learning_evaluation import authored_records
from evals.dispatch_supervision_evaluation import authored_pool, authored_reviews


def reference():
    return [row for row in authored_records() if row.split == "train"]


def batch_for(pool=None, **policy):
    return build_review_batch(pool if pool is not None else authored_pool(), fit_advisor(reference()),
        created_at="2026-10-02T00:00:00Z", policy=ReviewQueuePolicy(review_budget_slots=10,
            adjudication_reserve_slots=2, max_per_context=2, **policy))


def reidentify(snapshot, tag):
    row = snapshot.model_copy(deep=True)
    row.source_receipt_sha256 = fingerprint(tag)
    row.record_id = fingerprint([row.source_receipt_sha256, row.candidate_id])
    return row.seal()


def adjudication_for(batch, reviews, *, reason="execution_evidence", role="defensive", reviewer="third-reviewer"):
    snapshot = batch.snapshots[0]
    return DispatchAdjudication(record_id=snapshot.record_id, snapshot_fingerprint=snapshot.fingerprint,
        batch_fingerprint=batch.fingerprint, reviewer_id=reviewer, preferred_role=role, label_source="human",
        independent_review=True, reviewed_at=max(row.reviewed_at for row in reviews) + timedelta(hours=1), reason=reason).seal()


def test_selection_is_order_invariant_budgeted_diverse_and_blinded():
    pool = authored_pool()
    artifact = fit_advisor(reference())
    batch = batch_for(pool)
    assert batch.verify() and batch == build_review_batch(list(reversed(pool)), artifact,
        created_at=batch.created_at, policy=batch.policy)
    assert batch.planned_review_slots == 8 and len(batch.snapshots) == 4
    assert batch.planned_review_slots + batch.policy.adjudication_reserve_slots <= batch.policy.review_budget_slots
    assert len({row.family_id for row in batch.snapshots}) == 4
    assert sum(row.selection == "audit" for row in batch.selections) == 1
    assert any(row.reason == "unseen_context" for row in batch.selections)
    packet = batch.reviewer_packet()
    assert set(packet) == {"schema_version", "batch_fingerprint", "snapshots"}
    encoded = str(packet)
    assert all(key not in encoded for key in ("proposed_role", "priority", "selections", "baseline_role", "reason", "artifact_fingerprint"))
    before = batch.fingerprint
    pool[0].context.patch_size = "empty"
    assert batch.verify() and batch.fingerprint == before


def test_training_and_reserved_holdout_families_are_never_selected():
    pool = authored_pool()
    reserved = pool[0].family_id
    reused = reidentify(pool[1], "same-training-family")
    reused.family_id = reference()[0].snapshot.family_id
    reused.seal()
    batch = batch_for(pool + [reused], reserved_families=[reserved])
    assert batch.excluded_records == 2
    assert reserved not in {row.family_id for row in batch.snapshots}
    assert reused.family_id not in {row.family_id for row in batch.snapshots}


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 5, 6])
def test_small_human_budgets_never_overspend(budget):
    policy = ReviewQueuePolicy(review_budget_slots=budget, adjudication_reserve_slots=0)
    batch = build_review_batch(authored_pool(), fit_advisor(reference()), created_at="2026-10-02T00:00:00Z", policy=policy)
    assert batch.verify() and batch.planned_review_slots <= budget
    assert len(batch.snapshots) <= budget // 2


@pytest.mark.parametrize("mutation", ["duplicate", "digest", "policy", "scope", "future"])
def test_invalid_pools_fail_closed(mutation):
    pool = authored_pool()
    if mutation == "duplicate":
        pool.append(pool[0])
    elif mutation == "digest":
        pool[0].family_id = "altered"
    elif mutation in {"policy", "scope"}:
        setattr(pool[0], f"{mutation}_sha256", "b" * 64)
        pool[0].seal()
    else:
        pool[0].observed_at += timedelta(days=2)
        pool[0].seal()
    with pytest.raises(ValueError):
        batch_for(pool)


def test_family_duplicates_cannot_dominate_selection():
    pool = authored_pool(count=4, unseen=2)
    for number in range(30):
        pool.append(reidentify(pool[0], ["family-repeat", number]))
    batch = batch_for(pool)
    assert len({row.family_id for row in batch.snapshots}) == len(batch.snapshots)
    assert sum(row.family_id == pool[0].family_id for row in batch.snapshots) <= 1


def test_routine_low_agreement_and_thin_support_priorities_are_recorded():
    rows = reference()
    for row in rows:
        row.preferred_role = "boundary"
    artifact = fit_advisor(rows)
    batch = build_review_batch(authored_pool(unseen=0), artifact, created_at="2026-10-02T00:00:00Z")
    assert {row.reason for row in batch.selections} == {"routine"}
    thin = build_review_batch(authored_pool(unseen=0), fit_advisor(reference()[:2]), created_at=batch.created_at)
    assert {row.reason for row in thin.selections} == {"thin_support"}
    rows[0].preferred_role = rows[1].preferred_role = "defensive"
    ambiguous = build_review_batch(authored_pool(unseen=0), fit_advisor(rows), created_at=batch.created_at)
    assert {row.reason for row in ambiguous.selections} == {"low_agreement"}


def test_missing_and_conflicting_reviews_do_not_generate_training_labels():
    batch = batch_for()
    empty = consolidate_reviews(batch, [])
    assert not empty["records"] and empty["pairwise_review_agreement"] is None
    reviews = authored_reviews(batch)
    report = consolidate_reviews(batch, reviews)
    assert report["unresolved_records"] == 1 and len(report["records"]) == 3
    assert report["pairwise_review_agreement"] == 0.75
    assert report == consolidate_reviews(batch, list(reversed(reviews)))
    assert report["training_only"] and not report["serving_activation"]
    assert all(ReviewedDecision.model_validate(row).split == "train" for row in report["records"])


def test_independent_adjudication_preserves_provenance_and_synthetic_source():
    batch = batch_for()
    reviews = authored_reviews(batch)
    adjudication = adjudication_for(batch, reviews)
    report = consolidate_reviews(batch, reviews, [adjudication])
    assert report["unresolved_records"] == 0 and report["used_review_slots"] == 9
    assert all(row["label_source"] == "synthetic" for row in report["records"])
    audit = next(row for row in report["audit"] if row["record_id"] == adjudication.record_id)
    assert audit["disposition"] == "adjudicated" and audit["adjudication_fingerprint"] == adjudication.fingerprint
    assert len(audit["review_fingerprints"]) == 2
    unresolved = consolidate_reviews(batch, reviews, [adjudication_for(batch, reviews, reason="insufficient_evidence", role=None)])
    assert unresolved["unresolved_records"] == 1 and len(unresolved["records"]) == 3


def test_unanimous_human_reviews_create_only_training_records():
    batch = batch_for()
    reviews = authored_reviews(batch, conflict=False)
    for row in reviews:
        row.label_source = "human"
        row.seal()
    report = consolidate_reviews(batch, reviews)
    assert report["unresolved_records"] == 0
    assert all(row["label_source"] == "human" and row["split"] == "train" for row in report["records"])
    assert fit_advisor([ReviewedDecision.model_validate(row) for row in report["records"]]).verify()


@pytest.mark.parametrize("mutation", ["digest", "batch", "snapshot", "unknown", "used", "timestamp", "duplicate", "overspend", "independence"])
def test_invalid_reviews_are_rejected(mutation):
    batch = batch_for()
    reviews = authored_reviews(batch)
    row = reviews[0]
    if mutation == "digest":
        row.reviewer_id = "changed"
    elif mutation in {"batch", "snapshot"}:
        setattr(row, f"{mutation}_fingerprint", "b" * 64)
        row.seal()
    elif mutation == "unknown":
        row.record_id = "b" * 64
        row.seal()
    elif mutation == "used":
        row.preferred_role = "minimal"
        row.seal()
    elif mutation == "timestamp":
        row.reviewed_at = batch.created_at - timedelta(seconds=1)
        row.seal()
    elif mutation == "duplicate":
        reviews[1].reviewer_id = row.reviewer_id
        reviews[1].seal()
    elif mutation == "overspend":
        reviews.append(row)
    else:
        row.independent_review = False
        row.seal()
    with pytest.raises(ValueError):
        consolidate_reviews(batch, reviews)


@pytest.mark.parametrize("mutation", ["same_reviewer", "early", "unanimous", "incomplete", "duplicate", "budget"])
def test_adjudication_cannot_bypass_conflicts_or_budget(mutation):
    batch = batch_for()
    reviews = authored_reviews(batch, conflict=mutation != "unanimous")
    adjudication = adjudication_for(batch, reviews)
    if mutation == "same_reviewer":
        adjudication.reviewer_id = reviews[0].reviewer_id
        adjudication.seal()
    elif mutation == "early":
        adjudication.reviewed_at = batch.created_at
        adjudication.seal()
    elif mutation == "incomplete":
        reviews = [row for row in reviews if row.fingerprint != reviews[0].fingerprint]
    values = [adjudication] * (3 if mutation == "budget" else 2 if mutation == "duplicate" else 1)
    with pytest.raises(ValueError):
        consolidate_reviews(batch, reviews, values)


@pytest.mark.parametrize("mutation", ["slots", "families", "mask", "reserve", "selection"])
def test_rehashed_invalid_batches_are_rejected(mutation):
    batch = batch_for()
    if mutation == "slots":
        batch.planned_review_slots += 1
    elif mutation == "families":
        batch.snapshots[1].family_id = batch.snapshots[0].family_id
        batch.snapshots[1].seal()
    elif mutation == "mask":
        batch.snapshots[0].context.used_roles = ["minimal", "minimal"]
        batch.snapshots[0].seal()
    elif mutation == "reserve":
        batch.policy.adjudication_reserve_slots = 11
    else:
        batch.selections[0].record_id = "b" * 64
    batch.seal()
    assert not batch.verify()
    with pytest.raises(ValueError):
        consolidate_reviews(batch, [])


def test_shift_audit_detects_coarse_shift_and_insufficient_data():
    rows = reference()
    artifact = fit_advisor(rows)
    stable = audit_distribution_shift(rows, authored_pool(unseen=0), artifact)
    shifted = audit_distribution_shift(rows, authored_pool(unseen=20), artifact)
    assert stable["status"] == "stable" and stable["unseen_context_rate"] == 0
    assert shifted["status"] == "shifted" and shifted["features"]["patch_size"]["total_variation"] == 1
    assert "patch_size:feature_shift" in shifted["signals"] and not shifted["serving_activation"]
    assert shifted == audit_distribution_shift(list(reversed(rows)), list(reversed(authored_pool(unseen=20))), artifact)
    empty = audit_distribution_shift(rows, [], artifact)
    assert empty["status"] == "insufficient_data" and empty["unseen_context_rate"] is None
    assert all(value["total_variation"] is None for value in empty["features"].values())
    small = audit_distribution_shift(rows[:2], authored_pool(unseen=0), fit_advisor(rows[:2]))
    assert small["status"] == "insufficient_data"


def test_shift_is_family_balanced_and_never_retrains():
    rows = reference()
    artifact = fit_advisor(rows)
    pool = authored_pool(count=2, unseen=1)
    for number in range(50):
        pool.append(reidentify(pool[0], ["duplicate-attempt", number]))
    before = artifact.model_dump_json()
    report = audit_distribution_shift(rows, pool, artifact, ShiftAuditPolicy(min_current_families=2))
    assert report["unseen_context_rate"] == 0.5
    assert report["features"]["patch_size"]["total_variation"] == 0.5
    assert artifact.model_dump_json() == before


@pytest.mark.parametrize("mutation", ["future_labels", "family_overlap", "reference_labels", "reference_split"])
def test_shift_rejects_training_lineage_and_temporal_leakage(mutation):
    rows = reference()
    artifact = fit_advisor(rows)
    pool = authored_pool()
    if mutation == "future_labels":
        pool[0].observed_at = rows[0].snapshot.observed_at
        pool[0].seal()
    elif mutation == "family_overlap":
        pool[0].family_id = rows[0].snapshot.family_id
        pool[0].seal()
    elif mutation == "reference_labels":
        rows[0].preferred_role = "boundary"
    else:
        rows[0].split = "test"
    with pytest.raises(ValueError):
        audit_distribution_shift(rows, pool, artifact)


def test_policy_validation_and_explicit_abstention():
    with pytest.raises(ValueError):
        ReviewQueuePolicy(review_budget_slots=1)
    with pytest.raises(ValueError):
        ReviewQueuePolicy(reserved_families=["same", "same"])
    with pytest.raises(ValueError):
        ReviewQueuePolicy(reserved_families=["unsafe\nfamily"])
    batch = batch_for()
    with pytest.raises(ValueError):
        adjudication_for(batch, authored_reviews(batch), reason="insufficient_evidence")


@pytest.mark.parametrize("mutation", ["context_quota", "future_observation", "reserved_family"])
def test_batch_cannot_be_rehashed_to_bypass_selection_quota_or_cutoff(mutation):
    batch = batch_for()
    if mutation == "context_quota":
        batch.policy.max_per_context = 1
    elif mutation == "future_observation":
        batch.snapshots[0].observed_at = batch.created_at + timedelta(seconds=1)
        batch.snapshots[0].seal()
    else:
        batch.policy.reserved_families = [batch.snapshots[0].family_id]
    batch.seal()
    assert not batch.verify()
