"""Budgeted, blinded supervision and coarse shift audits for dispatch advisors."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from code_agent.adaptive_dispatch import next_perspective
from code_agent.dispatch_learning import (
    SHA,
    DecisionSnapshot,
    DispatchAdvisor,
    DispatchAdvisorArtifact,
    ReviewedDecision,
    Role,
    StrictModel,
    fingerprint,
    fit_advisor,
)


class SealedModel(StrictModel):
    fingerprint: str = ""

    def seal(self):
        self.fingerprint = fingerprint(self.model_dump(mode="json", exclude={"fingerprint"}))
        return self

    def verify(self) -> bool:
        try:
            type(self).model_validate(self.model_dump(mode="json"))
            return self.fingerprint == fingerprint(self.model_dump(mode="json", exclude={"fingerprint"}))
        except (ValueError, TypeError):
            return False


FamilyId = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._:-]+$")]


class ReviewQueuePolicy(StrictModel):
    review_budget_slots: int = Field(default=24, ge=0, le=1000)
    reviewers_per_record: int = Field(default=2, ge=2, le=5)
    adjudication_reserve_slots: int = Field(default=2, ge=0, le=100)
    max_per_context: int = Field(default=3, ge=1, le=100)
    audit_fraction: float = Field(default=0.25, ge=0, le=1)
    seed: int = Field(default=17, ge=0, le=2**31 - 1)
    reserved_families: list[FamilyId] = Field(default_factory=list, max_length=10000)

    @model_validator(mode="after")
    def valid_reserve(self):
        if self.adjudication_reserve_slots > self.review_budget_slots:
            raise ValueError("adjudication reserve exceeds human review budget")
        if len(set(self.reserved_families)) != len(self.reserved_families):
            raise ValueError("duplicate reserved families")
        return self


class QueueSelection(StrictModel):
    record_id: str = Field(pattern=SHA)
    selection: Literal["audit", "targeted"]
    priority: int = Field(ge=0, le=500)
    reason: Literal["unseen_context", "thin_support", "low_agreement", "heuristic_disagreement", "routine"]


class ReviewBatch(SealedModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_fingerprint: str = Field(pattern=SHA)
    policy_sha256: str = Field(pattern=SHA)
    scope_sha256: str = Field(pattern=SHA)
    pool_fingerprint: str = Field(pattern=SHA)
    created_at: AwareDatetime
    policy: ReviewQueuePolicy
    snapshots: list[DecisionSnapshot] = Field(max_length=500)
    selections: list[QueueSelection] = Field(max_length=500)
    excluded_records: int = Field(ge=0)
    planned_review_slots: int = Field(ge=0)
    training_only: Literal[True] = True
    serving_changed: Literal[False] = False

    @model_validator(mode="after")
    def consistent_batch(self):
        ids = [row.record_id for row in self.snapshots]
        capacity = (self.policy.review_budget_slots - self.policy.adjudication_reserve_slots) // self.policy.reviewers_per_record
        if (len(ids) != len(set(ids)) or ids != [row.record_id for row in self.selections]
                or len(ids) > capacity or self.planned_review_slots != len(ids) * self.policy.reviewers_per_record):
            raise ValueError("invalid review-slot allocation or selection binding")
        if len({row.family_id for row in self.snapshots}) != len(ids):
            raise ValueError("at most one decision per task family may be selected")
        contexts = Counter(fingerprint(row.context.model_dump(mode="json")) for row in self.snapshots)
        if any(count > self.policy.max_per_context for count in contexts.values()):
            raise ValueError("context review quota exceeded")
        for row in self.snapshots:
            if (not row.verify() or row.observed_at > self.created_at or row.family_id in self.policy.reserved_families
                    or row.scope_sha256 != self.scope_sha256 or row.policy_sha256 != self.policy_sha256):
                raise ValueError("invalid, future, reserved or mismatched snapshot")
        return self

    def reviewer_packet(self) -> dict:
        """Do not leak model proposal, rank, priority, reason, or private batch metadata."""
        return {"schema_version": "1.0", "batch_fingerprint": self.fingerprint,
                "snapshots": [row.model_dump(mode="json") for row in sorted(self.snapshots, key=lambda row: row.record_id)]}


def validate_pool(rows: list[DecisionSnapshot], artifact: DispatchAdvisorArtifact) -> None:
    if not artifact.verify() or len(rows) > 10000 or len({row.record_id for row in rows}) != len(rows):
        raise ValueError("valid artifact and bounded, unique snapshot pool required")
    for row in rows:
        DecisionSnapshot.model_validate(row.model_dump(mode="json"))
        if (not row.verify() or row.scope_sha256 != artifact.scope_sha256 or row.policy_sha256 != artifact.policy_sha256):
            raise ValueError("snapshot integrity or lineage mismatch")


def build_review_batch(rows: list[DecisionSnapshot], artifact: DispatchAdvisorArtifact, *, created_at: datetime | str,
                       policy: ReviewQueuePolicy | None = None) -> ReviewBatch:
    validate_pool(rows, artifact)
    policy = ReviewQueuePolicy.model_validate((policy or ReviewQueuePolicy()).model_dump())
    advisor = DispatchAdvisor(artifact)
    # Validate the timestamp once, including timezone awareness, before comparisons.
    timestamp = ReviewBatch(artifact_fingerprint=artifact.fingerprint, policy_sha256=artifact.policy_sha256,
        scope_sha256=artifact.scope_sha256, pool_fingerprint=fingerprint([]), created_at=created_at,
        policy=policy, snapshots=[], selections=[], excluded_records=0, planned_review_slots=0).created_at
    excluded = set(artifact.training_families) | set(policy.reserved_families)
    if any(row.observed_at > timestamp for row in rows):
        raise ValueError("cannot schedule unobserved decisions")
    pool = [row for row in rows if row.family_id not in excluded]
    candidates = {}
    for row in pool:
        value = advisor.recommend(row.context, candidate_id=row.candidate_id,
            baseline_role=next_perspective(row.context.category, set(row.context.used_roles)),
            policy_sha256=row.policy_sha256, scope_sha256=row.scope_sha256)
        if value.disposition == "unseen_context":
            priority, reason = 400, "unseen_context"
        elif value.disposition == "insufficient_support":
            priority, reason = 350, "thin_support"
        elif value.disposition == "low_agreement":
            priority, reason = 300 + round(100 * (1 - value.label_agreement)), "low_agreement"
        elif value.proposed_role != value.baseline_role:
            priority, reason = 250, "heuristic_disagreement"
        else:
            priority, reason = 100, "routine"
        candidates[row.record_id] = QueueSelection(record_id=row.record_id, selection="targeted", priority=priority, reason=reason)
    capacity = (policy.review_budget_slots - policy.adjudication_reserve_slots) // policy.reviewers_per_record
    chosen, selected, families, contexts = [], [], set(), Counter()

    def take(row, mode):
        context = fingerprint(row.context.model_dump(mode="json"))
        if row.family_id in families or contexts[context] >= policy.max_per_context:
            return False
        families.add(row.family_id)
        contexts[context] += 1
        chosen.append(row.model_copy(deep=True))
        selected.append(candidates[row.record_id].model_copy(update={"selection": mode}))
        return True

    audit_slots = min(capacity, math.ceil(capacity * policy.audit_fraction))
    for row in sorted(pool, key=lambda row: fingerprint([policy.seed, row.record_id])):
        if len(chosen) >= audit_slots:
            break
        take(row, "audit")
    for row in sorted(pool, key=lambda row: (-candidates[row.record_id].priority, row.record_id)):
        if len(chosen) >= capacity:
            break
        take(row, "targeted")
    return ReviewBatch(artifact_fingerprint=artifact.fingerprint, policy_sha256=artifact.policy_sha256,
        scope_sha256=artifact.scope_sha256, pool_fingerprint=fingerprint([row.model_dump(mode="json") for row in sorted(rows, key=lambda row: row.record_id)]),
        created_at=timestamp, policy=policy, snapshots=chosen, selections=selected,
        excluded_records=len(rows) - len(pool), planned_review_slots=len(chosen) * policy.reviewers_per_record).seal()


class IndependentDispatchReview(SealedModel):
    record_id: str = Field(pattern=SHA)
    snapshot_fingerprint: str = Field(pattern=SHA)
    batch_fingerprint: str = Field(pattern=SHA)
    reviewer_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._:-]+$")
    preferred_role: Role
    label_source: Literal["human", "synthetic"]
    independent_review: Literal[True]
    reviewed_at: AwareDatetime


class DispatchAdjudication(IndependentDispatchReview):
    preferred_role: Role | None = None
    reason: Literal["execution_evidence", "localization_evidence", "contract_evidence", "insufficient_evidence"]

    @model_validator(mode="after")
    def explicit_resolution(self):
        if (self.reason == "insufficient_evidence") != (self.preferred_role is None):
            raise ValueError("unresolved evidence must abstain; other adjudications require a role")
        return self


def _bind_review(value: IndependentDispatchReview, batch: ReviewBatch, snapshots: dict) -> None:
    snapshot = snapshots.get(value.record_id)
    if (not value.verify() or snapshot is None or value.batch_fingerprint != batch.fingerprint
            or value.snapshot_fingerprint != snapshot.fingerprint or value.reviewed_at < batch.created_at
            or value.preferred_role in snapshot.context.used_roles):
        raise ValueError("review integrity, timing, role mask or batch binding failed")


def consolidate_reviews(batch: ReviewBatch, reviews: list[IndependentDispatchReview],
                        adjudications: list[DispatchAdjudication] = ()) -> dict:
    if not batch.verify():
        raise ValueError("invalid review batch")
    if len(reviews) > batch.planned_review_slots or len(adjudications) > batch.policy.adjudication_reserve_slots:
        raise ValueError("human review budget exceeded")
    snapshots = {row.record_id: row for row in batch.snapshots}
    groups = defaultdict(list)
    for review in reviews:
        IndependentDispatchReview.model_validate(review.model_dump(mode="json"))
        _bind_review(review, batch, snapshots)
        groups[review.record_id].append(review)
    resolved = {}
    for review in adjudications:
        DispatchAdjudication.model_validate(review.model_dump(mode="json"))
        _bind_review(review, batch, snapshots)
        if review.record_id in resolved:
            raise ValueError("duplicate adjudication")
        resolved[review.record_id] = review
    records, audit, pairs_agree, pairs_total = [], [], 0, 0
    for snapshot in sorted(batch.snapshots, key=lambda row: row.record_id):
        values = sorted(groups[snapshot.record_id], key=lambda row: row.reviewer_id)
        if len({row.reviewer_id for row in values}) != len(values) or len(values) > batch.policy.reviewers_per_record:
            raise ValueError("duplicate reviewers or per-record budget exceeded")
        for index, left in enumerate(values):
            for right in values[index + 1:]:
                pairs_total += 1
                pairs_agree += left.preferred_role == right.preferred_role
        adjudication = resolved.get(snapshot.record_id)
        role, disposition = None, "awaiting_reviews"
        if len(values) == batch.policy.reviewers_per_record:
            if len({row.preferred_role for row in values}) == 1:
                role, disposition = values[0].preferred_role, "unanimous"
            else:
                disposition = "conflict"
        if adjudication:
            if (disposition != "conflict" or adjudication.reviewer_id in {row.reviewer_id for row in values}
                    or adjudication.reviewed_at < max(row.reviewed_at for row in values)):
                raise ValueError("adjudication requires a later, separate reviewer and complete conflicting reviews")
            if adjudication.reason != "insufficient_evidence":
                role, disposition = adjudication.preferred_role, "adjudicated"
            else:
                disposition = "unresolved"
        sources = values + ([adjudication] if adjudication else [])
        if role:
            records.append(ReviewedDecision(snapshot=snapshot.model_copy(deep=True), split="train", preferred_role=role,
                reviewer_id="consensus:" + fingerprint([row.fingerprint for row in sources])[:24], independent_review=True,
                label_source="human" if all(row.label_source == "human" for row in sources) else "synthetic",
                reviewed_at=max(row.reviewed_at for row in sources)).model_dump(mode="json"))
        audit.append({"record_id": snapshot.record_id, "disposition": disposition,
                      "review_fingerprints": [row.fingerprint for row in values],
                      "adjudication_fingerprint": adjudication.fingerprint if adjudication else None})
    result = {"schema_version": "1.0", "batch_fingerprint": batch.fingerprint, "records": records, "audit": audit,
              "used_review_slots": len(reviews) + len(adjudications), "training_only": True, "serving_activation": False,
              "pairwise_review_agreement": pairs_agree / pairs_total if pairs_total else None,
              "unresolved_records": sum(row["disposition"] not in {"unanimous", "adjudicated"} for row in audit),
              "limitations": ["Agreement is not correctness. Reviewer independence and identity are operator declarations.",
                              "Targeted and audit selections are training-only, not unbiased evaluation or logged propensities.",
                              "Synthetic input reviews remain synthetic even after human adjudication."]}
    result["fingerprint"] = fingerprint(result)
    return result


class ShiftAuditPolicy(StrictModel):
    min_reference_families: int = Field(default=5, ge=2, le=1000)
    min_current_families: int = Field(default=20, ge=2, le=1000)
    max_feature_total_variation: float = Field(default=0.35, ge=0, le=1)
    max_unseen_context_rate: float = Field(default=0.4, ge=0, le=1)


def _distribution(rows: list[DecisionSnapshot], feature: str) -> dict[str, float]:
    groups = defaultdict(list)
    for row in rows:
        groups[row.family_id].append(row)
    counts = defaultdict(Fraction)
    for family in sorted(groups):
        values = groups[family]
        for row in values:
            value = getattr(row.context, feature)
            key = ",".join(value) if isinstance(value, list) else value
            counts[key] += Fraction(1, len(values) * len(groups))
    return {key: float(value) for key, value in sorted(counts.items())}


def audit_distribution_shift(reference: list[ReviewedDecision], current: list[DecisionSnapshot],
                             artifact: DispatchAdvisorArtifact, policy: ShiftAuditPolicy | None = None) -> dict:
    policy = ShiftAuditPolicy.model_validate((policy or ShiftAuditPolicy()).model_dump())
    if fit_advisor(reference, artifact.thresholds) != artifact:
        raise ValueError("exact training-only reference lineage required")
    validate_pool(current, artifact)
    if set(artifact.training_families) & {row.family_id for row in current}:
        raise ValueError("current families overlap training reference")
    if any(row.observed_at <= max(value.reviewed_at for value in reference) for row in current):
        raise ValueError("current observations must follow available reference labels")
    original = [row.snapshot for row in reference]
    features = ("category", "used_roles", "changed_files", "patch_size", "call_headroom")
    shifts = {}
    for feature in features:
        before, after = _distribution(original, feature), _distribution(current, feature)
        shifts[feature] = {"reference": before, "current": after,
                          "total_variation": sum(abs(before.get(key, 0) - after.get(key, 0)) for key in sorted(before.keys() | after.keys())) / 2 if current else None}
    groups = defaultdict(list)
    for row in current:
        groups[row.family_id].append(row)
    unseen = sum(float(sum(fingerprint(row.context.model_dump(mode="json")) not in artifact.cells for row in groups[family]) / len(groups[family]))
                 for family in sorted(groups)) / len(groups) if groups else None
    reasons = [f"{feature}:feature_shift" for feature in features if shifts[feature]["total_variation"] is not None
               and shifts[feature]["total_variation"] > policy.max_feature_total_variation]
    if unseen is not None and unseen > policy.max_unseen_context_rate:
        reasons.append("unseen_context_shift")
    enough = len(artifact.training_families) >= policy.min_reference_families and len(groups) >= policy.min_current_families
    result = {"schema_version": "1.0", "artifact_fingerprint": artifact.fingerprint,
              "pool_fingerprint": fingerprint([row.model_dump(mode="json") for row in sorted(current, key=lambda row: row.record_id)]),
              "policy": policy.model_dump(mode="json"), "reference_families": len(artifact.training_families),
              "current_families": len(groups), "features": shifts, "unseen_context_rate": unseen,
              "status": "insufficient_data" if not enough else "shifted" if reasons else "stable",
              "signals": reasons if enough else ["insufficient_independent_families"], "serving_activation": False,
              "limitations": ["Coarse marginal shift signals are not semantic OOD detection or evidence of coding-quality change.",
                              "Thresholds are engineering diagnostics, not calibrated statistical tests or risk guarantees.",
                              "Family identity and observation chronology are declared metadata, not external attestation."]}
    result["fingerprint"] = fingerprint(result)
    return result
