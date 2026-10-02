"""Prospectively registered, content-free comparisons of two real selectors."""

from __future__ import annotations

import hmac
import time
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field

from agent.adaptive_compute import (
    ComputePlan,
    ComputePolicy,
    DeliberationReceipt,
    verify_plan,
    verify_receipt,
)
from agent.execution_replay import ExecutionReplayStore, ReviewLabel, digest
from agent.preference_ranking import PreferenceRanker, StrictModel
from agent.prospective_validation import SignedRecord

HEX = r"^[a-f0-9]{64}$"


class ShadowStudy(SignedRecord):
    schema_version: Literal["1.0"] = "1.0"
    study_id: str = Field(pattern=HEX)
    tenant: str = Field(pattern=HEX)
    artifact_fingerprint: str = Field(pattern=HEX)
    compute_policy_fingerprint: str = Field(pattern=HEX)
    registered_at: float = Field(ge=0)
    simulation: bool


class ShadowCandidate(StrictModel):
    event_id: str = Field(pattern=HEX)
    observation_fingerprint: str = Field(pattern=HEX)
    confidence: float = Field(ge=0, le=1)
    grounded: bool
    conformal_decision: Literal["release", "abstain", "not_evaluated"]
    eligible: bool
    consensus: float = Field(ge=0, le=1)
    process_reward: float | None = Field(default=None, ge=0, le=1)


class ShadowComparison(SignedRecord):
    study_id: str = Field(pattern=HEX)
    tenant: str = Field(pattern=HEX)
    request_group: str = Field(pattern=HEX)
    artifact_fingerprint: str = Field(pattern=HEX)
    baseline_receipt: str = Field(pattern=HEX)
    shadow_receipt: str = Field(pattern=HEX)
    route: str
    high_risk: bool
    status: Literal["released", "abstained", "budget_exhausted"]
    baseline_event: str = Field(pattern=r"^(|[a-f0-9]{64})$")
    shadow_event: str = Field(pattern=r"^(|[a-f0-9]{64})$")
    required_consensus: float = Field(ge=0, le=1)
    minimum_confidence: float = Field(ge=0, le=1)
    ranking_status: Literal["reranked", "fallback"]
    ranking_reason: str
    improvement_lcb: float
    observed_at: float = Field(ge=0)
    candidates: list[ShadowCandidate] = Field(min_length=1, max_length=5)

    def verify(self, key: bytes) -> None:
        super().verify(key)
        rows = {row.event_id: row for row in self.candidates}
        if len(rows) != len(self.candidates):
            raise ValueError("duplicate shadow candidate")
        for event in (self.baseline_event, self.shadow_event):
            if self.status == "released":
                row = rows.get(event)
                if (
                    not row
                    or not row.eligible
                    or not row.grounded
                    or row.conformal_decision == "abstain"
                    or row.confidence < self.minimum_confidence
                    or row.consensus < self.required_consensus
                ):
                    raise ValueError("shadow choice violates original release gates")
            elif event:
                raise ValueError("shadow cannot release a baseline abstention")
        if self.ranking_status == "fallback" and self.baseline_event != self.shadow_event:
            raise ValueError("shadow fallback changed selection")


class ShadowMember(StrictModel):
    comparison: ShadowComparison
    task_family: str = Field(pattern=HEX)
    family_fingerprint: str = Field(pattern=HEX)
    first_seen: float = Field(ge=0)
    baseline_label: ReviewLabel | None
    shadow_label: ReviewLabel | None


class ShadowCohort(SignedRecord):
    study: ShadowStudy
    frozen_at: float = Field(ge=0)
    embargo_seconds: float = Field(ge=0)
    members: list[ShadowMember]
    exclusions: dict[str, int]

    def verify(self, key: bytes) -> None:
        super().verify(key)
        self.study.verify(key)
        families = set()
        for member in self.members:
            row = member.comparison
            row.verify(key)
            if (
                member.task_family in families
                or row.study_id != self.study.study_id
                or row.tenant != self.study.tenant
                or row.artifact_fingerprint != self.study.artifact_fingerprint
                or row.status != "released"
                or member.first_seen <= self.study.registered_at + self.embargo_seconds
                or member.first_seen > row.observed_at
                or row.observed_at > self.frozen_at
            ):
                raise ValueError("invalid forward-time shadow cohort")
            families.add(member.task_family)
            observations = {item.event_id: item for item in row.candidates}
            for label, event in (
                (member.baseline_label, row.baseline_event),
                (member.shadow_label, row.shadow_event),
            ):
                if label is None:
                    continue
                if label.fingerprint != digest(key, "ReviewLabel", label.model_dump(exclude={"fingerprint"})):
                    raise ValueError("shadow label integrity failed")
                if (
                    label.event_id != event
                    or label.observation_fingerprint != observations[event].observation_fingerprint
                    or label.reviewed_at < row.observed_at
                    or label.reviewed_at > self.frozen_at
                ):
                    raise ValueError("invalid delayed shadow label binding")


class ShadowApproval(SignedRecord):
    """A bounded, artifact/policy/scope-specific lease, not automatic deployment."""

    tenant: str = Field(pattern=HEX)
    artifact_fingerprint: str = Field(pattern=HEX)
    compute_policy_fingerprint: str = Field(pattern=HEX)
    study_id: str = Field(pattern=HEX)
    cohort_fingerprint: str = Field(pattern=HEX)
    report_fingerprint: str = Field(pattern=HEX)
    scopes: list[str] = Field(min_length=1)
    reviewed_families: int = Field(ge=20)
    review_coverage: float = Field(ge=0.8, le=1)
    error_upper_95: float = Field(ge=0, le=0.2)
    utility_lower_95: float = Field(ge=0)
    unsafe_selections: Literal[0] = 0
    issued_at: float = Field(ge=0)
    expires_at: float = Field(ge=0)

    def verify(self, key: bytes) -> None:
        super().verify(key)
        if not 0 < self.expires_at - self.issued_at <= 7 * 86400 or len(set(self.scopes)) != len(self.scopes):
            raise ValueError("invalid shadow approval lease")


def validate_approval(
    path: str | Path,
    ranker: PreferenceRanker,
    key: bytes,
    policy: ComputePolicy,
    *,
    route: str,
    high_risk: bool,
    now: float | None = None,
) -> ShadowApproval:
    approval = ShadowApproval.model_validate_json(Path(path).read_text(encoding="utf-8"))
    return check_approval(approval, ranker, key, policy, route=route, high_risk=high_risk, now=now)


def check_approval(
    approval: ShadowApproval,
    ranker: PreferenceRanker,
    key: bytes,
    policy: ComputePolicy,
    *,
    route: str,
    high_risk: bool,
    now: float | None = None,
) -> ShadowApproval:
    """Validate a lease in memory or loaded from the legacy file interface."""
    approval.verify(key)
    now = time.time() if now is None else now
    if (
        approval.tenant != digest(key, "preference-tenant", ranker.tenant)
        or approval.artifact_fingerprint != ranker.artifact.fingerprint
        or approval.compute_policy_fingerprint != digest(key, "approved-compute-policy", policy.model_dump())
        or f"{route}:{int(high_risk)}" not in approval.scopes
        or not approval.issued_at <= now < approval.expires_at
        or ranker.artifact.simulation
    ):
        raise ValueError("shadow approval is expired or incompatible")
    return approval


class PreferenceShadowStore:
    def __init__(self, replay: ExecutionReplayStore):
        self.replay = replay
        with replay._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS preference_shadow_studies (study_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS preference_shadows (study_id TEXT NOT NULL REFERENCES preference_shadow_studies(study_id), request_group TEXT NOT NULL REFERENCES request_families(request_group) ON DELETE CASCADE, payload TEXT NOT NULL, PRIMARY KEY (study_id, request_group))"
            )

    def register(
        self, name: str, tenant: str, ranker: PreferenceRanker, policy: ComputePolicy
    ) -> ShadowStudy:
        if not name.strip() or len(name) > 128 or not tenant.strip() or tenant != ranker.tenant:
            raise ValueError("shadow registration requires study name and tenant")
        key = self.replay.key
        study_id = digest(key, "shadow-study", [tenant, name])
        study = ShadowStudy(
            study_id=study_id,
            tenant=digest(key, "tenant", tenant),
            artifact_fingerprint=ranker.artifact.fingerprint,
            compute_policy_fingerprint=digest(key, "shadow-compute-policy", policy.model_dump()),
            registered_at=self.replay.clock(),
            simulation=ranker.artifact.simulation,
        ).seal(key)
        with self.replay._db() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT payload FROM preference_shadow_studies WHERE study_id=?", (study_id,)
            ).fetchone()
            if existing:
                previous = ShadowStudy.model_validate_json(existing[0])
                previous.verify(key)
                ignored = {"fingerprint", "registered_at"}
                if previous.model_dump(exclude=ignored) != study.model_dump(exclude=ignored):
                    raise ValueError("shadow study identity is immutable")
                return previous
            db.execute(
                "INSERT INTO preference_shadow_studies VALUES (?, ?, ?)",
                (study_id, study.tenant, study.model_dump_json()),
            )
        return study

    def study(self, study_id: str, tenant: str) -> ShadowStudy:
        with self.replay._db() as db:
            row = db.execute(
                "SELECT payload, tenant FROM preference_shadow_studies WHERE study_id=?", (study_id,)
            ).fetchone()
        if row is None:
            raise ValueError("register shadow study before execution")
        study = ShadowStudy.model_validate_json(row[0])
        study.verify(self.replay.key)
        if (
            study.study_id != study_id
            or study.tenant != row[1]
            or study.tenant != digest(self.replay.key, "tenant", tenant)
        ):
            raise ValueError("shadow study tenant/index mismatch")
        return study

    def capture(
        self,
        study_id: str,
        tenant: str,
        request_id: str,
        plan: ComputePlan,
        baseline: DeliberationReceipt,
        shadow: DeliberationReceipt,
        policy: ComputePolicy,
        *,
        consent: bool,
        compute_key: bytes | None = None,
    ) -> ShadowComparison | None:
        if consent is not True or not request_id or not tenant:
            return None
        if not baseline.candidate_summaries:
            return None
        key = self.replay.key
        study = self.study(study_id, tenant)
        if not verify_plan(plan, compute_key) or not all(
            verify_receipt(row, compute_key) for row in (baseline, shadow)
        ):
            raise ValueError("shadow compute integrity failed")
        audit = shadow.preference_ranking or {}
        if (
            baseline.preference_ranking is not None
            or baseline.plan_fingerprint != plan.plan_fingerprint
            or shadow.plan_fingerprint != plan.plan_fingerprint
            or baseline.status != shadow.status
            or baseline.candidate_summaries != shadow.candidate_summaries
            or baseline.extra_tokens != shadow.extra_tokens
            or baseline.latency_ms != shadow.latency_ms
            or baseline.attempted_candidates != shadow.attempted_candidates
            or policy.version != plan.policy_version
            or policy.version != baseline.policy_version
            or digest(key, "shadow-compute-policy", policy.model_dump()) != study.compute_policy_fingerprint
            or (
                baseline.status == "released"
                and audit.get("artifact_fingerprint") != study.artifact_fingerprint
            )
        ):
            raise ValueError("shadow does not compare the registered selectors on the same pool")
        group = digest(key, "request", [tenant, request_id])
        observations, _ = self.replay.snapshot(tenant)
        rows = {obs.candidate: (obs, label) for obs, label in observations if obs.request_group == group}
        candidates = []
        for summary in baseline.candidate_summaries:
            obs, _ = rows.get(digest(key, "candidate", summary.candidate_id), (None, None))
            if (
                obs is None
                or obs.receipt != digest(key, "receipt", baseline.receipt_fingerprint)
                or obs.origin != ("synthetic" if study.simulation else "runtime")
                or obs.selected != (summary.candidate_id == baseline.selected_candidate_id)
                or obs.confidence != summary.confidence
                or obs.grounded != summary.grounded
                or obs.estimated_output_tokens != summary.token_count
                or obs.latency_ms != summary.latency_ms
                or obs.route != plan.route
                or obs.high_risk != plan.high_risk
            ):
                raise ValueError("shadow replay source binding failed")
            candidates.append(
                ShadowCandidate(
                    event_id=obs.event_id,
                    observation_fingerprint=obs.fingerprint,
                    confidence=summary.confidence,
                    grounded=summary.grounded,
                    conformal_decision=summary.conformal_decision,
                    eligible=summary.eligible,
                    consensus=summary.consensus,
                    process_reward=summary.process_reward,
                )
            )
        if len(candidates) != len(rows):
            raise ValueError("shadow must capture the complete evaluated pool")

        def event_for(candidate):
            return digest(key, "event", [group, digest(key, "candidate", candidate)]) if candidate else ""

        comparison = ShadowComparison(
            study_id=study_id,
            tenant=study.tenant,
            request_group=group,
            artifact_fingerprint=study.artifact_fingerprint,
            baseline_receipt=digest(key, "receipt", baseline.receipt_fingerprint),
            shadow_receipt=digest(key, "receipt", shadow.receipt_fingerprint),
            route=plan.route,
            high_risk=plan.high_risk,
            status=baseline.status,
            baseline_event=event_for(baseline.selected_candidate_id),
            shadow_event=event_for(shadow.selected_candidate_id),
            required_consensus=policy.high_risk_min_consensus if plan.high_risk else policy.min_consensus,
            minimum_confidence=min(1, plan.initial_confidence + policy.min_confidence_gain),
            ranking_status=audit.get("status", "fallback"),
            ranking_reason=audit.get("reason", "no_releasable_candidate"),
            improvement_lcb=audit.get("improvement_lcb", 0),
            observed_at=min(obs.observed_at for obs, _ in rows.values()),
            candidates=candidates,
        ).seal(key)
        comparison.verify(key)
        with self.replay._db() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT payload FROM preference_shadows WHERE study_id=? AND request_group=?",
                (study_id, group),
            ).fetchone()
            if existing:
                if ShadowComparison.model_validate_json(existing[0]) != comparison:
                    raise ValueError("conflicting shadow capture retry")
                return comparison
            # No post-outcome creation of purported prospective comparisons.
            if any(
                db.execute("SELECT 1 FROM labels WHERE event_id=?", (candidate.event_id,)).fetchone()
                for candidate in candidates
            ):
                raise ValueError("shadow comparison must precede review labels")
            if comparison.observed_at < study.registered_at:
                raise ValueError("shadow request preceded study registration")
            db.execute(
                "INSERT INTO preference_shadows VALUES (?, ?, ?)",
                (study_id, group, comparison.model_dump_json()),
            )
        return comparison

    def comparisons(self, study_id: str, tenant: str) -> list[ShadowComparison]:
        study = self.study(study_id, tenant)
        with self.replay._db() as db:
            rows = db.execute(
                "SELECT request_group, payload FROM preference_shadows WHERE study_id=?", (study_id,)
            ).fetchall()
        result = []
        for group, payload in rows:
            item = ShadowComparison.model_validate_json(payload)
            item.verify(self.replay.key)
            if (
                item.request_group != group
                or item.study_id != study_id
                or item.tenant != study.tenant
                or item.artifact_fingerprint != study.artifact_fingerprint
            ):
                raise ValueError("shadow comparison index integrity failed")
            result.append(item)
        return sorted(result, key=lambda item: (item.observed_at, item.request_group))

    def freeze(
        self, study_id: str, tenant: str, *, embargo_seconds: float = 3600, frozen_at: float | None = None
    ) -> ShadowCohort:
        study = self.study(study_id, tenant)
        frozen_at = self.replay.clock() if frozen_at is None else frozen_at
        if frozen_at > self.replay.clock() or frozen_at < study.registered_at:
            raise ValueError("shadow snapshot time is outside observed history")
        observations, families = self.replay.snapshot(tenant)
        current = {obs.event_id: (obs, label) for obs, label in observations}
        earliest = {}
        for obs, _ in observations:
            family = families.get(obs.request_group)
            if family and family.task_family and obs.observed_at <= frozen_at:
                value = (obs.observed_at, obs.request_group)
                earliest[family.task_family] = min(earliest.get(family.task_family, value), value)
        exclusions = Counter()
        members = []
        for row in self.comparisons(study_id, tenant):
            if row.observed_at > frozen_at:
                continue
            family = families.get(row.request_group)
            if not family or not family.task_family:
                exclusions["missing_preassigned_family"] += 1
                continue
            first_seen, group = earliest[family.task_family]
            if first_seen <= study.registered_at + embargo_seconds:
                exclusions["preexisting_family_or_embargo"] += 1
                continue
            if group != row.request_group:
                exclusions["repeated_family_request"] += 1
                continue
            if row.status != "released":
                exclusions["baseline_abstention"] += 1
                continue
            for candidate in row.candidates:
                obs, _ = current.get(candidate.event_id, (None, None))
                if obs is None or obs.fingerprint != candidate.observation_fingerprint:
                    raise ValueError("shadow source lineage revoked")
            labels = []
            for event in (row.baseline_event, row.shadow_event):
                label = current[event][1]
                labels.append(label if label is not None and label.reviewed_at <= frozen_at else None)
            members.append(
                ShadowMember(
                    comparison=row,
                    task_family=family.task_family,
                    family_fingerprint=family.fingerprint,
                    first_seen=first_seen,
                    baseline_label=labels[0],
                    shadow_label=labels[1],
                )
            )
        cohort = ShadowCohort(
            study=study,
            frozen_at=frozen_at,
            embargo_seconds=embargo_seconds,
            members=members,
            exclusions=dict(exclusions),
        ).seal(self.replay.key)
        cohort.verify(self.replay.key)
        return cohort

    def validate_lineage(self, cohort: ShadowCohort, tenant: str) -> None:
        cohort.verify(self.replay.key)
        if cohort.study != self.study(cohort.study.study_id, tenant):
            raise ValueError("shadow study lineage changed")
        # Recreate the exact as-of snapshot; later labels cannot improve cached evidence.
        current = self.freeze(
            cohort.study.study_id, tenant, embargo_seconds=cohort.embargo_seconds, frozen_at=cohort.frozen_at
        )
        if not hmac.compare_digest(current.fingerprint, cohort.fingerprint):
            raise ValueError("shadow cohort lineage revoked or changed")

    def review_queue(self, study_id: str, tenant: str, *, audit_percent: int = 10) -> list[dict]:
        if not 0 <= audit_percent <= 100:
            raise ValueError("audit percent must be between zero and 100")
        records, _ = self.replay.snapshot(tenant)
        labels = {obs.event_id: label for obs, label in records}
        queue = []
        for row in self.comparisons(study_id, tenant):
            if row.status != "released":
                continue
            missing = [
                event for event in sorted({row.baseline_event, row.shadow_event}) if labels.get(event) is None
            ]
            if not missing:
                continue
            disagreement = row.baseline_event != row.shadow_event
            audit = (
                int(digest(self.replay.key, "shadow-review-audit", row.request_group)[:8], 16) % 100
                < audit_percent
            )
            if disagreement or audit:
                queue.append(
                    {
                        "request_group": row.request_group,
                        "event_ids": missing,
                        "priority": "disagreement" if disagreement else "audit",
                        "improvement_lcb": row.improvement_lcb,
                    }
                )
        return sorted(queue, key=lambda item: (item["priority"] != "disagreement", item["request_group"]))
