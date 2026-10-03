"""Preregistered, non-serving validation of an independently trained verifier."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import Field

from agent.adaptive_compute import (
    CandidateAssessment,
    ComputePlan,
    ComputePolicy,
    DeliberationReceipt,
    _agreement,
    select_candidate,
)
from agent.execution_replay import ExecutionReplayStore, Observation, ReviewLabel, digest
from agent.preference_ranking import StrictModel
from agent.process_reward import ProcessRewardScorer
from agent.process_supervision import (
    ProcessCohort,
    ProcessSnapshot,
    ProcessSupervisionStore,
    ReviewedProcessCandidate,
)
from agent.prospective_validation import SignedRecord

HEX = r"^[a-f0-9]{64}$"


class VerifierTrialPolicy(StrictModel):
    primary_threshold: float = Field(default=0.6, ge=0, le=1)
    curve_thresholds: list[float] = Field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.6, 0.75, 1.0], min_length=1, max_length=16
    )
    embargo_seconds: float = Field(default=60.0, ge=0)
    min_reviewed_releases_per_scope: int = Field(default=20, ge=20)
    min_release_coverage: float = Field(default=0.5, ge=0.5, le=1)
    min_review_coverage: float = Field(default=0.8, ge=0.8, le=1)
    max_error_upper_95: float = Field(default=0.2, gt=0, le=0.2)

    def check(self):
        if (
            self.curve_thresholds != sorted(set(self.curve_thresholds))
            or self.primary_threshold not in self.curve_thresholds
            or any(not 0 <= value <= 1 for value in self.curve_thresholds)
        ):
            raise ValueError("register sorted unique thresholds including the primary threshold")


class VerifierStudy(SignedRecord):
    study_id: str = Field(pattern=HEX)
    tenant: str = Field(pattern=HEX)
    registered_at: float = Field(ge=0)
    incumbent_fingerprint: str = Field(pattern=r"^(confidence-only|[a-f0-9]{64})$")
    compute_policy: ComputePolicy
    trial_policy: VerifierTrialPolicy
    candidate: ReviewedProcessCandidate
    training_cohort: ProcessCohort
    training_report: dict


class VerifierChoice(StrictModel):
    snapshot: ProcessSnapshot
    eligible: bool
    conformal_decision: Literal["release", "abstain", "not_evaluated"]
    consensus: float = Field(ge=0, le=1)
    baseline_reward: float | None = Field(default=None, ge=0, le=1)
    score: float = Field(ge=0, le=1)
    tie_rank: int = Field(ge=0, le=4)


class VerifierComparison(SignedRecord):
    study_id: str = Field(pattern=HEX)
    request_group: str = Field(pattern=HEX)
    captured_at: float = Field(ge=0)
    baseline_event: str = Field(pattern=r"^(|[a-f0-9]{64})$")
    proposed_event: str = Field(pattern=r"^(|[a-f0-9]{64})$")
    minimum_confidence: float = Field(ge=0, le=1)
    required_consensus: float = Field(ge=0, le=1)
    choices: list[VerifierChoice] = Field(min_length=1, max_length=5)

    def releasable(self, shadow=False):
        return [
            row
            for row in self.choices
            if row.eligible
            and row.snapshot.source.grounded
            and row.conformal_decision != "abstain"
            and row.snapshot.source.confidence >= self.minimum_confidence
            and row.consensus >= self.required_consensus
        ]

    def select(self, threshold: float) -> str:
        row = next((row for row in self.choices if row.snapshot.source.event_id == self.proposed_event), None)
        return self.proposed_event if row is not None and row.score >= threshold else ""

    def verify(self, key: bytes) -> None:
        super().verify(key)
        events, ranks = set(), set()
        for row in self.choices:
            row.snapshot.verify(key)
            source = row.snapshot.source
            if (
                source.event_id in events
                or row.tie_rank in ranks
                or source.request_group != self.request_group
                or row.snapshot.captured_at > self.captured_at
            ):
                raise ValueError("verifier comparison pool binding failed")
            events.add(source.event_id)
            ranks.add(row.tie_rank)
        if ranks != set(range(len(self.choices))):
            raise ValueError("invalid verifier tie ranks")
        if (
            len(
                {
                    (
                        row.snapshot.source.tenant,
                        row.snapshot.source.receipt,
                        row.snapshot.source.route,
                        row.snapshot.source.high_risk,
                        row.snapshot.family.fingerprint,
                    )
                    for row in self.choices
                }
            )
            != 1
        ):
            raise ValueError("verifier pool crosses source requests")

        def choose(shadow):
            selected = max(
                self.releasable(shadow),
                key=lambda row: (
                    row.score
                    if shadow
                    else row.baseline_reward
                    if row.baseline_reward is not None
                    else row.snapshot.source.confidence,
                    row.snapshot.source.confidence,
                    row.consensus,
                    -row.snapshot.source.estimated_output_tokens,
                    row.tie_rank,
                ),
                default=None,
            )
            return selected.snapshot.source.event_id if selected else ""

        if (
            self.baseline_event != choose(False)
            or self.proposed_event != choose(True)
            or [row.snapshot.source.event_id for row in self.choices if row.snapshot.source.selected]
            != ([self.baseline_event] if self.baseline_event else [])
        ):
            raise ValueError("verifier choice violates frozen release policy")


class VerifierMember(StrictModel):
    comparison: VerifierComparison
    task_family: str = Field(pattern=HEX)
    labels: list[ReviewLabel | None]


class VerifierCohort(SignedRecord):
    study: VerifierStudy
    frozen_at: float = Field(ge=0)
    members: list[VerifierMember]
    exclusions: dict[str, int]


class VerifierShadowStore:
    # Fixed schemas/tables let other NON-SERVING verifier studies reuse source,
    # eligibility, chronology and review controls without changing old payloads.
    study_schema = VerifierStudy
    choice_schema = VerifierChoice
    comparison_schema = VerifierComparison
    member_schema = VerifierMember
    cohort_schema = VerifierCohort
    study_table = "verifier_studies"
    comparison_table = "verifier_comparisons"

    def __init__(self, replay: ExecutionReplayStore, model_key: bytes):
        if len(model_key) < 32 or model_key == replay.key:
            raise ValueError("verifier shadow requires an independent model key")
        self.replay, self.model_key = replay, model_key
        if (self.study_table, self.comparison_table) not in {
            ("verifier_studies", "verifier_comparisons"),
            ("ensemble_outcome_studies", "ensemble_outcome_comparisons"),
        }:
            raise ValueError("unsupported verifier shadow tables")
        self.process = ProcessSupervisionStore(replay)
        with replay._db() as db:
            db.execute(
                f"CREATE TABLE IF NOT EXISTS {self.study_table} (study_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            # Deliberately retain comparisons on individual source deletion: lineage must fail, not shrink silently.
            db.execute(
                f"CREATE TABLE IF NOT EXISTS {self.comparison_table} (study_id TEXT NOT NULL REFERENCES {self.study_table}(study_id) ON DELETE CASCADE, request_group TEXT NOT NULL, tenant TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(study_id, request_group))"
            )

    def check_study(self, study: VerifierStudy, tenant: str):
        key = self.replay.key
        study.verify(key)
        study.trial_policy.check()
        study.candidate.verify(self.model_key)
        report = study.training_report
        if (
            study.tenant != digest(key, "tenant", tenant)
            or study.candidate.tenant != digest(self.model_key, "process-model-tenant", tenant)
            or study.candidate.cohort_fingerprint != study.training_cohort.fingerprint
            or study.candidate.simulation != study.training_cohort.simulation
            or study.training_cohort.frozen_at > study.registered_at
            or report.get("fingerprint")
            != digest(
                key, "process-supervision-report", {k: v for k, v in report.items() if k != "fingerprint"}
            )
            or study.candidate.report_fingerprint != report.get("fingerprint")
            or report.get("cohort_fingerprint") != study.training_cohort.fingerprint
            or report.get("gate_passed") is not True
            or report.get("policy", {}).get("version") != "explicit-workflow-supervision-v1"
            or (not study.candidate.simulation and report.get("candidate_ready_for_review") is not True)
        ):
            raise ValueError("verifier training provenance or tenant mismatch")
        self.process.validate_lineage(tenant, study.training_cohort)

    def register(
        self,
        name: str,
        tenant: str,
        candidate: ReviewedProcessCandidate,
        training: ProcessCohort,
        report: dict,
        compute: ComputePolicy,
        policy: VerifierTrialPolicy,
        incumbent_fingerprint: str = "confidence-only",
        **study_fields,
    ) -> VerifierStudy:
        if not tenant.strip() or not name.strip() or len(name) > 128:
            raise ValueError("verifier study requires name and tenant")
        study = self.study_schema(
            study_id=digest(self.replay.key, "verifier-study", [tenant, name]),
            tenant=digest(self.replay.key, "tenant", tenant),
            registered_at=self.replay.clock(),
            incumbent_fingerprint=incumbent_fingerprint,
            compute_policy=compute,
            trial_policy=policy,
            candidate=candidate,
            training_cohort=training,
            training_report=report,
            **study_fields,
        ).seal(self.replay.key)
        self.check_study(study, tenant)
        with self.replay._db() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute(
                f"SELECT payload FROM {self.study_table} WHERE study_id=?", (study.study_id,)
            ).fetchone()
            if previous:
                old = self.study_schema.model_validate_json(previous[0])
                self.check_study(old, tenant)
                ignored = {"registered_at", "fingerprint"}
                if old.model_dump(exclude=ignored) != study.model_dump(exclude=ignored):
                    raise ValueError("registered verifier study is immutable")
                return old
            db.execute(
                f"INSERT INTO {self.study_table} VALUES (?, ?, ?)",
                (study.study_id, study.tenant, study.model_dump_json()),
            )
        return study

    def study(self, study_id: str, tenant: str) -> VerifierStudy:
        with self.replay._db() as db:
            row = db.execute(
                f"SELECT tenant, payload FROM {self.study_table} WHERE study_id=?", (study_id,)
            ).fetchone()
        if row is None:
            raise ValueError("register verifier study before capture")
        study = self.study_schema.model_validate_json(row[1])
        self.check_study(study, tenant)
        if study.study_id != study_id or row[0] != study.tenant:
            raise ValueError("verifier study index mismatch")
        return study

    def assess_snapshot(self, study, snapshot):
        score = ProcessRewardScorer(study.candidate.artifact).score_steps(
            [step.process_step(i) for i, step in enumerate(snapshot.steps)], snapshot.source.high_risk
        )
        return score, {}

    def proposal_allowed(self, extras):
        return True

    def comparison_extras(self, study, choices):
        return {}

    def review_events(self, study, row):
        return {row.baseline_event, *(row.select(t) for t in study.trial_policy.curve_thresholds)} - {""}

    def report_extras(self, cohort):
        return {}

    def capture(
        self,
        study_id: str,
        tenant: str,
        request_id: str,
        plan: ComputePlan,
        baseline: DeliberationReceipt,
        candidates: list[CandidateAssessment],
        policy: ComputePolicy,
        *,
        consent: bool,
        incumbent_fingerprint: str = "confidence-only",
        compute_key: bytes | None = None,
    ):
        if consent is not True or not tenant.strip() or not request_id.strip():
            return None
        study = self.study(study_id, tenant)
        rebuilt = select_candidate(plan, candidates, policy, compute_key, baseline.attempted_candidates)
        if (
            rebuilt != baseline
            or baseline.preference_ranking is not None
            or policy != study.compute_policy
            or incumbent_fingerprint != study.incumbent_fingerprint
            or (
                incumbent_fingerprint == "confidence-only"
                and any(row.process_reward is not None for row in candidates)
            )
        ):
            raise ValueError("verifier shadow incumbent or compute policy mismatch")
        group = digest(self.replay.key, "request", [tenant, request_id])
        rows = self.process.rows(tenant)
        snapshots = {
            snap.source.candidate: snap for snap, _ in rows.values() if snap.source.request_group == group
        }
        assessments = {candidate: self.assess_snapshot(study, snap) for candidate, snap in snapshots.items()}
        scores = {candidate: assessment[0] for candidate, assessment in assessments.items()}
        alternate = select_candidate(
            plan,
            [
                row.model_copy(
                    update={
                        "process_reward": scores.get(digest(self.replay.key, "candidate", row.candidate_id))
                    }
                )
                for row in candidates
            ],
            policy,
            compute_key,
            baseline.attempted_candidates,
        )
        if [row.model_dump(exclude={"process_reward"}) for row in alternate.candidate_summaries] != [
            row.model_dump(exclude={"process_reward"}) for row in baseline.candidate_summaries
        ]:
            raise ValueError("verifier shadow changed eligibility or candidate pool")
        ranks = {
            name: index
            for index, name in enumerate(sorted(row.candidate_id for row in baseline.candidate_summaries))
        }
        choices = []
        assessed = {candidate.candidate_id: candidate for candidate in candidates}
        eligible = [assessed[row.candidate_id] for row in baseline.candidate_summaries if row.eligible]
        for row in baseline.candidate_summaries:
            # Preserve the selector's unrounded Jaccard value at threshold boundaries.
            consensus = max(
                (
                    _agreement(assessed[row.candidate_id], other)
                    for other in eligible
                    if other.candidate_id != row.candidate_id
                ),
                default=0.0,
            )
            if round(consensus, 6) != row.consensus:
                raise ValueError("verifier consensus binding failed")
            snap = snapshots.get(digest(self.replay.key, "candidate", row.candidate_id))
            if (
                snap is None
                or snap.source.receipt != digest(self.replay.key, "receipt", baseline.receipt_fingerprint)
                or snap.source.origin != ("synthetic" if study.candidate.simulation else "runtime")
                or snap.source.confidence != row.confidence
                or snap.source.grounded != row.grounded
                or snap.source.estimated_output_tokens != row.token_count
                or snap.source.latency_ms != row.latency_ms
                or snap.source.route != plan.route
                or snap.source.high_risk != plan.high_risk
                or row.eligible
                != (
                    row.grounded
                    and row.conformal_decision != "abstain"
                    and row.confidence >= min(1, plan.initial_confidence + policy.min_confidence_gain)
                )
            ):
                raise ValueError("verifier snapshot source binding failed")
            choices.append(
                self.choice_schema(
                    snapshot=snap,
                    eligible=row.eligible,
                    conformal_decision=row.conformal_decision,
                    consensus=consensus,
                    baseline_reward=row.process_reward,
                    score=scores[snap.source.candidate],
                    tie_rank=ranks[row.candidate_id],
                    **assessments[snap.source.candidate][1],
                )
            )
        if not choices or len(choices) != len(snapshots):
            raise ValueError("verifier requires complete workflow pool")

        def event_for(name):
            return snapshots[digest(self.replay.key, "candidate", name)].source.event_id if name else ""

        proposed = event_for(alternate.selected_candidate_id)
        if proposed and not self.proposal_allowed(
            assessments[digest(self.replay.key, "candidate", alternate.selected_candidate_id)][1]
        ):
            proposed = ""
        comparison = self.comparison_schema(
            study_id=study_id,
            request_group=group,
            captured_at=self.replay.clock(),
            baseline_event=event_for(baseline.selected_candidate_id),
            proposed_event=proposed,
            minimum_confidence=min(1, plan.initial_confidence + policy.min_confidence_gain),
            required_consensus=policy.high_risk_min_consensus if plan.high_risk else policy.min_consensus,
            choices=choices,
            **self.comparison_extras(study, choices),
        ).seal(self.replay.key)
        comparison.verify(self.replay.key)
        with self.replay._db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                f"SELECT payload FROM {self.comparison_table} WHERE study_id=? AND request_group=?",
                (study_id, group),
            ).fetchone()
            if old:
                previous = self.comparison_schema.model_validate_json(old[0])
                previous.verify(self.replay.key)
                if previous.model_dump(exclude={"captured_at", "fingerprint"}) != comparison.model_dump(
                    exclude={"captured_at", "fingerprint"}
                ):
                    raise ValueError("verifier comparison is immutable")
                return previous
            for choice in choices:
                source = choice.snapshot.source
                current = db.execute(
                    "SELECT payload FROM process_snapshots WHERE event_id=?", (source.event_id,)
                ).fetchone()
                observed = db.execute(
                    "SELECT payload FROM observations WHERE event_id=?", (source.event_id,)
                ).fetchone()
                if (
                    current is None
                    or ProcessSnapshot.model_validate_json(current[0]) != choice.snapshot
                    or observed is None
                    or Observation.model_validate_json(observed[0]) != source
                ):
                    raise ValueError("verifier source changed during capture")
                if (
                    db.execute("SELECT 1 FROM labels WHERE event_id=?", (source.event_id,)).fetchone()
                    or db.execute(
                        "SELECT 1 FROM process_step_reviews WHERE event_id=?", (source.event_id,)
                    ).fetchone()
                ):
                    raise ValueError("verifier comparison must precede all reviews")
                if source.observed_at <= study.registered_at or comparison.captured_at < source.observed_at:
                    raise ValueError("verifier request must follow registration")
            db.execute(
                f"INSERT INTO {self.comparison_table} VALUES (?, ?, ?, ?)",
                (study_id, group, study.tenant, comparison.model_dump_json()),
            )
        return comparison

    def comparisons(self, study_id: str, tenant: str):
        study = self.study(study_id, tenant)
        with self.replay._db() as db:
            data = db.execute(
                f"SELECT request_group, tenant, payload FROM {self.comparison_table} WHERE study_id=?",
                (study_id,),
            ).fetchall()
        result = []
        snapshots = self.process.rows(tenant)
        for group, tenant_hash, payload in data:
            row = self.comparison_schema.model_validate_json(payload)
            row.verify(self.replay.key)
            if row.request_group != group or row.study_id != study_id or tenant_hash != study.tenant:
                raise ValueError("verifier comparison index mismatch")
            for choice in row.choices:
                snap = choice.snapshot
                score, extras = self.assess_snapshot(study, snap)
                if (
                    snapshots.get(snap.source.event_id, (None,))[0] != snap
                    or choice.score != score
                    or any(getattr(choice, name) != value for name, value in extras.items())
                ):
                    raise ValueError("verifier workflow lineage or score changed")
            result.append(row)
        return sorted(result, key=lambda row: (row.captured_at, row.request_group))

    def freeze(self, study_id: str, tenant: str, *, frozen_at: float | None = None):
        study = self.study(study_id, tenant)
        frozen_at = self.replay.clock() if frozen_at is None else frozen_at
        if not study.registered_at <= frozen_at <= self.replay.clock():
            raise ValueError("invalid verifier freeze time")
        observations, families = self.replay.snapshot(tenant)
        labels = {obs.event_id: label for obs, label in observations}
        earliest = {}
        for obs, _ in observations:
            family = families.get(obs.request_group)
            if family and family.task_family and obs.observed_at <= frozen_at:
                value = (obs.observed_at, obs.request_group)
                earliest[family.task_family] = min(earliest.get(family.task_family, value), value)
        excluded, members = Counter(), []
        for row in self.comparisons(study_id, tenant):
            if row.captured_at > frozen_at:
                continue
            family = families[row.request_group]
            first, group = earliest[family.task_family]
            if first <= study.registered_at + study.trial_policy.embargo_seconds:
                excluded["preexisting_family_or_embargo"] += 1
                continue
            if group != row.request_group:
                excluded["repeat_family"] += 1
                continue
            if any(
                label is not None and label.reviewed_at < row.captured_at
                for choice in row.choices
                if (label := labels.get(choice.snapshot.source.event_id)) is not None
            ):
                raise ValueError("verifier review predates comparison")
            members.append(
                self.member_schema(
                    comparison=row,
                    task_family=family.task_family,
                    labels=[
                        label
                        if (label := labels.get(choice.snapshot.source.event_id)) is not None
                        and label.reviewed_at <= frozen_at
                        else None
                        for choice in row.choices
                    ],
                )
            )
        return self.cohort_schema(
            study=study, frozen_at=frozen_at, members=members, exclusions=dict(excluded)
        ).seal(self.replay.key)

    def validate_lineage(self, cohort: VerifierCohort, tenant: str):
        cohort.verify(self.replay.key)
        if self.freeze(cohort.study.study_id, tenant, frozen_at=cohort.frozen_at) != cohort:
            raise ValueError("verifier frozen lineage changed")

    def queue(self, study_id: str, tenant: str):
        cohort = self.freeze(study_id, tenant)
        result = []
        for member in cohort.members:
            row = member.comparison
            # Review the union of every preregistered curve choice and the incumbent, not only disagreements.
            events = self.review_events(cohort.study, row)
            known = {
                choice.snapshot.source.event_id
                for choice, label in zip(row.choices, member.labels, strict=True)
                if label is not None and label.verdict != "ambiguous"
            }
            if events - known:
                result.append(
                    {
                        "request_group": row.request_group,
                        "event_ids": sorted(events - known),
                        "priority": "disagreement"
                        if row.baseline_event != row.select(cohort.study.trial_policy.primary_threshold)
                        else "audit",
                    }
                )
        return sorted(result, key=lambda row: (row["priority"] != "disagreement", row["request_group"]))
