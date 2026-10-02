"""Frozen workflow features and genuinely independent per-step annotations."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import Field

from agent.execution_replay import (
    ExecutionReplayStore,
    Observation,
    RequestFamily,
    ReviewLabel,
    _verify,
    digest,
)
from agent.preference_ranking import StrictModel
from agent.process_reward import ProcessRewardArtifact, ProcessStep, ProcessTrace, StepKind
from agent.prospective_validation import SignedRecord

HEX = r"^[a-f0-9]{64}$"


class ReviewedProcessCandidate(SignedRecord):
    tenant: str = Field(pattern=HEX)
    cohort_fingerprint: str = Field(pattern=HEX)
    report_fingerprint: str = Field(pattern=HEX)
    simulation: bool
    calibration_temperature: float = Field(ge=0.25, le=4)
    explicit_training_steps: int = Field(ge=1)
    artifact: ProcessRewardArtifact

    def verify(self, key: bytes) -> None:
        super().verify(key)
        if not self.artifact.verify():
            raise ValueError("reviewed process candidate contains invalid model")


class WorkflowStep(StrictModel):
    kind: StepKind
    has_evidence: bool
    citation_valid: bool
    policy_allowed: bool
    error: bool
    confidence: float = Field(ge=0, le=1)

    def process_step(self, index: int, target: float | None = None) -> ProcessStep:
        return ProcessStep(step_id=f"{index}-{self.kind}", step_label=target, **self.model_dump())


class ProcessSnapshot(SignedRecord):
    source: Observation
    family: RequestFamily
    captured_at: float = Field(ge=0)
    steps: list[WorkflowStep] = Field(min_length=1, max_length=64)

    def verify(self, key: bytes) -> None:
        super().verify(key)
        _verify(self.source, key)
        _verify(self.family, key)
        if (
            self.family.tenant != self.source.tenant
            or self.family.request_group != self.source.request_group
            or not self.family.task_family
            or self.captured_at < self.source.observed_at
            or self.steps[-1].kind != "answer"
            or self.steps[-1].confidence != self.source.confidence
        ):
            raise ValueError("invalid workflow snapshot source binding")


class StepAnnotation(SignedRecord):
    event_id: str = Field(pattern=HEX)
    step_index: int = Field(ge=0, le=63)
    snapshot_fingerprint: str = Field(pattern=HEX)
    verdict: Literal["correct", "incorrect", "ambiguous"]
    reviewer: str = Field(pattern=HEX)
    reviewed_at: float = Field(ge=0)


class ProcessMember(StrictModel):
    snapshot: ProcessSnapshot
    split: Literal["train", "validation", "test"]
    labels: list[StepAnnotation | None]
    outcome: ReviewLabel | None = None


class ProcessCohort(SignedRecord):
    tenant: str = Field(pattern=HEX)
    train_cutoff: float = Field(gt=0)
    validation_cutoff: float = Field(gt=0)
    embargo_seconds: float = Field(ge=0)
    frozen_at: float = Field(gt=0)
    simulation: bool
    members: list[ProcessMember]
    exclusions: dict[str, int]

    def verify(self, key: bytes) -> None:
        super().verify(key)
        if (
            not self.train_cutoff + self.embargo_seconds < self.validation_cutoff
            or not self.validation_cutoff + self.embargo_seconds < self.frozen_at
        ):
            raise ValueError("invalid process cohort time windows")
        families, events = {}, set()
        for member in self.members:
            snap = member.snapshot
            snap.verify(key)
            if snap.source.tenant != self.tenant or (not self.simulation and snap.source.origin != "runtime"):
                raise ValueError("process cohort tenant or origin mismatch")
            group = (snap.source.request_group, member.split)
            family = snap.family.task_family
            if (
                families.setdefault(family, group) != group
                or snap.source.event_id in events
                or len(member.labels) != len(snap.steps)
            ):
                raise ValueError("process family crosses folds or duplicate candidate")
            events.add(snap.source.event_id)
            cutoff = (
                self.train_cutoff
                if member.split == "train"
                else self.validation_cutoff
                if member.split == "validation"
                else self.frozen_at
            )
            start = (
                0
                if member.split == "train"
                else self.train_cutoff + self.embargo_seconds
                if member.split == "validation"
                else self.validation_cutoff + self.embargo_seconds
            )
            if not start < snap.source.observed_at <= snap.captured_at <= cutoff:
                raise ValueError("process snapshot violates chronological fold")
            for index, label in enumerate(member.labels):
                if label is None:
                    continue
                label.verify(key)
                if (
                    label.event_id != snap.source.event_id
                    or label.step_index != index
                    or label.snapshot_fingerprint != snap.fingerprint
                    or not snap.captured_at <= label.reviewed_at <= cutoff
                ):
                    raise ValueError("invalid independent step annotation binding")
            if member.outcome:
                _verify(member.outcome, key)
                if (
                    member.outcome.event_id != snap.source.event_id
                    or member.outcome.observation_fingerprint != snap.source.fingerprint
                    or not snap.source.observed_at <= member.outcome.reviewed_at <= cutoff
                ):
                    raise ValueError("process outcome chronology or source binding mismatch")

    def traces(self) -> list[ProcessTrace]:
        traces = []
        for member in self.members:
            snap = member.snapshot
            outcome = member.outcome
            known = outcome is not None and outcome.verdict != "ambiguous"
            traces.append(
                ProcessTrace(
                    trace_id=snap.source.event_id,
                    group_id=snap.family.task_family,
                    query="metadata-only observed workflow checks",
                    high_risk=snap.source.high_risk,
                    self_confidence=snap.source.confidence,
                    steps=[
                        step.process_step(
                            i,
                            float(label.verdict == "correct")
                            if label and label.verdict != "ambiguous"
                            else None,
                        )
                        for i, (step, label) in enumerate(zip(snap.steps, member.labels, strict=True))
                    ],
                    outcome_quality=float(outcome.verdict == "correct" and not outcome.unsafe)
                    if known
                    else 0.5,
                    safe=known and not outcome.unsafe,
                    split=member.split,
                    review_status="synthetic_seed" if self.simulation else "human_reviewed",
                )
            )
        return traces


class ProcessSupervisionStore:
    def __init__(self, replay: ExecutionReplayStore):
        self.replay = replay
        with replay._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS process_snapshots (event_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS process_step_reviews (event_id TEXT NOT NULL REFERENCES process_snapshots(event_id) ON DELETE CASCADE, step_index INTEGER NOT NULL, tenant TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(event_id, step_index))"
            )

    def snapshot(self, tenant: str, event_id: str) -> ProcessSnapshot:
        rows = self.rows(tenant)
        if event_id not in rows:
            raise ValueError("workflow snapshot not found for tenant")
        return rows[event_id][0]

    def rows(self, tenant: str) -> dict[str, tuple[ProcessSnapshot, dict[int, StepAnnotation]]]:
        records, families = self.replay.snapshot(tenant)
        sources = {obs.event_id: obs for obs, _ in records}
        tenant_hash = digest(self.replay.key, "tenant", tenant)
        with self.replay._db() as db:
            db.execute("BEGIN")
            snapshots = db.execute(
                "SELECT event_id, payload FROM process_snapshots WHERE tenant=?", (tenant_hash,)
            ).fetchall()
            labels = db.execute(
                "SELECT event_id, step_index, payload FROM process_step_reviews WHERE tenant=?",
                (tenant_hash,),
            ).fetchall()
        result = {}
        for event_id, payload in snapshots:
            snapshot = ProcessSnapshot.model_validate_json(payload)
            snapshot.verify(self.replay.key)
            if (
                snapshot.source.event_id != event_id
                or snapshot.source.tenant != tenant_hash
                or sources.get(event_id) != snapshot.source
                or families.get(snapshot.source.request_group) != snapshot.family
            ):
                raise ValueError("workflow source lineage changed")
            result[event_id] = (snapshot, {})
        for event_id, index, payload in labels:
            label = StepAnnotation.model_validate_json(payload)
            label.verify(self.replay.key)
            if (
                event_id not in result
                or index != label.step_index
                or label.event_id != event_id
                or index >= len(result[event_id][0].steps)
                or label.snapshot_fingerprint != result[event_id][0].fingerprint
                or label.reviewed_at < result[event_id][0].captured_at
            ):
                raise ValueError("step review index or source binding failed")
            result[event_id][1][index] = label
        return result

    def capture(
        self, tenant: str, event_id: str, steps: list[ProcessStep], *, consent: bool
    ) -> ProcessSnapshot | None:
        if consent is not True or not tenant.strip():
            return None
        if any(step.step_label is not None for step in steps):
            raise ValueError("cannot capture targets as pre-review features")
        records, families = self.replay.snapshot(tenant)
        source = next((obs for obs, _ in records if obs.event_id == event_id), None)
        if source is None or source.request_group not in families:
            raise ValueError("process capture requires original replay observation and family")
        features = [
            WorkflowStep(**{name: getattr(step, name) for name in WorkflowStep.model_fields})
            for step in steps
        ]
        snapshot = ProcessSnapshot(
            source=source,
            family=families[source.request_group],
            captured_at=self.replay.clock(),
            steps=features,
        ).seal(self.replay.key)
        snapshot.verify(self.replay.key)
        with self.replay._db() as db:
            db.execute("BEGIN IMMEDIATE")
            original = db.execute("SELECT payload FROM observations WHERE event_id=?", (event_id,)).fetchone()
            family = db.execute(
                "SELECT payload FROM request_families WHERE request_group=?", (source.request_group,)
            ).fetchone()
            if (
                original is None
                or family is None
                or Observation.model_validate_json(original[0]) != source
                or RequestFamily.model_validate_json(family[0]) != snapshot.family
            ):
                raise ValueError("process source changed before capture")
            existing = db.execute(
                "SELECT payload FROM process_snapshots WHERE event_id=?", (event_id,)
            ).fetchone()
            if existing:
                previous = ProcessSnapshot.model_validate_json(existing[0])
                previous.verify(self.replay.key)
                if previous.model_dump(exclude={"captured_at", "fingerprint"}) != snapshot.model_dump(
                    exclude={"captured_at", "fingerprint"}
                ):
                    raise ValueError("workflow features are immutable")
                return previous
            labelled = db.execute(
                "SELECT 1 FROM observations o LEFT JOIN labels l ON l.event_id=o.event_id LEFT JOIN process_step_reviews r ON r.event_id=o.event_id WHERE o.tenant=? AND json_extract(o.payload, '$.request_group')=? AND (l.event_id IS NOT NULL OR r.event_id IS NOT NULL) LIMIT 1",
                (source.tenant, source.request_group),
            ).fetchone()
            if labelled:
                raise ValueError("complete workflow pool must precede any review")
            db.execute(
                "INSERT INTO process_snapshots VALUES (?, ?, ?)",
                (event_id, source.tenant, snapshot.model_dump_json()),
            )
        return snapshot

    def review(self, tenant: str, event_id: str, index: int, verdict: str, reviewer: str) -> StepAnnotation:
        if not reviewer.strip():
            raise ValueError("step reviewer identity is required")
        snapshot = self.snapshot(tenant, event_id)
        if not 0 <= index < len(snapshot.steps):
            raise ValueError("unknown workflow step index")
        label = StepAnnotation(
            event_id=event_id,
            step_index=index,
            snapshot_fingerprint=snapshot.fingerprint,
            verdict=verdict,
            reviewer=digest(self.replay.key, "step-reviewer", reviewer),
            reviewed_at=self.replay.clock(),
        ).seal(self.replay.key)
        if label.reviewed_at < snapshot.captured_at:
            raise ValueError("step review precedes feature capture")
        with self.replay._db() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT payload FROM process_snapshots WHERE event_id=?", (event_id,)
            ).fetchone()
            if current is None or ProcessSnapshot.model_validate_json(current[0]) != snapshot:
                raise ValueError("workflow snapshot changed before review")
            original = db.execute("SELECT payload FROM observations WHERE event_id=?", (event_id,)).fetchone()
            if original is None or Observation.model_validate_json(original[0]) != snapshot.source:
                raise ValueError("workflow source changed before review")
            existing = db.execute(
                "SELECT payload FROM process_step_reviews WHERE event_id=? AND step_index=?",
                (event_id, index),
            ).fetchone()
            if existing:
                previous = StepAnnotation.model_validate_json(existing[0])
                previous.verify(self.replay.key)
                if previous.model_dump(exclude={"reviewed_at", "fingerprint"}) != label.model_dump(
                    exclude={"reviewed_at", "fingerprint"}
                ):
                    raise ValueError("step annotations are immutable")
                return previous
            db.execute(
                "INSERT INTO process_step_reviews VALUES (?, ?, ?, ?)",
                (event_id, index, snapshot.source.tenant, label.model_dump_json()),
            )
        return label

    def freeze(
        self,
        tenant: str,
        *,
        train_cutoff: float,
        validation_cutoff: float,
        embargo_seconds: float,
        frozen_at: float | None = None,
        simulation: bool = False,
    ) -> ProcessCohort:
        frozen_at = self.replay.clock() if frozen_at is None else frozen_at
        if frozen_at > self.replay.clock():
            raise ValueError("cannot freeze future process evidence")
        rows = self.rows(tenant)
        observations, families = self.replay.snapshot(tenant)
        groups, first = {}, {}
        for obs, outcome in observations:
            if obs.observed_at > frozen_at:
                continue
            groups.setdefault(obs.request_group, []).append((obs, outcome))
            family = families.get(obs.request_group)
            if family and family.task_family:
                value = (obs.observed_at, obs.request_group)
                first[family.task_family] = min(first.get(family.task_family, value), value)
        excluded, members = Counter(), []
        for group, sources in sorted(groups.items()):
            if not any(obs.event_id in rows for obs, _ in sources):
                continue
            family = families[group]
            start, earliest = first[family.task_family]
            if group != earliest:
                excluded["repeat_family"] += 1
                continue
            if start <= train_cutoff:
                split, cutoff = "train", train_cutoff
            elif train_cutoff + embargo_seconds < start <= validation_cutoff:
                split, cutoff = "validation", validation_cutoff
            elif validation_cutoff + embargo_seconds < start:
                split, cutoff = "test", frozen_at
            else:
                excluded["embargo"] += 1
                continue
            if any(obs.event_id not in rows for obs, _ in sources):
                raise ValueError("incomplete workflow candidate pool")
            if any(rows[obs.event_id][0].captured_at > cutoff for obs, _ in sources):
                excluded["snapshot_after_fold_cutoff"] += 1
                continue
            for obs, outcome in sources:
                snapshot, labels = rows[obs.event_id]
                members.append(
                    ProcessMember(
                        snapshot=snapshot,
                        split=split,
                        labels=[
                            labels.get(i) if labels.get(i) and labels[i].reviewed_at <= cutoff else None
                            for i in range(len(snapshot.steps))
                        ],
                        outcome=outcome if outcome and outcome.reviewed_at <= cutoff else None,
                    )
                )
        cohort = ProcessCohort(
            tenant=digest(self.replay.key, "tenant", tenant),
            train_cutoff=train_cutoff,
            validation_cutoff=validation_cutoff,
            embargo_seconds=embargo_seconds,
            frozen_at=frozen_at,
            simulation=simulation,
            members=members,
            exclusions=dict(excluded),
        ).seal(self.replay.key)
        cohort.verify(self.replay.key)
        return cohort

    def validate_lineage(self, tenant: str, cohort: ProcessCohort) -> None:
        cohort.verify(self.replay.key)
        current = self.freeze(
            tenant,
            train_cutoff=cohort.train_cutoff,
            validation_cutoff=cohort.validation_cutoff,
            embargo_seconds=cohort.embargo_seconds,
            frozen_at=cohort.frozen_at,
            simulation=cohort.simulation,
        )
        if current != cohort:
            raise ValueError("frozen process lineage changed")

    def queue(self, tenant: str, *, limit: int = 100) -> list[dict]:
        result = []
        for event_id, (snapshot, labels) in sorted(self.rows(tenant).items()):
            missing = [index for index in range(len(snapshot.steps)) if index not in labels]
            if missing:
                result.append(
                    {
                        "event_id": event_id,
                        "step_indexes": missing,
                        "steps": [step.model_dump() for step in snapshot.steps],
                        "route": snapshot.source.route,
                        "high_risk": snapshot.source.high_risk,
                    }
                )
        return result[: max(1, min(limit, 1000))]
