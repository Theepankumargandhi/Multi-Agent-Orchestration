"""Signed chronological, family-separated cohorts and a holdout exposure ledger."""

from __future__ import annotations

import hmac
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.execution_replay import ExecutionReplayStore, digest
from agent.uncertainty import CalibrationExample


class SignedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    fingerprint: str = ""

    def seal(self, key: bytes):
        self.fingerprint = digest(key, type(self).__name__, self.model_dump(exclude={"fingerprint"}))
        return self

    def verify(self, key: bytes) -> None:
        if len(key) < 32:
            raise ValueError("validation requires a key of at least 32 bytes")
        expected = digest(key, type(self).__name__, self.model_dump(exclude={"fingerprint"}))
        if not hmac.compare_digest(expected, self.fingerprint):
            raise ValueError("prospective validation integrity verification failed")


class CohortExample(CalibrationExample):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CohortMember(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    example: CohortExample
    request_group: str = Field(pattern=r"^[a-f0-9]{64}$")
    task_family: str = Field(pattern=r"^[a-f0-9]{64}$")
    family_first_seen: float = Field(ge=0)
    observed_at: float = Field(ge=0)
    reviewed_at: float = Field(ge=0)
    observation_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    label_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    family_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    unsafe: bool
    origin: Literal["runtime", "synthetic"]


class FrozenCohort(SignedRecord):
    schema_version: Literal["1.0"] = "1.0"
    tenant: str = Field(pattern=r"^[a-f0-9]{64}$")
    calibration_cutoff: float = Field(gt=0)
    embargo_seconds: float = Field(ge=0)
    frozen_at: float = Field(gt=0)
    simulation: bool = False
    selection_policy: Literal["earliest-request-per-family-selected-candidate-v1"] = (
        "earliest-request-per-family-selected-candidate-v1"
    )
    members: list[CohortMember]
    excluded_groups: dict[str, int]

    def verify(self, key: bytes) -> None:
        super().verify(key)
        test_start = self.calibration_cutoff + self.embargo_seconds
        if self.frozen_at <= test_start or any(value < 0 for value in self.excluded_groups.values()):
            raise ValueError("invalid cohort time window or exclusions")
        families, groups, ids = set(), set(), set()
        for member in self.members:
            if member.task_family in families or member.request_group in groups or member.example.id in ids:
                raise ValueError("cohort requires one independent representative per task family")
            families.add(member.task_family)
            groups.add(member.request_group)
            ids.add(member.example.id)
            if not member.family_first_seen <= member.observed_at <= member.reviewed_at <= self.frozen_at:
                raise ValueError("invalid cohort observation/label chronology")
            if member.example.split == "calibration":
                if member.reviewed_at > self.calibration_cutoff:
                    raise ValueError("future calibration labels are prohibited")
            elif member.observed_at <= self.calibration_cutoff or member.family_first_seen < test_start:
                raise ValueError("test task family predates prospective window")
            if not self.simulation and member.origin != "runtime":
                raise ValueError("synthetic observations cannot enter a production cohort")
            if member.unsafe and member.example.correct:
                raise ValueError("unsafe observations cannot be correct calibration targets")


def freeze_cohort(
    store: ExecutionReplayStore,
    tenant: str,
    *,
    calibration_cutoff: float,
    embargo_seconds: float,
    frozen_at: float,
    allow_synthetic: bool = False,
) -> FrozenCohort:
    if (
        not all(math.isfinite(value) for value in [calibration_cutoff, embargo_seconds, frozen_at])
        or calibration_cutoff <= 0
        or embargo_seconds < 0
    ):
        raise ValueError("invalid cutoff or embargo")
    if frozen_at <= calibration_cutoff + embargo_seconds:
        raise ValueError("snapshot must follow the embargo")
    records, assignments = store.snapshot(tenant)
    groups: dict[str, list] = {}
    for event, label in records:
        groups.setdefault(event.request_group, []).append((event, label))
    exclusions: dict[str, int] = {}

    def exclude(reason):
        exclusions[reason] = exclusions.get(reason, 0) + 1

    families: dict[str, list] = {}
    for group, rows in groups.items():
        assignment = assignments.get(group)
        if assignment is None or not assignment.task_family:
            exclude("missing_pre_execution_family")
            continue
        families.setdefault(assignment.task_family, []).append((group, rows, assignment))
    members = []
    for family, requests in sorted(families.items()):
        # Choose before consulting ANY correctness or review-availability labels.
        group, rows, assignment = min(
            requests, key=lambda request: (min(row[0].observed_at for row in request[1]), request[0])
        )
        for _ in requests[1:]:
            exclude("repeated_task_family")
        first_seen = min(row[0].observed_at for _, family_rows, _ in requests for row in family_rows)
        observed_at = max(row[0].observed_at for row in rows)
        event, label = min(rows, key=lambda row: (not row[0].selected, row[0].event_id))
        reason = (
            "future_observation"
            if observed_at > frozen_at
            else "synthetic"
            if not allow_synthetic and any(row[0].origin != "runtime" for row in rows)
            else "unreviewed"
            if label is None
            else "ambiguous"
            if label.verdict == "ambiguous"
            else "label_after_snapshot"
            if label.reviewed_at > frozen_at
            else "invalid_label_chronology"
            if label.reviewed_at < observed_at
            else "unsupported_route"
            if event.route not in {"web", "hybrid", "rag", "kg", "math"}
            else ""
        )
        split = "calibration" if observed_at <= calibration_cutoff else "test"
        if not reason and split == "calibration" and label.reviewed_at > calibration_cutoff:
            reason = "delayed_training_label"
        if not reason and split == "test" and first_seen < calibration_cutoff + embargo_seconds:
            reason = "family_seen_before_test_window"
        if reason:
            exclude(reason)
            continue
        members.append(
            CohortMember(
                example=CohortExample(
                    id=event.event_id,
                    split=split,
                    route=event.route,
                    confidence=event.confidence,
                    correct=label.verdict == "correct" and not label.unsafe,
                    high_risk=event.high_risk,
                ),
                request_group=group,
                task_family=family,
                family_first_seen=first_seen,
                observed_at=observed_at,
                reviewed_at=label.reviewed_at,
                observation_fingerprint=event.fingerprint,
                label_fingerprint=label.fingerprint,
                family_fingerprint=assignment.fingerprint,
                unsafe=label.unsafe,
                origin=event.origin,
            )
        )
    cohort = FrozenCohort(
        tenant=digest(store.key, "tenant", tenant),
        calibration_cutoff=calibration_cutoff,
        embargo_seconds=embargo_seconds,
        frozen_at=frozen_at,
        simulation=allow_synthetic,
        members=members,
        excluded_groups=dict(sorted(exclusions.items())),
    ).seal(store.key)
    cohort.verify(store.key)
    return cohort


def validate_live_lineage(cohort: FrozenCohort, store: ExecutionReplayStore, tenant: str) -> None:
    cohort.verify(store.key)
    if cohort.tenant != digest(store.key, "tenant", tenant):
        raise ValueError("cohort does not belong to tenant")
    records, assignments = store.snapshot(tenant)
    by_id = {event.event_id: (event, label) for event, label in records}
    group_times: dict[str, float] = {}
    family_times: dict[str, float] = {}
    for event, _ in records:
        group_times[event.request_group] = max(group_times.get(event.request_group, 0), event.observed_at)
        assignment = assignments.get(event.request_group)
        if assignment and assignment.task_family:
            family_times[assignment.task_family] = min(
                family_times.get(assignment.task_family, event.observed_at), event.observed_at
            )
    for member in cohort.members:
        record = by_id.get(member.example.id)
        assignment = assignments.get(member.request_group)
        if record is None or record[1] is None or assignment is None:
            raise ValueError("cohort lineage revoked or unavailable")
        event, label = record
        if (
            event.fingerprint != member.observation_fingerprint
            or label.fingerprint != member.label_fingerprint
            or assignment.fingerprint != member.family_fingerprint
            or event.request_group != member.request_group
            or assignment.task_family != member.task_family
            or member.example.route != event.route
            or member.example.confidence != event.confidence
            or member.example.high_risk != event.high_risk
            or member.unsafe != label.unsafe
            or member.example.correct != (label.verdict == "correct" and not label.unsafe)
            or member.origin != event.origin
            or member.reviewed_at != label.reviewed_at
            or member.observed_at != group_times[event.request_group]
            or member.family_first_seen != family_times[assignment.task_family]
        ):
            raise ValueError("cohort lineage changed")


def save_cohort(path: str | Path, cohort: FrozenCohort, key: bytes) -> None:
    """Exclusive creation prevents concurrent operators replacing a frozen file."""
    cohort.verify(key)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as stream:
            stream.write(cohort.model_dump_json(indent=2))
    except FileExistsError:
        previous = FrozenCohort.model_validate_json(target.read_text(encoding="utf-8"))
        previous.verify(key)
        if previous.fingerprint != cohort.fingerprint:
            raise ValueError("cohort paths are immutable; choose a new output path")


class StudyTicket(SignedRecord):
    study_id: str
    cohort_fingerprint: str
    test_families: list[str]
    status: Literal["pending", "complete"] = "pending"
    report: dict | None = None


class HoldoutLedger:
    """Persist exposure reservations BEFORE consulting held-out labels.

    Exact completed retries return cached evidence; partial overlap or a crashed
    pending study cannot quietly reuse families. This is an operator control,
    not an append-only trusted hardware log: protect the DB and signing key.
    """

    def __init__(self, path: str | Path, key: bytes):
        if len(key) < 32:
            raise ValueError("ledger requires a key of at least 32 bytes")
        self.path, self.key = Path(path), key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS studies (study_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS exposures (family_id TEXT PRIMARY KEY, study_id TEXT NOT NULL REFERENCES studies(study_id))"
            )

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def _ticket(self, db, study_id):
        row = db.execute("SELECT payload FROM studies WHERE study_id=?", (study_id,)).fetchone()
        if row is None:
            raise ValueError("holdout ledger index integrity verification failed")
        ticket = StudyTicket.model_validate_json(row[0])
        ticket.verify(self.key)
        actual = [
            row[0]
            for row in db.execute(
                "SELECT family_id FROM exposures WHERE study_id=? ORDER BY family_id", (study_id,)
            )
        ]
        if ticket.study_id != study_id or actual != sorted(ticket.test_families):
            raise ValueError("holdout ledger index integrity verification failed")
        if (ticket.status == "complete") != (ticket.report is not None):
            raise ValueError("invalid holdout ledger state")
        return ticket

    def reserve(self, study_id: str, cohort: FrozenCohort) -> dict | None:
        cohort.verify(self.key)
        families = sorted(member.task_family for member in cohort.members if member.example.split == "test")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            # Missing exposure rows must not silently permit reuse in a new study.
            for (previous_id,) in db.execute("SELECT study_id FROM studies").fetchall():
                self._ticket(db, previous_id)
            existing = db.execute("SELECT 1 FROM studies WHERE study_id=?", (study_id,)).fetchone()
            if existing:
                ticket = self._ticket(db, study_id)
                if ticket.cohort_fingerprint != cohort.fingerprint or ticket.test_families != families:
                    raise ValueError("conflicting holdout study retry")
                if ticket.status == "pending":
                    raise ValueError("holdout study pending; operator audit required")
                return ticket.report
            for family in families:
                exposed = db.execute("SELECT study_id FROM exposures WHERE family_id=?", (family,)).fetchone()
                if exposed:
                    self._ticket(db, exposed[0])
                    raise ValueError("held-out task family already exposed; collect fresh families")
            ticket = StudyTicket(
                study_id=study_id, cohort_fingerprint=cohort.fingerprint, test_families=families
            ).seal(self.key)
            db.execute("INSERT INTO studies VALUES (?, ?)", (study_id, ticket.model_dump_json()))
            db.executemany("INSERT INTO exposures VALUES (?, ?)", [(family, study_id) for family in families])
        return None

    def finish(self, study_id: str, report: dict) -> None:
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            ticket = self._ticket(db, study_id)
            if ticket.status != "pending":
                raise ValueError("holdout study is already complete")
            ticket.status, ticket.report = "complete", report
            ticket.seal(self.key)
            db.execute("UPDATE studies SET payload=? WHERE study_id=?", (ticket.model_dump_json(), study_id))
