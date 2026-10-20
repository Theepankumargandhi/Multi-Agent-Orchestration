"""Frozen repair-memory experiments with owner-attested lineage and leak checks.

Digests detect drift, not a malicious host recomputing them. Text overlap is a
conservative warning heuristic, not proof of semantic independence.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from code_agent.models import RepairMemoryReceipt
from code_agent.repair_memory import RepairLesson, RepairMemoryStore, digest, pack_lessons, scope_hash
from code_agent.security_policy import classify_untrusted_content


def prepare_memory(cases, policy, snapshot, workflow):
    if snapshot is None:
        if policy.repair_memory_enabled:
            raise ValueError("mutable repair memory is not allowed in frozen benchmarks; a frozen audited snapshot is required")
        return None
    if not policy.repair_memory_enabled or workflow not in {"single_agent", "verified_pr"}:
        raise ValueError("snapshot experiments require memory enabled and a supported single_agent/verified_pr workflow")
    if not isinstance(snapshot, FrozenRepairMemory):
        raise ValueError("benchmark memory requires an audited frozen snapshot")
    pinned = FrozenRepairMemory.prepare(MemorySnapshot.model_validate_json(snapshot.payload), cases)
    if (pinned.fingerprint, pinned.dataset_sha256, pinned.audit_sha256, pinned.allowed_cases) != (
            snapshot.fingerprint, snapshot.dataset_sha256, snapshot.audit_sha256, snapshot.allowed_cases):
        raise ValueError("frozen memory selection or fingerprint drifted")
    return pinned


def issue_identity(issue: str):
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", issue).casefold())[:3000]
    normalized = " ".join(words)
    shingles = sorted({digest(words[index:index + 5]) for index in range(max(0, len(words) - 4))})
    return digest(normalized), shingles


class SourceLineage(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    case_id: str = Field(min_length=1, max_length=300)
    family_id: str = Field(min_length=1, max_length=200)
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    issue: str = Field(min_length=10, max_length=12000)
    completed_at: float = Field(ge=0)

    @field_validator("case_id", "family_id")
    @classmethod
    def explicit_identity(cls, value):
        if not value.strip():
            raise ValueError("source identities must not be blank")
        return value


class SnapshotEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    lesson: RepairLesson
    lesson_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_case_id: str = Field(min_length=1, max_length=300)
    source_family_id: str = Field(min_length=1, max_length=200)
    source_dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_issue_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_issue_shingles: list[str] = Field(max_length=3000)
    source_completed_at: float = Field(ge=0)


class SnapshotRepository(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    entries: list[SnapshotEntry] = Field(max_length=200)


class MemorySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: str = "1.0"
    snapshot_type: str = "reviewed-repair-memory"
    frozen_at: float = Field(ge=0)
    repositories: dict[str, SnapshotRepository] = Field(min_length=1, max_length=100)
    fingerprint: str = ""

    def unsigned(self):
        return self.model_dump(mode="json", exclude={"fingerprint"})

    def seal(self):
        self.fingerprint = digest(self.unsigned())
        return self

    @model_validator(mode="after")
    def bounded_records(self):
        if sum(len(repo.entries) for repo in self.repositories.values()) > 5000:
            raise ValueError("snapshot record bound exceeded")
        return self


def validate_snapshot(snapshot: MemorySnapshot):
    if (snapshot.schema_version != "1.0" or snapshot.snapshot_type != "reviewed-repair-memory"
            or digest(snapshot.unsigned()) != snapshot.fingerprint):
        raise ValueError("snapshot binding is invalid")
    identifiers = set()
    for repository, pack in snapshot.repositories.items():
        from code_agent.models import CodeTask
        CodeTask(repository=repository, issue="Validate snapshot repository scope.")
        for entry in pack.entries:
            lesson = entry.lesson
            if (entry.lesson_id != lesson.identifier or entry.lesson_id in identifiers
                    or lesson.scope_sha256 != pack.scope_sha256
                    or not entry.source_completed_at <= lesson.created_at <= snapshot.frozen_at < lesson.expires_at
                    or lesson.expires_at - lesson.created_at > 90 * 86400
                    or any(not re.fullmatch(r"[a-f0-9]{64}", value) for value in entry.source_issue_shingles)
                    or entry.source_issue_shingles != sorted(set(entry.source_issue_shingles))
                    or set(classify_untrusted_content(lesson.text)) & {"instruction_injection", "secret_material"}):
                raise ValueError("snapshot lesson or lineage is invalid")
            identifiers.add(entry.lesson_id)


def freeze_memory(store: RepairMemoryStore, *, project: str, namespace: str,
                  repositories: list[str], lineage: dict[str, SourceLineage], frozen_at: float | None = None):
    clock = time.time() if frozen_at is None else frozen_at
    packs = {}
    used = set()
    if len(repositories) != len(set(repositories)):
        raise ValueError("snapshot repositories must be unique")
    for repository in sorted(repositories):
        scope = scope_hash(project, namespace, repository)
        records, invalid = store.records(scope=scope, now=clock)
        if invalid:
            raise ValueError("cannot freeze invalid lesson records")
        entries = []
        for lesson in sorted(records, key=lambda record: record.identifier):
            source = lineage.get(lesson.identifier)
            if source is None:
                raise ValueError("every frozen lesson requires explicit source-task lineage")
            issue_hash, shingles = issue_identity(source.issue)
            entries.append(SnapshotEntry(lesson=lesson, lesson_id=lesson.identifier,
                source_case_id=source.case_id, source_family_id=source.family_id,
                source_dataset_sha256=source.dataset_sha256, source_issue_sha256=issue_hash,
                source_issue_shingles=shingles, source_completed_at=source.completed_at))
            used.add(lesson.identifier)
        packs[repository] = SnapshotRepository(scope_sha256=scope, entries=entries)
    if set(lineage) != used:
        raise ValueError("lineage must match the exported active lesson selection exactly")
    snapshot = MemorySnapshot(frozen_at=clock, repositories=packs).seal()
    validate_snapshot(snapshot)
    return snapshot


def audit_snapshot(snapshot: MemorySnapshot, cases, *, now: float | None = None):
    validate_snapshot(snapshot)
    clock = time.time() if now is None else now
    if not cases or len(cases) > 500 or len({case.id for case in cases}) != len(cases):
        raise ValueError("memory audit requires a nonempty unique case selection")
    families, identities = {}, {}
    for case in cases:
        family = case.metadata.get("family_id")
        if not isinstance(family, str) or not family.strip() or len(family) > 200:
            raise ValueError("memory experiments require explicit evaluation family_id metadata")
        if case.repository not in snapshot.repositories:
            raise ValueError("snapshot must explicitly cover every evaluated repository")
        families[case.id] = family
        issue_hash, shingles = issue_identity(case.issue)
        identities[case.id] = (issue_hash, set(shingles))
    findings = []
    if snapshot.frozen_at > clock:
        findings.append({"code": "future_snapshot"})
    dataset_hash = digest([case.model_dump(mode="json") for case in cases])
    for pack in snapshot.repositories.values():
        for entry in pack.entries:
            source_shingles = set(entry.source_issue_shingles)
            problem_hash, problem_shingles = issue_identity(entry.lesson.problem)
            problem_shingles = set(problem_shingles)
            lesson_hash, lesson_shingles = issue_identity(entry.lesson.lesson)
            lesson_shingles = set(lesson_shingles)
            for case in cases:
                issue_hash, shingles = identities[case.id]
                overlap = len(shingles & source_shingles)
                denominator = min(len(shingles), len(source_shingles))
                codes = []
                if entry.source_case_id == case.id:
                    codes.append("source_case_overlap")
                if entry.source_family_id == families[case.id]:
                    codes.append("source_family_overlap")
                if entry.source_dataset_sha256 == dataset_hash:
                    codes.append("source_dataset_overlap")
                if entry.source_issue_sha256 == issue_hash:
                    codes.append("source_issue_exact_match")
                if denominator >= 4 and overlap / denominator >= 0.60:
                    codes.append("source_issue_near_duplicate")
                # The reviewed lesson itself must not simply restate a held-out issue.
                count = len(shingles & problem_shingles)
                bound = min(len(shingles), len(problem_shingles))
                if problem_hash == issue_hash or (bound >= 4 and count / bound >= 0.60):
                    codes.append("lesson_problem_overlap")
                count, bound = len(shingles & lesson_shingles), min(len(shingles), len(lesson_shingles))
                if lesson_hash == issue_hash or (bound >= 4 and count / bound >= 0.60):
                    codes.append("lesson_text_overlap")
                for code in codes:
                    findings.append({"code": code, "lesson_id": entry.lesson_id, "case_id_sha256": digest(case.id)})
                if len(findings) >= 1000:
                    break
            if len(findings) >= 1000:
                break
        if len(findings) >= 1000:
            break
    report = {"schema_version": "1.0", "audit_policy": "case-family-dataset-text-v1",
              "snapshot_sha256": snapshot.fingerprint, "dataset_sha256": dataset_hash,
              "cases": len(cases), "lessons": sum(len(pack.entries) for pack in snapshot.repositories.values()),
              "passed": not findings, "findings": findings,
              "semantic_independence_proven": False}
    report["findings_capped"] = len(findings) >= 1000
    report["audit_sha256"] = digest(report)
    return report


@dataclass(frozen=True)
class FrozenRepairMemory:
    """Immutable serialized content pinned to one audited case selection."""

    payload: str
    fingerprint: str
    dataset_sha256: str
    audit_sha256: str
    allowed_cases: tuple[tuple[str, str], ...]

    @classmethod
    def prepare(cls, snapshot: MemorySnapshot, cases, *, now: float | None = None):
        audit = audit_snapshot(snapshot, cases, now=now)
        if not audit["passed"]:
            raise ValueError("repair-memory leakage audit rejected this case selection")
        return cls(snapshot.model_dump_json(), snapshot.fingerprint, audit["dataset_sha256"],
                   audit["audit_sha256"], tuple((case.id, case.repository) for case in cases))

    @classmethod
    def load(cls, path: Path, fingerprint: str, cases):
        if not re.fullmatch(r"[a-f0-9]{64}", fingerprint) or path.stat().st_size > 50_000_000:
            raise ValueError("expected snapshot fingerprint and bounded file are required")
        try:
            snapshot = MemorySnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raise ValueError("memory snapshot could not be read or validated") from None
        if snapshot.fingerprint != fingerprint:
            raise ValueError("snapshot does not match the expected fingerprint")
        return cls.prepare(snapshot, cases)

    def retrieve(self, task, workspace, *, stage: str, iteration: int):
        if (task.benchmark_case_id, task.repository) not in self.allowed_cases:
            raise ValueError("frozen memory task is outside the audited case selection")
        snapshot = MemorySnapshot.model_validate_json(self.payload)
        pack = snapshot.repositories[task.repository]
        context, scores = pack_lessons([entry.lesson for entry in pack.entries], task.issue[:4000],
                                      set(workspace.list_files(task.policy.max_files)))
        receipt = RepairMemoryReceipt(stage=stage, iteration=iteration,
            status="selected" if context else "no_match", scope_sha256=pack.scope_sha256,
            query_sha256=digest(task.issue[:4000]), snapshot_sha256=self.fingerprint,
            context_sha256=digest(context) if context else "", lesson_scores=scores,
            approximate_tokens=(len(context) + 3) // 4 if context else 0)
        return context, receipt

    def valid_receipts(self, task, receipts):
        """Check exposure evidence against the pinned repository, without storing text."""
        if (task.benchmark_case_id, task.repository) not in self.allowed_cases or not receipts:
            return False
        pack = MemorySnapshot.model_validate_json(self.payload).repositories[task.repository]
        identifiers = {entry.lesson_id for entry in pack.entries}
        for receipt in receipts:
            if (receipt.snapshot_sha256 != self.fingerprint or receipt.scope_sha256 != pack.scope_sha256
                    or not re.fullmatch(r"[a-f0-9]{64}", receipt.query_sha256)
                    or receipt.invalid_records or receipt.status not in {"selected", "no_match"}):
                return False
            if receipt.status == "no_match":
                if receipt.lesson_scores or receipt.context_sha256 or receipt.approximate_tokens:
                    return False
            elif (not 1 <= len(receipt.lesson_scores) <= 3
                    or not set(receipt.lesson_scores) <= identifiers
                    or not all(math.isfinite(value) and 0.2 <= value <= 1.000001 for value in receipt.lesson_scores.values())
                    or not re.fullmatch(r"[a-f0-9]{64}", receipt.context_sha256)
                    or not 1 <= receipt.approximate_tokens <= 750):
                return False
        return True
