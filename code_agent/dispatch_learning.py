"""Family-balanced categorical dispatch learning with non-serving recommendations."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictInt, model_validator

from code_agent.adaptive_dispatch import next_perspective, stopping_reason
from code_agent.models import (
    DispatchShadowEvidence,
    RepairCandidateEvidence,
    RepairTournamentPolicy,
    RepairTournamentReport,
)

Role = Literal["boundary", "compatibility", "defensive"]
ROLES = ("boundary", "compatibility", "defensive")
ALL_ROLES = ("minimal", *ROLES)
SHA = r"^[a-f0-9]{64}$"


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def policy_fingerprint(policy: RepairTournamentPolicy) -> str:
    payload = policy.model_dump(mode="json")
    if payload.get("adaptive_dispatch"):
        payload["adaptive_dispatch"].pop("advisor_shadow", None)
    return fingerprint(payload)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DispatchContext(StrictModel):
    category: Literal["wrong_localization", "incomplete_fix", "dependency_or_contract", "regression", "unknown"]
    used_roles: list[Literal["minimal", "boundary", "compatibility", "defensive"]] = Field(min_length=1, max_length=3)
    changed_files: Literal["none", "few", "many"]
    patch_size: Literal["empty", "small", "large"]
    call_headroom: Literal["low", "medium", "high"]

    @model_validator(mode="after")
    def canonical_mask(self):
        if self.used_roles != [role for role in ALL_ROLES if role in self.used_roles] or "minimal" not in self.used_roles:
            raise ValueError("used-role mask must be unique, canonical and include minimal")
        return self


def context_for(row: RepairCandidateEvidence, used: set[str], calls: int, policy: RepairTournamentPolicy) -> DispatchContext:
    if not row.diagnosis or not policy.adaptive_dispatch or calls < 0 or calls > policy.max_model_calls:
        raise ValueError("bounded pre-decision execution evidence required")
    remaining = max(0, policy.max_model_calls - calls - policy.adaptive_dispatch.verification_model_calls)
    return DispatchContext(category=row.diagnosis.category,
        used_roles=[role for role in ALL_ROLES if role in used],
        changed_files="none" if row.changed_files == 0 else "few" if row.changed_files <= 3 else "many",
        patch_size="empty" if row.patch_chars == 0 else "small" if row.patch_chars <= 8000 else "large",
        call_headroom="low" if remaining * 4 <= policy.max_model_calls else "medium" if remaining * 2 <= policy.max_model_calls else "high")


class DecisionSnapshot(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    record_id: str = Field(pattern=SHA)
    family_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._:-]+$")
    scope_sha256: str = Field(pattern=SHA)
    policy_sha256: str = Field(pattern=SHA)
    source_receipt_sha256: str = Field(pattern=SHA)
    candidate_id: str = Field(pattern=r"^candidate-[1-4]$")
    observed_at: AwareDatetime
    context: DispatchContext
    fingerprint: str = ""

    def seal(self):
        self.fingerprint = fingerprint(self.model_dump(mode="json", exclude={"fingerprint"}))
        return self

    def verify(self) -> bool:
        return (self.fingerprint == fingerprint(self.model_dump(mode="json", exclude={"fingerprint"}))
                and self.record_id == fingerprint([self.source_receipt_sha256, self.candidate_id]))


class ReviewedDecision(StrictModel):
    snapshot: DecisionSnapshot
    split: Literal["train", "validation", "test"]
    preferred_role: Role
    reviewer_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._:-]+$")
    independent_review: Literal[True]
    label_source: Literal["human", "synthetic"]
    reviewed_at: AwareDatetime

    @model_validator(mode="after")
    def valid_review(self):
        if not self.snapshot.verify() or self.preferred_role in self.snapshot.context.used_roles:
            raise ValueError("invalid snapshot or already-used review target")
        if self.reviewed_at < self.snapshot.observed_at:
            raise ValueError("review cannot predate observation")
        return self


class AdvisorThresholds(StrictModel):
    min_supporting_families: int = Field(default=5, ge=2, le=1000)
    min_label_agreement: float = Field(default=0.75, ge=0.5, le=1)
    min_margin: float = Field(default=0.2, ge=0, le=1)
    min_agreement_lower_bound: float = Field(default=0.4, ge=0, le=1)
    min_eval_families: int = Field(default=5, ge=2, le=1000)
    min_coverage: float = Field(default=0.5, ge=0, le=1)
    max_review_disagreement_upper_bound: float = Field(default=0.25, ge=0, le=1)


class DispatchCell(StrictModel):
    context: DispatchContext
    votes: dict[Role, StrictInt]

    @model_validator(mode="after")
    def valid_votes(self):
        if (set(self.votes) != set(ROLES) or any(value < 0 for value in self.votes.values())
                or sum(self.votes.values()) == 0 or any(self.votes[role] for role in self.context.used_roles if role in ROLES)):
            raise ValueError("invalid family vote counts or used-role votes")
        return self


class DispatchAdvisorArtifact(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    model_type: Literal["family-balanced-categorical-advisor"] = "family-balanced-categorical-advisor"
    policy_sha256: str = Field(pattern=SHA)
    scope_sha256: str = Field(pattern=SHA)
    training_fingerprint: str = Field(pattern=SHA)
    training_families: list[str] = Field(min_length=1, max_length=10000)
    label_source: Literal["human", "synthetic_or_mixed"]
    thresholds: AdvisorThresholds
    cells: dict[str, DispatchCell] = Field(min_length=1, max_length=10000)
    fingerprint: str = ""

    def seal(self):
        self.fingerprint = fingerprint(self.model_dump(mode="json", exclude={"fingerprint"}))
        return self

    def verify(self) -> bool:
        try:
            DispatchAdvisorArtifact.model_validate(self.model_dump(mode="json"))
        except (ValueError, TypeError):
            return False
        return (self.fingerprint == fingerprint(self.model_dump(mode="json", exclude={"fingerprint"}))
                and self.training_families == sorted(set(self.training_families))
                and all(key == fingerprint(cell.context.model_dump(mode="json"))
                        and sum(cell.votes.values()) <= len(self.training_families) for key, cell in self.cells.items()))


def validate_protocol(records: list[ReviewedDecision]) -> None:
    if not records or len(records) > 10000:
        raise ValueError("bounded nonempty reviewed dataset required")
    if len({row.snapshot.record_id for row in records}) != len(records):
        raise ValueError("duplicate decision records")
    if len({(row.snapshot.policy_sha256, row.snapshot.scope_sha256) for row in records}) != 1:
        raise ValueError("mixed policy or scope lineage")
    families: dict[str, str] = {}
    for row in records:
        ReviewedDecision.model_validate(row.model_dump())
        family = row.snapshot.family_id
        if family in families and families[family] != row.split:
            raise ValueError("task-family leakage across splits")
        families[family] = row.split
    groups = [[row for row in records if row.split == split] for split in ("train", "validation", "test")]
    if any(not group for group in groups):
        raise ValueError("train, validation and test splits are required")
    for previous, following in zip(groups, groups[1:]):
        if max(row.reviewed_at for row in previous) >= min(row.snapshot.observed_at for row in following):
            raise ValueError("chronology or label-availability leakage across splits")


def fit_advisor(records: list[ReviewedDecision], thresholds: AdvisorThresholds | None = None) -> DispatchAdvisorArtifact:
    if not records or len(records) > 10000 or any(row.split != "train" for row in records):
        raise ValueError("training-only reviewed records required")
    if len({row.snapshot.record_id for row in records}) != len(records):
        raise ValueError("duplicate training decisions")
    if len({(row.snapshot.policy_sha256, row.snapshot.scope_sha256) for row in records}) != 1:
        raise ValueError("mixed policy or scope lineage")
    family_votes: dict[tuple[str, str], str] = {}
    cells: dict[str, DispatchCell] = {}
    for row in records:
        ReviewedDecision.model_validate(row.model_dump())
        key = fingerprint(row.snapshot.context.model_dump(mode="json"))
        family_key = (key, row.snapshot.family_id)
        if family_key in family_votes and family_votes[family_key] != row.preferred_role:
            raise ValueError("conflicting reviews within a family/context")
        if family_key in family_votes:
            continue
        family_votes[family_key] = row.preferred_role
        if key not in cells:
            # Construct only once a positive family vote is available.
            cells[key] = DispatchCell(context=row.snapshot.context.model_copy(deep=True), votes={role: int(role == row.preferred_role) for role in ROLES})
        else:
            cells[key].votes[row.preferred_role] += 1
    ordered = sorted(records, key=lambda row: row.snapshot.record_id)
    return DispatchAdvisorArtifact(policy_sha256=records[0].snapshot.policy_sha256,
        scope_sha256=records[0].snapshot.scope_sha256,
        training_fingerprint=fingerprint([row.model_dump(mode="json") for row in ordered]),
        training_families=sorted({row.snapshot.family_id for row in records}),
        label_source="human" if all(row.label_source == "human" for row in records) else "synthetic_or_mixed",
        thresholds=thresholds.model_copy(deep=True) if thresholds else AdvisorThresholds(), cells=cells).seal()


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    p, denominator = successes / total, 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0, center - radius), min(1, center + radius))


class DispatchAdvisor:
    def __init__(self, artifact: DispatchAdvisorArtifact):
        if not artifact.verify():
            raise ValueError("dispatch artifact integrity failed")
        self.artifact = artifact.model_copy(deep=True)

    def recommend(self, context: DispatchContext, *, candidate_id: str, baseline_role: str,
                  policy_sha256: str, scope_sha256: str) -> DispatchShadowEvidence:
        key = fingerprint(context.model_dump(mode="json"))
        shadow = DispatchShadowEvidence(candidate_id=candidate_id, context_sha256=key,
            artifact_fingerprint=self.artifact.fingerprint, baseline_role=baseline_role, disposition="unseen_context")
        if (policy_sha256, scope_sha256) != (self.artifact.policy_sha256, self.artifact.scope_sha256):
            shadow.disposition = "lineage_mismatch"
            return shadow
        cell = self.artifact.cells.get(key)
        if cell is None:
            return shadow
        total = sum(cell.votes.values())
        shadow.supporting_families = total
        thresholds = self.artifact.thresholds
        if total < thresholds.min_supporting_families:
            shadow.disposition = "insufficient_support"
            return shadow
        available = [role for role in ROLES if role not in context.used_roles]
        ranked = sorted(available, key=lambda role: (-cell.votes[role], ROLES.index(role)))
        winner = ranked[0]
        agreement = (cell.votes[winner] + 1) / (total + len(available))
        second = (cell.votes[ranked[1]] + 1) / (total + len(available)) if len(ranked) > 1 else 0
        shadow.label_agreement = agreement
        if (agreement < thresholds.min_label_agreement or agreement - second < thresholds.min_margin
                or wilson_interval(cell.votes[winner], total)[0] < thresholds.min_agreement_lower_bound):
            shadow.disposition = "low_agreement"
            return shadow
        shadow.disposition, shadow.proposed_role = "supported", winner
        return shadow


def load_advisor(path: Path, expected_fingerprint: str) -> DispatchAdvisor:
    if path.is_symlink() or path.stat().st_size > 2_000_000:
        raise ValueError("bounded regular advisor artifact required")
    artifact = DispatchAdvisorArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    if artifact.fingerprint != expected_fingerprint:
        raise ValueError("advisor artifact does not match the operator pin")
    return DispatchAdvisor(artifact)


def snapshots_from_report(report: RepairTournamentReport, *, family_id: str, scope_sha256: str,
                          observed_at: str) -> list[DecisionSnapshot]:
    from code_agent.repair_tournament import verify_tournament_report
    if not verify_tournament_report(report) or not report.policy.adaptive_dispatch:
        raise ValueError("valid adaptive receipt required")
    snapshots, used, prefix, calls = [], set(), [], report.shared_model_calls
    for index, row in enumerate(report.candidates):
        if row.status == "skipped":
            break
        used.add(row.perspective)
        calls += row.model_calls
        prefix.append(row)
        if index + 1 >= len(report.dispatch_decisions) or stopping_reason(prefix, report):
            continue
        if not next_perspective(row.diagnosis.category, used):
            continue
        snapshots.append(DecisionSnapshot(record_id=fingerprint([report.fingerprint, row.candidate_id]),
            family_id=family_id, scope_sha256=scope_sha256, policy_sha256=policy_fingerprint(report.policy),
            source_receipt_sha256=report.fingerprint, candidate_id=row.candidate_id, observed_at=observed_at,
            context=context_for(row, used, calls, report.policy)).seal())
    return snapshots


def evaluate_advisor(records: list[ReviewedDecision], artifact: DispatchAdvisorArtifact) -> dict:
    validate_protocol(records)
    regenerated = fit_advisor([row for row in records if row.split == "train"], artifact.thresholds)
    if regenerated != artifact:
        raise ValueError("artifact is not reproducible from the declared training split")
    advisor = DispatchAdvisor(artifact)
    metrics = {}
    curves = []
    for split in ("validation", "test"):
        rows = [row for row in records if row.split == split]
        proposals = [advisor.recommend(row.snapshot.context, candidate_id=row.snapshot.candidate_id,
            baseline_role=next_perspective(row.snapshot.context.category, set(row.snapshot.context.used_roles)),
            policy_sha256=row.snapshot.policy_sha256, scope_sha256=row.snapshot.scope_sha256) for row in rows]
        metrics[split] = _summarize(rows, proposals)
        for threshold in (0.5, 0.75, 0.9):
            curves.append({"split": split, "agreement_threshold": threshold,
                           **_summarize(rows, proposals, threshold=threshold)})
    holds = []
    if artifact.label_source != "human" or any(row.label_source != "human" for row in records):
        holds.append("synthetic_or_mixed_review_labels")
    for split, values in metrics.items():
        if values["families"] < artifact.thresholds.min_eval_families:
            holds.append(f"{split}:insufficient_families")
        if values["coverage"] < artifact.thresholds.min_coverage:
            holds.append(f"{split}:insufficient_coverage")
        if values["family_disagreement_upper_bound"] > artifact.thresholds.max_review_disagreement_upper_bound:
            holds.append(f"{split}:review_disagreement_risk")
    report = {"schema_version": "1.0", "artifact_fingerprint": artifact.fingerprint,
              "dataset_fingerprint": fingerprint([row.model_dump(mode="json") for row in sorted(records, key=lambda row: row.snapshot.record_id)]),
              "metrics": metrics, "risk_coverage_curve": curves, "blocking_reasons": holds,
              "decision": "held" if holds else "ready_for_owner_review", "serving_activation": False,
              "limitations": ["Learns independently declared preferred specialists, not patch correctness or counterfactual utility.",
                              "Label agreement is not a probability of solving an issue; confidence intervals concern reviewed recommendations.",
                              "Shadow mode makes no additional model calls and never changes served dispatch.",
                              "Declared review independence/timestamps are audit metadata, not external attestation."]}
    report["fingerprint"] = fingerprint(report)
    return report


def _summarize(rows: list[ReviewedDecision], proposals: list[DispatchShadowEvidence], *, threshold: float = 0) -> dict:
    groups = defaultdict(list)
    for row, proposal in zip(rows, proposals, strict=True):
        groups[row.snapshot.family_id].append((row, proposal))
    coverages, agreements, baseline_agreements, family_errors, paired_deltas = [], [], [], [], []
    for family_id in sorted(groups):
        values = groups[family_id]
        accepted = [(row, proposal) for row, proposal in values if proposal.proposed_role and proposal.label_agreement >= threshold]
        coverages.append(len(accepted) / len(values))
        baseline_agreements.append(sum(proposal.baseline_role == row.preferred_role for row, proposal in values) / len(values))
        if accepted:
            correct = sum(proposal.proposed_role == row.preferred_role for row, proposal in accepted)
            agreements.append(correct / len(accepted))
            family_errors.append(int(correct != len(accepted)))
            baseline_correct = sum(proposal.baseline_role == row.preferred_role for row, proposal in accepted)
            paired_deltas.append((correct - baseline_correct) / len(accepted))
    interval = None
    if paired_deltas:
        rng = random.Random(fingerprint(sorted(groups)))
        draws = sorted(sum(rng.choice(paired_deltas) for _ in paired_deltas) / len(paired_deltas) for _ in range(500))
        interval = [draws[12], draws[487]]
    return {"families": len(groups), "covered_families": len(family_errors),
            "coverage": sum(coverages) / len(coverages), "accepted_review_agreement": sum(agreements) / len(agreements) if agreements else None,
            "baseline_review_agreement": sum(baseline_agreements) / len(baseline_agreements),
            "family_disagreement_upper_bound": wilson_interval(sum(family_errors), len(family_errors))[1],
            "paired_review_agreement_delta_on_accepted": sum(paired_deltas) / len(paired_deltas) if paired_deltas else None,
            "paired_family_bootstrap_95_ci": interval,
            "dispositions": {value: sum(proposal.disposition == value for proposal in proposals) for value in sorted({proposal.disposition for proposal in proposals})}}
