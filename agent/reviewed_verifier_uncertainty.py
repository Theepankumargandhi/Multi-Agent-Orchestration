"""Explicit-label, family-bootstrap STEP uncertainty. Never a serving selector."""

from __future__ import annotations

import math
import random
import statistics

from pydantic import Field, model_validator

from agent.execution_replay import digest
from agent.preference_ranking import StrictModel
from agent.process_reward import (
    ProcessRewardArtifact,
    ProcessRewardScorer,
    ProcessStep,
    ProcessTrace,
    train_process_reward_model,
)
from agent.prospective_validation import SignedRecord

HEX = r"^[a-f0-9]{64}$"


def _temperature(probability: float, temperature: float) -> float:
    clipped = min(1 - 1e-9, max(1e-9, probability))
    logit = math.log(clipped / (1 - clipped)) / temperature
    return 1 / (1 + math.exp(-max(-30, min(30, logit))))


class UncertaintyPolicy(StrictModel):
    version: str = "reviewed-step-uncertainty-v1"
    member_count: int = Field(default=5, ge=3, le=15)
    epochs: int = Field(default=300, ge=1, le=1000)
    seed: int = Field(default=41, ge=0)
    spread_penalty: float = Field(default=1.5, ge=0, le=5)
    support_margin: float = Field(default=0.05, ge=0, le=0.1)
    primary_confidence: float = Field(default=0.8, ge=0.5, le=1)
    maximum_spread: float = Field(default=0.15, gt=0, le=0.5)
    curve_confidences: list[float] = Field(default_factory=lambda: [0.5, 0.7, 0.8, 0.9, 0.95])

    @model_validator(mode="after")
    def check(self):
        if (
            self.version != "reviewed-step-uncertainty-v1"
            or not 1 <= len(self.curve_confidences) <= 16
            or self.curve_confidences != sorted(set(self.curve_confidences))
            or self.primary_confidence not in self.curve_confidences
            or any(not 0.5 <= value <= 1 for value in self.curve_confidences)
        ):
            raise ValueError("invalid preregistered uncertainty policy")
        return self


def pattern(step: ProcessStep, high_risk: bool) -> str:
    # Fixed typed proxies only: no labels, IDs, answers, queries or reviewer identities.
    return ":".join(
        [step.kind]
        + [
            str(int(value))
            for value in (step.has_evidence, step.citation_valid, step.policy_allowed, step.error, high_risk)
        ]
    )


class FeatureSupport(StrictModel):
    pattern: str = Field(pattern=r"^(plan|retrieve|tool|reason|verify|answer|refuse)(:[01]){5}$")
    progress_min: float = Field(gt=0, le=1)
    progress_max: float = Field(gt=0, le=1)
    confidence_min: float = Field(ge=0, le=1)
    confidence_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def check(self):
        if self.progress_min > self.progress_max or self.confidence_min > self.confidence_max:
            raise ValueError("invalid training support range")
        return self


class ReviewedStepEnsemble(SignedRecord):
    tenant: str = Field(pattern=HEX)
    cohort_fingerprint: str = Field(pattern=HEX)
    report_fingerprint: str = Field(pattern=HEX)
    simulation: bool
    policy: UncertaintyPolicy
    members: list[ProcessRewardArtifact] = Field(min_length=3, max_length=15)
    calibration_temperature: float = Field(ge=0.25, le=4)
    support: list[FeatureSupport] = Field(min_length=1)
    # Digests attest whole-family sampling without disclosing sampled family IDs.
    bootstrap_fingerprints: list[str]
    explicit_training_steps: int = Field(ge=1)

    def verify(self, key: bytes) -> None:
        super().verify(key)
        UncertaintyPolicy.model_validate(self.policy.model_dump())
        if (
            len(self.members) != self.policy.member_count
            or len(self.bootstrap_fingerprints) != len(self.members)
            or len({row.pattern for row in self.support}) != len(self.support)
            or not all(member.verify() for member in self.members)
        ):
            raise ValueError("invalid reviewed step ensemble")


class StepEstimate(StrictModel):
    mean: float = Field(ge=0, le=1)
    spread: float = Field(ge=0, le=1)
    conservative_confidence: float = Field(ge=0, le=1)
    predicted_correct: bool
    feature_supported: bool
    accepted: bool


class ReviewedStepScorer:
    def __init__(self, candidate: ReviewedStepEnsemble, key: bytes):
        candidate.verify(key)
        self.candidate = candidate
        self.scorers = [ProcessRewardScorer(member) for member in candidate.members]
        self.support = {row.pattern: row for row in candidate.support}

    def estimates(self, steps: list[ProcessStep], high_risk: bool = False) -> list[StepEstimate]:
        if not steps:
            raise ValueError("step uncertainty requires a nonempty workflow")
        policy = self.candidate.policy
        probabilities = [scorer.step_probabilities(steps, high_risk) for scorer in self.scorers]
        result = []
        for index, step in enumerate(steps):
            calibrated = [
                _temperature(values[index], self.candidate.calibration_temperature)
                for values in probabilities
            ]
            mean, spread = statistics.fmean(calibrated), statistics.pstdev(calibrated)
            support = self.support.get(pattern(step, high_risk))
            progress, margin = (index + 1) / len(steps), policy.support_margin
            supported = support is not None and (
                support.progress_min - margin <= progress <= support.progress_max + margin
                and support.confidence_min - margin <= step.confidence <= support.confidence_max + margin
            )
            confidence = max(0.0, max(mean, 1 - mean) - policy.spread_penalty * spread)
            result.append(
                StepEstimate(
                    mean=mean,
                    spread=spread,
                    conservative_confidence=confidence,
                    predicted_correct=mean >= 0.5,
                    feature_supported=supported,
                    accepted=supported
                    and spread <= policy.maximum_spread
                    and confidence >= policy.primary_confidence,
                )
            )
        return result


def fit_ensemble(traces: list[ProcessTrace], policy: UncertaintyPolicy, key: bytes) -> dict:
    """Train only on explicit labels; calibrate only validation step labels.

    The test fold is deliberately ignored even when it is supplied. A family
    draw includes ALL its candidate traces, unlike the legacy trace bootstrap.
    """
    training = [trace for trace in traces if trace.split == "train"]
    validation = [trace for trace in traces if trace.split == "validation"]
    groups = {}
    for trace in training:
        if any(step.step_label is not None for step in trace.steps):
            groups.setdefault(trace.group_id, []).append(trace)
    if not groups or not validation:
        raise ValueError("reviewed ensemble requires labelled training families and validation steps")
    # Unlabelled siblings must stay with their family too (they add no targets).
    groups = {family: [trace for trace in training if trace.group_id == family] for family in groups}
    members, draws = [], []
    families = sorted(groups)
    for member_index in range(policy.member_count):
        rng = random.Random(policy.seed + member_index)
        sampled = [rng.choice(families) for _ in families]
        sampled_traces = [trace for family in sampled for trace in groups[family]]
        members.append(
            train_process_reward_model(sampled_traces, epochs=policy.epochs, explicit_steps_only=True)
        )
        draws.append(digest(key, "step-ensemble-family-draw", sampled))
    scorers = [ProcessRewardScorer(member) for member in members]
    validation_rows = []
    for trace in validation:
        predictions = [scorer.step_probabilities(trace.steps, trace.high_risk) for scorer in scorers]
        for index, step in enumerate(trace.steps):
            if step.step_label is not None:
                validation_rows.append(
                    (trace.group_id, step.step_label, [values[index] for values in predictions])
                )
    if not validation_rows:
        raise ValueError("reviewed ensemble requires explicit validation labels")

    def loss(temperature):
        errors = {}
        for family, target, values in validation_rows:
            probability = statistics.fmean(_temperature(value, temperature) for value in values)
            errors.setdefault(family, []).append((probability - target) ** 2)
        return statistics.fmean(statistics.fmean(values) for values in errors.values())

    temperature = min((i / 20 for i in range(5, 81)), key=lambda t: (loss(t), abs(t - 1)))
    support_rows = {}
    for trace in training:
        for index, step in enumerate(trace.steps):
            support_rows.setdefault(pattern(step, trace.high_risk), []).append(
                ((index + 1) / len(trace.steps), step.confidence)
            )
    support = [
        FeatureSupport(
            pattern=name,
            progress_min=min(a for a, _ in values),
            progress_max=max(a for a, _ in values),
            confidence_min=min(b for _, b in values),
            confidence_max=max(b for _, b in values),
        )
        for name, values in sorted(support_rows.items())
    ]
    return {
        "members": members,
        "calibration_temperature": temperature,
        "support": support,
        "bootstrap_fingerprints": draws,
        "explicit_training_steps": sum(
            step.step_label is not None for trace in training for step in trace.steps
        ),
    }
