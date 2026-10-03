"""Calibrated ensemble uncertainty for process-reward verification."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from agent.process_reward import (
    ProcessRewardArtifact,
    ProcessRewardScorer,
    ProcessStep,
    ProcessTrace,
    train_process_reward_model,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, value))))


def _calibrate(probability: float, temperature: float) -> float:
    clipped = min(1 - 1e-6, max(1e-6, probability))
    logit = math.log(clipped / (1 - clipped))
    return _sigmoid(logit / temperature)


class RewardEstimate(BaseModel):
    mean: float = Field(ge=0, le=1)
    standard_deviation: float = Field(ge=0, le=1)
    lower_confidence_bound: float = Field(ge=0, le=1)
    out_of_distribution: bool
    member_scores: list[float] = Field(min_length=1, max_length=15)


class ProcessRewardEnsembleArtifact(BaseModel):
    schema_version: str = "1.0"
    model_type: str = "bootstrapped-process-reward-ensemble"
    members: list[ProcessRewardArtifact] = Field(min_length=3, max_length=15)
    calibration_temperature: float = Field(ge=0.25, le=4)
    risk_multiplier: float = Field(ge=0, le=5)
    disagreement_threshold: float = Field(gt=0, le=0.5)
    calibration_examples: int = Field(ge=1)
    dataset_fingerprint: str
    artifact_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_schema(self) -> "ProcessRewardEnsembleArtifact":
        if self.schema_version != "1.0" or self.model_type != "bootstrapped-process-reward-ensemble":
            raise ValueError("unsupported process-reward ensemble schema")
        if not all(member.verify() for member in self.members):
            raise ValueError("process-reward ensemble contains an invalid member")
        return self

    def seal(self) -> None:
        self.artifact_fingerprint = hashlib.sha256(
            _canonical(self.model_dump(mode="json", exclude={"artifact_fingerprint"}))
        ).hexdigest()

    def verify(self) -> bool:
        expected = hashlib.sha256(
            _canonical(self.model_dump(mode="json", exclude={"artifact_fingerprint"}))
        ).hexdigest()
        return bool(self.artifact_fingerprint) and self.artifact_fingerprint == expected

    def save(self, path: Path) -> None:
        self.seal()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ProcessRewardEnsembleArtifact":
        artifact = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if not artifact.verify():
            raise ValueError("process-reward ensemble integrity verification failed")
        return artifact


def _raw_scores(
    members: list[ProcessRewardArtifact], steps: list[ProcessStep], high_risk: bool
) -> list[float]:
    return [
        ProcessRewardScorer(member).score_steps(steps, high_risk)
        for member in members
    ]


def train_process_reward_ensemble(
    traces: list[ProcessTrace],
    *,
    member_count: int = 5,
    seed: int = 41,
    epochs: int = 500,
    risk_multiplier: float = 1.5,
) -> ProcessRewardEnsembleArtifact:
    if not 3 <= member_count <= 15:
        raise ValueError("process-reward ensemble requires 3 to 15 members")
    training = [trace for trace in traces if trace.split == "train"]
    validation = [trace for trace in traces if trace.split == "validation"]
    if not training or not validation:
        raise ValueError("ensemble training requires train and validation traces")
    members = []
    for member_index in range(member_count):
        rng = random.Random(seed + member_index)
        bootstrapped = [rng.choice(training) for _ in training]
        members.append(
            train_process_reward_model(
                bootstrapped + validation,
                epochs=epochs,
                learning_rate=0.12,
            )
        )

    validation_rows = []
    for trace in validation:
        scores = _raw_scores(members, trace.steps, trace.high_risk)
        validation_rows.append(
            (
                statistics.fmean(scores),
                float(trace.safe and trace.outcome_quality >= 0.8),
                statistics.pstdev(scores),
            )
        )
    temperature = min(
        (step / 20 for step in range(5, 81)),
        key=lambda candidate: (
            sum(
                (_calibrate(score, candidate) - target) ** 2
                for score, target, _ in validation_rows
            ),
            abs(candidate - 1),
        ),
    )
    disagreement_threshold = min(
        0.5,
        max(0.025, max(row[2] for row in validation_rows) * 1.5),
    )
    dataset_fingerprint = hashlib.sha256(
        b"\n".join(_canonical(trace.model_dump(mode="json")) for trace in training + validation)
    ).hexdigest()
    artifact = ProcessRewardEnsembleArtifact(
        members=members,
        calibration_temperature=temperature,
        risk_multiplier=risk_multiplier,
        disagreement_threshold=disagreement_threshold,
        calibration_examples=len(validation),
        dataset_fingerprint=dataset_fingerprint,
    )
    artifact.seal()
    return artifact


class EnsembleProcessRewardScorer:
    def __init__(
        self,
        artifact: ProcessRewardEnsembleArtifact,
        *,
        verify_artifact: bool = True,
    ) -> None:
        if verify_artifact and not artifact.verify():
            raise ValueError("process-reward ensemble integrity verification failed")
        self.artifact = artifact
        self._member_scorers = [ProcessRewardScorer(member) for member in artifact.members]

    @classmethod
    def load(cls, path: Path) -> "EnsembleProcessRewardScorer":
        return cls(ProcessRewardEnsembleArtifact.load(path))

    def score_steps_with_uncertainty(
        self, steps: list[ProcessStep], high_risk: bool = False
    ) -> RewardEstimate:
        scores = [
            scorer.score_steps(steps, high_risk) for scorer in self._member_scorers
        ]
        raw_mean = statistics.fmean(scores)
        calibrated_mean = _calibrate(raw_mean, self.artifact.calibration_temperature)
        disagreement = statistics.pstdev(scores)
        multiplier = self.artifact.risk_multiplier * (1.5 if high_risk else 1.0)
        lower_bound = max(0.0, calibrated_mean - multiplier * disagreement)
        return RewardEstimate(
            mean=calibrated_mean,
            standard_deviation=disagreement,
            lower_confidence_bound=lower_bound,
            out_of_distribution=disagreement > self.artifact.disagreement_threshold,
            member_scores=scores,
        )

    def score_steps(self, steps: list[ProcessStep], high_risk: bool = False) -> float:
        """Return the conservative score so existing selection code becomes risk-aware."""
        return self.score_steps_with_uncertainty(
            steps, high_risk
        ).lower_confidence_bound
