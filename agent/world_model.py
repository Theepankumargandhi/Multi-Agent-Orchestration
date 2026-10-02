"""Integrity-sealed action-conditioned dynamics model for agent planning."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

WorldAction = Literal["retrieve", "reason", "verify", "answer", "abstain"]
ROUTES = ("rag", "web", "kg", "code", "general")
ACTIONS = ("retrieve", "reason", "verify", "answer", "abstain")
FEATURE_NAMES = (
    "bias",
    "confidence",
    "evidence",
    "reasoned",
    "verified",
    "high_risk",
    *(f"route_{route}" for route in ROUTES),
    *(f"action_{action}" for action in ACTIONS),
    *(f"route_action_{route}_{action}" for route in ROUTES for action in ACTIONS),
    *(f"action_confidence_{action}" for action in ACTIONS),
    *(f"action_evidence_{action}" for action in ACTIONS),
    *(f"action_reasoned_{action}" for action in ACTIONS),
    *(f"action_verified_{action}" for action in ACTIONS),
    *(f"action_high_risk_{action}" for action in ACTIONS),
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, value))))


class TransitionExample(BaseModel):
    transition_id: str = Field(min_length=2, max_length=120)
    route: str = Field(min_length=2, max_length=40)
    action: WorldAction
    high_risk: bool = False
    before_evidence: int = Field(ge=0, le=100)
    before_confidence: float = Field(ge=0, le=1)
    before_reasoned: bool = False
    before_verified: bool = False
    after_evidence: int = Field(ge=0, le=100)
    after_confidence: float = Field(ge=0, le=1)
    transition_succeeded: bool = True
    split: Literal["train", "validation", "test"] = "train"
    review_status: Literal["synthetic_seed", "human_reviewed"] = "synthetic_seed"


class WorldModelMember(BaseModel):
    confidence_weights: list[float]
    evidence_weights: list[float]
    success_weights: list[float]
    confidence_residual_std: float = Field(ge=0, le=1)
    evidence_residual_std: float = Field(ge=0, le=10)
    member_fingerprint: str = ""

    def seal(self) -> None:
        self.member_fingerprint = _hash(
            self.model_dump(mode="json", exclude={"member_fingerprint"})
        )

    def verify(self) -> bool:
        return bool(self.member_fingerprint) and self.member_fingerprint == _hash(
            self.model_dump(mode="json", exclude={"member_fingerprint"})
        )


class WorldModelArtifact(BaseModel):
    schema_version: str = "1.0"
    model_type: str = "bootstrapped-action-conditioned-dynamics"
    feature_names: list[str] = Field(default_factory=lambda: list(FEATURE_NAMES))
    members: list[WorldModelMember] = Field(min_length=3, max_length=15)
    supported_routes: list[str]
    supported_actions: list[WorldAction]
    confidence_ood_threshold: float = Field(gt=0, le=1)
    success_floor: float = Field(ge=0, le=1)
    success_temperature: float = Field(ge=0.25, le=4)
    risk_multiplier: float = Field(ge=0, le=5)
    training_examples: int = Field(ge=1)
    validation_examples: int = Field(ge=1)
    human_reviewed_examples: int = Field(ge=0)
    dataset_fingerprint: str
    artifact_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_schema(self) -> "WorldModelArtifact":
        if self.schema_version != "1.0" or self.model_type != "bootstrapped-action-conditioned-dynamics":
            raise ValueError("unsupported world-model artifact schema")
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError("world-model feature schema mismatch")
        width = len(FEATURE_NAMES)
        for member in self.members:
            if not member.verify():
                raise ValueError("world-model member integrity verification failed")
            if any(
                len(weights) != width
                for weights in (
                    member.confidence_weights,
                    member.evidence_weights,
                    member.success_weights,
                )
            ):
                raise ValueError("world-model member width mismatch")
        return self

    def seal(self) -> None:
        self.artifact_fingerprint = _hash(
            self.model_dump(mode="json", exclude={"artifact_fingerprint"})
        )

    def verify(self) -> bool:
        return bool(self.artifact_fingerprint) and self.artifact_fingerprint == _hash(
            self.model_dump(mode="json", exclude={"artifact_fingerprint"})
        )

    def save(self, path: Path) -> None:
        self.seal()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "WorldModelArtifact":
        artifact = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if not artifact.verify():
            raise ValueError("world-model artifact integrity verification failed")
        return artifact


class WorldModelPrediction(BaseModel):
    confidence_delta_mean: float
    confidence_delta_std: float = Field(ge=0)
    confidence_delta_lcb: float
    evidence_delta_mean: float
    evidence_delta_std: float = Field(ge=0)
    evidence_delta_lcb: float
    success_probability: float = Field(ge=0, le=1)
    success_lcb: float = Field(ge=0, le=1)
    epistemic_uncertainty: float = Field(ge=0)
    out_of_distribution: bool
    model_fingerprint: str


def load_transitions(path: Path) -> list[TransitionExample]:
    examples = [
        TransitionExample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not examples:
        raise ValueError("world-model transition dataset is empty")
    ids = [item.transition_id for item in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("world-model transition ids must be unique")
    return examples


def transition_features(
    *,
    route: str,
    action: WorldAction,
    high_risk: bool,
    evidence_count: int,
    confidence: float,
    reasoned: bool,
    verified: bool,
) -> list[float]:
    normalized_route = route.casefold()
    normalized_evidence = min(evidence_count, 5) / 5
    return [
        1.0,
        confidence,
        normalized_evidence,
        float(reasoned),
        float(verified),
        float(high_risk),
        *(float(normalized_route == item) for item in ROUTES),
        *(float(action == item) for item in ACTIONS),
        *(
            float(normalized_route == route and action == candidate)
            for route in ROUTES
            for candidate in ACTIONS
        ),
        *(float(action == item) * confidence for item in ACTIONS),
        *(float(action == item) * normalized_evidence for item in ACTIONS),
        *(float(action == item and reasoned) for item in ACTIONS),
        *(float(action == item and verified) for item in ACTIONS),
        *(float(action == item and high_risk) for item in ACTIONS),
    ]


def _targets(item: TransitionExample) -> tuple[float, float, float]:
    return (
        item.after_confidence - item.before_confidence,
        float(item.after_evidence - item.before_evidence),
        float(item.transition_succeeded),
    )


def _fit_linear(
    rows: list[tuple[list[float], float]], *, epochs: int, learning_rate: float, l2: float
) -> list[float]:
    weights = [0.0] * len(FEATURE_NAMES)
    for _ in range(epochs):
        gradient = [0.0] * len(weights)
        for features, target in rows:
            error = _dot(weights, features) - target
            for index, value in enumerate(features):
                gradient[index] += error * value
        for index in range(len(weights)):
            penalty = 0 if index == 0 else l2 * weights[index]
            weights[index] -= learning_rate * (gradient[index] / len(rows) + penalty)
    return weights


def _fit_logistic(
    rows: list[tuple[list[float], float]], *, epochs: int, learning_rate: float, l2: float
) -> list[float]:
    weights = [0.0] * len(FEATURE_NAMES)
    for _ in range(epochs):
        gradient = [0.0] * len(weights)
        for features, target in rows:
            error = _sigmoid(_dot(weights, features)) - target
            for index, value in enumerate(features):
                gradient[index] += error * value
        for index in range(len(weights)):
            penalty = 0 if index == 0 else l2 * weights[index]
            weights[index] -= learning_rate * (gradient[index] / len(rows) + penalty)
    return weights


def _feature_row(item: TransitionExample) -> list[float]:
    return transition_features(
        route=item.route,
        action=item.action,
        high_risk=item.high_risk,
        evidence_count=item.before_evidence,
        confidence=item.before_confidence,
        reasoned=item.before_reasoned,
        verified=item.before_verified,
    )


def _train_member(
    examples: list[TransitionExample], *, epochs: int
) -> WorldModelMember:
    rows = [(_feature_row(item), _targets(item)) for item in examples]
    confidence_rows = [(features, target[0]) for features, target in rows]
    evidence_rows = [(features, target[1]) for features, target in rows]
    success_rows = [(features, target[2]) for features, target in rows]
    confidence_weights = _fit_linear(
        confidence_rows, epochs=epochs, learning_rate=0.08, l2=0.0001
    )
    evidence_weights = _fit_linear(
        evidence_rows, epochs=epochs, learning_rate=0.08, l2=0.0001
    )
    success_weights = _fit_logistic(
        success_rows, epochs=epochs, learning_rate=0.12, l2=0.01
    )
    confidence_residuals = [
        target - _dot(confidence_weights, features)
        for features, target in confidence_rows
    ]
    evidence_residuals = [
        target - _dot(evidence_weights, features)
        for features, target in evidence_rows
    ]
    member = WorldModelMember(
        confidence_weights=confidence_weights,
        evidence_weights=evidence_weights,
        success_weights=success_weights,
        confidence_residual_std=statistics.pstdev(confidence_residuals),
        evidence_residual_std=statistics.pstdev(evidence_residuals),
    )
    member.seal()
    return member


def _fit_success_temperature(
    members: list[WorldModelMember], validation: list[TransitionExample]
) -> float:
    candidates = [0.5 + index * 0.05 for index in range(41)]

    def brier(temperature: float) -> float:
        errors = []
        for item in validation:
            features = _feature_row(item)
            probability = statistics.fmean(
                _sigmoid(_dot(member.success_weights, features) / temperature)
                for member in members
            )
            errors.append((probability - float(item.transition_succeeded)) ** 2)
        return statistics.fmean(errors)

    return min(candidates, key=lambda candidate: (brier(candidate), candidate))


def train_world_model(
    examples: list[TransitionExample],
    *,
    member_count: int = 3,
    seed: int = 73,
    epochs: int = 1000,
    risk_multiplier: float = 1.25,
) -> WorldModelArtifact:
    if not 3 <= member_count <= 15:
        raise ValueError("world model requires 3 to 15 members")
    training = [item for item in examples if item.split == "train"]
    validation = [item for item in examples if item.split == "validation"]
    if not training or not validation:
        raise ValueError("world-model training requires train and validation examples")
    members = []
    strata: dict[tuple[str, str, bool, bool], list[TransitionExample]] = {}
    for item in training:
        strata.setdefault(
            (item.action, item.route.casefold(), item.high_risk, item.transition_succeeded),
            [],
        ).append(item)
    for member_index in range(member_count):
        rng = random.Random(seed + member_index)
        bootstrap = training + [
            rng.choice(group)
            for group in strata.values()
            for _ in range(len(group))
        ]
        members.append(_train_member(bootstrap, epochs=epochs))
    provisional = WorldModelArtifact(
        members=members,
        supported_routes=sorted({item.route.casefold() for item in training}),
        supported_actions=sorted({item.action for item in training}),
        confidence_ood_threshold=0.05,
        success_floor=0.55,
        success_temperature=_fit_success_temperature(members, validation),
        risk_multiplier=risk_multiplier,
        training_examples=len(training),
        validation_examples=len(validation),
        human_reviewed_examples=sum(
            item.review_status == "human_reviewed" for item in training
        ),
        dataset_fingerprint=_hash(
            [item.model_dump(mode="json") for item in training + validation]
        ),
    )
    provisional.seal()
    model = AgentWorldModel(provisional)
    disagreements = [
        model.predict_example(item).epistemic_uncertainty for item in validation
    ]
    provisional.confidence_ood_threshold = min(
        0.5, max(0.015, max(disagreements) * 1.5)
    )
    provisional.seal()
    return provisional


class AgentWorldModel:
    def __init__(self, artifact: WorldModelArtifact, *, verify_artifact: bool = True):
        if verify_artifact and not artifact.verify():
            raise ValueError("world-model artifact integrity verification failed")
        self.artifact = artifact

    @classmethod
    def load(cls, path: Path) -> "AgentWorldModel":
        return cls(WorldModelArtifact.load(path))

    def predict(
        self,
        *,
        route: str,
        action: WorldAction,
        high_risk: bool,
        evidence_count: int,
        confidence: float,
        reasoned: bool,
        verified: bool,
    ) -> WorldModelPrediction:
        features = transition_features(
            route=route,
            action=action,
            high_risk=high_risk,
            evidence_count=evidence_count,
            confidence=confidence,
            reasoned=reasoned,
            verified=verified,
        )
        confidence_scores = [
            _dot(member.confidence_weights, features) for member in self.artifact.members
        ]
        evidence_scores = [
            _dot(member.evidence_weights, features) for member in self.artifact.members
        ]
        success_scores = [
            _sigmoid(
                _dot(member.success_weights, features)
                / self.artifact.success_temperature
            )
            for member in self.artifact.members
        ]
        confidence_mean = statistics.fmean(confidence_scores)
        evidence_mean = statistics.fmean(evidence_scores)
        success_mean = statistics.fmean(success_scores)
        confidence_epistemic = statistics.pstdev(confidence_scores)
        evidence_epistemic = statistics.pstdev(evidence_scores)
        success_epistemic = statistics.pstdev(success_scores)
        confidence_aleatoric = statistics.fmean(
            member.confidence_residual_std for member in self.artifact.members
        )
        evidence_aleatoric = statistics.fmean(
            member.evidence_residual_std for member in self.artifact.members
        )
        confidence_std = math.hypot(confidence_epistemic, confidence_aleatoric)
        evidence_std = math.hypot(evidence_epistemic, evidence_aleatoric)
        multiplier = self.artifact.risk_multiplier * (1.5 if high_risk else 1.0)
        normalized_route = route.casefold()
        ood = (
            normalized_route not in self.artifact.supported_routes
            or action not in self.artifact.supported_actions
            or confidence_epistemic > self.artifact.confidence_ood_threshold
        )
        return WorldModelPrediction(
            confidence_delta_mean=confidence_mean,
            confidence_delta_std=confidence_std,
            confidence_delta_lcb=confidence_mean - multiplier * confidence_std,
            evidence_delta_mean=evidence_mean,
            evidence_delta_std=evidence_std,
            evidence_delta_lcb=evidence_mean - multiplier * evidence_std,
            success_probability=success_mean,
            success_lcb=max(0.0, success_mean - multiplier * success_epistemic),
            epistemic_uncertainty=max(
                confidence_epistemic, evidence_epistemic, success_epistemic
            ),
            out_of_distribution=ood,
            model_fingerprint=self.artifact.artifact_fingerprint,
        )

    def predict_example(self, item: TransitionExample) -> WorldModelPrediction:
        return self.predict(
            route=item.route,
            action=item.action,
            high_risk=item.high_risk,
            evidence_count=item.before_evidence,
            confidence=item.before_confidence,
            reasoned=item.before_reasoned,
            verified=item.before_verified,
        )
