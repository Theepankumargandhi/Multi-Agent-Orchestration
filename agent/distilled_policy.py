"""Masked search-visit distillation with a small, integrity-checked policy head."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.world_model import ACTIONS, ROUTES, WorldAction

FEATURES = (
    "bias",
    "confidence",
    "evidence",
    "reasoned",
    "verified",
    "high_risk",
    "retrieval_available",
    "verification_available",
    "remaining_tokens",
    "remaining_retrievals",
    "missing_evidence",
    "answer_ready",
    *(f"route_{route}" for route in ROUTES),
    *(f"legal_{action}" for action in ACTIONS),
)


def fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class DistillationContext(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    route: str = Field(min_length=2, max_length=80)
    confidence: float = Field(ge=0, le=1)
    evidence_count: int = Field(ge=0, le=100)
    reasoned: bool = False
    verified: bool = False
    high_risk: bool = False
    retrieval_available: bool = True
    verification_available: bool = True
    remaining_tokens: int = Field(ge=0, le=100_000)
    remaining_retrievals: int = Field(ge=0, le=8)
    legal_actions: list[WorldAction] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def unique_actions(self) -> "DistillationContext":
        if len(set(self.legal_actions)) != len(self.legal_actions):
            raise ValueError("legal actions must be unique")
        return self


def features(context: DistillationContext) -> list[float]:
    required = 2 if context.high_risk else 1
    return [
        1.0,
        context.confidence,
        min(context.evidence_count, 5) / 5,
        float(context.reasoned),
        float(context.verified),
        float(context.high_risk),
        float(context.retrieval_available),
        float(context.verification_available),
        min(context.remaining_tokens, 2000) / 2000,
        context.remaining_retrievals / 8,
        float(context.evidence_count < required),
        float(
            context.reasoned
            and context.verified
            and context.evidence_count >= required
            and context.confidence >= 0.62
        ),
        *(float(context.route.casefold() == route) for route in ROUTES),
        *(float(action in context.legal_actions) for action in ACTIONS),
    ]


class SearchTeacherExample(BaseModel):
    example_id: str
    split: Literal["train", "validation", "test"]
    context: DistillationContext
    target_probabilities: dict[WorldAction, float]
    teacher_plan_fingerprint: str
    teacher_model_fingerprint: str

    @model_validator(mode="after")
    def validate_target(self) -> "SearchTeacherExample":
        if set(self.target_probabilities) != set(self.context.legal_actions):
            raise ValueError("teacher targets must match the legal action mask")
        values = list(self.target_probabilities.values())
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError("invalid teacher probabilities")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-8):
            raise ValueError("teacher target probabilities must sum to one")
        return self


class DistilledPolicyArtifact(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    schema_version: Literal["1.0"] = "1.0"
    model_type: Literal["masked-search-visit-softmax"] = "masked-search-visit-softmax"
    feature_names: list[str] = Field(default_factory=lambda: list(FEATURES))
    weights: dict[WorldAction, list[float]]
    supported_routes: list[str]
    entropy_threshold: float = Field(default=0.98, gt=0, le=1)
    training_examples: int = Field(ge=1)
    validation_examples: int = Field(ge=1)
    training_fingerprint: str
    teacher_model_fingerprints: list[str]
    teacher_policy_fingerprint: str
    validation_kl: float = Field(ge=0)
    validation_uniform_kl: float = Field(ge=0)
    artifact_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_weights(self) -> "DistilledPolicyArtifact":
        if self.feature_names != list(FEATURES) or set(self.weights) != set(ACTIONS):
            raise ValueError("distilled-policy feature or action schema mismatch")
        if any(len(weights) != len(FEATURES) for weights in self.weights.values()):
            raise ValueError("distilled-policy weight dimensions mismatch")
        if not all(math.isfinite(value) for weights in self.weights.values() for value in weights):
            raise ValueError("distilled-policy weights must be finite")
        return self

    def seal(self) -> None:
        self.artifact_fingerprint = fingerprint(
            self.model_dump(mode="json", exclude={"artifact_fingerprint"})
        )

    def verify(self) -> bool:
        return bool(self.artifact_fingerprint) and self.artifact_fingerprint == fingerprint(
            self.model_dump(mode="json", exclude={"artifact_fingerprint"})
        )

    def save(self, path: Path) -> None:
        self.seal()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DistilledPolicyArtifact":
        artifact = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if not artifact.verify():
            raise ValueError("distilled-policy artifact integrity verification failed")
        return artifact


class DistilledPrior(BaseModel):
    probabilities: dict[WorldAction, float]
    normalized_entropy: float = Field(ge=0, le=1)
    fallback: bool
    fallback_reason: str
    policy_fingerprint: str


def _probabilities(
    weights: dict[WorldAction, list[float]],
    context: DistillationContext,
) -> dict[WorldAction, float]:
    vector = features(context)
    scores = {
        action: sum(a * b for a, b in zip(weights[action], vector, strict=True))
        for action in context.legal_actions
    }
    peak = max(scores.values())
    exponents = {action: math.exp(score - peak) for action, score in scores.items()}
    total = sum(exponents.values())
    return {action: value / total for action, value in exponents.items()}


def _kl(target: dict[WorldAction, float], predicted: dict[WorldAction, float]) -> float:
    return max(
        0.0,
        sum(
            value * math.log(value / max(1e-12, predicted[action]))
            for action, value in target.items()
            if value > 0
        ),
    )


def train_distilled_policy(
    examples: list[SearchTeacherExample],
    *,
    teacher_policy_fingerprint: str,
    epochs: int = 600,
    learning_rate: float = 0.15,
) -> DistilledPolicyArtifact:
    """Fit only training rows; validation selects the checkpoint, test rows are forbidden."""
    if any(item.split == "test" for item in examples):
        raise ValueError("test examples cannot enter policy distillation")
    if len({item.example_id for item in examples}) != len(examples):
        raise ValueError("distillation example ids must be unique")
    train = [item for item in examples if item.split == "train"]
    validation = [item for item in examples if item.split == "validation"]
    if not train or not validation or epochs < 1:
        raise ValueError("distillation requires train, validation, and positive epochs")
    train_contexts = {fingerprint(item.context.model_dump(mode="json")) for item in train}
    if any(fingerprint(item.context.model_dump(mode="json")) in train_contexts for item in validation):
        raise ValueError("train and validation contexts overlap")
    weights = {action: [0.0] * len(FEATURES) for action in ACTIONS}
    best_weights = {action: values.copy() for action, values in weights.items()}
    uniform_kl = sum(
        _kl(
            item.target_probabilities,
            {action: 1 / len(item.context.legal_actions) for action in item.context.legal_actions},
        )
        for item in validation
    ) / len(validation)
    best_kl = uniform_kl
    vectors = [features(item.context) for item in train]
    for epoch in range(epochs):
        gradient = {action: [0.0] * len(FEATURES) for action in ACTIONS}
        for item, vector in zip(train, vectors, strict=True):
            probabilities = _probabilities(weights, item.context)
            for action in item.context.legal_actions:
                error = probabilities[action] - item.target_probabilities[action]
                for index, value in enumerate(vector):
                    gradient[action][index] += error * value
        for action in ACTIONS:
            for index in range(len(FEATURES)):
                weights[action][index] -= learning_rate * (
                    gradient[action][index] / len(train) + 0.0001 * weights[action][index]
                )
        if epoch % 10 == 0 or epoch == epochs - 1:
            validation_kl = sum(
                _kl(item.target_probabilities, _probabilities(weights, item.context)) for item in validation
            ) / len(validation)
            if validation_kl < best_kl:
                best_kl = validation_kl
                best_weights = {action: values.copy() for action, values in weights.items()}
    artifact = DistilledPolicyArtifact(
        weights=best_weights,
        supported_routes=sorted({item.context.route.casefold() for item in train}),
        training_examples=len(train),
        validation_examples=len(validation),
        training_fingerprint=fingerprint([item.model_dump(mode="json") for item in examples]),
        teacher_model_fingerprints=sorted({item.teacher_model_fingerprint for item in examples}),
        teacher_policy_fingerprint=teacher_policy_fingerprint,
        validation_kl=best_kl,
        validation_uniform_kl=uniform_kl,
    )
    artifact.seal()
    return artifact


class DistilledPlanningPolicy:
    def __init__(self, artifact: DistilledPolicyArtifact):
        if not artifact.verify():
            raise ValueError("distilled-policy artifact integrity verification failed")
        self.artifact = artifact

    @classmethod
    def load(cls, path: Path) -> "DistilledPlanningPolicy":
        return cls(DistilledPolicyArtifact.load(path))

    def priors(self, context: DistillationContext) -> DistilledPrior:
        probabilities = _probabilities(self.artifact.weights, context)
        count = len(probabilities)
        entropy = (
            (-sum(p * math.log(max(p, 1e-12)) for p in probabilities.values()) / math.log(count))
            if count > 1
            else 0.0
        )
        entropy = max(0.0, min(1.0, entropy))
        reason = ""
        if context.route.casefold() not in self.artifact.supported_routes:
            reason = "unsupported_route"
        elif entropy > self.artifact.entropy_threshold:
            reason = "high_policy_entropy"
        if reason:
            probabilities = {action: 1 / count for action in probabilities}
        return DistilledPrior(
            probabilities=probabilities,
            normalized_entropy=entropy,
            fallback=bool(reason),
            fallback_reason=reason,
            policy_fingerprint=self.artifact.artifact_fingerprint,
        )
