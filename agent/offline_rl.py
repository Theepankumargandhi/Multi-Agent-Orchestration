"""Conservative offline-RL action priors for bounded agent planning."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from agent.world_model import FEATURE_NAMES, WorldAction, transition_features


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _softmax(values: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in values]
    peak = max(scaled)
    exponents = [math.exp(max(-30.0, min(30.0, value - peak))) for value in scaled]
    total = sum(exponents)
    return [value / total for value in exponents]


class PlanningState(BaseModel):
    evidence_count: int = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    reasoned: bool = False
    verified: bool = False


class LoggedPlanningTransition(BaseModel):
    event_id: str = Field(min_length=2, max_length=160)
    episode_id: str = Field(min_length=2, max_length=160)
    timestep: int = Field(ge=0, le=20)
    route: str = Field(min_length=2, max_length=80)
    high_risk: bool = False
    state: PlanningState
    action: WorldAction
    reward: float = Field(ge=-1, le=1)
    next_state: PlanningState
    done: bool = False
    safe: bool = True
    behavior_propensity: float = Field(gt=0, le=1)
    split: Literal["train", "validation", "test"] = "train"
    review_status: Literal["synthetic_seed", "human_reviewed"] = "synthetic_seed"


class QEnsembleMember(BaseModel):
    weights: list[float]
    member_fingerprint: str = ""

    def seal(self) -> None:
        self.member_fingerprint = _hash(
            self.model_dump(mode="json", exclude={"member_fingerprint"})
        )

    def verify(self) -> bool:
        return bool(self.member_fingerprint) and self.member_fingerprint == _hash(
            self.model_dump(mode="json", exclude={"member_fingerprint"})
        )


class OfflineRLArtifact(BaseModel):
    schema_version: str = "1.0"
    model_type: str = "bootstrapped-linear-cql"
    feature_names: list[str] = Field(default_factory=lambda: list(FEATURE_NAMES))
    members: list[QEnsembleMember] = Field(min_length=3, max_length=15)
    supported_routes: list[str]
    supported_actions: list[WorldAction]
    gamma: float = Field(gt=0, lt=1)
    conservative_alpha: float = Field(ge=0, le=2)
    policy_temperature: float = Field(gt=0, le=2)
    risk_multiplier: float = Field(ge=0, le=5)
    uncertainty_threshold: float = Field(gt=0)
    training_examples: int = Field(ge=1)
    validation_examples: int = Field(ge=1)
    human_reviewed_examples: int = Field(ge=0)
    dataset_fingerprint: str
    artifact_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_schema(self) -> "OfflineRLArtifact":
        if self.schema_version != "1.0" or self.model_type != "bootstrapped-linear-cql":
            raise ValueError("unsupported offline-RL artifact schema")
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError("offline-RL feature schema mismatch")
        for member in self.members:
            if not member.verify():
                raise ValueError("offline-RL member integrity verification failed")
            if len(member.weights) != len(FEATURE_NAMES):
                raise ValueError("offline-RL member width mismatch")
            if not all(math.isfinite(value) for value in member.weights):
                raise ValueError("offline-RL artifact contains non-finite values")
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
    def load(cls, path: Path) -> "OfflineRLArtifact":
        artifact = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if not artifact.verify():
            raise ValueError("offline-RL artifact integrity verification failed")
        return artifact


class ActionPrior(BaseModel):
    action: WorldAction
    q_mean: float
    q_std: float = Field(ge=0)
    q_lcb: float
    probability: float = Field(ge=0, le=1)


class PolicyPriorEstimate(BaseModel):
    actions: list[ActionPrior]
    max_uncertainty: float = Field(ge=0)
    out_of_distribution: bool
    policy_fingerprint: str


def load_planning_replay(path: Path) -> list[LoggedPlanningTransition]:
    transitions = [
        LoggedPlanningTransition.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not transitions:
        raise ValueError("offline-RL replay dataset is empty")
    event_ids = [item.event_id for item in transitions]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("offline-RL event ids must be unique")
    episode_steps: dict[str, list[int]] = {}
    for item in transitions:
        episode_steps.setdefault(item.episode_id, []).append(item.timestep)
    if any(sorted(steps) != list(range(len(steps))) for steps in episode_steps.values()):
        raise ValueError("offline-RL episode timesteps must be contiguous and zero-based")
    return transitions


def safe_actions(state: PlanningState, *, high_risk: bool) -> list[WorldAction]:
    actions: list[WorldAction] = []
    if not state.reasoned:
        actions.append("retrieve")
        if state.evidence_count > 0:
            actions.append("reason")
    if state.reasoned and not state.verified:
        actions.append("verify")
    required_evidence = 2 if high_risk else 1
    if (
        state.reasoned
        and state.verified
        and state.evidence_count >= required_evidence
        and state.confidence >= 0.62
    ):
        actions.append("answer")
    actions.append("abstain")
    return actions


def _features(
    route: str,
    high_risk: bool,
    state: PlanningState,
    action: WorldAction,
) -> list[float]:
    return transition_features(
        route=route,
        action=action,
        high_risk=high_risk,
        evidence_count=state.evidence_count,
        confidence=state.confidence,
        reasoned=state.reasoned,
        verified=state.verified,
    )


def _train_member(
    transitions: list[LoggedPlanningTransition],
    *,
    gamma: float,
    conservative_alpha: float,
    epochs: int,
) -> QEnsembleMember:
    weights = [0.0] * len(FEATURE_NAMES)
    learning_rate = 0.035
    for _ in range(epochs):
        gradient = [0.0] * len(weights)
        for item in transitions:
            logged_features = _features(item.route, item.high_risk, item.state, item.action)
            target = item.reward
            if not item.done:
                next_actions = safe_actions(item.next_state, high_risk=item.high_risk)
                target += gamma * max(
                    _dot(
                        weights,
                        _features(item.route, item.high_risk, item.next_state, action),
                    )
                    for action in next_actions
                )
            prediction = _dot(weights, logged_features)
            td_error = prediction - target
            candidates = safe_actions(item.state, high_risk=item.high_risk)
            candidate_features = [
                _features(item.route, item.high_risk, item.state, action)
                for action in candidates
            ]
            probabilities = _softmax(
                [_dot(weights, features) for features in candidate_features], 1.0
            )
            for index in range(len(weights)):
                conservative_gradient = sum(
                    probability * features[index]
                    for probability, features in zip(
                        probabilities, candidate_features, strict=True
                    )
                ) - logged_features[index]
                l2 = 0.0005 * weights[index] if index else 0.0
                gradient[index] += (
                    td_error * logged_features[index]
                    + conservative_alpha * conservative_gradient
                    + l2
                )
        scale = learning_rate / len(transitions)
        weights = [
            weight - scale * value
            for weight, value in zip(weights, gradient, strict=True)
        ]
    member = QEnsembleMember(weights=weights)
    member.seal()
    return member


def train_offline_rl_policy(
    transitions: list[LoggedPlanningTransition],
    *,
    member_count: int = 3,
    seed: int = 211,
    gamma: float = 0.9,
    conservative_alpha: float = 0.08,
    epochs: int = 900,
    policy_temperature: float = 0.18,
    risk_multiplier: float = 1.25,
) -> OfflineRLArtifact:
    if not 3 <= member_count <= 15:
        raise ValueError("offline-RL policy requires 3 to 15 members")
    training = [item for item in transitions if item.split == "train"]
    validation = [item for item in transitions if item.split == "validation"]
    if not training or not validation:
        raise ValueError("offline-RL training requires train and validation transitions")
    episodes: dict[str, list[LoggedPlanningTransition]] = {}
    for item in training:
        episodes.setdefault(item.episode_id, []).append(item)
    members = []
    episode_groups = list(episodes.values())
    for member_index in range(member_count):
        rng = random.Random(seed + member_index)
        bootstrap = training + [
            item
            for _ in range(len(episode_groups))
            for item in rng.choice(episode_groups)
        ]
        members.append(
            _train_member(
                bootstrap,
                gamma=gamma,
                conservative_alpha=conservative_alpha,
                epochs=epochs,
            )
        )
    artifact = OfflineRLArtifact(
        members=members,
        supported_routes=sorted({item.route.casefold() for item in training}),
        supported_actions=sorted({item.action for item in training}),
        gamma=gamma,
        conservative_alpha=conservative_alpha,
        policy_temperature=policy_temperature,
        risk_multiplier=risk_multiplier,
        uncertainty_threshold=0.05,
        training_examples=len(training),
        validation_examples=len(validation),
        human_reviewed_examples=sum(
            item.review_status == "human_reviewed" for item in training
        ),
        dataset_fingerprint=_hash(
            [item.model_dump(mode="json") for item in training + validation]
        ),
    )
    artifact.seal()
    policy = ConservativePlanningPolicy(artifact)
    uncertainties = [
        policy.priors(
            route=item.route,
            high_risk=item.high_risk,
            state=item.state,
            actions=safe_actions(item.state, high_risk=item.high_risk),
        ).max_uncertainty
        for item in validation
    ]
    artifact.uncertainty_threshold = max(0.02, max(uncertainties) * 1.5)
    artifact.seal()
    return artifact


class ConservativePlanningPolicy:
    def __init__(self, artifact: OfflineRLArtifact, *, verify_artifact: bool = True):
        if verify_artifact and not artifact.verify():
            raise ValueError("offline-RL artifact integrity verification failed")
        self.artifact = artifact

    @classmethod
    def load(cls, path: Path) -> "ConservativePlanningPolicy":
        return cls(OfflineRLArtifact.load(path))

    def priors(
        self,
        *,
        route: str,
        high_risk: bool,
        state: PlanningState,
        actions: list[WorldAction],
    ) -> PolicyPriorEstimate:
        if not actions:
            raise ValueError("offline-RL prior requires at least one candidate action")
        estimates: list[tuple[WorldAction, float, float, float]] = []
        for action in actions:
            features = _features(route, high_risk, state, action)
            scores = [_dot(member.weights, features) for member in self.artifact.members]
            mean = statistics.fmean(scores)
            deviation = statistics.pstdev(scores)
            multiplier = self.artifact.risk_multiplier * (1.5 if high_risk else 1.0)
            estimates.append((action, mean, deviation, mean - multiplier * deviation))
        maximum_uncertainty = max(item[2] for item in estimates)
        out_of_distribution = (
            route.casefold() not in self.artifact.supported_routes
            or maximum_uncertainty > self.artifact.uncertainty_threshold
        )
        probabilities = (
            [1 / len(estimates)] * len(estimates)
            if out_of_distribution
            else _softmax(
                [item[3] for item in estimates], self.artifact.policy_temperature
            )
        )
        return PolicyPriorEstimate(
            actions=[
                ActionPrior(
                    action=item[0],
                    q_mean=item[1],
                    q_std=item[2],
                    q_lcb=item[3],
                    probability=probability,
                )
                for item, probability in zip(estimates, probabilities, strict=True)
            ],
            max_uncertainty=maximum_uncertainty,
            out_of_distribution=out_of_distribution,
            policy_fingerprint=self.artifact.artifact_fingerprint,
        )
