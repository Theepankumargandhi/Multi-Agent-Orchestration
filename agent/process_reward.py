"""Integrity-checked process reward model for verifier-guided agent search."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

StepKind = Literal["plan", "retrieve", "tool", "reason", "verify", "answer", "refuse"]
FEATURE_NAMES = (
    "bias",
    "progress",
    "retrieve",
    "tool",
    "verify",
    "answer",
    "has_evidence",
    "citation_valid",
    "policy_allowed",
    "error",
    "confidence",
    "high_risk",
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, value))))


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class ProcessStep(BaseModel):
    step_id: str = Field(min_length=1, max_length=120)
    kind: StepKind
    has_evidence: bool = False
    citation_valid: bool = False
    policy_allowed: bool = True
    error: bool = False
    confidence: float = Field(default=0.5, ge=0, le=1)
    tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    step_label: float | None = Field(default=None, ge=0, le=1)


class ProcessTrace(BaseModel):
    trace_id: str = Field(min_length=2, max_length=160)
    group_id: str = Field(min_length=2, max_length=160)
    query: str = Field(min_length=1, max_length=20_000)
    high_risk: bool = False
    self_confidence: float = Field(ge=0, le=1)
    steps: list[ProcessStep] = Field(min_length=1, max_length=64)
    outcome_quality: float = Field(ge=0, le=1)
    safe: bool
    split: Literal["train", "validation", "test"] = "train"
    review_status: Literal["synthetic_seed", "human_reviewed"] = "synthetic_seed"


class ProcessRewardArtifact(BaseModel):
    schema_version: str = "1.0"
    model_type: str = "process-logistic-regression"
    feature_names: list[str] = Field(default_factory=lambda: list(FEATURE_NAMES))
    weights: list[float]
    stopping_threshold: float = Field(ge=0, le=1)
    temporal_discount: float = Field(gt=0, le=1)
    training_examples: int = Field(ge=1)
    human_labeled_steps: int = Field(ge=0)
    dataset_fingerprint: str
    artifact_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_schema(self) -> "ProcessRewardArtifact":
        if self.schema_version != "1.0" or self.model_type != "process-logistic-regression":
            raise ValueError("unsupported process-reward artifact schema")
        if tuple(self.feature_names) != FEATURE_NAMES or len(self.weights) != len(FEATURE_NAMES):
            raise ValueError("process-reward feature schema mismatch")
        if not all(math.isfinite(value) and abs(value) <= 100 for value in self.weights):
            raise ValueError("process-reward weights must be finite and bounded")
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
    def load(cls, path: Path) -> "ProcessRewardArtifact":
        artifact = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if not artifact.verify():
            raise ValueError("process-reward artifact integrity verification failed")
        return artifact


class StepReward(BaseModel):
    step_id: str
    reward: float = Field(ge=0, le=1)
    credit: float
    pruned: bool
    reason: str = ""


class ProcessRewardReceipt(BaseModel):
    schema_version: str = "1.0"
    trace_id: str
    policy_fingerprint: str
    trajectory_score: float = Field(ge=0, le=1)
    accepted: bool
    pruned_at_step: str = ""
    step_rewards: list[StepReward]
    receipt_fingerprint: str = ""


def step_features(
    step: ProcessStep, index: int, total: int, *, high_risk: bool = False
) -> list[float]:
    progress = (index + 1) / max(1, total)
    return [
        1.0,
        progress,
        float(step.kind == "retrieve"),
        float(step.kind == "tool"),
        float(step.kind == "verify"),
        float(step.kind == "answer"),
        float(step.has_evidence),
        float(step.citation_valid),
        float(step.policy_allowed),
        float(step.error),
        step.confidence,
        float(high_risk),
    ]


def load_traces(path: Path) -> list[ProcessTrace]:
    traces = [
        ProcessTrace.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not traces:
        raise ValueError("process-supervision dataset is empty")
    ids = [trace.trace_id for trace in traces]
    if len(ids) != len(set(ids)):
        raise ValueError("process-supervision trace ids must be unique")
    return traces


def _step_target(trace: ProcessTrace, index: int, discount: float) -> float:
    step = trace.steps[index]
    if step.step_label is not None:
        return step.step_label
    terminal = trace.outcome_quality if trace.safe else 0.0
    remaining = len(trace.steps) - index - 1
    return terminal * discount**remaining


def train_process_reward_model(
    traces: list[ProcessTrace],
    *,
    epochs: int = 800,
    learning_rate: float = 0.15,
    l2: float = 0.01,
    temporal_discount: float = 0.9,
    explicit_steps_only: bool = False,
) -> ProcessRewardArtifact:
    training = [trace for trace in traces if trace.split == "train"]
    if not training:
        raise ValueError("at least one training trace is required")
    examples: list[tuple[list[float], float, float]] = []
    human_labeled = 0
    for trace in training:
        for index, step in enumerate(trace.steps):
            if explicit_steps_only and step.step_label is None:
                continue
            target = _step_target(trace, index, temporal_discount)
            if step.step_label is not None:
                human_labeled += int(trace.review_status == "human_reviewed")
            # Later steps receive more credit because they are closer to the verified outcome.
            weight = temporal_discount ** (len(trace.steps) - index - 1)
            examples.append(
                (step_features(step, index, len(trace.steps), high_risk=trace.high_risk), target, weight)
            )
    if not examples:
        raise ValueError("explicit process training requires labelled steps")
    weights = [0.0] * len(FEATURE_NAMES)
    for _ in range(epochs):
        gradient = [0.0] * len(weights)
        total_weight = sum(item[2] for item in examples)
        for features, target, example_weight in examples:
            error = (_sigmoid(_dot(weights, features)) - target) * example_weight
            for index, value in enumerate(features):
                gradient[index] += error * value
        for index in range(len(weights)):
            penalty = 0 if index == 0 else l2 * weights[index]
            weights[index] -= learning_rate * (gradient[index] / total_weight + penalty)
    dataset_fingerprint = hashlib.sha256(
        b"\n".join(_canonical(trace.model_dump(mode="json")) for trace in training)
    ).hexdigest()
    artifact = ProcessRewardArtifact(
        weights=weights,
        stopping_threshold=0.35,
        temporal_discount=temporal_discount,
        training_examples=len(examples),
        human_labeled_steps=human_labeled,
        dataset_fingerprint=dataset_fingerprint,
    )
    scorer = ProcessRewardScorer(artifact, verify_artifact=False)
    validation = [trace for trace in traces if trace.split == "validation"]
    if validation and not explicit_steps_only:
        artifact.stopping_threshold = _tune_stopping_threshold(scorer, validation)
    artifact.seal()
    return artifact


def _tune_stopping_threshold(
    scorer: "ProcessRewardScorer", traces: list[ProcessTrace]
) -> float:
    best: tuple[float, float, float] | None = None
    for threshold in [step / 100 for step in range(15, 76, 5)]:
        detected = 0
        false_prunes = 0
        failures = 0
        successes = 0
        for trace in traces:
            should_fail = not trace.safe or trace.outcome_quality < 0.8
            pruned = scorer.prune_index(trace.steps, trace.high_risk, threshold) is not None
            failures += int(should_fail)
            successes += int(not should_fail)
            detected += int(should_fail and pruned)
            false_prunes += int(not should_fail and pruned)
        recall = detected / max(1, failures)
        false_rate = false_prunes / max(1, successes)
        candidate = (recall - false_rate, -false_rate, threshold)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best else 0.35


class ProcessRewardScorer:
    def __init__(self, artifact: ProcessRewardArtifact, *, verify_artifact: bool = True):
        if verify_artifact and not artifact.verify():
            raise ValueError("process-reward artifact integrity verification failed")
        self.artifact = artifact

    @classmethod
    def load(cls, path: Path) -> "ProcessRewardScorer":
        return cls(ProcessRewardArtifact.load(path))

    def step_probabilities(
        self, steps: list[ProcessStep], high_risk: bool = False
    ) -> list[float]:
        return [
            _sigmoid(
                _dot(
                    self.artifact.weights,
                    step_features(step, index, len(steps), high_risk=high_risk),
                )
            )
            for index, step in enumerate(steps)
        ]

    def score_steps(self, steps: list[ProcessStep], high_risk: bool = False) -> float:
        probabilities = self.step_probabilities(steps, high_risk)
        if not probabilities:
            return 0.0
        score = 0.5 * probabilities[-1] + 0.3 * (sum(probabilities) / len(probabilities))
        score += 0.2 * min(probabilities)
        if any(not step.policy_allowed for step in steps):
            score *= 0.05
        if any(step.error for step in steps):
            score *= 0.25
        return max(0.0, min(1.0, score))

    def prune_index(
        self,
        steps: list[ProcessStep],
        high_risk: bool = False,
        threshold: float | None = None,
    ) -> int | None:
        threshold = self.artifact.stopping_threshold if threshold is None else threshold
        probabilities = self.step_probabilities(steps, high_risk)
        low_streak = 0
        for index, (step, probability) in enumerate(zip(steps, probabilities, strict=True)):
            if not step.policy_allowed or step.error:
                return index
            low_streak = low_streak + 1 if probability < threshold else 0
            if low_streak >= 2:
                return index
        return None

    def evaluate_trace(self, trace: ProcessTrace) -> ProcessRewardReceipt:
        probabilities = self.step_probabilities(trace.steps, trace.high_risk)
        full_score = self.score_steps(trace.steps, trace.high_risk)
        prune_index = self.prune_index(trace.steps, trace.high_risk)
        rewards = []
        for index, (step, reward) in enumerate(zip(trace.steps, probabilities, strict=True)):
            without = trace.steps[:index] + trace.steps[index + 1 :]
            counterfactual = self.score_steps(without, trace.high_risk) if without else 0.0
            pruned = prune_index is not None and index == prune_index
            reason = ""
            if pruned:
                reason = "policy_or_execution_failure" if not step.policy_allowed or step.error else "low_process_reward"
            rewards.append(
                StepReward(
                    step_id=step.step_id,
                    reward=reward,
                    credit=full_score - counterfactual,
                    pruned=pruned,
                    reason=reason,
                )
            )
        receipt = ProcessRewardReceipt(
            trace_id=trace.trace_id,
            policy_fingerprint=self.artifact.artifact_fingerprint,
            trajectory_score=full_score,
            accepted=prune_index is None,
            pruned_at_step=trace.steps[prune_index].step_id if prune_index is not None else "",
            step_rewards=rewards,
        )
        receipt.receipt_fingerprint = hashlib.sha256(
            _canonical(receipt.model_dump(mode="json", exclude={"receipt_fingerprint"}))
        ).hexdigest()
        return receipt

    def verify_receipt(self, receipt: ProcessRewardReceipt) -> bool:
        expected = hashlib.sha256(
            _canonical(receipt.model_dump(mode="json", exclude={"receipt_fingerprint"}))
        ).hexdigest()
        return bool(receipt.receipt_fingerprint) and receipt.receipt_fingerprint == expected
