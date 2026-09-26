"""Small, auditable learned router for quality-constrained model selection."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

FEATURE_NAMES = (
    "bias",
    "length",
    "multi_step",
    "freshness",
    "local_context",
    "relationship",
    "coding",
    "high_risk",
    "ambiguity",
)


class RoutingObservation(BaseModel):
    id: str
    query: str
    small_success: bool
    strong_success: bool
    small_cost_usd: float = Field(ge=0)
    strong_cost_usd: float = Field(ge=0)
    split: str = "train"
    high_risk: bool = False


class RouterEvaluation(BaseModel):
    threshold: float
    success_rate: float
    total_cost_usd: float
    fixed_strong_success_rate: float
    fixed_strong_cost_usd: float
    cost_reduction: float
    strong_selection_rate: float


@dataclass
class CostAwareRouter:
    weights: list[float]
    threshold: float = 0.5
    training_fingerprint: str = ""

    @staticmethod
    def features(query: str, high_risk: bool = False) -> list[float]:
        text = query.lower()
        tokens = re.findall(r"[a-z0-9_]+", text)
        return [
            1.0,
            min(len(tokens), 100) / 100.0,
            float(bool(re.search(r"\b(step[- ]by[- ]step|compare|analy[sz]e|plan|multi[- ]step)\b", text))),
            float(bool(re.search(r"\b(latest|current|today|news|recent)\b", text))),
            float(bool(re.search(r"\b(repository|codebase|document|pdf|local|uploaded)\b", text))),
            float(bool(re.search(r"\b(connect|depend|relationship|impact|trace|architecture)\w*\b", text))),
            float(bool(re.search(r"\b(code|function|class|api|database|debug|refactor)\w*\b", text))),
            float(high_risk or bool(re.search(r"\b(medical|legal|financial|security|delete|credential)\w*\b", text))),
            float(len(tokens) <= 3 or bool(re.search(r"\b(this|that|it|them)\b", text))),
        ]

    def probability_strong(self, query: str, high_risk: bool = False) -> float:
        values = self.features(query, high_risk)
        logit = sum(weight * value for weight, value in zip(self.weights, values, strict=True))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))

    def select(self, query: str, high_risk: bool = False) -> str:
        if high_risk:
            return "strong"
        return "strong" if self.probability_strong(query, high_risk) >= self.threshold else "small"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "feature_names": FEATURE_NAMES,
                    "weights": self.weights,
                    "threshold": self.threshold,
                    "training_fingerprint": self.training_fingerprint,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "CostAwareRouter":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if tuple(payload.get("feature_names") or ()) != FEATURE_NAMES:
            raise ValueError("router feature schema mismatch")
        return cls(
            weights=[float(item) for item in payload["weights"]],
            threshold=float(payload["threshold"]),
            training_fingerprint=str(payload.get("training_fingerprint") or ""),
        )


def _training_target(observation: RoutingObservation) -> float:
    return float(observation.strong_success and not observation.small_success or observation.high_risk)


def train_router(
    observations: list[RoutingObservation],
    epochs: int = 500,
    learning_rate: float = 0.2,
    regularization: float = 0.01,
) -> CostAwareRouter:
    train = [item for item in observations if item.split == "train"]
    if not train:
        raise ValueError("at least one training observation is required")
    weights = [0.0] * len(FEATURE_NAMES)
    for _ in range(epochs):
        gradient = [0.0] * len(weights)
        for observation in train:
            features = CostAwareRouter.features(observation.query, observation.high_risk)
            logit = sum(weight * value for weight, value in zip(weights, features, strict=True))
            prediction = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
            error = prediction - _training_target(observation)
            for index, value in enumerate(features):
                gradient[index] += error * value
        for index in range(len(weights)):
            penalty = 0.0 if index == 0 else regularization * weights[index]
            weights[index] -= learning_rate * (gradient[index] / len(train) + penalty)
    payload = "\n".join(item.model_dump_json() for item in train).encode()
    return CostAwareRouter(weights=weights, training_fingerprint=hashlib.sha256(payload).hexdigest())


def evaluate_router(router: CostAwareRouter, observations: list[RoutingObservation]) -> RouterEvaluation:
    if not observations:
        raise ValueError("at least one evaluation observation is required")
    successes = 0
    cost = 0.0
    strong_selected = 0
    for observation in observations:
        choice = router.select(observation.query, observation.high_risk)
        if choice == "strong":
            strong_selected += 1
            successes += int(observation.strong_success)
            cost += observation.strong_cost_usd
        else:
            successes += int(observation.small_success)
            cost += observation.small_cost_usd
    fixed_cost = sum(item.strong_cost_usd for item in observations)
    fixed_success = sum(item.strong_success for item in observations) / len(observations)
    return RouterEvaluation(
        threshold=router.threshold,
        success_rate=successes / len(observations),
        total_cost_usd=cost,
        fixed_strong_success_rate=fixed_success,
        fixed_strong_cost_usd=fixed_cost,
        cost_reduction=(fixed_cost - cost) / fixed_cost if fixed_cost else 0.0,
        strong_selection_rate=strong_selected / len(observations),
    )


def tune_threshold(
    router: CostAwareRouter,
    validation: list[RoutingObservation],
    minimum_success_rate: float,
) -> RouterEvaluation:
    candidates: list[RouterEvaluation] = []
    for step in range(5, 100, 5):
        router.threshold = step / 100
        report = evaluate_router(router, validation)
        if report.success_rate >= minimum_success_rate:
            candidates.append(report)
    if not candidates:
        raise ValueError("no threshold satisfies the requested success rate")
    best = min(candidates, key=lambda item: (item.total_cost_usd, -item.success_rate, item.threshold))
    router.threshold = best.threshold
    return best
