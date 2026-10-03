"""Safety-constrained contextual bandit routing and offline policy evaluation.

The implementation intentionally uses small, auditable linear models. It learns from
logged feedback, emits propensities required for unbiased evaluation, and refuses to
promote policies that violate safety constraints or lack adequate action support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from evals.adaptive_router import FEATURE_NAMES, CostAwareRouter


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [
        [*row, *[float(index == column) for column in range(size)]]
        for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("singular bandit design matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column], strict=True)
            ]
    return [row[size:] for row in augmented]


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


class BanditAction(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,31}$")
    model: str = Field(min_length=1, max_length=160)
    max_cost_usd: float = Field(gt=0)
    max_latency_ms: float = Field(gt=0)
    allow_high_risk: bool = False


class LoggedBanditEvent(BaseModel):
    event_id: str = Field(min_length=2, max_length=160)
    query: str = Field(min_length=1, max_length=20_000)
    action: str
    propensity: float = Field(gt=0, le=1)
    quality: float = Field(ge=0, le=1)
    cost_usd: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    safe: bool = True
    high_risk: bool = False
    split: Literal["train", "validation", "test"] = "train"


class LinearActionModel(BaseModel):
    theta: list[float]
    covariance: list[list[float]]
    observations: int = Field(ge=1)


class BanditArtifact(BaseModel):
    schema_version: str = "1.0"
    model_type: str = "disjoint-linucb"
    feature_names: list[str] = Field(default_factory=lambda: list(FEATURE_NAMES))
    actions: list[BanditAction]
    models: dict[str, LinearActionModel]
    ridge: float = Field(gt=0)
    exploration_alpha: float = Field(ge=0)
    cost_weight: float = Field(ge=0)
    latency_weight: float = Field(ge=0)
    unsafe_penalty: float = Field(gt=0)
    training_fingerprint: str
    artifact_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_schema(self) -> "BanditArtifact":
        if self.schema_version != "1.0" or self.model_type != "disjoint-linucb":
            raise ValueError("unsupported contextual-bandit artifact schema")
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError("contextual-bandit feature schema mismatch")
        action_names = {item.name for item in self.actions}
        if action_names != set(self.models):
            raise ValueError("each configured action must have exactly one learned model")
        dimensions = len(FEATURE_NAMES)
        for model in self.models.values():
            if len(model.theta) != dimensions or any(
                len(row) != dimensions for row in model.covariance
            ):
                raise ValueError("invalid contextual-bandit model dimensions")
            values = model.theta + [value for row in model.covariance for value in row]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("contextual-bandit artifact contains non-finite values")
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
    def load(cls, path: Path) -> "BanditArtifact":
        artifact = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if not artifact.verify():
            raise ValueError("contextual-bandit artifact integrity check failed")
        return artifact


class ActionEstimate(BaseModel):
    action: str
    expected_utility: float
    uncertainty: float = Field(ge=0)
    score: float


def check_policy_reproducibility(artifact: BanditArtifact, reference_path: Path) -> dict:
    """Strict source/configuration identity; allow only bounded coefficient roundoff.

    Each artifact must independently verify its exact digest. The reference hash
    is never assigned to regenerated weights, and promotion is checked separately.
    """
    reference = BanditArtifact.load(reference_path)
    if not artifact.verify():
        raise ValueError("contextual-bandit artifact integrity check failed")
    exclude = {"models", "artifact_fingerprint"}
    if (artifact.model_dump(mode="json", exclude=exclude) != reference.model_dump(mode="json", exclude=exclude)
            or set(artifact.models) != set(reference.models)):
        raise ValueError("contextual-bandit training identity or configuration changed")
    maximum_delta = 0.0
    for name, model in artifact.models.items():
        expected = reference.models[name]
        if (model.model_dump(exclude={"theta", "covariance"}) != expected.model_dump(exclude={"theta", "covariance"})
                or len(model.theta) != len(expected.theta) or len(model.covariance) != len(expected.covariance)
                or any(len(row) != len(other) for row, other in zip(model.covariance, expected.covariance, strict=True))):
            raise ValueError("contextual-bandit model structure changed")
        values = model.theta + [value for row in model.covariance for value in row]
        expected_values = expected.theta + [value for row in expected.covariance for value in row]
        for actual, frozen in zip(values, expected_values, strict=True):
            if (not math.isfinite(actual) or not math.isfinite(frozen)
                    or not math.isclose(actual, frozen, rel_tol=0.0, abs_tol=1e-12)):
                raise ValueError("contextual-bandit learned coefficients changed beyond roundoff tolerance")
            maximum_delta = max(maximum_delta, abs(actual - frozen))
    return {
        "reproducible": True,
        "comparison": "strict-metadata-absolute-coefficient-tolerance",
        "absolute_tolerance": 1e-12,
        "maximum_coefficient_delta": maximum_delta,
        "exact_match": artifact.artifact_fingerprint == reference.artifact_fingerprint,
        "reference_fingerprint": reference.artifact_fingerprint,
        "generated_fingerprint": artifact.artifact_fingerprint,
    }


class BanditDecision(BaseModel):
    action: str
    model: str
    propensity: float = Field(gt=0, le=1)
    exploration: bool
    high_risk: bool
    feasible_actions: list[str]
    estimates: list[ActionEstimate]
    policy_fingerprint: str


class OfflinePolicyReport(BaseModel):
    schema_version: str = "1.0"
    dataset_fingerprint: str
    policy_fingerprint: str
    events: int
    matched_events: int
    support_violations: int
    effective_sample_size: float
    behavior_utility: float
    ips_utility: float
    snips_utility: float
    doubly_robust_utility: float
    confidence_low: float
    confidence_high: float
    estimated_lift: float
    target_safety_violations: int
    promoted: bool
    reasons: list[str]


class ContextualBanditPolicy:
    """Runtime selector with hard constraints and deterministic epsilon exploration."""

    def __init__(self, artifact: BanditArtifact):
        if not artifact.verify():
            raise ValueError("contextual-bandit artifact integrity check failed")
        self.artifact = artifact
        self.actions = {item.name: item for item in artifact.actions}

    @staticmethod
    def features(query: str, high_risk: bool = False) -> list[float]:
        return CostAwareRouter.features(query, high_risk)

    def _feasible(
        self,
        high_risk: bool,
        max_cost_usd: float | None,
        max_latency_ms: float | None,
    ) -> list[BanditAction]:
        feasible = [
            action
            for action in self.artifact.actions
            if (not high_risk or action.allow_high_risk)
            and (max_cost_usd is None or action.max_cost_usd <= max_cost_usd)
            and (max_latency_ms is None or action.max_latency_ms <= max_latency_ms)
        ]
        if not feasible:
            raise ValueError("no bandit action satisfies the safety and budget constraints")
        return feasible

    def estimates(
        self,
        query: str,
        *,
        high_risk: bool = False,
        max_cost_usd: float | None = None,
        max_latency_ms: float | None = None,
    ) -> list[ActionEstimate]:
        features = self.features(query, high_risk)
        estimates = []
        for action in self._feasible(high_risk, max_cost_usd, max_latency_ms):
            model = self.artifact.models[action.name]
            expected = _dot(model.theta, features)
            variance = max(0.0, _dot(features, _matvec(model.covariance, features)))
            uncertainty = math.sqrt(variance)
            estimates.append(
                ActionEstimate(
                    action=action.name,
                    expected_utility=expected,
                    uncertainty=uncertainty,
                    score=expected + self.artifact.exploration_alpha * uncertainty,
                )
            )
        return sorted(estimates, key=lambda item: (-item.score, item.action))

    def decide(
        self,
        query: str,
        *,
        request_id: str,
        high_risk: bool = False,
        max_cost_usd: float | None = None,
        max_latency_ms: float | None = None,
        epsilon: float = 0.0,
    ) -> BanditDecision:
        if not 0 <= epsilon < 1:
            raise ValueError("epsilon must be in [0, 1)")
        estimates = self.estimates(
            query,
            high_risk=high_risk,
            max_cost_usd=max_cost_usd,
            max_latency_ms=max_latency_ms,
        )
        selected = estimates[0]
        exploring = False
        if epsilon and len(estimates) > 1:
            seed = hashlib.sha256(
                f"{self.artifact.artifact_fingerprint}:{request_id}".encode()
            ).digest()
            draw = int.from_bytes(seed[:8], "big") / (2**64 - 1)
            if draw < epsilon:
                selected = estimates[int.from_bytes(seed[8:12], "big") % len(estimates)]
                exploring = selected.action != estimates[0].action
        count = len(estimates)
        propensity = epsilon / count
        if selected.action == estimates[0].action:
            propensity += 1 - epsilon
        action = self.actions[selected.action]
        return BanditDecision(
            action=action.name,
            model=action.model,
            propensity=propensity,
            exploration=exploring,
            high_risk=high_risk,
            feasible_actions=[item.action for item in estimates],
            estimates=estimates,
            policy_fingerprint=self.artifact.artifact_fingerprint,
        )


def event_utility(
    event: LoggedBanditEvent,
    *,
    cost_weight: float,
    latency_weight: float,
    unsafe_penalty: float,
) -> float:
    utility = event.quality - cost_weight * event.cost_usd
    utility -= latency_weight * event.latency_ms / 1000
    if not event.safe:
        utility -= unsafe_penalty
    return utility


def load_events(path: Path) -> list[LoggedBanditEvent]:
    events = [
        LoggedBanditEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not events:
        raise ValueError("bandit feedback dataset is empty")
    return events


def train_policy(
    events: list[LoggedBanditEvent],
    actions: list[BanditAction],
    *,
    ridge: float = 0.01,
    exploration_alpha: float = 0.15,
    cost_weight: float = 10.0,
    latency_weight: float = 0.05,
    unsafe_penalty: float = 2.0,
) -> BanditArtifact:
    training = [event for event in events if event.split == "train"]
    if not training:
        raise ValueError("at least one training event is required")
    action_names = {action.name for action in actions}
    unknown = {event.action for event in training} - action_names
    if unknown:
        raise ValueError(f"feedback references unknown actions: {sorted(unknown)}")
    dimensions = len(FEATURE_NAMES)
    models: dict[str, LinearActionModel] = {}
    for action in sorted(actions, key=lambda item: item.name):
        selected = [event for event in training if event.action == action.name]
        if not selected:
            raise ValueError(f"no training feedback for action {action.name}")
        design = [
            [ridge * float(row == column) for column in range(dimensions)]
            for row in range(dimensions)
        ]
        target = [0.0] * dimensions
        for event in selected:
            features = CostAwareRouter.features(event.query, event.high_risk)
            reward = event_utility(
                event,
                cost_weight=cost_weight,
                latency_weight=latency_weight,
                unsafe_penalty=unsafe_penalty,
            )
            for row in range(dimensions):
                target[row] += features[row] * reward
                for column in range(dimensions):
                    design[row][column] += features[row] * features[column]
        covariance = _inverse(design)
        models[action.name] = LinearActionModel(
            theta=_matvec(covariance, target),
            covariance=covariance,
            observations=len(selected),
        )
    fingerprint = hashlib.sha256(
        b"\n".join(_canonical(event.model_dump(mode="json")) for event in training)
    ).hexdigest()
    artifact = BanditArtifact(
        actions=sorted(actions, key=lambda item: item.name),
        models=models,
        ridge=ridge,
        exploration_alpha=exploration_alpha,
        cost_weight=cost_weight,
        latency_weight=latency_weight,
        unsafe_penalty=unsafe_penalty,
        training_fingerprint=fingerprint,
    )
    artifact.seal()
    return artifact


def evaluate_policy(
    artifact: BanditArtifact,
    events: list[LoggedBanditEvent],
    *,
    split: str = "test",
    minimum_effective_sample_size: float = 3.0,
    minimum_lift: float = 0.0,
    regression_tolerance: float = 0.05,
) -> OfflinePolicyReport:
    selected = [event for event in events if event.split == split]
    if not selected:
        raise ValueError(f"no {split} events are available for offline evaluation")
    policy = ContextualBanditPolicy(artifact)
    weighted_rewards: list[float] = []
    weights: list[float] = []
    dr_values: list[float] = []
    behavior_rewards: list[float] = []
    support_violations = 0
    target_safety_violations = 0
    matched = 0
    supported_contexts: list[tuple[str, bool]] = []
    target_actions: dict[tuple[str, bool], str] = {}
    logged_actions: dict[tuple[str, bool], set[str]] = {}
    for event in selected:
        context = (event.query, event.high_risk)
        logged_actions.setdefault(context, set()).add(event.action)
        reward = event_utility(
            event,
            cost_weight=artifact.cost_weight,
            latency_weight=artifact.latency_weight,
            unsafe_penalty=artifact.unsafe_penalty,
        )
        behavior_rewards.append(reward)
        try:
            decision = policy.decide(
                event.query,
                request_id=event.event_id,
                high_risk=event.high_risk,
            )
        except ValueError:
            support_violations += 1
            continue
        estimate_by_action = {item.action: item.expected_utility for item in decision.estimates}
        target_actions[context] = decision.action
        target_estimate = estimate_by_action[decision.action]
        logged_estimate = estimate_by_action.get(event.action, 0.0)
        target_probability = float(event.action == decision.action)
        importance = target_probability / event.propensity
        if target_probability:
            matched += 1
            if not event.safe:
                target_safety_violations += 1
        weights.append(importance)
        weighted_rewards.append(importance * reward)
        dr_values.append(target_estimate + importance * (reward - logged_estimate))
        supported_contexts.append((event.query, event.high_risk))
    if not dr_values:
        raise ValueError("offline evaluation found no supported events")
    support_violations += sum(
        target not in logged_actions.get(context, set())
        for context, target in target_actions.items()
    )
    count = len(dr_values)
    ips = sum(weighted_rewards) / count
    weight_sum = sum(weights)
    snips = sum(weighted_rewards) / weight_sum if weight_sum else 0.0
    dr = sum(dr_values) / count
    # Logged rows for the same context are correlated because the seed/evaluation
    # protocol records every available action. Use a cluster-robust standard error
    # over contexts instead of pretending those counterfactual rows are independent.
    clusters: dict[tuple[str, bool], list[float]] = {}
    for context, value in zip(supported_contexts, dr_values, strict=True):
        clusters.setdefault(context, []).append(value)
    cluster_values = [sum(values) / len(values) for values in clusters.values()]
    cluster_mean = sum(cluster_values) / len(cluster_values)
    variance = sum((value - cluster_mean) ** 2 for value in cluster_values) / max(
        1, len(cluster_values) - 1
    )
    margin = 1.96 * math.sqrt(variance / len(cluster_values))
    behavior = sum(behavior_rewards) / len(behavior_rewards)
    ess = weight_sum**2 / sum(value**2 for value in weights) if any(weights) else 0.0
    reasons = []
    if support_violations:
        reasons.append("target policy has unsupported contexts")
    if target_safety_violations:
        reasons.append("target policy matched unsafe logged outcomes")
    if ess < minimum_effective_sample_size:
        reasons.append("effective sample size is below the promotion floor")
    if dr - behavior < minimum_lift:
        reasons.append("estimated utility lift is below the promotion floor")
    if dr - margin < behavior - regression_tolerance:
        reasons.append("confidence interval fails the no-regression guardrail")
    dataset_fingerprint = hashlib.sha256(
        b"\n".join(_canonical(event.model_dump(mode="json")) for event in selected)
    ).hexdigest()
    return OfflinePolicyReport(
        dataset_fingerprint=dataset_fingerprint,
        policy_fingerprint=artifact.artifact_fingerprint,
        events=len(selected),
        matched_events=matched,
        support_violations=support_violations,
        effective_sample_size=ess,
        behavior_utility=behavior,
        ips_utility=ips,
        snips_utility=snips,
        doubly_robust_utility=dr,
        confidence_low=dr - margin,
        confidence_high=dr + margin,
        estimated_lift=dr - behavior,
        target_safety_violations=target_safety_violations,
        promoted=not reasons,
        reasons=reasons,
    )


def default_actions() -> list[BanditAction]:
    return [
        BanditAction(
            name="economy",
            model="gpt-4o-mini",
            max_cost_usd=0.01,
            max_latency_ms=2500,
            allow_high_risk=False,
        ),
        BanditAction(
            name="balanced",
            model="gpt-4.1-mini",
            max_cost_usd=0.03,
            max_latency_ms=5000,
            allow_high_risk=True,
        ),
        BanditAction(
            name="quality",
            model="gpt-4.1",
            max_cost_usd=0.12,
            max_latency_ms=12_000,
            allow_high_risk=True,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and gate a contextual model-routing policy.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--artifact", type=Path, default=Path("data/evaluations/bandit/policy.json"))
    parser.add_argument("--report", type=Path, default=Path("data/evaluations/bandit/report.json"))
    parser.add_argument("--check", type=Path)
    parser.add_argument("--require-promotion", action="store_true")
    args = parser.parse_args()
    events = load_events(args.dataset)
    artifact = train_policy(events, default_actions())
    if args.check:
        try:
            comparison = check_policy_reproducibility(artifact, args.check)
        except ValueError as exc:
            raise SystemExit(f"contextual-bandit artifact is not reproducible: {exc}") from exc
        print(json.dumps({"reproducibility": comparison}))
    artifact.save(args.artifact)
    report = evaluate_policy(artifact, events)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.model_dump(mode="json"), indent=2))
    if args.require_promotion and not report.promoted:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
