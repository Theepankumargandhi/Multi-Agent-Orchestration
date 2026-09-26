"""Privacy-safe online evaluation, drift detection, and canary governance."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, variance
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.model_gateway import InferenceReceipt


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _hash(value: object, key: bytes | None = None) -> str:
    payload = _canonical(value)
    if key:
        return hmac.new(key, payload, hashlib.sha256).hexdigest()
    return hashlib.sha256(payload).hexdigest()


class OnlineInferenceEvent(BaseModel):
    """Content-free production signal emitted after a controlled model call."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    event_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    request_id: str = Field(min_length=3, max_length=160)
    tenant_fingerprint: str = Field(min_length=16, max_length=128)
    occurred_at: float = Field(ge=0)
    variant: Literal["control", "canary"]
    selected_provider: str = Field(min_length=1, max_length=80)
    outcome: Literal["success", "blocked", "unavailable", "error"]
    quality_score: float | None = Field(default=None, ge=0, le=1)
    safety_violation: bool | None = None
    latency_ms: float = Field(ge=0)
    cost_usd: float = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cache_hit: bool = False
    fallback_used: bool = False
    high_risk: bool = False
    receipt_fingerprint: str = Field(min_length=32, max_length=128)
    event_fingerprint: str = ""


class DelayedFeedback(BaseModel):
    """Human or asynchronous grader signal joined without storing request content."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    feedback_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    event_id: str = Field(min_length=8, max_length=128)
    tenant_fingerprint: str = Field(min_length=16, max_length=128)
    quality_score: float = Field(ge=0, le=1)
    safety_violation: bool
    reviewer_fingerprint: str = Field(min_length=16, max_length=128)
    occurred_at: float = Field(ge=0)
    feedback_fingerprint: str = ""


class OnlineEvalPolicy(BaseModel):
    version: str = "online-slo-v1"
    short_window: int = Field(default=10, ge=2, le=10_000)
    long_window: int = Field(default=30, ge=4, le=100_000)
    retention_events: int = Field(default=10_000, ge=10, le=1_000_000)
    min_canary_samples: int = Field(default=8, ge=2, le=100_000)
    min_quality: float = Field(default=0.75, ge=0, lt=1)
    max_safety_rate: float = Field(default=0.05, gt=0, le=1)
    max_p95_latency_ms: float = Field(default=500, gt=0)
    max_average_cost_usd: float = Field(default=0.01, gt=0)
    quality_noninferiority_margin: float = Field(default=0.03, ge=0, le=0.5)
    max_latency_ratio: float = Field(default=1.25, ge=1)
    max_cost_ratio: float = Field(default=1.25, ge=1)
    short_burn_threshold: float = Field(default=2, gt=0)
    long_burn_threshold: float = Field(default=1, gt=0)
    provider_mix_js_threshold: float = Field(default=0.10, ge=0, le=1)

    @model_validator(mode="after")
    def validate_windows(self) -> "OnlineEvalPolicy":
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window")
        return self


class WindowMetrics(BaseModel):
    name: str
    samples: int
    scored_samples: int
    quality_mean: float | None
    quality_pass_rate: float | None
    safety_rate: float | None
    p95_latency_ms: float
    average_cost_usd: float
    success_rate: float
    cache_hit_rate: float
    fallback_rate: float
    quality_error_budget_burn: float | None
    safety_error_budget_burn: float | None


class OnlineAlert(BaseModel):
    code: str
    severity: Literal["warning", "critical"]
    variant: Literal["control", "canary", "all"]
    metric: str
    short_observed: float
    long_observed: float
    threshold: float
    evidence_fingerprint: str


class CanaryDecision(BaseModel):
    action: Literal["promote", "hold", "rollback"]
    reason: str
    control_samples: int
    canary_samples: int
    quality_delta: float | None = None
    quality_delta_ci_low: float | None = None
    quality_delta_ci_high: float | None = None
    latency_ratio: float | None = None
    cost_ratio: float | None = None
    decision_fingerprint: str = ""


class OnlineEvaluationReport(BaseModel):
    schema_version: str = "1.0"
    generated_by: str = "agentforge-online-eval-controller"
    generated_at: str
    policy_version: str
    dataset_fingerprint: str
    accepted_events: int
    duplicate_events: int
    joined_feedback: int
    event_integrity_rate: float
    privacy_violations: int
    provider_mix_js_divergence: float
    drift_detected: bool
    windows: list[WindowMetrics]
    alerts: list[OnlineAlert]
    canary_decision: CanaryDecision
    report_fingerprint: str = ""


def _seal_model(model: BaseModel, field: str, key: bytes | None = None) -> None:
    payload = model.model_dump(mode="json", exclude={field})
    setattr(model, field, _hash(payload, key))


def _verify_model(model: BaseModel, field: str, key: bytes | None = None) -> bool:
    expected = _hash(model.model_dump(mode="json", exclude={field}), key)
    return hmac.compare_digest(str(getattr(model, field)), expected)


def _percentile95(values: list[float]) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _window_metrics(name: str, events: list[OnlineInferenceEvent], policy: OnlineEvalPolicy) -> WindowMetrics:
    if not events:
        return WindowMetrics(
            name=name,
            samples=0,
            scored_samples=0,
            quality_mean=None,
            quality_pass_rate=None,
            safety_rate=None,
            p95_latency_ms=0,
            average_cost_usd=0,
            success_rate=0,
            cache_hit_rate=0,
            fallback_rate=0,
            quality_error_budget_burn=None,
            safety_error_budget_burn=None,
        )
    quality = [item.quality_score for item in events if item.quality_score is not None]
    safety = [item.safety_violation for item in events if item.safety_violation is not None]
    quality_pass = (
        sum(score >= policy.min_quality for score in quality) / len(quality) if quality else None
    )
    safety_rate = sum(safety) / len(safety) if safety else None
    return WindowMetrics(
        name=name,
        samples=len(events),
        scored_samples=len(quality),
        quality_mean=mean(quality) if quality else None,
        quality_pass_rate=quality_pass,
        safety_rate=safety_rate,
        p95_latency_ms=_percentile95([item.latency_ms for item in events]),
        average_cost_usd=mean(item.cost_usd for item in events),
        success_rate=sum(item.outcome == "success" for item in events) / len(events),
        cache_hit_rate=sum(item.cache_hit for item in events) / len(events),
        fallback_rate=sum(item.fallback_used for item in events) / len(events),
        quality_error_budget_burn=(1 - quality_pass) / (1 - policy.min_quality)
        if quality_pass is not None
        else None,
        safety_error_budget_burn=safety_rate / policy.max_safety_rate
        if safety_rate is not None
        else None,
    )


def _js_divergence(left: list[OnlineInferenceEvent], right: list[OnlineInferenceEvent]) -> float:
    if not left or not right:
        return 0
    keys = set(item.selected_provider for item in left + right)
    lp = Counter(item.selected_provider for item in left)
    rp = Counter(item.selected_provider for item in right)
    result = 0.0
    for key in keys:
        p = lp[key] / len(left)
        q = rp[key] / len(right)
        midpoint = (p + q) / 2
        if p:
            result += 0.5 * p * math.log2(p / midpoint)
        if q:
            result += 0.5 * q * math.log2(q / midpoint)
    return result


def _mean_delta_ci(control: list[float], canary: list[float]) -> tuple[float, float, float] | None:
    if len(control) < 2 or len(canary) < 2:
        return None
    delta = mean(canary) - mean(control)
    standard_error = math.sqrt(variance(control) / len(control) + variance(canary) / len(canary))
    return delta, delta - 1.96 * standard_error, delta + 1.96 * standard_error


class OnlineEvaluationController:
    """Bounded exactly-once signal store with deterministic online release decisions."""

    def __init__(self, policy: OnlineEvalPolicy | None = None, *, integrity_key: bytes | None = None):
        self.policy = policy or OnlineEvalPolicy()
        self.integrity_key = integrity_key
        self._events: dict[str, OnlineInferenceEvent] = {}
        self._feedback: dict[str, DelayedFeedback] = {}
        self.duplicate_events = 0
        self.joined_feedback = 0

    @property
    def events(self) -> list[OnlineInferenceEvent]:
        return sorted(self._events.values(), key=lambda item: (item.occurred_at, item.event_id))

    def ingest(self, event: OnlineInferenceEvent) -> bool:
        candidate = event.model_copy(deep=True)
        if not candidate.event_fingerprint:
            _seal_model(candidate, "event_fingerprint", self.integrity_key)
        elif not _verify_model(candidate, "event_fingerprint", self.integrity_key):
            raise ValueError("event fingerprint verification failed")
        existing = self._events.get(candidate.event_id)
        if existing:
            if hmac.compare_digest(existing.event_fingerprint, candidate.event_fingerprint):
                self.duplicate_events += 1
                return False
            raise ValueError("conflicting duplicate event_id")
        self._events[candidate.event_id] = candidate
        overflow = len(self._events) - self.policy.retention_events
        if overflow > 0:
            for old in self.events[:overflow]:
                del self._events[old.event_id]
        return True

    def join_feedback(self, feedback: DelayedFeedback) -> bool:
        candidate = feedback.model_copy(deep=True)
        if not candidate.feedback_fingerprint:
            _seal_model(candidate, "feedback_fingerprint", self.integrity_key)
        elif not _verify_model(candidate, "feedback_fingerprint", self.integrity_key):
            raise ValueError("feedback fingerprint verification failed")
        if candidate.feedback_id in self._feedback:
            if hmac.compare_digest(
                self._feedback[candidate.feedback_id].feedback_fingerprint,
                candidate.feedback_fingerprint,
            ):
                return False
            raise ValueError("conflicting duplicate feedback_id")
        event = self._events.get(candidate.event_id)
        if event is None:
            raise ValueError("feedback references an unknown event")
        if not hmac.compare_digest(event.tenant_fingerprint, candidate.tenant_fingerprint):
            raise ValueError("feedback tenant does not match event tenant")
        updated = event.model_copy(
            update={
                "quality_score": candidate.quality_score,
                "safety_violation": candidate.safety_violation,
                "event_fingerprint": "",
            }
        )
        _seal_model(updated, "event_fingerprint", self.integrity_key)
        self._events[event.event_id] = updated
        self._feedback[candidate.feedback_id] = candidate
        self.joined_feedback += 1
        return True

    def verify_event(self, event: OnlineInferenceEvent) -> bool:
        return _verify_model(event, "event_fingerprint", self.integrity_key)

    def evaluate(
        self, *, dataset_fingerprint: str = "", privacy_violations: int = 0
    ) -> OnlineEvaluationReport:
        ordered = self.events
        control = [item for item in ordered if item.variant == "control"]
        canary = [item for item in ordered if item.variant == "canary"]
        windows: list[WindowMetrics] = []
        indexed: dict[str, WindowMetrics] = {}
        for variant, items in (("control", control), ("canary", canary), ("all", ordered)):
            for label, size in (("short", self.policy.short_window), ("long", self.policy.long_window)):
                name = f"{variant}_{label}"
                metric = _window_metrics(name, items[-size:], self.policy)
                windows.append(metric)
                indexed[name] = metric

        alerts: list[OnlineAlert] = []
        for variant in ("control", "canary", "all"):
            short = indexed[f"{variant}_short"]
            long = indexed[f"{variant}_long"]
            for metric_name, short_burn, long_burn in (
                ("quality", short.quality_error_budget_burn, long.quality_error_budget_burn),
                ("safety", short.safety_error_budget_burn, long.safety_error_budget_burn),
            ):
                if short_burn is None or long_burn is None:
                    continue
                if (
                    short_burn >= self.policy.short_burn_threshold
                    and long_burn >= self.policy.long_burn_threshold
                ):
                    evidence = {
                        "variant": variant,
                        "metric": metric_name,
                        "short": short_burn,
                        "long": long_burn,
                        "event_ids": [item.event_id for item in ordered[-self.policy.long_window :]],
                    }
                    alerts.append(
                        OnlineAlert(
                            code=f"{variant}_{metric_name}_budget_burn",
                            severity="critical",
                            variant=variant,
                            metric=metric_name,
                            short_observed=short_burn,
                            long_observed=long_burn,
                            threshold=self.policy.long_burn_threshold,
                            evidence_fingerprint=_hash(evidence, self.integrity_key),
                        )
                    )
            for metric_name, short_observed, long_observed, threshold in (
                (
                    "latency",
                    short.p95_latency_ms,
                    long.p95_latency_ms,
                    self.policy.max_p95_latency_ms,
                ),
                (
                    "cost",
                    short.average_cost_usd,
                    long.average_cost_usd,
                    self.policy.max_average_cost_usd,
                ),
            ):
                if short.samples and short_observed > threshold and long_observed > threshold:
                    evidence = {
                        "variant": variant,
                        "metric": metric_name,
                        "short": short_observed,
                        "long": long_observed,
                        "threshold": threshold,
                    }
                    alerts.append(
                        OnlineAlert(
                            code=f"{variant}_{metric_name}_slo",
                            severity="critical",
                            variant=variant,
                            metric=metric_name,
                            short_observed=short_observed,
                            long_observed=long_observed,
                            threshold=threshold,
                            evidence_fingerprint=_hash(evidence, self.integrity_key),
                        )
                    )

        control_long = indexed["control_long"]
        canary_long = indexed["canary_long"]
        quality_ci = _mean_delta_ci(
            [item.quality_score for item in control[-self.policy.long_window :] if item.quality_score is not None],
            [item.quality_score for item in canary[-self.policy.long_window :] if item.quality_score is not None],
        )
        latency_ratio = (
            canary_long.p95_latency_ms / control_long.p95_latency_ms
            if control_long.p95_latency_ms and canary_long.samples
            else None
        )
        cost_ratio = (
            canary_long.average_cost_usd / control_long.average_cost_usd
            if control_long.average_cost_usd and canary_long.samples
            else None
        )
        critical_canary = any(item.variant == "canary" and item.severity == "critical" for item in alerts)
        if len(canary) < self.policy.min_canary_samples or len(control) < 2:
            action, reason = "hold", "insufficient statistically useful canary evidence"
        elif critical_canary:
            action, reason = "rollback", "multi-window canary SLO or error-budget threshold exceeded"
        elif canary_long.quality_mean is not None and canary_long.quality_mean < self.policy.min_quality:
            action, reason = "rollback", "canary absolute quality SLO was missed"
        elif (
            canary_long.safety_rate is not None
            and canary_long.safety_rate > self.policy.max_safety_rate
        ):
            action, reason = "rollback", "canary absolute safety SLO was missed"
        elif canary_long.p95_latency_ms > self.policy.max_p95_latency_ms:
            action, reason = "rollback", "canary absolute latency SLO was missed"
        elif canary_long.average_cost_usd > self.policy.max_average_cost_usd:
            action, reason = "rollback", "canary absolute cost SLO was missed"
        elif quality_ci and quality_ci[2] < -self.policy.quality_noninferiority_margin:
            action, reason = "rollback", "canary quality is statistically below the non-inferiority margin"
        elif latency_ratio is not None and latency_ratio > self.policy.max_latency_ratio:
            action, reason = "rollback", "canary p95 latency ratio exceeded"
        elif cost_ratio is not None and cost_ratio > self.policy.max_cost_ratio:
            action, reason = "rollback", "canary average cost ratio exceeded"
        elif quality_ci and quality_ci[1] >= -self.policy.quality_noninferiority_margin:
            action, reason = "promote", "canary met quality, safety, latency, and cost release criteria"
        else:
            action, reason = "hold", "confidence interval still crosses the release boundary"

        decision = CanaryDecision(
            action=action,
            reason=reason,
            control_samples=len(control),
            canary_samples=len(canary),
            quality_delta=quality_ci[0] if quality_ci else None,
            quality_delta_ci_low=quality_ci[1] if quality_ci else None,
            quality_delta_ci_high=quality_ci[2] if quality_ci else None,
            latency_ratio=latency_ratio,
            cost_ratio=cost_ratio,
        )
        _seal_model(decision, "decision_fingerprint", self.integrity_key)
        js_divergence = _js_divergence(control, canary)
        integrity_rate = (
            sum(self.verify_event(item) for item in ordered) / len(ordered) if ordered else 0
        )
        report = OnlineEvaluationReport(
            generated_at=datetime.now(UTC).isoformat(),
            policy_version=self.policy.version,
            dataset_fingerprint=dataset_fingerprint or _hash(
                [item.model_dump(mode="json") for item in ordered], self.integrity_key
            ),
            accepted_events=len(ordered),
            duplicate_events=self.duplicate_events,
            joined_feedback=self.joined_feedback,
            event_integrity_rate=integrity_rate,
            privacy_violations=privacy_violations,
            provider_mix_js_divergence=js_divergence,
            drift_detected=js_divergence >= self.policy.provider_mix_js_threshold
            or action == "rollback",
            windows=windows,
            alerts=alerts,
            canary_decision=decision,
        )
        _seal_model(report, "report_fingerprint", self.integrity_key)
        return report

    def verify_report(self, report: OnlineEvaluationReport) -> bool:
        return _verify_model(report, "report_fingerprint", self.integrity_key) and _verify_model(
            report.canary_decision, "decision_fingerprint", self.integrity_key
        )


def event_from_gateway_receipt(
    receipt: InferenceReceipt, *, integrity_key: bytes | None = None
) -> OnlineInferenceEvent:
    """Convert a gateway receipt into an operational signal without model content."""
    failed_attempts = {"error", "timeout", "circuit_open", "budget_rejected"}
    event = OnlineInferenceEvent(
        event_id=f"gateway-{_hash([receipt.request_id, receipt.receipt_fingerprint])[:32]}",
        request_id=f"request-{_hash(receipt.request_id, integrity_key)[:32]}",
        tenant_fingerprint=receipt.tenant_fingerprint,
        occurred_at=datetime.fromisoformat(receipt.generated_at).timestamp(),
        variant="canary" if receipt.canary_selected else "control",
        selected_provider=receipt.selected_provider or "none",
        outcome=receipt.outcome,
        latency_ms=sum(item.latency_ms for item in receipt.attempts),
        cost_usd=receipt.cost_usd + (receipt.shadow.cost_usd if receipt.shadow else 0),
        prompt_tokens=receipt.prompt_tokens,
        completion_tokens=receipt.completion_tokens,
        cache_hit=receipt.cache_hit,
        fallback_used=any(item.outcome in failed_attempts for item in receipt.attempts),
        high_risk=receipt.high_risk,
        receipt_fingerprint=receipt.receipt_fingerprint,
    )
    _seal_model(event, "event_fingerprint", integrity_key)
    return event


def append_online_event(event: OnlineInferenceEvent, path: Path, *, max_bytes: int = 10_000_000) -> None:
    """Append a bounded JSONL operational signal; never serializes prompt or response content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= max_bytes:
        rotated = path.with_suffix(path.suffix + ".1")
        path.replace(rotated)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(event.model_dump_json() + "\n")
