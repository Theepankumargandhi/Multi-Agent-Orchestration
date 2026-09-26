"""Deterministic fault injection, recovery evaluation, and SLO release gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Fault = Literal[
    "none",
    "rate_limit",
    "timeout",
    "malformed_output",
    "unavailable",
    "empty_result",
    "worker_crash",
    "context_overflow",
    "policy_denial",
    "stream_disconnect",
    "authentication_error",
]
Strategy = Literal[
    "none",
    "retry",
    "fallback_model",
    "repair_output",
    "retrieval_fallback",
    "graceful_degradation",
    "checkpoint_resume",
    "lease_recovery",
    "compress_context",
    "stream_resume",
    "fail_closed",
]
Status = Literal["healthy", "recovered", "contained", "failed"]


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


class FaultStep(BaseModel):
    result: Fault
    latency_ms: float = Field(ge=0, le=120_000)
    tokens: int = Field(default=0, ge=0, le=1_000_000)
    cost_usd: float = Field(default=0.0, ge=0, le=1000)


class ReliabilityScenario(BaseModel):
    id: str = Field(min_length=2, max_length=120, pattern=r"^[a-z0-9-]+$")
    title: str = Field(min_length=5, max_length=240)
    component: Literal[
        "model", "retrieval", "tool", "persistence", "worker", "context", "stream", "policy"
    ]
    fault: Fault
    strategy: Strategy
    expected_status: Literal["healthy", "recovered", "contained"]
    primary: list[FaultStep] = Field(min_length=1, max_length=6)
    fallback: list[FaultStep] = Field(default_factory=list, max_length=4)
    max_attempts: int = Field(default=2, ge=1, le=6)
    max_recovery_ms: float = Field(ge=0, le=300_000)
    max_cost_usd: float = Field(default=1000, ge=0, le=1000)
    tags: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_plan(self) -> "ReliabilityScenario":
        if self.primary[0].result != self.fault:
            raise ValueError("first primary step must inject the declared fault")
        if self.expected_status == "contained" and self.strategy != "fail_closed":
            raise ValueError("contained scenarios must use fail_closed")
        if self.strategy == "fail_closed" and self.expected_status != "contained":
            raise ValueError("fail_closed scenarios must expect containment")
        if self.strategy in {
            "fallback_model",
            "retrieval_fallback",
            "graceful_degradation",
            "checkpoint_resume",
            "lease_recovery",
            "stream_resume",
        } and not self.fallback:
            raise ValueError("fallback recovery strategies require fallback steps")
        return self


class ReliabilityEvent(BaseModel):
    sequence: int = Field(ge=1)
    phase: Literal["primary", "control", "fallback"]
    action: str
    result: str
    latency_ms: float = Field(ge=0)
    tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)


class ReliabilityRun(BaseModel):
    status: Status
    attempts: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
    tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    events: list[ReliabilityEvent]
    trace_fingerprint: str


class ReliabilityOutcome(BaseModel):
    scenario_id: str
    title: str
    component: str
    fault: Fault
    strategy: Strategy
    expected_status: str
    baseline: ReliabilityRun
    resilient: ReliabilityRun
    recovered: bool
    safely_contained: bool
    slo_compliant: bool
    passed: bool
    recovery_latency_delta_ms: float
    additional_tokens: int
    additional_cost_usd: float


class ReliabilityMetrics(BaseModel):
    availability_rate: float
    safe_containment_rate: float
    scenario_pass_rate: float
    slo_compliance_rate: float
    mean_attempts: float
    p50_duration_ms: float
    p95_duration_ms: float
    total_tokens: int
    total_cost_usd: float


class ReliabilityReport(BaseModel):
    schema_version: str = "1.0"
    generated_at: str
    generated_by: str = "agentforge-reliability-lab"
    dataset_path: str
    dataset_fingerprint: str
    policy_fingerprint: str
    report_fingerprint: str = ""
    scenarios: int
    faults: int
    benign_controls: int
    passed: int
    baseline: ReliabilityMetrics
    resilient: ReliabilityMetrics
    recovery_success_rate: float
    improvement_percentage_points: float
    outcomes: list[ReliabilityOutcome]


class InjectedFault(RuntimeError):
    """Typed dependency failure carrying only bounded operational metadata."""

    def __init__(self, step: FaultStep):
        super().__init__(step.result)
        self.step = step


class ScriptedDependency:
    """Dependency double that executes a declared outcome sequence without sleeping."""

    def __init__(self, steps: list[FaultStep]):
        self._steps = deque(steps)

    def invoke(self) -> FaultStep:
        if not self._steps:
            raise InjectedFault(FaultStep(result="unavailable", latency_ms=0))
        step = self._steps.popleft()
        if step.result != "none":
            raise InjectedFault(step)
        return step


_PRIMARY_RECOVERY = {"retry", "repair_output", "compress_context"}
_FALLBACK_RECOVERY = {
    "fallback_model",
    "retrieval_fallback",
    "graceful_degradation",
    "checkpoint_resume",
    "lease_recovery",
    "stream_resume",
}
_CONTROL_ACTION = {
    "retry": "bounded_backoff_retry",
    "fallback_model": "switch_model",
    "repair_output": "repair_structured_output",
    "retrieval_fallback": "switch_retriever",
    "graceful_degradation": "degrade_without_tool",
    "checkpoint_resume": "resume_checkpoint",
    "lease_recovery": "recover_expired_lease",
    "compress_context": "compress_context",
    "stream_resume": "resume_stream",
    "fail_closed": "fail_closed",
}


def _event(
    events: list[ReliabilityEvent],
    *,
    phase: Literal["primary", "control", "fallback"],
    action: str,
    step: FaultStep,
) -> None:
    events.append(
        ReliabilityEvent(
            sequence=len(events) + 1,
            phase=phase,
            action=action,
            result=step.result,
            latency_ms=step.latency_ms,
            tokens=step.tokens,
            cost_usd=step.cost_usd,
        )
    )


def _finalize(status: Status, events: list[ReliabilityEvent]) -> ReliabilityRun:
    attempts = sum(item.phase in {"primary", "fallback"} for item in events)
    canonical = [item.model_dump(mode="json") for item in events]
    return ReliabilityRun(
        status=status,
        attempts=attempts,
        duration_ms=sum(item.latency_ms for item in events),
        tokens=sum(item.tokens for item in events),
        cost_usd=sum(item.cost_usd for item in events),
        events=events,
        trace_fingerprint=_canonical_hash(canonical),
    )


def execute_scenario(scenario: ReliabilityScenario, *, resilient: bool) -> ReliabilityRun:
    """Execute one fault plan through the baseline or recovery state machine."""
    primary = ScriptedDependency(scenario.primary)
    events: list[ReliabilityEvent] = []
    attempts = scenario.max_attempts if resilient else 1
    for _ in range(attempts):
        try:
            step = primary.invoke()
            _event(events, phase="primary", action="invoke", step=step)
            status: Status = "healthy" if len(events) == 1 else "recovered"
            return _finalize(status, events)
        except InjectedFault as exc:
            _event(events, phase="primary", action="invoke", step=exc.step)
            if not resilient:
                return _finalize("failed", events)
            if scenario.strategy == "fail_closed":
                control = FaultStep(result="none", latency_ms=0.1)
                _event(events, phase="control", action="fail_closed", step=control)
                return _finalize("contained", events)
            if scenario.strategy in _PRIMARY_RECOVERY:
                control = FaultStep(result="none", latency_ms=0.1)
                _event(
                    events,
                    phase="control",
                    action=_CONTROL_ACTION[scenario.strategy],
                    step=control,
                )
                continue
            break

    if resilient and scenario.strategy in _FALLBACK_RECOVERY:
        control = FaultStep(result="none", latency_ms=0.1)
        _event(
            events,
            phase="control",
            action=_CONTROL_ACTION[scenario.strategy],
            step=control,
        )
        fallback = ScriptedDependency(scenario.fallback)
        for _ in range(min(scenario.max_attempts, len(scenario.fallback))):
            try:
                step = fallback.invoke()
                _event(events, phase="fallback", action="invoke_fallback", step=step)
                return _finalize("recovered", events)
            except InjectedFault as exc:
                _event(events, phase="fallback", action="invoke_fallback", step=exc.step)
    return _finalize("failed", events)


def load_reliability_scenarios(path: Path) -> list[ReliabilityScenario]:
    scenarios: list[ReliabilityScenario] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            scenario = ReliabilityScenario.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"invalid reliability scenario at line {line_number}: {exc}") from exc
        if scenario.id in seen:
            raise ValueError(f"duplicate reliability scenario id {scenario.id!r}")
        seen.add(scenario.id)
        scenarios.append(scenario)
    if not scenarios:
        raise ValueError("reliability dataset is empty")
    return scenarios


def dataset_fingerprint(scenarios: list[ReliabilityScenario]) -> str:
    return _canonical_hash([item.model_dump(mode="json") for item in scenarios])


def _metrics(runs: list[ReliabilityRun], passed: list[bool]) -> ReliabilityMetrics:
    availability = sum(item.status in {"healthy", "recovered"} for item in runs)
    containment_candidates = [item for item in runs if item.status in {"contained", "failed"}]
    return ReliabilityMetrics(
        availability_rate=availability / len(runs),
        safe_containment_rate=(
            sum(item.status == "contained" for item in containment_candidates)
            / len(containment_candidates)
            if containment_candidates
            else 1.0
        ),
        scenario_pass_rate=sum(passed) / len(passed),
        slo_compliance_rate=sum(passed) / len(passed),
        mean_attempts=statistics.fmean(item.attempts for item in runs),
        p50_duration_ms=statistics.median(item.duration_ms for item in runs),
        p95_duration_ms=_percentile([item.duration_ms for item in runs], 0.95),
        total_tokens=sum(item.tokens for item in runs),
        total_cost_usd=sum(item.cost_usd for item in runs),
    )


def _report_fingerprint(report: ReliabilityReport) -> str:
    return _canonical_hash(
        report.model_dump(mode="json", exclude={"generated_at", "report_fingerprint"})
    )


def verify_report(report: ReliabilityReport) -> bool:
    traces_valid = all(
        outcome.baseline.trace_fingerprint
        == _canonical_hash([item.model_dump(mode="json") for item in outcome.baseline.events])
        and outcome.resilient.trace_fingerprint
        == _canonical_hash([item.model_dump(mode="json") for item in outcome.resilient.events])
        for outcome in report.outcomes
    )
    return traces_valid and report.report_fingerprint == _report_fingerprint(report)


def evaluate_reliability(
    scenarios: list[ReliabilityScenario], *, dataset_path: str = ""
) -> ReliabilityReport:
    outcomes: list[ReliabilityOutcome] = []
    policy_payload = []
    for scenario in scenarios:
        baseline = execute_scenario(scenario, resilient=False)
        resilient_run = execute_scenario(scenario, resilient=True)
        recovered = resilient_run.status == "recovered"
        safely_contained = resilient_run.status == "contained"
        expected = resilient_run.status == scenario.expected_status
        slo = (
            resilient_run.duration_ms <= scenario.max_recovery_ms
            and resilient_run.cost_usd <= scenario.max_cost_usd
            and resilient_run.attempts <= scenario.max_attempts
        )
        outcomes.append(
            ReliabilityOutcome(
                scenario_id=scenario.id,
                title=scenario.title,
                component=scenario.component,
                fault=scenario.fault,
                strategy=scenario.strategy,
                expected_status=scenario.expected_status,
                baseline=baseline,
                resilient=resilient_run,
                recovered=recovered,
                safely_contained=safely_contained,
                slo_compliant=slo,
                passed=expected and slo,
                recovery_latency_delta_ms=resilient_run.duration_ms - baseline.duration_ms,
                additional_tokens=resilient_run.tokens - baseline.tokens,
                additional_cost_usd=resilient_run.cost_usd - baseline.cost_usd,
            )
        )
        policy_payload.append(
            {
                "id": scenario.id,
                "strategy": scenario.strategy,
                "max_attempts": scenario.max_attempts,
                "max_recovery_ms": scenario.max_recovery_ms,
                "max_cost_usd": scenario.max_cost_usd,
            }
        )
    resilient_passed = [item.passed for item in outcomes]
    baseline_passed = [
        item.baseline.status == item.expected_status
        and item.baseline.duration_ms
        <= next(s.max_recovery_ms for s in scenarios if s.id == item.scenario_id)
        for item in outcomes
    ]
    baseline_metrics = _metrics([item.baseline for item in outcomes], baseline_passed)
    resilient_metrics = _metrics([item.resilient for item in outcomes], resilient_passed)
    recovery_cases = [item for item in outcomes if item.expected_status == "recovered"]
    report = ReliabilityReport(
        generated_at=datetime.now(UTC).isoformat(),
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint(scenarios),
        policy_fingerprint=_canonical_hash(policy_payload),
        scenarios=len(scenarios),
        faults=sum(item.fault != "none" for item in scenarios),
        benign_controls=sum(item.fault == "none" for item in scenarios),
        passed=sum(resilient_passed),
        baseline=baseline_metrics,
        resilient=resilient_metrics,
        recovery_success_rate=(
            sum(item.recovered for item in recovery_cases) / len(recovery_cases)
            if recovery_cases
            else 1.0
        ),
        improvement_percentage_points=(
            resilient_metrics.scenario_pass_rate - baseline_metrics.scenario_pass_rate
        )
        * 100,
        outcomes=outcomes,
    )
    report.report_fingerprint = _report_fingerprint(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic agent fault injection and recovery SLO evaluation"
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("evals/datasets/agent_reliability_scenarios.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-recovery-rate", type=float, default=1.0)
    parser.add_argument("--max-p95-ms", type=float, default=250.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    scenarios = load_reliability_scenarios(args.dataset)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "scenarios": len(scenarios),
                    "dataset_fingerprint": dataset_fingerprint(scenarios),
                },
                sort_keys=True,
            )
        )
        return
    report = evaluate_reliability(scenarios, dataset_path=args.dataset.as_posix())
    payload = report.model_dump_json(indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    failed = (
        not verify_report(report)
        or report.resilient.scenario_pass_rate
        < max(0.0, min(args.min_pass_rate, 1.0))
        or report.recovery_success_rate < max(0.0, min(args.min_recovery_rate, 1.0))
        or report.resilient.p95_duration_ms > max(0.0, args.max_p95_ms)
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
