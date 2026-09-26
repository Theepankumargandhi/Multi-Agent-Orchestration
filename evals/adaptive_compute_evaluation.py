"""Offline ablation for budgeted adaptive test-time compute."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.adaptive_compute import (
    ComputePolicy,
    ComputeSignals,
    candidate_assessment,
    plan_compute,
    select_candidate,
    verify_plan,
    verify_receipt,
)

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "adaptive_compute_scenarios.jsonl"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class SimulatedCandidate(BaseModel):
    id: str
    correct: bool
    confidence: float = Field(ge=0, le=1)
    grounded: bool
    conformal_decision: Literal["release", "abstain", "not_evaluated"]
    claim_keys: list[str]
    tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)


class AdaptiveComputeScenario(BaseModel):
    id: str
    route: str
    initial_confidence: float = Field(ge=0, le=1)
    initial_correct: bool
    grounding_action: Literal["pass", "repair", "abstain", "not_required"]
    uncertainty_decision: Literal["release", "abstain", "not_evaluated", "unavailable"]
    out_of_distribution: bool
    high_risk: bool
    evidence_count: int = Field(ge=0)
    expected_status: Literal["released", "abstained", "budget_exhausted"]
    recoverable: bool
    candidates: list[SimulatedCandidate]


class AdaptiveComputeOutcome(BaseModel):
    scenario_id: str
    route: str
    plan_action: str
    final_status: str
    baseline_released: bool
    baseline_correct: bool
    final_correct: bool
    recovered: bool
    selected_candidate_id: str
    attempted_candidates: int
    extra_tokens: int
    latency_ms: float
    budget_violation: bool
    unsafe_release: bool
    integrity_verified: bool
    passed: bool
    outcome_fingerprint: str = ""


class AdaptiveComputeReport(BaseModel):
    schema_version: str = "1.0"
    generated_by: str = "agentforge-adaptive-compute-eval"
    dataset_path: str
    dataset_fingerprint: str
    scenario_count: int
    pass_rate: float
    baseline_selective_accuracy: float
    adaptive_selective_accuracy: float
    answer_coverage: float
    recovery_rate: float
    unsafe_release_rate: float
    budget_violation_rate: float
    early_exit_rate: float
    candidate_call_reduction: float
    mean_extra_tokens: float
    mean_extra_latency_ms: float
    receipt_integrity_rate: float
    outcomes: list[AdaptiveComputeOutcome]
    report_fingerprint: str = ""


def load_scenarios(path: Path = DEFAULT_DATASET) -> list[AdaptiveComputeScenario]:
    scenarios = [
        AdaptiveComputeScenario.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [item.id for item in scenarios]
    if not scenarios:
        raise ValueError("adaptive compute dataset is empty")
    if len(ids) != len(set(ids)):
        raise ValueError("adaptive compute scenario ids must be unique")
    return scenarios


def evaluate_adaptive_compute(
    scenarios: list[AdaptiveComputeScenario],
    dataset_path: str = "",
    *,
    policy: ComputePolicy | None = None,
    integrity_key: bytes | None = None,
) -> AdaptiveComputeReport:
    policy = policy or ComputePolicy()
    outcomes: list[AdaptiveComputeOutcome] = []
    for scenario in scenarios:
        plan = plan_compute(
            ComputeSignals(
                route=scenario.route,
                grounding_action=scenario.grounding_action,
                grounding_confidence=scenario.initial_confidence,
                uncertainty_decision=scenario.uncertainty_decision,
                out_of_distribution=scenario.out_of_distribution,
                high_risk=scenario.high_risk,
                evidence_count=scenario.evidence_count,
            ),
            policy,
            integrity_key,
        )
        baseline_released = scenario.grounding_action in {"pass", "not_required"}
        baseline_correct = baseline_released and scenario.initial_correct
        selected_id = ""
        attempted = 0
        extra_tokens = 0
        latency_ms = 0.0
        integrity_verified = verify_plan(plan, integrity_key)
        if plan.action == "early_exit":
            final_status = "released"
            final_correct = scenario.initial_correct
        elif plan.action == "abstain":
            final_status = "abstained"
            final_correct = False
        else:
            candidates = [
                candidate_assessment(
                    candidate_id=item.id,
                    answer=f"simulated::{scenario.id}::{item.id}",
                    confidence=item.confidence,
                    grounded=item.grounded,
                    conformal_decision=item.conformal_decision,
                    claim_keys=item.claim_keys,
                    token_count=item.tokens,
                    latency_ms=item.latency_ms,
                )
                for item in scenario.candidates
            ]
            receipt = select_candidate(plan, candidates, policy, integrity_key)
            final_status = receipt.status
            selected_id = receipt.selected_candidate_id
            attempted = receipt.attempted_candidates
            extra_tokens = receipt.extra_tokens
            latency_ms = receipt.latency_ms
            integrity_verified = integrity_verified and verify_receipt(
                receipt, integrity_key
            )
            correctness = {item.id: item.correct for item in scenario.candidates}
            final_correct = bool(selected_id and correctness.get(selected_id))
        budget_violation = (
            extra_tokens > plan.token_budget or latency_ms > plan.latency_budget_ms
        )
        unsafe_release = final_status == "released" and not final_correct
        outcome = AdaptiveComputeOutcome(
            scenario_id=scenario.id,
            route=scenario.route,
            plan_action=plan.action,
            final_status=final_status,
            baseline_released=baseline_released,
            baseline_correct=baseline_correct,
            final_correct=final_correct,
            recovered=scenario.recoverable and final_status == "released" and final_correct,
            selected_candidate_id=selected_id,
            attempted_candidates=attempted,
            extra_tokens=extra_tokens,
            latency_ms=latency_ms,
            budget_violation=budget_violation,
            unsafe_release=unsafe_release,
            integrity_verified=integrity_verified,
            passed=(
                final_status == scenario.expected_status
                and not unsafe_release
                and not budget_violation
                and integrity_verified
            ),
        )
        outcome.outcome_fingerprint = _hash(
            outcome.model_dump(mode="json", exclude={"outcome_fingerprint"})
        )
        outcomes.append(outcome)

    released_baseline = [item for item in outcomes if item.baseline_released]
    released_adaptive = [item for item in outcomes if item.final_status == "released"]
    recoverable_count = sum(item.recoverable for item in scenarios)
    always_compute_calls = len(scenarios) * policy.max_candidates
    attempted_calls = sum(item.attempted_candidates for item in outcomes)
    report = AdaptiveComputeReport(
        dataset_path=dataset_path,
        dataset_fingerprint=_hash(
            [item.model_dump(mode="json") for item in scenarios]
        ),
        scenario_count=len(scenarios),
        pass_rate=sum(item.passed for item in outcomes) / len(outcomes),
        baseline_selective_accuracy=sum(item.baseline_correct for item in released_baseline)
        / max(1, len(released_baseline)),
        adaptive_selective_accuracy=sum(item.final_correct for item in released_adaptive)
        / max(1, len(released_adaptive)),
        answer_coverage=len(released_adaptive) / len(outcomes),
        recovery_rate=sum(item.recovered for item in outcomes) / max(1, recoverable_count),
        unsafe_release_rate=sum(item.unsafe_release for item in outcomes) / len(outcomes),
        budget_violation_rate=sum(item.budget_violation for item in outcomes) / len(outcomes),
        early_exit_rate=sum(item.plan_action == "early_exit" for item in outcomes)
        / len(outcomes),
        candidate_call_reduction=1 - attempted_calls / max(1, always_compute_calls),
        mean_extra_tokens=sum(item.extra_tokens for item in outcomes) / len(outcomes),
        mean_extra_latency_ms=sum(item.latency_ms for item in outcomes) / len(outcomes),
        receipt_integrity_rate=sum(item.integrity_verified for item in outcomes)
        / len(outcomes),
        outcomes=outcomes,
    )
    report.report_fingerprint = _hash(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    )
    return report


def verify_report(report: AdaptiveComputeReport) -> bool:
    if report.report_fingerprint != _hash(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    ):
        return False
    return all(
        item.outcome_fingerprint
        == _hash(item.model_dump(mode="json", exclude={"outcome_fingerprint"}))
        for item in report.outcomes
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adaptive test-time compute")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-recovery-rate", type=float, default=1.0)
    parser.add_argument("--max-unsafe-release-rate", type=float, default=0.0)
    parser.add_argument("--max-budget-violation-rate", type=float, default=0.0)
    args = parser.parse_args()
    integrity_key = os.getenv("ADAPTIVE_COMPUTE_INTEGRITY_KEY", "").encode() or None
    report = evaluate_adaptive_compute(
        load_scenarios(args.dataset),
        args.dataset.as_posix(),
        integrity_key=integrity_key,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "pass_rate": report.pass_rate,
                "baseline_selective_accuracy": report.baseline_selective_accuracy,
                "adaptive_selective_accuracy": report.adaptive_selective_accuracy,
                "recovery_rate": report.recovery_rate,
                "unsafe_release_rate": report.unsafe_release_rate,
                "candidate_call_reduction": report.candidate_call_reduction,
                "report": str(args.output) if args.output else "",
            }
        )
    )
    if (
        report.pass_rate < args.min_pass_rate
        or report.recovery_rate < args.min_recovery_rate
        or report.unsafe_release_rate > args.max_unsafe_release_rate
        or report.budget_violation_rate > args.max_budget_violation_rate
        or not verify_report(report)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
