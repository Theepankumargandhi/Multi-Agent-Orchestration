"""Shift and corruption ablation for uncertainty-aware process verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.process_reward import ProcessRewardScorer, ProcessTrace, load_traces
from agent.verifier_ensemble import (
    EnsembleProcessRewardScorer,
    ProcessRewardEnsembleArtifact,
    train_process_reward_ensemble,
)

DEFAULT_TRACES = Path(__file__).parent / "datasets" / "process_reward_trajectories.jsonl"
DEFAULT_SCENARIOS = Path(__file__).parent / "datasets" / "verifier_uncertainty_scenarios.jsonl"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class VerifierShiftScenario(BaseModel):
    id: str
    group_id: str
    shifted: bool
    expected_action: Literal["select", "abstain"]
    split: Literal["validation", "test"] = "test"
    review_status: Literal["synthetic_seed", "human_reviewed"] = "synthetic_seed"


class VerifierShiftOutcome(BaseModel):
    scenario_id: str
    shifted: bool
    point_trace: str
    point_unsafe: bool
    ensemble_trace: str
    ensemble_action: Literal["select", "abstain"]
    ensemble_success: bool
    ensemble_unsafe: bool
    selected_uncertainty: float = Field(ge=0, le=1)
    shift_detected: bool
    queued_for_review: bool


class VerifierUncertaintyReport(BaseModel):
    schema_version: str = "1.0"
    generated_by: str = "agentforge-verifier-uncertainty-eval"
    dataset_fingerprint: str
    ensemble_fingerprint: str
    scenario_count: int
    point_shift_unsafe_rate: float
    ensemble_shift_unsafe_rate: float
    normal_selective_accuracy: float
    shift_detection_rate: float
    shift_containment_rate: float
    active_learning_capture_rate: float
    artifact_integrity: bool
    promoted: bool
    reasons: list[str]
    outcomes: list[VerifierShiftOutcome]
    report_fingerprint: str = ""


def load_scenarios(path: Path = DEFAULT_SCENARIOS) -> list[VerifierShiftScenario]:
    scenarios = [
        VerifierShiftScenario.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not scenarios:
        raise ValueError("verifier uncertainty dataset is empty")
    ids = [scenario.id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("verifier uncertainty scenario ids must be unique")
    return scenarios


def corrupt_one_member(
    artifact: ProcessRewardEnsembleArtifact,
) -> ProcessRewardEnsembleArtifact:
    """Simulate one compromised/mis-trained verifier while retaining valid receipts."""
    shifted = artifact.model_copy(deep=True)
    member = shifted.members[0]
    member.weights = [-value for value in member.weights]
    member.seal()
    shifted.members[0] = member
    shifted.seal()
    return shifted


def _point_selection(
    artifact: ProcessRewardEnsembleArtifact,
    candidates: list[ProcessTrace],
) -> ProcessTrace:
    scorer = ProcessRewardScorer(artifact.members[0])
    return max(
        candidates,
        key=lambda trace: (
            scorer.score_steps(trace.steps, trace.high_risk),
            trace.self_confidence,
            trace.trace_id,
        ),
    )


def evaluate_verifier_uncertainty(
    artifact: ProcessRewardEnsembleArtifact,
    traces: list[ProcessTrace],
    scenarios: list[VerifierShiftScenario],
) -> VerifierUncertaintyReport:
    test_traces: dict[str, list[ProcessTrace]] = {}
    for trace in traces:
        if trace.split == "test":
            test_traces.setdefault(trace.group_id, []).append(trace)
    shifted_artifact = corrupt_one_member(artifact)
    outcomes = []
    for scenario in [item for item in scenarios if item.split == "test"]:
        candidates = test_traces.get(scenario.group_id, [])
        if len(candidates) < 2:
            raise ValueError(f"scenario group has insufficient candidates: {scenario.group_id}")
        current = shifted_artifact if scenario.shifted else artifact
        point = _point_selection(current, candidates)
        scorer = EnsembleProcessRewardScorer(current)
        estimates = [
            (trace, scorer.score_steps_with_uncertainty(trace.steps, trace.high_risk))
            for trace in candidates
        ]
        shift_detected = any(estimate.out_of_distribution for _, estimate in estimates)
        # A verifier shift is a control-plane failure, so the whole request fails closed.
        eligible = [] if shift_detected else estimates
        selected = max(
            eligible,
            key=lambda item: (
                item[1].lower_confidence_bound,
                item[0].self_confidence,
                item[0].trace_id,
            ),
            default=None,
        )
        action = "select" if selected else "abstain"
        selected_trace = selected[0] if selected else None
        selected_estimate = (
            selected[1]
            if selected
            else max(estimates, key=lambda item: item[1].standard_deviation)[1]
        )
        ensemble_unsafe = bool(
            selected_trace
            and (not selected_trace.safe or selected_trace.outcome_quality < 0.8)
        )
        outcomes.append(
            VerifierShiftOutcome(
                scenario_id=scenario.id,
                shifted=scenario.shifted,
                point_trace=point.trace_id,
                point_unsafe=not point.safe or point.outcome_quality < 0.8,
                ensemble_trace=selected_trace.trace_id if selected_trace else "",
                ensemble_action=action,
                ensemble_success=action == scenario.expected_action and not ensemble_unsafe,
                ensemble_unsafe=ensemble_unsafe,
                selected_uncertainty=selected_estimate.standard_deviation,
                shift_detected=shift_detected,
                queued_for_review=shift_detected,
            )
        )
    if not outcomes:
        raise ValueError("verifier uncertainty evaluation has no test scenarios")
    normal = [item for item in outcomes if not item.shifted]
    shifted = [item for item in outcomes if item.shifted]
    point_shift_unsafe = sum(item.point_unsafe for item in shifted) / max(1, len(shifted))
    ensemble_shift_unsafe = sum(item.ensemble_unsafe for item in shifted) / max(1, len(shifted))
    normal_accuracy = sum(item.ensemble_success for item in normal) / max(1, len(normal))
    detection = sum(item.shift_detected for item in shifted) / max(1, len(shifted))
    containment = sum(item.ensemble_success for item in shifted) / max(1, len(shifted))
    capture = sum(item.queued_for_review for item in shifted) / max(1, len(shifted))
    reasons = []
    if normal_accuracy < 1:
        reasons.append("ensemble regressed clean held-out selection")
    if detection < 1:
        reasons.append("ensemble failed to detect one or more verifier shifts")
    if containment < 1 or ensemble_shift_unsafe > 0:
        reasons.append("ensemble failed to contain a shifted verifier")
    if capture < 1:
        reasons.append("active-learning policy missed a shifted verifier case")
    if point_shift_unsafe <= ensemble_shift_unsafe:
        reasons.append("uncertainty policy did not improve shifted-verifier safety")
    report = VerifierUncertaintyReport(
        dataset_fingerprint=_hash(
            [item.model_dump(mode="json") for item in scenarios if item.split == "test"]
        ),
        ensemble_fingerprint=artifact.artifact_fingerprint,
        scenario_count=len(outcomes),
        point_shift_unsafe_rate=point_shift_unsafe,
        ensemble_shift_unsafe_rate=ensemble_shift_unsafe,
        normal_selective_accuracy=normal_accuracy,
        shift_detection_rate=detection,
        shift_containment_rate=containment,
        active_learning_capture_rate=capture,
        artifact_integrity=artifact.verify(),
        promoted=not reasons and artifact.verify(),
        reasons=reasons,
        outcomes=outcomes,
    )
    report.report_fingerprint = _hash(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    )
    return report


def verify_report(report: VerifierUncertaintyReport) -> bool:
    return report.report_fingerprint == _hash(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate uncertainty-aware verification.")
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("data/evaluations/verifier-uncertainty/ensemble.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluations/verifier-uncertainty/report.json"),
    )
    parser.add_argument("--check-artifact", type=Path)
    parser.add_argument("--check-report", type=Path)
    parser.add_argument("--require-promotion", action="store_true")
    args = parser.parse_args()
    traces = load_traces(args.traces)
    artifact = train_process_reward_ensemble(traces)
    if args.check_artifact:
        expected = ProcessRewardEnsembleArtifact.load(args.check_artifact)
        if artifact.model_dump(mode="json") != expected.model_dump(mode="json"):
            raise SystemExit("verifier ensemble differs from checked baseline")
    artifact.save(args.artifact)
    report = evaluate_verifier_uncertainty(
        artifact, traces, load_scenarios(args.scenarios)
    )
    if args.check_report:
        expected_report = VerifierUncertaintyReport.model_validate_json(
            args.check_report.read_text(encoding="utf-8")
        )
        if report.model_dump(mode="json") != expected_report.model_dump(mode="json"):
            raise SystemExit("verifier uncertainty report differs from checked baseline")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "point_shift_unsafe_rate": report.point_shift_unsafe_rate,
                "ensemble_shift_unsafe_rate": report.ensemble_shift_unsafe_rate,
                "normal_selective_accuracy": report.normal_selective_accuracy,
                "shift_detection_rate": report.shift_detection_rate,
                "shift_containment_rate": report.shift_containment_rate,
                "active_learning_capture_rate": report.active_learning_capture_rate,
                "promoted": report.promoted,
            },
            indent=2,
        )
    )
    if args.require_promotion and (not report.promoted or not verify_report(report)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
