"""Train and evaluate the selective-generation conformal risk gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel

from agent.uncertainty import (
    CalibrationExample,
    ConformalCalibrator,
    detect_confidence_drift,
    fit_calibrator,
    save_calibrator,
    selective_decision,
    verify_calibrator,
    verify_receipt,
)

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "uncertainty_calibration.jsonl"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class UncertaintyEvalOutcome(BaseModel):
    example_id: str
    route: str
    confidence: float
    correct: bool
    decision: str
    prediction_set: list[str]
    passed: bool
    receipt_verified: bool
    evidence_fingerprint: str = ""


class UncertaintyEvalReport(BaseModel):
    schema_version: str = "1.0"
    generated_by: str = "agentforge-uncertainty-eval"
    dataset_path: str
    dataset_fingerprint: str
    target_error_rate: float
    calibration_count: int
    test_count: int
    coverage_rate: float
    selective_accuracy: float
    empirical_error_rate: float
    incorrect_abstention_rate: float
    receipt_integrity_rate: float
    route_quantiles: dict[str, float]
    stable_js_divergence: float
    shifted_js_divergence: float
    shift_detected: bool
    outcomes: list[UncertaintyEvalOutcome]
    calibrator_fingerprint: str
    report_fingerprint: str = ""


def load_examples(path: Path = DEFAULT_DATASET) -> list[CalibrationExample]:
    examples = [
        CalibrationExample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [item.id for item in examples]
    if not examples:
        raise ValueError("uncertainty calibration dataset is empty")
    if len(ids) != len(set(ids)):
        raise ValueError("uncertainty calibration ids must be unique")
    if not any(item.split == "test" for item in examples):
        raise ValueError("uncertainty calibration dataset requires a test split")
    return examples


def evaluate_uncertainty(
    examples: list[CalibrationExample],
    dataset_path: str = "",
    *,
    target_error_rate: float = 0.2,
    integrity_key: bytes | None = None,
) -> tuple[ConformalCalibrator, UncertaintyEvalReport]:
    artifact = fit_calibrator(
        examples,
        target_error_rate=target_error_rate,
        integrity_key=integrity_key,
    )
    test = [item for item in examples if item.split == "test"]
    outcomes: list[UncertaintyEvalOutcome] = []
    for example in test:
        receipt = selective_decision(
            route=example.route,
            confidence=example.confidence,
            artifact=artifact,
            integrity_key=integrity_key,
        )
        released = receipt.decision == "release"
        passed = (released and example.correct) or (not released and not example.correct)
        outcome = UncertaintyEvalOutcome(
            example_id=example.id,
            route=example.route,
            confidence=example.confidence,
            correct=example.correct,
            decision=receipt.decision,
            prediction_set=list(receipt.prediction_set),
            passed=passed,
            receipt_verified=verify_receipt(receipt, integrity_key),
        )
        outcome.evidence_fingerprint = _hash(
            outcome.model_dump(mode="json", exclude={"evidence_fingerprint"})
        )
        outcomes.append(outcome)
    released = [item for item in outcomes if item.decision == "release"]
    incorrect = [item for item in outcomes if not item.correct]
    calibration_confidences = [
        item.confidence for item in examples if item.split == "calibration"
    ]
    stable = detect_confidence_drift(
        calibration_confidences,
        artifact,
        integrity_key=integrity_key,
    )
    shifted = detect_confidence_drift(
        [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16],
        artifact,
        integrity_key=integrity_key,
    )
    report = UncertaintyEvalReport(
        dataset_path=dataset_path,
        dataset_fingerprint=_hash([item.model_dump(mode="json") for item in examples]),
        target_error_rate=target_error_rate,
        calibration_count=artifact.calibration_count,
        test_count=len(test),
        coverage_rate=len(released) / len(test),
        selective_accuracy=sum(item.correct for item in released) / max(1, len(released)),
        empirical_error_rate=sum(not item.correct for item in released) / max(1, len(released)),
        incorrect_abstention_rate=sum(item.decision == "abstain" for item in incorrect)
        / max(1, len(incorrect)),
        receipt_integrity_rate=sum(item.receipt_verified for item in outcomes) / len(outcomes),
        route_quantiles=artifact.route_quantiles,
        stable_js_divergence=stable.js_divergence,
        shifted_js_divergence=shifted.js_divergence,
        shift_detected=shifted.drift_detected,
        outcomes=outcomes,
        calibrator_fingerprint=artifact.artifact_fingerprint,
    )
    report.report_fingerprint = _hash(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    )
    return artifact, report


def verify_evaluation_report(report: UncertaintyEvalReport) -> bool:
    if report.report_fingerprint != _hash(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    ):
        return False
    return all(
        item.evidence_fingerprint
        == _hash(item.model_dump(mode="json", exclude={"evidence_fingerprint"}))
        for item in report.outcomes
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selective-generation uncertainty control")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--target-error-rate", type=float, default=0.2)
    parser.add_argument("--min-selective-accuracy", type=float, default=1.0)
    parser.add_argument("--max-empirical-error", type=float, default=0)
    parser.add_argument("--min-incorrect-abstention", type=float, default=1.0)
    parser.add_argument("--require-drift-detection", action="store_true")
    args = parser.parse_args()
    examples = load_examples(args.dataset)
    integrity_key = os.getenv("UNCERTAINTY_INTEGRITY_KEY", "").encode() or None
    artifact, report = evaluate_uncertainty(
        examples,
        args.dataset.as_posix(),
        target_error_rate=args.target_error_rate,
        integrity_key=integrity_key,
    )
    if args.artifact:
        save_calibrator(args.artifact, artifact)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "coverage_rate": report.coverage_rate,
                "selective_accuracy": report.selective_accuracy,
                "empirical_error_rate": report.empirical_error_rate,
                "incorrect_abstention_rate": report.incorrect_abstention_rate,
                "shift_detected": report.shift_detected,
                "report": str(args.output) if args.output else "",
                "artifact": str(args.artifact) if args.artifact else "",
            }
        )
    )
    if (
        report.selective_accuracy < args.min_selective_accuracy
        or report.empirical_error_rate > args.max_empirical_error
        or report.incorrect_abstention_rate < args.min_incorrect_abstention
        or (args.require_drift_detection and not report.shift_detected)
        or not verify_calibrator(artifact, integrity_key)
        or not verify_evaluation_report(report)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
