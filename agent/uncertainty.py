"""Split-conformal selective answering with integrity-bound decisions."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.grounding import GroundingReport


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _fingerprint(value: object, key: bytes | None = None) -> str:
    payload = _canonical(value)
    return (
        hmac.new(key, payload, hashlib.sha256).hexdigest()
        if key
        else hashlib.sha256(payload).hexdigest()
    )


class CalibrationExample(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    split: Literal["calibration", "test"]
    route: Literal["web", "hybrid", "rag", "kg", "math"]
    confidence: float = Field(ge=0, le=1)
    correct: bool
    high_risk: bool = False


class ConformalCalibrator(BaseModel):
    schema_version: str = "1.0"
    method: str = "mondrian-split-conformal-binary-correctness"
    target_error_rate: float = Field(gt=0, lt=1)
    global_quantile: float = Field(ge=0, le=1)
    route_quantiles: dict[str, float]
    calibration_count: int
    route_counts: dict[str, int]
    confidence_histogram: list[float]
    confidence_p05: float
    confidence_p95: float
    dataset_fingerprint: str
    artifact_fingerprint: str = ""


class UncertaintyReceipt(BaseModel):
    schema_version: str = "1.0"
    route: str
    confidence: float
    quantile: float
    prediction_set: list[Literal["correct", "incorrect"]]
    decision: Literal["release", "abstain"]
    target_error_rate: float
    used_route_calibration: bool
    out_of_distribution: bool
    reason: str
    calibrator_fingerprint: str
    receipt_fingerprint: str = ""


class DriftReport(BaseModel):
    sample_count: int
    js_divergence: float
    drift_detected: bool
    threshold: float
    reference_fingerprint: str
    report_fingerprint: str = ""


def _nonconformity(example: CalibrationExample) -> float:
    return 1 - example.confidence if example.correct else example.confidence


def _quantile(scores: list[float], alpha: float) -> float:
    if not scores:
        raise ValueError("at least one calibration score is required")
    ordered = sorted(scores)
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1 - alpha)))
    return ordered[max(0, rank - 1)]


def _histogram(values: list[float], bins: int = 10) -> list[float]:
    counts = [0] * bins
    for value in values:
        counts[min(bins - 1, max(0, int(value * bins)))] += 1
    total = sum(counts) or 1
    return [count / total for count in counts]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def fit_calibrator(
    examples: list[CalibrationExample],
    *,
    target_error_rate: float = 0.2,
    min_route_samples: int = 5,
    integrity_key: bytes | None = None,
) -> ConformalCalibrator:
    calibration = [item for item in examples if item.split == "calibration"]
    if len(calibration) < 10:
        raise ValueError("at least 10 calibration examples are required")
    if not 0 < target_error_rate < 1:
        raise ValueError("target_error_rate must be between zero and one")
    grouped: dict[str, list[CalibrationExample]] = {}
    for item in calibration:
        grouped.setdefault(item.route, []).append(item)
    global_quantile = _quantile(
        [_nonconformity(item) for item in calibration], target_error_rate
    )
    route_quantiles = {
        route: _quantile([_nonconformity(item) for item in items], target_error_rate)
        for route, items in grouped.items()
        if len(items) >= min_route_samples
    }
    confidences = [item.confidence for item in calibration]
    # Bind the artifact only to the examples used to fit it. Held-out labels must
    # not influence any part of the deployable calibrator, including metadata.
    dataset_payload = [item.model_dump(mode="json") for item in calibration]
    artifact = ConformalCalibrator(
        target_error_rate=target_error_rate,
        global_quantile=round(global_quantile, 6),
        route_quantiles={key: round(value, 6) for key, value in sorted(route_quantiles.items())},
        calibration_count=len(calibration),
        route_counts=dict(sorted(Counter(item.route for item in calibration).items())),
        confidence_histogram=_histogram(confidences),
        confidence_p05=_percentile(confidences, 0.05),
        confidence_p95=_percentile(confidences, 0.95),
        dataset_fingerprint=_fingerprint(dataset_payload),
    )
    artifact.artifact_fingerprint = _fingerprint(
        artifact.model_dump(mode="json", exclude={"artifact_fingerprint"}), integrity_key
    )
    return artifact


def verify_calibrator(
    artifact: ConformalCalibrator, integrity_key: bytes | None = None
) -> bool:
    expected = _fingerprint(
        artifact.model_dump(mode="json", exclude={"artifact_fingerprint"}), integrity_key
    )
    return hmac.compare_digest(artifact.artifact_fingerprint, expected)


def save_calibrator(path: str | Path, artifact: ConformalCalibrator) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(target)


def load_calibrator(
    path: str | Path, integrity_key: bytes | None = None
) -> ConformalCalibrator:
    artifact = ConformalCalibrator.model_validate_json(Path(path).read_text(encoding="utf-8"))
    if not verify_calibrator(artifact, integrity_key):
        raise ValueError("conformal calibrator integrity verification failed")
    return artifact


def selective_decision(
    *,
    route: str,
    confidence: float,
    artifact: ConformalCalibrator,
    integrity_key: bytes | None = None,
) -> UncertaintyReceipt:
    normalized_route = route.casefold()
    used_route = normalized_route in artifact.route_quantiles
    quantile = artifact.route_quantiles.get(normalized_route, artifact.global_quantile)
    prediction_set: list[Literal["correct", "incorrect"]] = []
    if 1 - confidence <= quantile:
        prediction_set.append("correct")
    if confidence <= quantile:
        prediction_set.append("incorrect")
    decision = "release" if prediction_set == ["correct"] else "abstain"
    out_of_distribution = not (
        artifact.confidence_p05 <= confidence <= artifact.confidence_p95
    )
    receipt = UncertaintyReceipt(
        route=normalized_route,
        confidence=round(confidence, 6),
        quantile=quantile,
        prediction_set=prediction_set,
        decision=decision,
        target_error_rate=artifact.target_error_rate,
        used_route_calibration=used_route,
        out_of_distribution=out_of_distribution,
        reason=(
            "singleton_correctness_set"
            if decision == "release"
            else "ambiguous_or_empty_correctness_set"
        ),
        calibrator_fingerprint=artifact.artifact_fingerprint,
    )
    receipt.receipt_fingerprint = _fingerprint(
        receipt.model_dump(mode="json", exclude={"receipt_fingerprint"}), integrity_key
    )
    return receipt


def verify_receipt(receipt: UncertaintyReceipt, integrity_key: bytes | None = None) -> bool:
    expected = _fingerprint(
        receipt.model_dump(mode="json", exclude={"receipt_fingerprint"}), integrity_key
    )
    return hmac.compare_digest(receipt.receipt_fingerprint, expected)


def assess_grounding_report(
    report: GroundingReport,
    artifact: ConformalCalibrator,
    integrity_key: bytes | None = None,
) -> UncertaintyReceipt:
    return selective_decision(
        route=report.route,
        confidence=report.confidence,
        artifact=artifact,
        integrity_key=integrity_key,
    )


def _js_divergence(left: list[float], right: list[float]) -> float:
    midpoint = [(a + b) / 2 for a, b in zip(left, right, strict=True)]

    def kl(values: list[float], reference: list[float]) -> float:
        return sum(
            value * math.log(value / target, 2)
            for value, target in zip(values, reference, strict=True)
            if value > 0 and target > 0
        )

    return (kl(left, midpoint) + kl(right, midpoint)) / 2


def detect_confidence_drift(
    confidences: list[float],
    artifact: ConformalCalibrator,
    *,
    threshold: float = 0.15,
    integrity_key: bytes | None = None,
) -> DriftReport:
    if len(confidences) < 5:
        raise ValueError("at least five recent confidence scores are required")
    observed = _histogram([max(0.0, min(float(item), 1.0)) for item in confidences])
    divergence = _js_divergence(artifact.confidence_histogram, observed)
    report = DriftReport(
        sample_count=len(confidences),
        js_divergence=round(divergence, 6),
        drift_detected=divergence >= threshold,
        threshold=threshold,
        reference_fingerprint=artifact.artifact_fingerprint,
    )
    report.report_fingerprint = _fingerprint(
        report.model_dump(mode="json", exclude={"report_fingerprint"}), integrity_key
    )
    return report
