from pathlib import Path

import pytest

from agent.uncertainty import (
    CalibrationExample,
    ConformalCalibrator,
    detect_confidence_drift,
    fit_calibrator,
    load_calibrator,
    save_calibrator,
    selective_decision,
    verify_calibrator,
    verify_receipt,
)
from evals.uncertainty_evaluation import DEFAULT_DATASET, load_examples


def test_mondrian_calibrator_is_reproducible_integrity_bound_and_route_aware():
    examples = load_examples()
    first = fit_calibrator(examples, integrity_key=b"uncertainty-test-key")
    second = fit_calibrator(examples, integrity_key=b"uncertainty-test-key")
    assert first.model_dump() == second.model_dump()
    assert first.calibration_count == 25
    assert set(first.route_quantiles) == {"web", "hybrid", "rag", "kg", "math"}
    assert verify_calibrator(first, b"uncertainty-test-key")
    first.global_quantile = 0
    assert not verify_calibrator(first, b"uncertainty-test-key")


def test_selective_decision_releases_singleton_correct_set_and_abstains_on_ambiguity():
    artifact = fit_calibrator(load_examples())
    released = selective_decision(route="web", confidence=0.92, artifact=artifact)
    abstained = selective_decision(route="web", confidence=0.5, artifact=artifact)
    fallback = selective_decision(route="unknown", confidence=0.95, artifact=artifact)
    assert released.decision == "release" and released.prediction_set == ["correct"]
    assert abstained.decision == "abstain"
    assert set(abstained.prediction_set) == {"correct", "incorrect"}
    assert fallback.used_route_calibration is False
    assert verify_receipt(released)


def test_confidence_distribution_drift_detects_shift_but_not_reference():
    examples = load_examples()
    artifact = fit_calibrator(examples)
    reference = [item.confidence for item in examples if item.split == "calibration"]
    stable = detect_confidence_drift(reference, artifact)
    shifted = detect_confidence_drift([0.01, 0.03, 0.05, 0.07, 0.09, 0.11], artifact)
    assert not stable.drift_detected
    assert shifted.drift_detected
    assert shifted.js_divergence > stable.js_divergence


def test_calibrator_atomic_round_trip_and_tamper_detection(tmp_path: Path):
    artifact = fit_calibrator(load_examples(), integrity_key=b"uncertainty-test-key")
    path = tmp_path / "calibrator.json"
    save_calibrator(path, artifact)
    restored = load_calibrator(path, b"uncertainty-test-key")
    assert restored.artifact_fingerprint == artifact.artifact_fingerprint
    payload = restored.model_dump(mode="json")
    payload["global_quantile"] = 0.01
    path.write_text(ConformalCalibrator.model_validate(payload).model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_calibrator(path, b"uncertainty-test-key")


def test_calibrator_rejects_too_few_examples_and_drift_rejects_tiny_window():
    examples = [
        CalibrationExample(
            id=f"small-{index}",
            split="calibration",
            route="web",
            confidence=0.9,
            correct=True,
        )
        for index in range(5)
    ]
    with pytest.raises(ValueError, match="at least 10"):
        fit_calibrator(examples)
    artifact = fit_calibrator(load_examples())
    with pytest.raises(ValueError, match="at least five"):
        detect_confidence_drift([0.5], artifact)


def test_versioned_uncertainty_dataset_has_disjoint_splits_and_expected_size():
    examples = load_examples(DEFAULT_DATASET)
    calibration_ids = {item.id for item in examples if item.split == "calibration"}
    test_ids = {item.id for item in examples if item.split == "test"}
    assert len(examples) == 40
    assert len(calibration_ids) == 25 and len(test_ids) == 15
    assert calibration_ids.isdisjoint(test_ids)
