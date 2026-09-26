from pathlib import Path

import pytest

from evals.uncertainty_evaluation import (
    DEFAULT_DATASET,
    UncertaintyEvalReport,
    evaluate_uncertainty,
    load_examples,
    verify_evaluation_report,
)


def test_uncertainty_gate_is_selectively_accurate_and_detects_shift():
    artifact, report = evaluate_uncertainty(load_examples(), DEFAULT_DATASET.as_posix())
    assert artifact.calibration_count == 25
    assert report.test_count == 15
    assert report.coverage_rate == 2 / 3
    assert report.selective_accuracy == 1
    assert report.empirical_error_rate == 0
    assert report.incorrect_abstention_rate == 1
    assert report.receipt_integrity_rate == 1
    assert report.shift_detected
    assert verify_evaluation_report(report)


def test_uncertainty_evaluation_is_reproducible_and_tamper_evident():
    first_artifact, first = evaluate_uncertainty(load_examples())
    second_artifact, second = evaluate_uncertainty(load_examples())
    assert first_artifact.artifact_fingerprint == second_artifact.artifact_fingerprint
    assert first.report_fingerprint == second.report_fingerprint
    first.coverage_rate = 0
    assert not verify_evaluation_report(first)


def test_uncertainty_dataset_rejects_duplicate_ids(tmp_path: Path):
    row = DEFAULT_DATASET.read_text(encoding="utf-8").splitlines()[0]
    duplicate = tmp_path / "duplicates.jsonl"
    duplicate.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_examples(duplicate)


def test_uncertainty_report_json_round_trip():
    _, report = evaluate_uncertainty(load_examples())
    restored = UncertaintyEvalReport.model_validate_json(report.model_dump_json())
    assert verify_evaluation_report(restored)
