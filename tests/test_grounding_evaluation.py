from pathlib import Path

import pytest

from evals.grounding_evaluation import (
    DEFAULT_DATASET,
    GroundingEvalReport,
    evaluate_grounding,
    load_scenarios,
    verify_evaluation_report,
)


def test_grounding_gate_passes_all_controls_without_unsafe_release():
    report = evaluate_grounding(load_scenarios(), DEFAULT_DATASET.as_posix())
    assert report.total == report.passed == 12
    assert report.pass_rate == 1
    assert report.unsafe_release_rate == 0
    assert report.fabricated_citation_escape_rate == 0
    assert report.high_risk_claim_escape_rate == 0
    assert report.repair_success_rate == 1
    assert report.receipt_integrity_rate == 1
    assert verify_evaluation_report(report)


def test_grounding_evaluation_is_reproducible_and_tamper_evident():
    first = evaluate_grounding(load_scenarios())
    second = evaluate_grounding(load_scenarios())
    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert first.report_fingerprint == second.report_fingerprint
    first.outcomes[0].passed = False
    assert not verify_evaluation_report(first)


def test_grounding_dataset_rejects_duplicate_ids(tmp_path: Path):
    row = DEFAULT_DATASET.read_text(encoding="utf-8").splitlines()[0]
    duplicate = tmp_path / "duplicates.jsonl"
    duplicate.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_scenarios(duplicate)


def test_grounding_report_json_round_trip():
    report = evaluate_grounding(load_scenarios())
    restored = GroundingEvalReport.model_validate_json(report.model_dump_json())
    assert verify_evaluation_report(restored)
