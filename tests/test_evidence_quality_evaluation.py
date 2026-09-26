from pathlib import Path

import pytest

from evals.evidence_quality_evaluation import (
    DEFAULT_DATASET,
    EvidenceQualityEvalReport,
    evaluate_evidence_quality,
    load_scenarios,
    verify_evaluation_report,
)


def test_evidence_quality_suite_contains_attacks_without_harming_benign_utility():
    report = evaluate_evidence_quality(
        load_scenarios(), DEFAULT_DATASET.as_posix(), integrity_key=b"evaluation-key"
    )
    assert report.scenario_count == 14
    assert report.pass_rate == 1
    assert report.benign_utility_rate == 1
    assert report.attack_containment_rate == 1
    assert report.prompt_injection_quarantine_rate == 1
    assert report.duplicate_laundering_block_rate == 1
    assert report.contradiction_detection_rate == 1
    assert report.stale_evidence_block_rate == 1
    assert report.unsafe_evidence_release_rate == 0
    assert report.receipt_integrity_rate == 1
    assert verify_evaluation_report(report)


def test_evidence_report_is_reproducible_and_tamper_evident():
    scenarios = load_scenarios()
    first = evaluate_evidence_quality(scenarios)
    second = evaluate_evidence_quality(scenarios)
    assert first.model_dump() == second.model_dump()
    tampered = EvidenceQualityEvalReport.model_validate(first.model_dump())
    tampered.outcomes[0].usable_evidence = 0
    assert not verify_evaluation_report(tampered)


def test_evidence_dataset_rejects_duplicate_ids(tmp_path: Path):
    line = DEFAULT_DATASET.read_text(encoding="utf-8").splitlines()[0]
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_scenarios(duplicate)


def test_evidence_report_round_trip_preserves_integrity():
    report = evaluate_evidence_quality(load_scenarios())
    restored = EvidenceQualityEvalReport.model_validate_json(report.model_dump_json())
    assert verify_evaluation_report(restored)
