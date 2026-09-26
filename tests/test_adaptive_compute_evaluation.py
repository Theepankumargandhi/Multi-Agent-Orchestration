import json
from pathlib import Path

import pytest

from evals.adaptive_compute_evaluation import (
    DEFAULT_DATASET,
    AdaptiveComputeReport,
    evaluate_adaptive_compute,
    load_scenarios,
    verify_report,
)


def test_adaptive_compute_ablation_recovers_quality_with_bounded_compute():
    report = evaluate_adaptive_compute(
        load_scenarios(), DEFAULT_DATASET.as_posix(), integrity_key=b"eval-integrity-key"
    )
    assert report.scenario_count == 13
    assert report.pass_rate == 1
    assert report.adaptive_selective_accuracy == 1
    assert report.adaptive_selective_accuracy > report.baseline_selective_accuracy
    assert report.recovery_rate == 1
    assert report.unsafe_release_rate == 0
    assert report.budget_violation_rate == 0
    assert report.candidate_call_reduction > 0.5
    assert report.receipt_integrity_rate == 1
    assert verify_report(report)


def test_adaptive_compute_report_is_reproducible_and_tamper_evident():
    scenarios = load_scenarios()
    first = evaluate_adaptive_compute(scenarios)
    second = evaluate_adaptive_compute(scenarios)
    assert first.model_dump() == second.model_dump()
    tampered = AdaptiveComputeReport.model_validate(first.model_dump())
    tampered.outcomes[0].final_correct = False
    assert not verify_report(tampered)


def test_dataset_rejects_duplicate_ids(tmp_path: Path):
    line = DEFAULT_DATASET.read_text(encoding="utf-8").splitlines()[0]
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_scenarios(duplicate)


def test_report_json_round_trip_preserves_integrity():
    report = evaluate_adaptive_compute(load_scenarios())
    restored = AdaptiveComputeReport.model_validate_json(report.model_dump_json())
    assert verify_report(restored)
    assert json.loads(restored.model_dump_json())["generated_by"] == (
        "agentforge-adaptive-compute-eval"
    )
