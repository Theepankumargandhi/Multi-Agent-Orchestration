from pathlib import Path

import pytest

from evals.memory_evaluation import (
    DEFAULT_DATASET,
    MemoryEvalReport,
    evaluate_memory,
    load_scenarios,
    verify_report,
)


def test_memory_evaluation_passes_every_control_and_is_reproducible():
    scenarios = load_scenarios()
    first = evaluate_memory(scenarios, DEFAULT_DATASET.as_posix())
    second = evaluate_memory(scenarios, DEFAULT_DATASET.as_posix())
    assert first.total == first.passed == 12
    assert first.pass_rate == 1
    assert first.cross_tenant_leakage_rate == 0
    assert first.poisoning_attack_success_rate == 0
    assert first.stale_memory_rate == 0
    assert first.deletion_violation_rate == 0
    assert first.token_budget_violation_rate == 0
    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert [item.passed for item in first.outcomes] == [item.passed for item in second.outcomes]
    assert verify_report(first)


def test_memory_report_and_outcome_tampering_is_detected():
    report = evaluate_memory(load_scenarios())
    report.outcomes[0].observed = "tampered"
    assert not verify_report(report)
    restored = evaluate_memory(load_scenarios())
    restored.pass_rate = 0
    assert not verify_report(restored)


def test_memory_dataset_rejects_duplicates(tmp_path: Path):
    row = DEFAULT_DATASET.read_text(encoding="utf-8").splitlines()[0]
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_scenarios(duplicate)


def test_memory_report_json_round_trip():
    report = evaluate_memory(load_scenarios())
    restored = MemoryEvalReport.model_validate_json(report.model_dump_json())
    assert verify_report(restored)
