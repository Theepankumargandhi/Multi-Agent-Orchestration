from pathlib import Path

import pytest

from evals.model_gateway_evaluation import (
    GatewayEvalReport,
    _hash,
    evaluate_gateway,
    load_scenarios,
    verify_report,
)

DATASET = Path("evals/datasets/model_gateway_scenarios.jsonl")


@pytest.mark.asyncio
async def test_gateway_control_plane_report_is_reproducible_and_complete():
    scenarios = load_scenarios(DATASET)
    report = await evaluate_gateway(scenarios, DATASET.as_posix())
    assert len(scenarios) == 12
    assert _hash([item.model_dump(mode="json") for item in scenarios]) == (
        "30530d92ae9a590c7a9f34f1470b098f9e6f3098cc80eee19b49809c870563b9"
    )
    assert report.pass_rate == 1
    assert report.control_coverage == 1
    assert report.receipt_integrity_rate == 1
    assert report.privacy_violations == 0
    assert report.fallback_recovery_rate == 1
    assert report.cache_control_pass_rate == 1
    assert report.budget_control_pass_rate == 1
    assert report.canary_observed_percentage == 19.9
    assert verify_report(report)


@pytest.mark.asyncio
async def test_gateway_report_and_outcome_modification_is_detected():
    report = await evaluate_gateway(load_scenarios(DATASET), DATASET.as_posix())
    outcome_tamper = GatewayEvalReport.model_validate(report.model_dump())
    outcome_tamper.outcomes[0].selected_provider = "modified"
    assert not verify_report(outcome_tamper)
    report_tamper = GatewayEvalReport.model_validate(report.model_dump())
    report_tamper.pass_rate = 0
    assert not verify_report(report_tamper)


def test_gateway_dataset_rejects_duplicate_scenario_ids(tmp_path):
    line = DATASET.read_text(encoding="utf-8").splitlines()[0]
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate gateway scenario"):
        load_scenarios(duplicate)


@pytest.mark.asyncio
async def test_gateway_artifact_json_round_trip(tmp_path):
    original = await evaluate_gateway(load_scenarios(DATASET), DATASET.as_posix())
    output = tmp_path / "gateway.json"
    output.write_text(original.model_dump_json(), encoding="utf-8")
    restored = GatewayEvalReport.model_validate_json(output.read_text(encoding="utf-8"))
    assert verify_report(restored)
