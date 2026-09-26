import json
from pathlib import Path

import pytest

from evals.platform import (
    AgentRun,
    EvalCase,
    ExpectedBehavior,
    ExperimentConfig,
    ExperimentStore,
    VariantConfig,
    grade_case,
    load_cases,
    run_experiment,
)


def test_grade_case_combines_routing_citations_tools_and_latency():
    case = EvalCase(
        id="complete",
        input="Find evidence",
        expected=ExpectedBehavior(
            route="web",
            answer_contains=["supported"],
            min_citations=1,
            tool_calls=["web_search"],
            max_latency_ms=500,
        ),
    )
    run = AgentRun(
        answer="A supported answer [source](https://example.com).",
        route="web",
        tool_calls=["web_search"],
        latency_ms=100,
    )
    result = grade_case(case, run, {})
    assert result.passed
    assert result.quality_score == 1.0
    assert set(result.metrics) == {
        "route_accuracy",
        "required_term_recall",
        "citation_coverage",
        "tool_call_f1",
        "latency_slo",
    }


def test_grade_case_exposes_failure_instead_of_hiding_error():
    case = EvalCase(id="failed", input="hello", expected=ExpectedBehavior(route="general"))
    result = grade_case(case, AgentRun(error="TimeoutError"), {})
    assert not result.passed
    assert result.metrics["execution_success"].score == 0
    assert result.quality_score == 0


def test_dataset_loader_rejects_duplicate_ids(tmp_path: Path):
    dataset = tmp_path / "duplicate.jsonl"
    row = {"id": "same", "input": "hello", "expected": {"route": "general"}}
    dataset.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(dataset)


@pytest.mark.asyncio
async def test_experiment_compares_variants_and_persists_registry(tmp_path: Path):
    dataset = tmp_path / "routing.jsonl"
    rows = [
        {"id": "hello", "input": "hello", "expected": {"route": "general"}},
        {"id": "math", "input": "calculate 2 + 2", "expected": {"route": "math"}},
        {
            "id": "project",
            "input": "explain this repository",
            "expected": {"route": "rag"},
        },
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    config = ExperimentConfig(
        name="test-ablation",
        dataset=str(dataset),
        variants=[
            VariantConfig(name="baseline", adapter="keyword-baseline"),
            VariantConfig(name="production", adapter="research-router"),
        ],
        pass_threshold=0.5,
    )
    report = await run_experiment(config)
    assert report.winner in {"baseline", "production"}
    assert len(report.dataset_fingerprint) == 64
    assert any(item.pareto_optimal for item in report.reports)
    assert report.review_status_counts == {"synthetic_seed": 3}
    assert all(len(item.quality_confidence_interval) == 2 for item in report.reports)
    assert all("routing" not in item.slice_scores for item in report.reports)

    store = ExperimentStore(tmp_path / "results")
    output = store.save(report)
    assert output.exists()
    assert store.list()[0]["experiment_id"] == report.experiment_id
