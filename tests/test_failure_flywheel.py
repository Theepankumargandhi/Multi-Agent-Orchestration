import csv
from pathlib import Path

import pytest

from agent.research_assistant import _evaluation_instruction_suffix
from evals.flywheel import (
    CanaryPolicy,
    DeploymentRegistry,
    FailureTrace,
    OnlineWindow,
    PromotionPolicy,
    build_candidate_experiment,
    build_preference_examples,
    cluster_failures,
    evaluate_promotion,
    export_proposal_review,
    generate_adversarial_proposals,
    promote_reviewed_regressions,
    propose_regressions,
    redact_value,
    traces_from_experiment,
    verify_promotion_decision,
)
from evals.platform import (
    AgentRun,
    CaseResult,
    ExpectedBehavior,
    ExperimentReport,
    MetricResult,
    VariantConfig,
    VariantReport,
)


def _case(case_id: str, passed: bool, *, reviewed: bool = True) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        input=f"Find supported evidence for {case_id} user@example.com",
        expected=ExpectedBehavior(route="web", answer_contains=["supported"]),
        actual=AgentRun(
            answer="supported" if passed else "API_KEY=super-secret-token",
            route="web" if passed else "general",
            latency_ms=100,
            prompt_tokens=10,
            completion_tokens=5,
            estimated_cost_usd=0.01,
            trace=[{"agent": "intent_router_agent", "authorization": "Bearer abcdefghijk"}],
        ),
        metrics={
            "route_accuracy": MetricResult(
                score=float(passed), passed=passed, detail="route result"
            )
        },
        quality_score=float(passed),
        passed=passed,
        tags=["web", "grounding"],
        split="validation",
        review_status="human_reviewed" if reviewed else "synthetic_seed",
    )


def _variant(name: str, cases: list[CaseResult], *, cost: float = 0.02) -> VariantReport:
    return VariantReport(
        variant=VariantConfig(name=name, adapter="live-graph", model="fixture-model"),
        quality_score=sum(case.quality_score for case in cases) / len(cases),
        pass_rate=sum(case.passed for case in cases) / len(cases),
        metric_scores={"route_accuracy": sum(case.quality_score for case in cases) / len(cases)},
        p50_latency_ms=100,
        p95_latency_ms=100,
        total_cost_usd=cost,
        total_tokens=30,
        failure_categories={"routing_failure": sum(not case.passed for case in cases)},
        cases=cases,
    )


def _report(baseline: list[CaseResult], candidate: list[CaseResult]) -> ExperimentReport:
    return ExperimentReport(
        experiment_id="11111111-1111-1111-1111-111111111111",
        experiment_name="flywheel-test",
        created_at="2026-09-10T00:00:00+00:00",
        dataset_path="fixture.jsonl",
        dataset_fingerprint="a" * 64,
        winner="candidate",
        pass_threshold=0.8,
        evaluated_splits=["validation"],
        review_status_counts={"human_reviewed": len(baseline)},
        reports=[_variant("baseline", baseline), _variant("candidate", candidate)],
    )


def test_recursive_redaction_covers_keys_and_free_text():
    value = {
        "authorization": "Bearer abcdefghijk",
        "nested": ["contact user@example.com", "sk-abcdefghijklmnopqrstuvwxyz"],
    }
    redacted = redact_value(value)
    assert redacted["authorization"] == "[REDACTED]"
    assert "user@example.com" not in redacted["nested"][0]
    assert "sk-" not in redacted["nested"][1]


def test_report_ingestion_is_redacted_and_idempotent():
    report = _report([_case("one", False)], [_case("one", False)])
    first = traces_from_experiment(report)
    second = traces_from_experiment(report)
    assert [trace.trace_id for trace in first] == [trace.trace_id for trace in second]
    assert [trace.content_fingerprint for trace in first] == [trace.content_fingerprint for trace in second]
    assert all("user@example.com" not in trace.input for trace in first)
    assert all("super-secret-token" not in str(trace.actual) for trace in first)
    assert all(trace.failure_categories == ["routing_failure"] for trace in first)


def test_failure_clustering_and_candidate_matrix_are_failure_driven():
    traces = [
        FailureTrace(
            trace_id=f"trace-{index}",
            occurred_at="2026-09-10T00:00:00Z",
            source="online",
            source_run_id="run",
            case_id=f"case-{index}",
            input=f"retrieve supported project evidence {index}",
            failure_categories=["answer_grounding_failure"],
            passed=False,
            quality_score=0.2,
            tags=["rag"],
        ).redacted()
        for index in range(2)
    ]
    clusters = cluster_failures(traces, similarity_threshold=0.2)
    assert len(clusters) == 1
    assert clusters[0].size == 2
    matrix = build_candidate_experiment(
        dataset="reviewed.jsonl", clusters=clusters, model="fixture-model"
    )
    assert len(matrix["variants"]) == 2
    assert matrix["splits"] == ["validation"]
    assert "response_instruction_suffix" in matrix["variants"][1]["parameters"]


def test_regression_promotion_requires_explicit_reviewer(tmp_path: Path):
    trace = FailureTrace(
        trace_id="trace-one",
        occurred_at="2026-09-10T00:00:00Z",
        source="online",
        source_run_id="run",
        case_id="case-one",
        input="find project evidence",
        expected={"route": "rag"},
        failure_categories=["routing_failure"],
        passed=False,
        quality_score=0,
    ).redacted()
    proposals = propose_regressions([trace])
    review = tmp_path / "review.csv"
    export_proposal_review(proposals, review)
    rows = list(csv.DictReader(review.read_text(encoding="utf-8-sig").splitlines()))
    rows[0]["decision"] = "approve"
    rows[0]["reviewer"] = "engineer@example.com"
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "reviewed.jsonl"
    summary = promote_reviewed_regressions(proposals, review, output)
    assert summary == {"approved": 1, "rejected": 0, "pending": 0, "dataset_size": 1}
    assert '"review_status":"human_reviewed"' in output.read_text(encoding="utf-8")


def test_adversarial_variants_stay_quarantined_for_review():
    trace = FailureTrace(
        trace_id="trace-adversarial",
        occurred_at="2026-09-10T00:00:00Z",
        source="online",
        source_run_id="run",
        case_id="case",
        input="explain the local repository",
        expected={"route": "rag"},
        failure_categories=["routing_failure"],
        passed=False,
        quality_score=0,
    ).redacted()
    variants = generate_adversarial_proposals(propose_regressions([trace]))
    assert len(variants) == 2
    assert all(item.proposed_case.review_status == "synthetic_seed" for item in variants)
    assert all(item.proposed_case.split == "validation" for item in variants)
    assert {item.proposed_case.metadata["mutation"] for item in variants} == {
        "format-noise",
        "instruction-distractor",
    }


def test_preference_export_requires_human_backed_chosen_and_rejected():
    trace = FailureTrace(
        trace_id="trace-correction",
        occurred_at="2026-09-10T00:00:00Z",
        source="online",
        source_run_id="run",
        case_id="case",
        input="question",
        actual={"answer": "unsupported answer"},
        failure_categories=["human_rejection"],
        passed=False,
        quality_score=0,
        feedback={"decision": "corrected", "reviewer": "reviewer", "notes": "fixed"},
        metadata={"corrected_answer": "supported answer"},
    ).redacted()
    examples = build_preference_examples([trace])
    assert len(examples) == 1
    assert examples[0].provenance == "human_correction"
    assert examples[0].chosen == "supported answer"
    assert build_preference_examples(
        [trace.model_copy(update={"feedback": trace.feedback.model_copy(update={"decision": "unknown"})})]
    ) == []


def test_promotion_gate_approves_improvement_and_rejects_regression():
    improved = evaluate_promotion(
        _report([_case("one", False), _case("two", True)], [_case("one", True), _case("two", True)]),
        "baseline",
        "candidate",
    )
    assert improved.approved_for_canary
    assert len(improved.decision_fingerprint) == 64
    assert verify_promotion_decision(improved)
    assert not verify_promotion_decision(improved.model_copy(update={"candidate": "tampered"}))

    regressed = evaluate_promotion(
        _report([_case("one", True)], [_case("one", False)]),
        "baseline",
        "candidate",
        PromotionPolicy(require_human_reviewed_cases=False),
    )
    assert not regressed.approved_for_canary
    assert regressed.regressions == ["one"]


def test_canary_registry_promotes_and_automatically_rolls_back(tmp_path: Path):
    report = _report([_case("one", False)], [_case("one", True)])
    decision = evaluate_promotion(report, "baseline", "candidate")
    registry = DeploymentRegistry(tmp_path / "deployment.json")
    registry.initialize("production-v1")
    registry.start_canary("candidate-v2", decision)
    promoted = registry.observe(
        OnlineWindow(
            sample_size=10,
            quality_score=0.95,
            error_rate=0,
            p95_latency_ms=100,
            cost_per_request_usd=0.01,
        ),
        CanaryPolicy(minimum_samples=10),
    )
    assert promoted.active_version == "candidate-v2"
    assert promoted.previous_version == "production-v1"

    registry.start_canary("candidate-v3", decision)
    rolled_back = registry.observe(
        OnlineWindow(
            sample_size=10,
            quality_score=0.2,
            error_rate=0.5,
            p95_latency_ms=100,
            cost_per_request_usd=0.01,
        ),
        CanaryPolicy(minimum_samples=10),
    )
    assert rolled_back.status == "rolled_back"
    assert rolled_back.active_version == "candidate-v2"
    assert rolled_back.canary_version is None


def test_evaluation_prompt_suffix_is_not_available_to_public_runtime():
    assert _evaluation_instruction_suffix({"configurable": {"response_instruction_suffix": "inject"}}, "response_instruction_suffix") == ""
    assert _evaluation_instruction_suffix(
        {"configurable": {"evaluation_mode": True, "response_instruction_suffix": "candidate"}},
        "response_instruction_suffix",
    ) == "candidate"


def test_gate_requires_reviewed_evidence_by_default():
    report = _report([_case("one", False, reviewed=False)], [_case("one", True, reviewed=False)])
    report.review_status_counts = {"synthetic_seed": 1}
    decision = evaluate_promotion(report, "baseline", "candidate")
    assert not decision.approved_for_canary
    assert next(check for check in decision.checks if check.name == "human_reviewed_evidence").passed is False


def test_registry_rejects_failed_offline_decision(tmp_path: Path):
    decision = evaluate_promotion(
        _report([_case("one", True)], [_case("one", False)]), "baseline", "candidate"
    )
    registry = DeploymentRegistry(tmp_path / "deployment.json")
    registry.initialize("v1")
    with pytest.raises(ValueError, match="did not pass"):
        registry.start_canary("v2", decision)
