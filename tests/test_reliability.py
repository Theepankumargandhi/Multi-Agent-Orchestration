import asyncio
import json
from pathlib import Path

from evals.adaptive_compute_evaluation import evaluate_adaptive_compute
from evals.adaptive_compute_evaluation import (
    load_scenarios as load_adaptive_compute_scenarios,
)
from evals.arena import ArenaConfig, run_arena
from evals.arena import load_scenarios as load_arena_scenarios
from evals.evidence_quality_evaluation import evaluate_evidence_quality
from evals.evidence_quality_evaluation import (
    load_scenarios as load_evidence_quality_scenarios,
)
from evals.grounding_evaluation import evaluate_grounding
from evals.grounding_evaluation import load_scenarios as load_grounding_scenarios
from evals.memory_evaluation import evaluate_memory
from evals.memory_evaluation import load_scenarios as load_memory_scenarios
from evals.model_gateway_evaluation import (
    evaluate_gateway,
)
from evals.model_gateway_evaluation import (
    load_scenarios as load_gateway_scenarios,
)
from evals.online_monitor_evaluation import evaluate_dataset as evaluate_online_dataset
from evals.reliability import (
    ReliabilityReport,
    ReliabilityScenario,
    dataset_fingerprint,
    evaluate_reliability,
    execute_scenario,
    load_reliability_scenarios,
    verify_report,
)
from evals.reliability_dashboard import (
    arena_outcome_rows,
    incident_summary,
    latest_report,
    load_trace_records,
    trace_dot,
    trace_rows,
)
from evals.uncertainty_evaluation import evaluate_uncertainty, load_examples

DATASET = Path("evals/datasets/agent_reliability_scenarios.jsonl")
ARENA_DATASET = Path("evals/datasets/agent_arena_scenarios.jsonl")
ARENA_CONFIG = Path("evals/experiments/agent_arena.json")
GATEWAY_DATASET = Path("evals/datasets/model_gateway_scenarios.jsonl")


def _arena_report():
    config = ArenaConfig.model_validate_json(ARENA_CONFIG.read_text(encoding="utf-8"))
    return asyncio.run(run_arena(config, load_arena_scenarios(ARENA_DATASET)))


def _gateway_report():
    return asyncio.run(
        evaluate_gateway(load_gateway_scenarios(GATEWAY_DATASET), GATEWAY_DATASET.as_posix())
    )


def test_reliability_dataset_and_release_metrics_are_reproducible():
    scenarios = load_reliability_scenarios(DATASET)
    report = evaluate_reliability(scenarios, dataset_path=DATASET.as_posix())
    assert len(scenarios) == 15
    assert dataset_fingerprint(scenarios) == (
        "7f2e6ceea82e0c5bebaa07d18610311ee43d16ae3d5da8e5b60d6305e7811317"
    )
    assert report.faults == 12
    assert report.benign_controls == 3
    assert report.baseline.scenario_pass_rate == 0.2
    assert report.resilient.availability_rate == 13 / 15
    assert report.resilient.scenario_pass_rate == 1.0
    assert report.recovery_success_rate == 1.0
    assert report.resilient.p95_duration_ms == 157.1
    assert verify_report(report)


def test_recovery_engine_records_retry_fallback_and_fail_closed_controls():
    scenarios = {item.id: item for item in load_reliability_scenarios(DATASET)}
    retry = execute_scenario(scenarios["model-rate-limit-retry"], resilient=True)
    fallback = execute_scenario(scenarios["vector-store-fallback"], resilient=True)
    contained = execute_scenario(scenarios["policy-denial-containment"], resilient=True)
    assert retry.status == "recovered"
    assert [item.action for item in retry.events] == [
        "invoke",
        "bounded_backoff_retry",
        "invoke",
    ]
    assert fallback.status == "recovered"
    assert fallback.events[-1].phase == "fallback"
    assert contained.status == "contained"
    assert contained.events[-1].action == "fail_closed"


def test_exhausted_recovery_and_report_tampering_fail_closed():
    scenario = ReliabilityScenario.model_validate(
        {
            "id": "fallback-exhausted",
            "title": "Every configured model remains unavailable",
            "component": "model",
            "fault": "timeout",
            "strategy": "fallback_model",
            "expected_status": "recovered",
            "primary": [{"result": "timeout", "latency_ms": 10}],
            "fallback": [{"result": "timeout", "latency_ms": 10}],
            "max_attempts": 2,
            "max_recovery_ms": 50,
        }
    )
    assert execute_scenario(scenario, resilient=True).status == "failed"
    report = evaluate_reliability(load_reliability_scenarios(DATASET))
    tampered = ReliabilityReport.model_validate(report.model_dump())
    tampered.outcomes[0].resilient.events[0].latency_ms = 999
    assert not verify_report(tampered)


def test_dashboard_discovers_reports_and_builds_incident_summary(tmp_path):
    report = evaluate_reliability(load_reliability_scenarios(DATASET))
    path = tmp_path / "report.json"
    path.write_text(report.model_dump_json(), encoding="utf-8")
    selected = latest_report(
        tmp_path, lambda item: item.get("generated_by") == "agentforge-reliability-lab"
    )
    assert selected and selected[0] == path
    summary = incident_summary(report.outcomes[3].model_dump(mode="json"))
    assert "rate_limit" in summary
    assert "bounded_backoff_retry" in summary
    assert "recovered" in summary


def test_trace_debugger_loads_hierarchy_and_escapes_labels(tmp_path):
    trace = {
        "trace_id": "a" * 32,
        "service_name": "test",
        "spans": [
            {
                "span_id": "1" * 16,
                "parent_span_id": "",
                "name": 'root "quoted"',
                "kind": "agent",
                "status": "ok",
                "start_time_unix_nano": 1_000_000,
                "duration_ms": 2,
            },
            {
                "span_id": "2" * 16,
                "parent_span_id": "1" * 16,
                "name": "child",
                "kind": "tool",
                "status": "error",
                "start_time_unix_nano": 2_000_000,
                "duration_ms": 1,
            },
        ],
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    loaded = load_trace_records([path])
    assert len(loaded) == 1
    assert trace_rows(loaded[0])[1]["offset_ms"] == 1.0
    dot = trace_dot(loaded[0])
    assert '\\"quoted\\"' in dot
    assert "n0 -> n1" in dot


def test_reliability_command_center_renders_without_exceptions(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    results = tmp_path / "evaluations"
    output = results / "reliability" / "latest.json"
    output.parent.mkdir(parents=True)
    report = evaluate_reliability(load_reliability_scenarios(DATASET))
    output.write_text(report.model_dump_json(), encoding="utf-8")
    arena_output = results / "arena" / "latest.json"
    arena_output.parent.mkdir(parents=True)
    arena_output.write_text(_arena_report().model_dump_json(), encoding="utf-8")
    gateway_output = results / "model-gateway" / "latest.json"
    gateway_output.parent.mkdir(parents=True)
    gateway_output.write_text(_gateway_report().model_dump_json(), encoding="utf-8")
    online_output = results / "online-monitor" / "latest.json"
    online_output.parent.mkdir(parents=True)
    online_output.write_text(evaluate_online_dataset().model_dump_json(), encoding="utf-8")
    memory_output = results / "memory" / "latest.json"
    memory_output.parent.mkdir(parents=True)
    memory_output.write_text(
        evaluate_memory(load_memory_scenarios()).model_dump_json(), encoding="utf-8"
    )
    grounding_output = results / "grounding" / "latest.json"
    grounding_output.parent.mkdir(parents=True)
    grounding_output.write_text(
        evaluate_grounding(load_grounding_scenarios()).model_dump_json(), encoding="utf-8"
    )
    uncertainty_output = results / "uncertainty" / "latest.json"
    uncertainty_output.parent.mkdir(parents=True)
    _, uncertainty_report = evaluate_uncertainty(load_examples())
    uncertainty_output.write_text(uncertainty_report.model_dump_json(), encoding="utf-8")
    adaptive_output = results / "adaptive-compute" / "latest.json"
    adaptive_output.parent.mkdir(parents=True)
    adaptive_output.write_text(
        evaluate_adaptive_compute(load_adaptive_compute_scenarios()).model_dump_json(),
        encoding="utf-8",
    )
    evidence_output = results / "evidence-quality" / "latest.json"
    evidence_output.parent.mkdir(parents=True)
    evidence_output.write_text(
        evaluate_evidence_quality(load_evidence_quality_scenarios()).model_dump_json(),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTOPS_RESULTS_DIR", str(results))
    monkeypatch.setenv("AGENTOPS_DATASETS_DIR", str(DATASET.parent))
    app = AppTest.from_file("evals/reliability_dashboard.py").run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "AgentForge Reliability Command Center"


def test_arena_dashboard_flattens_counterfactual_trajectory_rows():
    report = _arena_report().model_dump(mode="json")
    rows = arena_outcome_rows(report)
    assert len(rows) == 20
    assert {item["variant"] for item in rows} == {
        "naive-policy-v1",
        "resilient-policy-v2",
    }
    assert all(len(item["trajectory"]) == 16 for item in rows)
