import pytest

from evals.counterfactual_evaluation import docker_drill, evaluate


def test_authored_counterfactual_controls_are_not_live_performance_claims():
    report = evaluate()
    assert report["passed"] == report["total"] == 9
    assert report["provider_calls"] == 0
    assert report["live_coding_resolution_measured"] is False
    assert report["repository_execution_measured"] is False


@pytest.mark.asyncio
async def test_optional_docker_drill_reports_unavailability_without_host_fallback(monkeypatch):
    monkeypatch.setattr("code_agent.sandbox.DockerSandbox.available", staticmethod(lambda: False))
    report = await docker_drill()
    assert report["status"] == "unavailable"
    assert report["reason"] == "docker_unavailable"
    assert report["sources_unchanged"] is True
