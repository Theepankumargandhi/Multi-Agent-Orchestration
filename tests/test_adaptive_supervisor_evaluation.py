import json
from pathlib import Path

import pytest

from code_agent.benchmark_release import BenchmarkVariant, load_matrix, paired_comparison
from code_agent.models import RepairTournamentPolicy, SandboxPolicy
from evals.adaptive_supervisor_evaluation import EXPECTED, main, markdown, run_controls
from tests.test_benchmark_release import _report


@pytest.mark.asyncio
async def test_three_way_controls_are_held_and_measure_only_authored_behavior():
    report = await run_controls()
    assert report["controls_passed"] and report["decision"] == "held" and not report["production_activation"]
    assert {row["scenario"] for row in report["controls"]} == set(EXPECTED)
    rows = {row["scenario"]: row["arms"] for row in report["controls"]}
    easy = rows["focused_fix"]
    assert easy["adaptive_supervisor"]["logical_model_calls"] < easy["repair_tournament"]["logical_model_calls"]
    assert rows["wrong_first_fix"]["adaptive_supervisor"]["dispatched_perspectives"] == ["minimal", "boundary"]
    assert rows["unsafe_first_fix"]["adaptive_supervisor"]["stop_reason"] == "safety_hold"
    assert rows["first_provider_outage"]["adaptive_supervisor"]["stop_reason"] == "provider_error"
    assert rows["shared_call_budget"]["adaptive_supervisor"]["stop_reason"] == "budget"
    assert all(arm["cost_per_resolved_usd"] is None and not arm["usage_accounting_complete"]
               for row in rows.values() for arm in row.values())
    assert "no activation" in markdown(report).lower()


def test_cli_preserves_outputs_and_cannot_promote_synthetic_controls(tmp_path):
    output = tmp_path / "controls.json"
    assert main(["--output", str(output)]) == 0
    before = output.read_bytes()
    assert json.loads(before)["controls_passed"] and output.with_suffix(".md").exists()
    assert main(["--output", str(output)]) == 2 and output.read_bytes() == before
    assert main(["--output", "evals/fixtures/repair_tournament/app.py"]) == 2
    assert main(["--output", str(tmp_path / "bad.md")]) == 2
    assert main(["--output", str(tmp_path / "strict.json"), "--require-gate"]) == 1


def test_frozen_matrix_has_three_distinct_workflows_and_explicit_policies():
    matrix = load_matrix(Path("evals/experiments/adaptive_supervisor_matrix.json"))
    assert [variant.workflow for variant in matrix.variants] == ["verified_pr", "repair_tournament", "adaptive_supervisor"]
    adaptive = matrix.variants[-1].tournament_policy
    assert adaptive.adaptive_dispatch and adaptive.max_parallel == 1
    assert matrix.variants[1].tournament_policy.adaptive_dispatch is None
    with pytest.raises(ValueError, match="explicit policy"):
        BenchmarkVariant(name="missing", model="authored", workflow="adaptive_supervisor")
    with pytest.raises(ValueError, match="adaptive_dispatch"):
        BenchmarkVariant(name="missing-adaptive", model="authored", workflow="adaptive_supervisor",
                         tournament_policy=RepairTournamentPolicy())


@pytest.mark.asyncio
async def test_worker_dispatches_opt_in_supervisor_without_parent_container(tmp_path, monkeypatch):
    from code_agent.worker import CodeAgentWorker
    from tests.test_code_execution import completed_result, make_store, request
    root = tmp_path / "repositories"
    (root / "fixture").mkdir(parents=True)
    store = make_store(tmp_path)
    job, _ = store.enqueue("owner", request())
    seen = []
    class Solver:
        async def solve(self, task, repository):
            seen.append(repository)
            return completed_result()
    monkeypatch.setattr("code_agent.repair_tournament.build_adaptive_supervisor", lambda _: Solver())
    monkeypatch.setattr("code_agent.worker.DockerSandbox", lambda *_: pytest.fail("unused parent container"))
    worker = CodeAgentWorker(store, root, SandboxPolicy(), workflow="adaptive_supervisor")
    assert (await worker._solve(job)).status == "completed" and seen == [(root / "fixture").resolve()]


@pytest.mark.asyncio
async def test_benchmark_records_policy_calls_prices_and_usage_coverage(tmp_path, monkeypatch):
    from code_agent.evaluation import run_benchmark
    from code_agent.evaluation_models import CodeBenchmarkCase
    from tests.test_code_benchmark import _successful_result
    (tmp_path / "fixture").mkdir()
    result = _successful_result()
    result.usage_accounting_complete = True
    seen = []
    class Solver:
        async def solve(self, task, repository):
            return result
    def builder(_, **kwargs):
        seen.append(kwargs["policy"])
        return Solver()
    monkeypatch.setattr("code_agent.repair_tournament.build_adaptive_supervisor", builder)
    monkeypatch.setattr("code_agent.evaluation.DockerSandbox", lambda *_: pytest.fail("unused parent container"))
    cases = [CodeBenchmarkCase(id="one", repository="fixture", issue="Fix the bounded fixture behavior.", test_command=["python"])]
    policy = RepairTournamentPolicy(max_parallel=1, adaptive_dispatch={})
    report = await run_benchmark(cases, repository_root=tmp_path, default_model="authored", policy=SandboxPolicy(),
        workflow="adaptive_supervisor", tournament_policy=policy, input_cost_per_million=1, output_cost_per_million=2)
    assert report.total_model_calls == report.average_model_calls == 3
    assert report.usage_accounting_complete and report.prices_configured and seen == [policy]
    result.usage_accounting_complete = False
    held = await run_benchmark(cases, repository_root=tmp_path, default_model="authored", policy=SandboxPolicy(),
        workflow="adaptive_supervisor", tournament_policy=policy)
    assert not held.usage_accounting_complete and not held.prices_configured
    assert report.config_fingerprint != held.config_fingerprint
    with pytest.raises(ValueError, match="adaptive_dispatch"):
        await run_benchmark(cases, repository_root=tmp_path, default_model="authored", policy=SandboxPolicy(),
                            workflow="adaptive_supervisor", tournament_policy=RepairTournamentPolicy())


def test_missing_usage_or_prices_holds_dollar_comparisons():
    baseline = _report("baseline", [True], cost=0.2, p95=100)
    candidate = _report("candidate", [True], cost=0.1, p95=100)
    assert not paired_comparison("base", baseline, "new", candidate).cost_comparison_eligible
    baseline.usage_accounting_complete = candidate.usage_accounting_complete = True
    baseline.prices_configured = candidate.prices_configured = True
    assert paired_comparison("base", baseline, "new", candidate).cost_comparison_eligible
    candidate.prices_configured = False
    assert not paired_comparison("base", baseline, "new", candidate).cost_comparison_eligible


def test_ci_runs_controls_not_live_or_paid_matrix():
    workflow = Path(".github/workflows/ci.yml").read_text()
    assert "python -m evals.adaptive_supervisor_evaluation" in workflow
    assert "adaptive-supervisor-controls" in workflow
    assert "adaptive_supervisor_matrix.json" not in workflow
