import json
from pathlib import Path

import pytest

from code_agent.benchmark_release import BenchmarkVariant, load_matrix
from code_agent.models import RepairTournamentPolicy, SandboxPolicy
from evals.repair_tournament_evaluation import FIXTURE, SCENARIOS, main, markdown, run_controls


@pytest.mark.asyncio
async def test_one_versus_three_team_controls_cover_fallbacks_and_vetoes():
    report = await run_controls()
    assert report["controls_passed"] and report["decision"] == "held" and not report["production_activation"]
    assert {row["scenario"] for row in report["controls"]} == set(SCENARIOS)
    rows = {row["scenario"]: row["arms"] for row in report["controls"]}
    assert rows["wrong_first_fix"]["one_team"]["status"] == "failed"
    assert rows["wrong_first_fix"]["three_teams"]["status"] == "completed"
    assert rows["focused_fix"]["three_teams"]["model_calls"] > rows["focused_fix"]["one_team"]["model_calls"]
    assert all(not arm["provider_usage_complete"] for arms in rows.values() for arm in arms.values() if arm["model_calls"])
    assert "authored" in markdown(report).lower()


def test_cli_protects_output_and_strict_review_holds(tmp_path):
    output = tmp_path / "controls.json"
    assert main(["--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["controls_passed"] and output.with_suffix(".md").exists()
    before = output.read_bytes()
    assert main(["--output", str(output)]) == 2 and output.read_bytes() == before
    assert main(["--output", str(FIXTURE / "app.py")]) == 2
    assert main(["--output", str(tmp_path / "controls.md")]) == 2
    assert main(["--output", str(tmp_path / "strict.json"), "--require-gate"]) == 1


def test_live_matrix_freezes_tournament_policies_without_running_models():
    matrix = load_matrix(Path("evals/experiments/repair_tournament_matrix.json"))
    assert [variant.workflow for variant in matrix.variants] == ["verified_pr", "repair_tournament", "repair_tournament"]
    assert matrix.variants[-1].tournament_policy.candidates == 3
    assert matrix.variants[1].tournament_policy.candidates == 1
    with pytest.raises(ValueError, match="explicit policy"):
        BenchmarkVariant(name="missing-policy", model="authored", workflow="repair_tournament")
    with pytest.raises(ValueError, match="omit"):
        BenchmarkVariant(name="wrong-policy", model="authored", tournament_policy=RepairTournamentPolicy())


@pytest.mark.asyncio
async def test_worker_dispatches_tournament_without_an_unused_parent_container(tmp_path, monkeypatch):
    from code_agent.worker import CodeAgentWorker
    from tests.test_code_execution import completed_result, make_store, request

    repositories = tmp_path / "repositories"
    (repositories / "fixture").mkdir(parents=True)
    store = make_store(tmp_path)
    job, _ = store.enqueue("owner", request())
    observed = []

    class Solver:
        async def solve(self, task, repository):
            observed.append((task, repository))
            return completed_result()

    monkeypatch.setattr("code_agent.repair_tournament.build_repair_tournament", lambda model: Solver())
    monkeypatch.setattr("code_agent.worker.DockerSandbox", lambda *_: pytest.fail("unused parent container"))
    worker = CodeAgentWorker(store, repositories, SandboxPolicy(), workflow="repair_tournament")
    assert (await worker._solve(job)).status == "completed"
    assert observed[0][1] == (repositories / "fixture").resolve()


@pytest.mark.asyncio
async def test_benchmark_captures_policy_and_uses_tournament_directly(tmp_path, monkeypatch):
    from code_agent.evaluation import run_benchmark
    from code_agent.evaluation_models import CodeBenchmarkCase
    from tests.test_code_benchmark import _successful_result

    (tmp_path / "fixture").mkdir()
    policies = []

    class Solver:
        async def solve(self, task, repository):
            assert repository == (tmp_path / "fixture").resolve()
            return _successful_result()

    def builder(model, **kwargs):
        policies.append(kwargs["policy"])
        return Solver()

    monkeypatch.setattr("code_agent.repair_tournament.build_repair_tournament", builder)
    monkeypatch.setattr("code_agent.evaluation.DockerSandbox", lambda *_: pytest.fail("unused parent container"))
    cases = [CodeBenchmarkCase(id="sample", repository="fixture", issue="Fix the requested boundary behavior.",
                               test_command=["python", "-m", "pytest", "-q"])]
    base = dict(repository_root=tmp_path, default_model="authored", policy=SandboxPolicy(), workflow="repair_tournament")
    first = await run_benchmark(cases, **base, tournament_policy=RepairTournamentPolicy())
    second = await run_benchmark(cases, **base, tournament_policy=RepairTournamentPolicy(max_model_calls=32))
    assert first.pass_at_1 == 1 and policies[0].candidates == 3
    assert first.config_fingerprint != second.config_fingerprint
    assert first.sandbox_policy["tournament_policy"]["max_model_calls"] == 64


def test_ci_controls_do_not_run_live_matrix():
    workflow = Path(".github/workflows/ci.yml").read_text()
    assert "python -m evals.repair_tournament_evaluation" in workflow
    assert "repair-tournament-controls" in workflow
    assert "repair_tournament_matrix.json" not in workflow
