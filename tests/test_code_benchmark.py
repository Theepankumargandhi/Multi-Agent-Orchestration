import hashlib
import json
from pathlib import Path

import pytest

from code_agent.dashboard import _safe_artifact, load_reports
from code_agent.evaluation import (
    CodeBenchmarkCase,
    dataset_fingerprint,
    load_code_cases,
    run_benchmark,
    score_code_result,
)
from code_agent.models import (
    AnalysisPlan,
    CodeAgentResult,
    ReviewVerdict,
    SandboxCommandResult,
    SandboxPolicy,
    ToolObservation,
    VerificationReport,
    VerificationRound,
)
from code_agent.swebench import convert_swebench_records, load_swebench_records, write_cases


def _successful_result() -> CodeAgentResult:
    return CodeAgentResult(
        status="completed",
        summary="Sandbox tests passed after repository changes.",
        patch="--- a/app.py\n+++ b/app.py\n-old\n+new\n",
        changed_files=["app.py"],
        baseline_test=SandboxCommandResult(
            command=["python", "-m", "pytest", "-q"], exit_code=1, duration_ms=10
        ),
        final_test=SandboxCommandResult(
            command=["python", "-m", "pytest", "-q"], exit_code=0, stdout="1 passed", duration_ms=12
        ),
        iterations=3,
        writes=1,
        tool_calls=4,
        model_calls=3,
        prompt_tokens=1000,
        completion_tokens=200,
        total_duration_ms=1250,
        observations=[
            ToolObservation(
                iteration=0,
                action="test",
                ok=False,
                summary="Baseline test command completed before agent edits.",
            ),
            ToolObservation(iteration=1, action="read", ok=True, summary="Inspected app.py."),
            ToolObservation(iteration=2, action="write", ok=True, summary="Changed the value."),
            ToolObservation(iteration=3, action="test", ok=True, summary="Tests passed."),
        ],
        model="test-model",
        sandbox={"original_repository_unchanged": True, "network_enabled": False},
    )


def test_swebench_adapter_excludes_gold_solution_fields(tmp_path: Path):
    source = tmp_path / "swebench.jsonl"
    source.write_text(
        json.dumps(
            {
                "instance_id": "org__repo-123",
                "repo": "org/repo",
                "base_commit": "1234567890abcdef",
                "problem_statement": "Fix the parser when a quoted value is empty.",
                "version": "1.0",
                "FAIL_TO_PASS": '["tests/test_parser.py::test_empty"]',
                "PASS_TO_PASS": [],
                "patch": "GOLD SOLUTION MUST NOT LEAK",
                "test_patch": "HIDDEN TEST PATCH",
                "language": "Java",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_swebench_records(source)
    cases = convert_swebench_records(records)
    encoded = json.dumps(cases)
    assert cases[0]["repository"] == "org__repo"
    assert cases[0]["metadata"]["base_commit"] == "1234567890abcdef"
    assert cases[0]["metadata"]["fail_to_pass"] == ["tests/test_parser.py::test_empty"]
    assert "language:java" in cases[0]["tags"]
    assert cases[0]["metadata"]["language"] == "Java"
    assert "GOLD SOLUTION" not in encoded
    assert "HIDDEN TEST" not in encoded


def test_dataset_fingerprint_is_stable_and_duplicate_ids_are_rejected(tmp_path: Path):
    case = CodeBenchmarkCase(
        id="case-1",
        repository="fixture",
        issue="Fix the deterministic fixture test failure.",
        test_command=["python", "-m", "pytest", "-q"],
    )
    assert dataset_fingerprint([case]) == dataset_fingerprint([case.model_copy(deep=True)])
    changed = case.model_copy(update={"issue": "Fix a different deterministic fixture failure."})
    assert dataset_fingerprint([case]) != dataset_fingerprint([changed])
    dataset = tmp_path / "cases.jsonl"
    write_cases([case.model_dump(mode="json"), case.model_dump(mode="json")], dataset)
    with pytest.raises(ValueError, match="duplicate"):
        load_code_cases(dataset)


def test_score_tracks_cost_tokens_and_regressions():
    result = _successful_result()
    score = score_code_result(
        result, input_cost_per_million=2.0, output_cost_per_million=10.0
    )
    assert score.resolved is True
    assert score.regression is False
    assert score.prompt_tokens == 1000
    assert score.estimated_cost_usd == pytest.approx(0.004)
    assert score.verification_passed is False

    result.verification = VerificationReport(
        analysis=AnalysisPlan(summary="Implement the focused correction."),
        rounds=[
            VerificationRound(
                round=0,
                verdict=ReviewVerdict(approved=True, summary="Verified."),
            )
        ],
        final_decision="verified",
    )
    verified = score_code_result(result)
    assert verified.verification_passed is True

    result.status = "failed"
    result.baseline_test.exit_code = 0
    result.final_test.exit_code = 1
    failed = score_code_result(result)
    assert failed.regression is True
    assert failed.resolved is False


@pytest.mark.asyncio
async def test_benchmark_writes_replayable_and_swebench_compatible_artifacts(
    tmp_path: Path, monkeypatch
):
    repository_root = tmp_path / "repositories"
    (repository_root / "fixture").mkdir(parents=True)
    output_root = tmp_path / "results"
    case = CodeBenchmarkCase(
        id="org__repo-123",
        repository="fixture",
        issue="Fix the deterministic fixture test failure.",
        test_command=["python", "-m", "pytest", "-q"],
        tags=["parser", "swe-bench"],
        source="swe-bench",
        metadata={"instance_id": "org__repo-123"},
    )

    class FakeDockerSandbox:
        def __init__(self, repository, policy):
            self.repository = repository

        def start(self):
            return self

        def close(self):
            return None

    class FakeCodingAgent:
        def __init__(self, model):
            self.model = model

        async def solve(self, task, sandbox):
            return _successful_result()

    monkeypatch.setattr("code_agent.evaluation.DockerSandbox", FakeDockerSandbox)
    monkeypatch.setattr("code_agent.evaluation.CodingAgent", FakeCodingAgent)
    monkeypatch.setattr("code_agent.evaluation.build_coding_model", lambda model: object())
    captured = {}

    def fake_verified_builder(model, *, context_strategy=None):
        captured.update({"model": model, "context_strategy": context_strategy})
        return FakeCodingAgent(object())

    monkeypatch.setattr("code_agent.evaluation.build_verified_pr_agent", fake_verified_builder)

    report = await run_benchmark(
        [case],
        repository_root=repository_root,
        default_model="test-model",
        policy=SandboxPolicy(),
        dataset_name="swe-bench-smoke",
        output_dir=output_root,
        input_cost_per_million=2.0,
        output_cost_per_million=10.0,
        workflow="verified_pr",
        context_strategy="lexical_graph",
    )
    assert report.pass_at_1 == 1.0
    assert report.failure_categories == {"resolved": 1}
    assert report.slice_metrics["tag:parser"]["pass_at_1"] == 1.0
    assert report.context_strategy == "lexical_graph"
    assert captured == {"model": "test-model", "context_strategy": "lexical_graph"}
    run_dir = output_root / report.run_id
    report_path = run_dir / "report.json"
    assert load_reports(output_root)[0][1].run_id == report.run_id
    outcome = report.outcomes[0]
    trajectory_path = _safe_artifact(report_path, outcome.trajectory_path)
    patch_path = _safe_artifact(report_path, outcome.patch_path)
    assert trajectory_path and patch_path
    patch = patch_path.read_text(encoding="utf-8")
    assert hashlib.sha256(patch.encode()).hexdigest() == outcome.patch_sha256
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert "patch" not in trajectory["result"]
    assert len(trajectory["result"]["observations"]) == 4
    prediction = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    assert prediction["instance_id"] == "org__repo-123"
    assert prediction["model_patch"] == patch
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"][outcome.patch_path]["sha256"] == outcome.patch_sha256


def test_checked_in_smoke_dataset_references_five_versioned_fixtures():
    root = Path(__file__).parents[1]
    cases = load_code_cases(root / "evals" / "datasets" / "code_agent_smoke.jsonl")
    repositories = root / "tests" / "fixtures" / "code_repositories"
    assert len(cases) == 5
    assert {case.split for case in cases} == {"validation", "test"}
    for case in cases:
        assert (repositories / case.repository / "fixture_tests.py").is_file()


def test_dashboard_rejects_artifacts_outside_run_directory(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report_path = run_dir / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    assert _safe_artifact(report_path, "../secret.txt") is None
