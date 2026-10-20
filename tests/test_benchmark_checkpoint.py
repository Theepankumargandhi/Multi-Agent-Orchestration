from __future__ import annotations

import asyncio
import json

import pytest

from code_agent.benchmark_checkpoint import BenchmarkCheckpoint
from code_agent.benchmark_release import build_release
from code_agent.dashboard import load_reports
from code_agent.evaluation import _summary, run_benchmark
from code_agent.evaluation_models import CodeBenchmarkCase
from code_agent.models import CodeAgentResult, SandboxCommandResult, SandboxPolicy
from code_agent.provider_reliability import ProviderFailure


@pytest.fixture
def runner(tmp_path, monkeypatch):
    root = tmp_path / "repositories"
    (root / "fixture").mkdir(parents=True)
    (root / "fixture" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    calls, responses, closed = [], [], []

    class Sandbox:
        def __init__(self, repository, policy):
            pass

        def start(self):
            pass

        def close(self):
            closed.append(True)

    class Agent:
        def __init__(self, model):
            pass

        async def solve(self, task, sandbox):
            calls.append(task.issue)
            value = responses.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value

    monkeypatch.setattr("code_agent.evaluation.DockerSandbox", Sandbox)
    monkeypatch.setattr("code_agent.evaluation.CodingAgent", Agent)
    monkeypatch.setattr("code_agent.evaluation.build_coding_model", lambda model: object())
    monkeypatch.setattr("code_agent.evaluation.image_identities", lambda images: {image: "sha256:" + "a" * 64 for image in images})
    cases = [CodeBenchmarkCase(id=f"case-{i}", repository="fixture", issue=f"Fix the deterministic value failure {i}.",
                               test_command=["python", "-m", "pytest"]) for i in range(3)]
    kwargs = dict(repository_root=root, default_model="scripted", policy=SandboxPolicy(),
                  checkpoint_dir=tmp_path / "results" / "durable", input_cost_per_million=1, output_cost_per_million=1)
    return cases, kwargs, calls, responses, closed


def result(*, quota=False, resolved=True):
    return CodeAgentResult(status="failed" if quota or not resolved else "completed", model="scripted", summary="Scripted execution evidence.",
        patch="--- a/app.py\n+++ b/app.py\n-old\n+new\n", changed_files=["app.py"],
        model_calls=1, prompt_tokens=100, completion_tokens=20, usage_accounting_complete=not quota,
        final_test=SandboxCommandResult(command=["python", "-m", "pytest"], exit_code=0 if resolved else 1, duration_ms=1),
        sandbox={"original_repository_unchanged": True},
        provider_failures=[ProviderFailure(category="quota", code="insufficient_quota", http_status=429)] if quota else [])


@pytest.mark.asyncio
async def test_hold_resume_keeps_completed_results_and_all_attempt_costs(runner):
    cases, kwargs, calls, responses, closed = runner
    responses.extend([result(), result(quota=True)])
    paused = await run_benchmark(cases, **kwargs)
    assert len(calls) == len(closed) == 2
    assert paused.execution_status == "paused" and paused.total == 3
    assert paused.pending_case_ids == ["case-1", "case-2"]
    assert paused.outcomes[-1].status == "not_run"
    assert _summary(paused)["pass_at_1"] is None
    assert _summary(paused)["cost_usd"] is None
    assert _summary(paused)["usage_accounting_complete"] is False
    assert load_reports(kwargs["checkpoint_dir"].parent) == []
    with pytest.raises(ValueError, match="retry-provider-hold"):
        await run_benchmark(cases, resume=True, **kwargs)
    assert len(calls) == 2
    responses.extend([result(), result()])
    completed = await run_benchmark(cases, resume=True, retry_provider_hold=True, **kwargs)
    assert len(calls) == 4 and completed.run_id == paused.run_id
    assert completed.resolved == 3 and completed.retained_attempts == 4
    assert completed.total_model_calls == 4
    assert completed.total_estimated_cost_usd == pytest.approx(0.00048)
    assert not completed.quality_metrics_valid and not completed.usage_accounting_complete
    assert completed.cost_per_resolved_usd is None
    assert completed.outcomes[0].patch_path == paused.outcomes[0].patch_path
    path = kwargs["checkpoint_dir"] / "report.json"
    with pytest.raises(ValueError, match="cannot be published"):
        build_release([("baseline", path, completed), ("candidate", path, completed)], title="No misleading score")


@pytest.mark.asyncio
async def test_completed_failed_cases_are_not_cherry_picked_or_rerun(runner):
    cases, kwargs, calls, responses, closed = runner
    responses.extend([result(resolved=False), result(), result()])
    report = await run_benchmark(cases, **kwargs)
    resumed = await run_benchmark(cases, resume=True, **kwargs)
    assert len(calls) == 3 and report.resolved == resumed.resolved == 2
    assert resumed.quality_metrics_valid and resumed.usage_accounting_complete


@pytest.mark.asyncio
async def test_localization_exposure_survives_completed_checkpoint_resume(runner):
    from code_agent.models import FaultLocalizationReceipt

    cases, kwargs, calls, responses, _closed = runner
    selected = result()
    selected.fault_localization = [FaultLocalizationReceipt(stage="implementation", iteration=0,
        status="selected", reason="authored_control")]
    abstained = result()
    abstained.fault_localization = [FaultLocalizationReceipt(stage="implementation", iteration=0,
        status="unsupported", reason="authored_control")]
    responses.extend([selected, abstained, result()])
    report = await run_benchmark(cases, **kwargs)
    resumed = await run_benchmark(cases, resume=True, **kwargs)
    assert len(calls) == 3
    assert report.fault_localized_cases == resumed.fault_localized_cases == 1
    assert report.fault_localization_status_counts == resumed.fault_localization_status_counts == {"selected": 1, "unsupported": 1}
    assert resumed.outcomes[0].fault_localization_status_counts == {"selected": 1}


@pytest.mark.asyncio
async def test_minimization_receipt_counts_survive_completed_checkpoint_resume(runner):
    from code_agent.models import (
        AnalysisPlan,
        PatchMinimizationReport,
        ReviewVerdict,
        VerificationReport,
        VerificationRound,
    )
    from code_agent.regression_challenges import fingerprint

    cases, kwargs, calls, responses, _closed = runner
    selected = result()
    proof = PatchMinimizationReport(status="unchanged", reason="authored_control")
    proof.fingerprint = fingerprint(proof.model_dump(mode="json", exclude={"fingerprint"}))
    selected.verification = VerificationReport(analysis=AnalysisPlan(summary="authored control"), rounds=[VerificationRound(round=0, verdict=ReviewVerdict(approved=True,
        summary="authored"), patch_minimization=proof)])
    responses.extend([selected, result(), result()])
    report = await run_benchmark(cases, **kwargs)
    resumed = await run_benchmark(cases, resume=True, **kwargs)
    assert len(calls) == 3
    assert report.patch_minimization_status_counts == resumed.patch_minimization_status_counts == {"unchanged": 1}
    assert report.minimization_completed_trials == resumed.minimization_completed_trials == 0


@pytest.mark.asyncio
async def test_crash_recovery_requires_explicit_retry_and_unknown_usage(runner):
    cases, kwargs, calls, responses, closed = runner
    responses.extend([result(), asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await run_benchmark(cases, **kwargs)
    assert len(closed) == 2
    with pytest.raises(ValueError, match="retry-interrupted"):
        await run_benchmark(cases, resume=True, **kwargs)
    responses.extend([result(), result()])
    report = await run_benchmark(cases, resume=True, retry_interrupted=True, **kwargs)
    assert len(calls) == 4 and report.recovery_usage_unknown
    assert not report.usage_accounting_complete and not report.quality_metrics_valid
    assert report.total_model_calls == 3  # Known calls only; prior in-flight usage unknown.


@pytest.mark.asyncio
async def test_cycle_detection_counts_survive_completed_checkpoint_resume(runner):
    from code_agent.models import AgentLoopReceipt
    cases, kwargs, calls, responses, _closed = runner
    selected = result()
    selected.loop_guard = [AgentLoopReceipt(stage="implementation", iteration=3, mode="shadow",
        disposition="shadow_limit", reason="repeated_tool_source_cycle", cycle_length=1, repetitions=3,
        source_sha256="a" * 64, witness_sha256=["b" * 64] * 3)]
    from code_agent.regression_challenges import fingerprint
    selected.loop_guard[0].fingerprint = fingerprint(selected.loop_guard[0].model_dump(mode="json", exclude={"fingerprint"}))
    responses.extend([selected, result(), result()])
    report = await run_benchmark(cases, **kwargs)
    resumed = await run_benchmark(cases, resume=True, **kwargs)
    assert len(calls) == 3
    assert report.loop_guard_status_counts == resumed.loop_guard_status_counts == {"shadow_limit": 1}
    assert resumed.outcomes[0].loop_guard_status_counts == {"shadow_limit": 1}
    assert _summary(resumed)["loop_guard_status_counts"] == {"shadow_limit": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["source", "dataset", "config", "image", "artifact", "journal", "interpreter", "packages"])
async def test_resume_rejects_drift_before_inference(runner, monkeypatch, change):
    cases, kwargs, calls, responses, closed = runner
    responses.extend([result(), result(), result()])
    report = await run_benchmark(cases, **kwargs)
    if change == "source":
        (kwargs["repository_root"] / "fixture" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    elif change == "dataset":
        cases = list(reversed(cases))
    elif change == "config":
        kwargs["default_model"] = "different-model"
    elif change == "image":
        monkeypatch.setattr("code_agent.evaluation.image_identities", lambda images: {image: "sha256:" + "b" * 64 for image in images})
    elif change == "artifact":
        (kwargs["checkpoint_dir"] / report.outcomes[0].patch_path).write_text("tampered", encoding="utf-8")
    elif change == "interpreter":
        monkeypatch.setattr("code_agent.benchmark_checkpoint.platform.python_version", lambda: "changed-runtime")
    elif change == "packages":
        monkeypatch.setattr("code_agent.benchmark_checkpoint.importlib.metadata.version", lambda name: "changed-package")
    else:
        path = kwargs["checkpoint_dir"] / "checkpoint.json"
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["state"]["run_id"] = "tampered"
        path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity|binding"):
        await run_benchmark(cases, resume=True, **kwargs)
    assert len(calls) == 3


def test_checkpoint_lock_and_containment(tmp_path):
    path = tmp_path / "private-run"
    with BenchmarkCheckpoint(path, resume=False) as first:
        first.bind("scope", "run")
        with pytest.raises(ValueError, match="already in use"):
            with BenchmarkCheckpoint(path, resume=True):
                pass
        with pytest.raises(ValueError):
            first.safe_path("../escape")
    with BenchmarkCheckpoint(path, resume=True) as second:
        assert second.bind("scope", "unused") == "run"


@pytest.mark.asyncio
async def test_cleanup_failure_leaves_attempt_interrupted(runner, monkeypatch):
    cases, kwargs, calls, responses, closed = runner
    responses.append(result())

    def failure(self):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr("code_agent.evaluation.DockerSandbox.close", failure)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await run_benchmark(cases, **kwargs)
    state = json.loads((kwargs["checkpoint_dir"] / "checkpoint.json").read_text(encoding="utf-8"))["state"]
    assert state["attempts"]["case-0"][-1]["status"] == "running"
