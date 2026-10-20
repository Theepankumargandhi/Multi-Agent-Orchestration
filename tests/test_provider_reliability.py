from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from code_agent.agent import CodingAgent
from code_agent.models import (
    AnalysisPlan,
    CodeAction,
    CodeTask,
    RepairTournamentPolicy,
    SandboxCommandResult,
    SandboxPolicy,
)
from code_agent.provider_reliability import (
    ProviderCircuitOpen,
    ProviderGuard,
    classify_provider_error,
    retry_delay,
)
from code_agent.repair_tournament import BudgetedModel, SharedModelBudget
from code_agent.verified_pr import VerifiedPRAgent
from code_agent.workspace import EphemeralWorkspace


class RateLimitError(Exception):
    def __init__(self, code="rate_limit_exceeded", status=429, retry_after=None):
        super().__init__("PRIVATE PROVIDER BODY AND CREDENTIAL")
        self.code, self.status_code = code, status
        self.response = SimpleNamespace(headers={"retry-after": retry_after} if retry_after else {})


class Sequence:
    def __init__(self, *values):
        self.values, self.calls = values, 0

    async def ainvoke(self, messages, config=None):
        value = self.values[self.calls]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return value


@pytest.mark.parametrize("code,status,category", [
    ("credit_balance_exhausted", 429, "quota"), ("insufficient_quota", 429, "quota"),
    ("organization_spend_limit_exceeded", 429, "quota"), ("invalid_api_key", 401, "authentication"),
    ("organization_usage_limit_exceeded", 429, "quota"),
    ("other", 403, "permission"), ("content_policy_violation", 400, "policy"),
    ("model_not_found", 404, "invalid_request"), ("rate_limit_exceeded", 429, "rate_limit"),
    ("server_error", 503, "transient"),
])
def test_classifier_prioritizes_structured_codes_and_redacts(code, status, category):
    failure = classify_provider_error(RateLimitError(code, status))
    assert failure.category == category
    assert "PRIVATE" not in failure.model_dump_json()
    if category not in {"transient", "rate_limit"}:
        assert failure.terminal and retry_delay(failure, 0, 100) is None


def test_nested_body_quota_and_retry_headers():
    error = RateLimitError(code="untrusted-code-secret")
    error.body = {"error": {"type": "insufficient_quota", "message": "PRIVATE"}}
    failure = classify_provider_error(error, stage="untrusted-stage")
    assert failure.category == "quota" and failure.stage == "model"
    error.body = {}
    error.response.headers["retry-after"] = "Thu, 01 Jan 2026 00:00:03 GMT"
    failure = classify_provider_error(error, now=datetime(2026, 1, 1, tzinfo=UTC))
    assert failure.code == "other" and retry_delay(failure, 0, 10) == 3
    assert retry_delay(failure, 0, 3) is None
    assert retry_delay(failure, 2, 100) is None
    error.response.headers["retry-after"] = "600"
    assert retry_delay(classify_provider_error(error), 0, 1000) is None


@pytest.mark.parametrize("value", ["not-a-date", "nan", "inf"])
def test_malformed_retry_after(value):
    assert classify_provider_error(RateLimitError(retry_after=value)).retry_after_seconds is None


def test_local_errors_are_not_provider_retry_candidates():
    assert not classify_provider_error(ValueError("PRIVATE")).retryable


@pytest.mark.asyncio
async def test_shared_hold_denies_before_budget_reservation():
    budget = SharedModelBudget(RepairTournamentPolicy())
    model = Sequence(RateLimitError("credit_balance_exhausted"))
    first = BudgetedModel(model, budget, "first")
    second = BudgetedModel(model, budget, "second")
    with pytest.raises(RateLimitError):
        await first.ainvoke([HumanMessage(content="task")])
    with pytest.raises(ProviderCircuitOpen):
        await second.ainvoke([HumanMessage(content="task")])
    assert model.calls == budget.calls == budget.guard.calls == 1
    assert "second" not in budget.by_candidate


def sandbox(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")

    class Sandbox:
        repository = root
        workspace = EphemeralWorkspace(root, SandboxPolicy())

        def run(self, command):
            return SandboxCommandResult(command=command, exit_code=0, duration_ms=1)

    value = Sandbox()
    value.workspace.prepare()
    return value


@pytest.mark.asyncio
async def test_quota_coding_call_is_not_retried(tmp_path, monkeypatch):
    waits = []

    async def sleep(delay):
        waits.append(delay)

    monkeypatch.setattr("code_agent.agent.asyncio.sleep", sleep)
    model = Sequence(RateLimitError("credit_balance_exhausted"))
    box = sandbox(tmp_path)
    try:
        result = await CodingAgent(model, capture_telemetry=False).solve(
            CodeTask(repository="repository", issue="Fix the value calculation correctly."), box)
    finally:
        box.workspace.cleanup()
    assert model.calls == result.model_calls == 1 and waits == []
    assert result.provider_failures[0].category == "quota"
    assert result.status == "failed" and not result.usage_accounting_complete


@pytest.mark.asyncio
async def test_temporary_coding_error_has_bounded_retry(tmp_path, monkeypatch):
    waits = []

    async def sleep(delay):
        waits.append(delay)

    monkeypatch.setattr("code_agent.agent.asyncio.sleep", sleep)
    model = Sequence(RateLimitError(retry_after="2"), CodeAction(kind="finish", rationale="No patch."))
    box = sandbox(tmp_path)
    try:
        result = await CodingAgent(model, capture_telemetry=False).solve(
            CodeTask(repository="repository", issue="Fix the value calculation correctly."), box)
    finally:
        box.workspace.cleanup()
    assert model.calls == result.model_calls == 2 and waits == [2]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["analysis", "implementation", "review"])
async def test_verified_workflow_stops_after_terminal_failure(tmp_path, stage):
    failure = RateLimitError("credit_balance_exhausted")
    analyst = Sequence(failure if stage == "analysis" else AnalysisPlan(summary="Inspect and fix the value."))
    action = Sequence(failure if stage == "implementation" else CodeAction(kind="finish", rationale="No patch."))
    reviewer = Sequence(failure)
    box = sandbox(tmp_path)
    try:
        result = await VerifiedPRAgent(analyst, action, reviewer).solve(
            CodeTask(repository="repository", issue="Fix the value calculation correctly."), box)
    finally:
        box.workspace.cleanup()
    assert result.status == "failed" and result.verification.final_decision == "blocked"
    assert result.provider_failures[-1].category == "quota"
    assert action.calls == (0 if stage == "analysis" else 1)
    assert reviewer.calls == (1 if stage == "review" else 0)
    assert result.model_calls == analyst.calls + action.calls + reviewer.calls


@pytest.mark.asyncio
async def test_guard_is_per_solve_not_global():
    first, second = ProviderGuard(), ProviderGuard()
    with pytest.raises(RateLimitError):
        await first.invoke(Sequence(RateLimitError("insufficient_quota")), [])
    assert await second.invoke(Sequence("healthy"), []) == "healthy"
    assert first.hold and not second.hold


@pytest.mark.asyncio
async def test_nested_guard_inherits_shared_hold_without_counting_denied_request():
    from code_agent.provider_reliability import GuardedModel

    shared, local = ProviderGuard(), ProviderGuard()
    with pytest.raises(RateLimitError):
        await shared.invoke(Sequence(RateLimitError("insufficient_quota")), [])
    with pytest.raises(ProviderCircuitOpen):
        await local.invoke(GuardedModel(Sequence("never dispatched"), shared, "analysis"), [])
    assert local.calls == 0 and local.hold and local.failures[-1].category == "quota"


@pytest.mark.asyncio
async def test_inflight_call_can_finish_but_no_new_call_after_hold():
    started, finish = asyncio.Event(), asyncio.Event()

    class Inflight:
        async def ainvoke(self, messages, config=None):
            started.set()
            await finish.wait()
            return "already in flight"

    guard = ProviderGuard()
    pending = asyncio.create_task(guard.invoke(Inflight(), []))
    await started.wait()
    with pytest.raises(RateLimitError):
        await guard.invoke(Sequence(RateLimitError("insufficient_quota")), [])
    finish.set()
    assert await pending == "already in flight"
    with pytest.raises(ProviderCircuitOpen):
        await guard.invoke(Sequence("must not run"), [])
    assert guard.calls == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_docker_checkpoint_executes_patch_and_skips_completed_case(tmp_path, monkeypatch):
    from code_agent.evaluation import run_benchmark
    from code_agent.evaluation_models import CodeBenchmarkCase
    from code_agent.sandbox import DockerSandbox

    policy = SandboxPolicy(command_timeout_seconds=30)
    if not DockerSandbox.available() or not DockerSandbox.image_available(policy.image):
        pytest.skip("Docker engine or sandbox image is unavailable")
    root = tmp_path / "repos"
    repository = root / "fixture"
    repository.mkdir(parents=True)
    (repository / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "test_app.py").write_text("from app import VALUE\ndef test_value():\n    assert VALUE == 2\n", encoding="utf-8")
    model = Sequence(CodeAction(kind="write", path="app.py", content="VALUE = 2\n", rationale="Repair fixture."),
                     CodeAction(kind="test", rationale="Execute owner test."))
    monkeypatch.setattr("code_agent.evaluation.build_coding_model", lambda name: model)
    cases = [CodeBenchmarkCase(id="real-container", repository="fixture", issue="Correct the value to satisfy the failing test.",
                               test_command=["python", "-m", "pytest", "-q"])]
    kwargs = dict(repository_root=root, default_model="scripted-no-inference", policy=policy,
                  checkpoint_dir=tmp_path / "durable")
    first = await run_benchmark(cases, **kwargs)
    second = await run_benchmark(cases, resume=True, **kwargs)
    assert first.resolved == second.resolved == 1 and model.calls == 2
    assert first.outcomes[0].score.baseline_passed is False
    assert first.outcomes[0].score.tests_passed is True
    assert second.retained_attempts == 1 and second.quality_metrics_valid
    assert (repository / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"
