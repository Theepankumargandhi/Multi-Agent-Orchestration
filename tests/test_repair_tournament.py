from __future__ import annotations

import asyncio
import json
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from code_agent.models import (
    AnalysisPlan,
    CodeAction,
    CodeAgentResult,
    CodeTask,
    RepairTournamentPolicy,
    ReviewFinding,
    ReviewVerdict,
    SandboxCommandResult,
    SandboxPolicy,
    VerificationReport,
    VerificationRound,
)
from code_agent.repair_tournament import (
    BudgetedModel,
    ModelBudgetExceeded,
    RepairTournament,
    SharedModelBudget,
    eligibility_reasons,
    verify_tournament_report,
)
from code_agent.security_policy import summarize_security
from code_agent.verification import repository_fingerprint, run_quality_gates
from code_agent.workspace import EphemeralWorkspace


class ControlledModel:
    def __init__(self, value=None, *, error=False, delay=0, usage=True):
        self.value, self.error, self.delay, self.usage = value, error, delay, usage
        self.calls = 0
        self.messages = []

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        self.messages.append(messages)
        await asyncio.sleep(self.delay)
        if self.error:
            raise RuntimeError("provider unavailable")
        if self.usage:
            generation = LLMResult(generations=[[ChatGeneration(message=AIMessage(
                content="structured response", response_metadata={"model_name": "authored-model"},
                usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
            ))]])
            for callback in (config or {}).get("callbacks", []):
                callback.on_llm_end(generation)
        return self.value


class ControlledSandbox:
    """No host execution: authored command outcomes, real confined workspace/diffs/gates."""
    def __init__(self, repository, policy, *, fail_fresh_test=False, startup_delay=0):
        self.repository, self.policy = repository, policy
        self.workspace = EphemeralWorkspace(repository, policy)
        self.closed = False
        self.fail_fresh_test, self.startup_delay = fail_fresh_test, startup_delay
        self.owner_runs = 0

    def start(self):
        time.sleep(self.startup_delay)
        self.workspace.prepare()
        return self

    def run(self, command):
        owner = command[:4] == ["python", "-m", "pytest", "-q"]
        if owner:
            self.owner_runs += 1
        if ".agentforge-coverage.json" in command:
            self.workspace.write_file(".agentforge-coverage.json", '{"files":{"app.py":{"executed_lines":[1]}}}')
        exit_code = 1 if owner and self.fail_fresh_test and self.owner_runs > 1 else 0
        return SandboxCommandResult(command=command, exit_code=exit_code, stdout="authored outcome", duration_ms=1)

    def close(self):
        self.workspace.cleanup()
        self.closed = True


class ControlledTeam:
    def __init__(self, candidate_id, budget, *, value=2, extra=False, delay=0, error=False, damaged=""):
        self.id, self.budget, self.value, self.extra = candidate_id, budget, value, extra
        self.delay, self.error, self.damaged = delay, error, damaged

    async def solve(self, task, sandbox):
        model = BudgetedModel(ControlledModel(error=self.error, delay=self.delay), self.budget, self.id)
        await model.ainvoke([HumanMessage(content="Plan a bounded repair")])
        before = repository_fingerprint(sandbox.repository, task.policy.max_files, task.policy.max_repository_bytes)
        sandbox.workspace.write_file("app.py", f"VALUE = {self.value}\n")
        if self.extra:
            sandbox.workspace.write_file("extra.py", "OTHER = 4\n")
        patch, changed = sandbox.workspace.unified_diff(), sandbox.workspace.changed_files()
        gates, final = await run_quality_gates(task, sandbox, patch=patch, changed_files=changed,
                                              source_fingerprint_before=before, baseline_lint=None)
        result = CodeAgentResult(status="completed", summary="Authored candidate", patch=patch, changed_files=changed,
            final_test=final, writes=1 + self.extra, model_calls=1, prompt_tokens=20, completion_tokens=10,
            verification=VerificationReport(analysis=AnalysisPlan(summary="Authored plan"), final_decision="verified",
                rounds=[VerificationRound(round=0, verdict=ReviewVerdict(approved=True, summary="Authored approval"),
                                          quality_gates=gates)]),
            security=summarize_security([], task.policy.security_policy_version))
        if self.damaged == "patch":
            result.patch += "stale patch"
        elif self.damaged == "gates":
            result.verification.rounds[-1].quality_gates = []
        elif self.damaged == "test":
            result.final_test.command = ["python", "unapproved.py"]
        elif self.damaged == "writes":
            result.writes = task.policy.max_workflow_writes + 1
        elif self.damaged == "source":
            sandbox.repository.joinpath("app.py").write_text("VALUE = 99\n")
        return result


def fixture_repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "app.py").write_text("VALUE = 1\n")
    return root


def code_task(**policy):
    return CodeTask(repository="fixture", issue="Change VALUE from one to two with regression coverage.",
                    model="authored-model", policy=SandboxPolicy(max_repair_rounds=0, **policy))


def tournament(*, options=None, challenge=None, sandboxes=None, sandbox_options=None, **policy):
    options, sandbox_options = options or {}, sandbox_options or {}
    sandboxes = sandboxes if sandboxes is not None else []

    def sandbox_factory(repository, branch_policy):
        sandbox = ControlledSandbox(repository, branch_policy, **sandbox_options)
        sandboxes.append(sandbox)
        return sandbox

    return RepairTournament(lambda candidate_id, _, budget: ControlledTeam(candidate_id, budget, **options.get(candidate_id, {})),
        challenge or ControlledModel(ReviewVerdict(approved=True, summary="No supported counterexample in supplied evidence.")),
        sandbox_factory=sandbox_factory, policy=RepairTournamentPolicy(candidates=2, max_parallel=2, **policy),
        input_cost_per_million=1, output_cost_per_million=2)


@pytest.mark.asyncio
async def test_isolated_teams_challenge_and_fresh_execution_select_smallest_patch(tmp_path):
    root = fixture_repository(tmp_path)
    sandboxes = []
    result = await tournament(options={"candidate-2": {"extra": True}}, sandboxes=sandboxes).solve(code_task(), root)
    assert result.status == "completed" and result.tournament.winner_id == "candidate-1"
    assert verify_tournament_report(result.tournament)
    assert result.tournament.source_unchanged and root.joinpath("app.py").read_text() == "VALUE = 1\n"
    assert "extra.py" not in result.patch
    assert len({sandbox.workspace.source for sandbox in sandboxes}) == 1
    assert all(sandbox.closed and sandbox.workspace.root is None and sandbox.owner_runs == 2 for sandbox in sandboxes)
    assert result.model_calls == 4 and result.prompt_tokens == 80 and result.completion_tokens == 40
    assert result.estimated_cost_usd == pytest.approx(0.00016)
    assert result.writes == 3 and result.tournament.usage_accounting_complete
    assert all(candidate.eligible and candidate.challenge_approved for candidate in result.tournament.candidates)
    assert all(candidate.final_gate_statuses["mandatory_tests"] == "passed" for candidate in result.tournament.candidates)


@pytest.mark.asyncio
@pytest.mark.parametrize("damaged", ["patch", "gates", "test", "writes"])
async def test_unsupported_candidate_claims_never_win(tmp_path, damaged):
    root = fixture_repository(tmp_path)
    result = await tournament(options={"candidate-1": {"damaged": damaged}}).solve(code_task(), root)
    assert not result.tournament.candidates[0].eligible
    assert result.tournament.winner_id == "candidate-2"


@pytest.mark.asyncio
async def test_approved_with_high_findings_is_blocked(tmp_path):
    review = ReviewVerdict(approved=True, summary="Looks good", findings=[ReviewFinding(
        severity="high", category="correctness", message="Boundary case lacks evidence.")])
    result = await tournament(challenge=ControlledModel(review)).solve(code_task(), fixture_repository(tmp_path))
    assert result.status == "failed" and not result.patch
    assert all(not candidate.eligible and candidate.challenge_blocking_findings == 1 for candidate in result.tournament.candidates)


@pytest.mark.asyncio
async def test_review_failure_and_fresh_test_failure_fail_closed(tmp_path):
    root = fixture_repository(tmp_path)
    result = await tournament(challenge=ControlledModel(error=True)).solve(code_task(), root)
    assert result.status == "failed" and not result.patch
    result = await tournament(sandbox_options={"fail_fresh_test": True}).solve(code_task(), root)
    assert result.status == "failed" and all("fresh_execution_gates_failed" in row.reasons for row in result.tournament.candidates)


@pytest.mark.asyncio
async def test_timeout_cleans_all_workspaces_without_selecting_late_results(tmp_path):
    sandboxes = []
    result = await tournament(options={"candidate-1": {"delay": 1}, "candidate-2": {"delay": 1}},
                              sandboxes=sandboxes, candidate_timeout_seconds=0.03).solve(code_task(), fixture_repository(tmp_path))
    assert result.status == "failed" and not result.patch
    assert all(row.status == "timed_out" for row in result.tournament.candidates)
    assert all(sandbox.closed and sandbox.workspace.root is None for sandbox in sandboxes)


@pytest.mark.asyncio
async def test_cancellation_during_startup_waits_then_cleans(tmp_path):
    sandboxes = []
    solver = tournament(sandboxes=sandboxes, sandbox_options={"startup_delay": 0.05})
    pending = asyncio.create_task(solver.solve(code_task(), fixture_repository(tmp_path)))
    await asyncio.sleep(0.02)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert sandboxes and all(sandbox.closed and sandbox.workspace.root is None for sandbox in sandboxes)


@pytest.mark.asyncio
async def test_provider_calls_are_reserved_atomically_before_dispatch():
    policy = RepairTournamentPolicy(max_model_calls=2)
    budget = SharedModelBudget(policy)
    provider = ControlledModel()
    models = [BudgetedModel(provider, budget, str(number)) for number in range(10)]
    results = await asyncio.gather(*(model.ainvoke([HumanMessage(content="x")]) for model in models), return_exceptions=True)
    assert provider.calls == budget.calls == 2 and budget.denied == 8
    assert sum(isinstance(result, ModelBudgetExceeded) for result in results) == 8


@pytest.mark.asyncio
async def test_failed_calls_consume_budget_and_prompt_limit_prevents_dispatch():
    budget = SharedModelBudget(RepairTournamentPolicy(max_model_calls=1))
    provider = ControlledModel(error=True)
    model = BudgetedModel(provider, budget, "one")
    with pytest.raises(RuntimeError):
        await model.ainvoke([HumanMessage(content="first")])
    with pytest.raises(ModelBudgetExceeded):
        await model.ainvoke([HumanMessage(content="second")])
    assert provider.calls == 1 and budget.calls_with_usage == 0
    provider = ControlledModel()
    model = BudgetedModel(provider, SharedModelBudget(RepairTournamentPolicy(max_estimated_prompt_tokens=64)), "one")
    with pytest.raises(ModelBudgetExceeded):
        await model.ainvoke([HumanMessage(content="x" * 1000)])
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_source_mutation_blocks_all_candidates(tmp_path):
    result = await tournament(options={"candidate-1": {"damaged": "source"}}).solve(code_task(), fixture_repository(tmp_path))
    assert result.status == "failed" and not result.patch and not result.tournament.source_unchanged


@pytest.mark.asyncio
async def test_one_candidate_error_does_not_discard_healthy_candidate(tmp_path):
    result = await tournament(options={"candidate-1": {"error": True}}).solve(code_task(), fixture_repository(tmp_path))
    assert result.status == "completed" and result.tournament.winner_id == "candidate-2"
    assert result.model_calls == 3 and not result.tournament.usage_accounting_complete


@pytest.mark.asyncio
async def test_insufficient_writes_rejects_before_starting(tmp_path):
    sandboxes = []
    with pytest.raises(ValueError, match="at least two"):
        await tournament(sandboxes=sandboxes).solve(code_task(max_workflow_writes=3), fixture_repository(tmp_path))
    assert not sandboxes


@pytest.mark.asyncio
async def test_real_verified_pr_team_integrates_with_tournament(tmp_path):
    from code_agent.verified_pr import VerifiedPRAgent
    from tests.test_verified_pr import SequenceModel

    root = fixture_repository(tmp_path)

    def factory(candidate_id, instructions, budget):
        actions = SequenceModel([CodeAction(kind="write", path="app.py", content="VALUE = 2\n"),
                                 CodeAction(kind="test"), CodeAction(kind="test")])
        return VerifiedPRAgent(BudgetedModel(ControlledModel(AnalysisPlan(summary="Change value.")), budget, candidate_id),
                              BudgetedModel(actions, budget, candidate_id, instructions),
                              BudgetedModel(ControlledModel(ReviewVerdict(approved=True, summary="Supported by gates.")), budget, candidate_id))

    solver = RepairTournament(factory, ControlledModel(ReviewVerdict(approved=True, summary="No blocker.")),
                              policy=RepairTournamentPolicy(candidates=1, max_parallel=1), sandbox_factory=ControlledSandbox)
    result = await solver.solve(code_task(), root)
    assert result.status == "completed" and result.tournament.winner_id == "candidate-1"
    assert result.model_calls == 6 and not result.tournament.usage_accounting_complete


def test_policy_validation_rejects_unknown_and_unbounded_settings():
    for values in ({"candidates": 5}, {"candidates": 1, "max_parallel": 2}, {"extra": "unsafe"},
                   {"candidate_timeout_seconds": float("nan")}, {"max_model_calls": 0}):
        with pytest.raises(ValueError):
            RepairTournamentPolicy(**values)


@pytest.mark.asyncio
async def test_challenge_timeout_is_identified_and_withholds_patch(tmp_path):
    result = await tournament(challenge=ControlledModel(ReviewVerdict(approved=True, summary="Late approval"), delay=0.2),
                              challenge_timeout_seconds=0.02).solve(code_task(), fixture_repository(tmp_path))
    assert result.status == "failed" and not result.patch
    assert all("challenge_deadline_exceeded" in candidate.reasons for candidate in result.tournament.candidates)


@pytest.mark.asyncio
async def test_global_deadline_withholds_even_early_eligible_candidates(tmp_path, monkeypatch):
    original_wait = asyncio.wait

    async def accelerated_deadline(tasks, *, timeout):
        return await original_wait(tasks, timeout=min(timeout, 0.1))

    monkeypatch.setattr("code_agent.repair_tournament.asyncio.wait", accelerated_deadline)
    result = await tournament(options={"candidate-2": {"delay": 0.5}}).solve(code_task(), fixture_repository(tmp_path))
    assert result.status == "failed" and not result.patch
    assert "tournament_deadline_exceeded" in result.tournament.blocking_reasons


@pytest.mark.asyncio
async def test_duplicate_eligible_patches_are_reported_not_claimed_as_diversity(tmp_path):
    result = await tournament().solve(code_task(), fixture_repository(tmp_path))
    assert result.tournament.unique_eligible_patches == 1
    assert sum(candidate.eligible for candidate in result.tournament.candidates) == 2
    assert result.tournament.winner_id == "candidate-1"


@pytest.mark.asyncio
async def test_invalid_factory_cannot_reuse_the_frozen_source_workspace(tmp_path):
    class BadSandbox(ControlledSandbox):
        def start(self):
            self.workspace.root = self.repository

        def close(self):
            self.closed = True

    solver = tournament()
    solver.sandbox_factory = BadSandbox
    result = await solver.solve(code_task(), fixture_repository(tmp_path))
    assert result.status == "failed" and not result.patch
    assert all(not candidate.eligible and "candidate_error:ValueError" in candidate.reasons for candidate in result.tournament.candidates)


@pytest.mark.asyncio
async def test_oversized_challenge_input_is_not_silently_truncated():
    result = CodeAgentResult(status="completed", summary="candidate", patch="x" * 100001,
        verification=VerificationReport(analysis=AnalysisPlan(summary="plan"), rounds=[VerificationRound(
            round=0, verdict=ReviewVerdict(approved=True, summary="approved"))]))
    with pytest.raises(ValueError, match="bounded input"):
        await tournament()._challenge(code_task(), result, ControlledModel(), None)


def test_nonfinite_costs_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        RepairTournament(lambda *_: None, ControlledModel(), input_cost_per_million=float("inf"))


def test_production_factory_wraps_all_specialist_models_and_keeps_explicit_policy(monkeypatch):
    from code_agent.repair_tournament import build_repair_tournament
    from code_agent.verified_pr import VerifiedPRAgent

    seen = []

    class Base:
        def with_structured_output(self, schema):
            return ControlledModel(ReviewVerdict(approved=True, summary="configured"))

    def team_builder(model, *, context_strategy=None):
        seen.append((model, context_strategy))
        return VerifiedPRAgent(ControlledModel(), ControlledModel(), ControlledModel())

    monkeypatch.setattr("code_agent.agent.build_chat_model", lambda _: Base())
    monkeypatch.setattr("code_agent.verified_pr.build_verified_pr_agent", team_builder)
    policy = RepairTournamentPolicy(candidates=1, max_parallel=1)
    solver = build_repair_tournament("configured-model", policy=policy, context_strategy="lexical")
    budget = SharedModelBudget(policy)
    team = solver.candidate_factory("candidate-1", "Minimal scope", budget)
    assert solver.policy == policy and seen == [("configured-model", "lexical")]
    assert all(isinstance(model, BudgetedModel) and model.budget is budget
               for model in (team.analysis_model, team.action_model, team.review_model))


def test_environment_policy_is_validated_without_activation(monkeypatch):
    from code_agent.repair_tournament import tournament_policy_from_environment

    monkeypatch.setenv("CODE_TOURNAMENT_CANDIDATES", "4")
    monkeypatch.setenv("CODE_TOURNAMENT_MAX_PARALLEL", "2")
    monkeypatch.setenv("CODE_TOURNAMENT_MAX_MODEL_CALLS", "40")
    assert tournament_policy_from_environment().candidates == 4
    assert tournament_policy_from_environment().max_model_calls == 40
    monkeypatch.setenv("CODE_TOURNAMENT_CANDIDATES", "500")
    with pytest.raises(ValueError):
        tournament_policy_from_environment()


def test_receipt_round_trip_and_tamper_detection(tmp_path):
    from code_agent.models import RepairCandidateEvidence, RepairTournamentReport
    from code_agent.repair_tournament import report_fingerprint

    report = RepairTournamentReport(policy=RepairTournamentPolicy(candidates=1, max_parallel=1),
        task_sha256="task", source_sha256="source", candidates=[RepairCandidateEvidence(candidate_id="candidate-1", perspective="minimal")])
    report.fingerprint = report_fingerprint(report)
    restored = RepairTournamentReport.model_validate_json(report.model_dump_json())
    assert verify_tournament_report(restored)
    restored.winner_id = "made-up"
    assert not verify_tournament_report(restored)
    restored.fingerprint = report_fingerprint(restored)
    assert not verify_tournament_report(restored)


def test_missing_verification_fails_without_trusting_summary():
    result = CodeAgentResult(status="completed", summary="All good", patch="patch", changed_files=["app.py"])
    assert "verification_not_passed" in eligibility_reasons(code_task(), result, "patch", ["app.py"])


@pytest.mark.asyncio
async def test_persistence_dossier_keeps_tournament_and_approval_gate(tmp_path):
    from code_agent.execution import JobStateError
    from code_agent.job_models import CodeTaskRequest
    from tests.test_code_execution import make_store

    result = await tournament().solve(code_task(), fixture_repository(tmp_path))
    store = make_store(tmp_path)
    queued, _ = store.enqueue("owner", CodeTaskRequest(repository="fixture", issue=code_task().issue))
    store.claim("worker")
    completed = store.complete(queued.task_id, "worker", result)
    assert completed.status == "awaiting_approval" and completed.result.patch == ""
    dossier = json.loads(store.dossier(queued.task_id, "owner", "json"))
    assert dossier["repair_tournament"]["winner_id"] == "candidate-1"
    assert "Repair tournament" in store.dossier(queued.task_id, "owner", "markdown")
    assert result.patch not in json.dumps(dossier)
    with pytest.raises(JobStateError):
        store.approved_patch(queued.task_id, "owner")
