import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from code_agent.agent import CodingAgent
from code_agent.loop_guard import NOTICE, AgentLoopGuard, event_digest, repeated_suffix, valid_loop_receipt
from code_agent.models import CodeAction, CodeTask, SandboxCommandResult, SandboxPolicy, ToolObservation
from code_agent.regression_challenges import fingerprint
from evals.loop_guard_evaluation import ScriptedModel, ScriptedSandbox, evaluate


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
    value = ScriptedSandbox(tmp_path, SandboxPolicy())
    try:
        yield value
    finally:
        value.workspace.cleanup()


def observation(iteration=1, output="VALUE = 0\n", kind="read", ok=True):
    return ToolObservation(iteration=iteration, action=kind, ok=ok, output=output, summary="controlled")


@pytest.mark.parametrize("history,period,repeats", [
    ([], 0, 0), (["a"], 0, 0), (["a", "a"], 1, 2), (["a"] * 20, 1, 3),
    (["a", "b"] * 3, 2, 3), (["a", "b", "c"] * 3, 3, 3), (["a", "b", "c", "d"] * 3, 4, 3),
    (["a", "b", "a", "c"], 0, 0), (["a", "b", "c", "d", "e"] * 3, 0, 0),
])
def test_bounded_suffix_matches_only_complete_short_cycles(history, period, repeats):
    assert repeated_suffix(history, 4, 3) == (period, repeats)


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_repeat_warning_stop_and_private_receipts(sandbox, mode):
    guard = AgentLoopGuard(SandboxPolicy(loop_guard_mode=mode), "implementation")
    action = CodeAction(kind="read", path="app.py", rationale="arbitrary model text")
    assert not guard.observe(sandbox.workspace, action, observation(1))
    assert not guard.observe(sandbox.workspace, action, observation(2))
    assert guard.receipts[-1].disposition == "warning"
    assert bool(guard.notice) == (mode == "enforce")
    assert guard.observe(sandbox.workspace, action, observation(3)) == (mode == "enforce")
    assert guard.receipts[-1].disposition == ("stopped" if mode == "enforce" else "shadow_limit")
    assert all(valid_loop_receipt(row) for row in guard.receipts)
    private = json.dumps([row.model_dump() for row in guard.receipts])
    assert "VALUE" not in private and "app.py" not in private and "arbitrary" not in private
    if mode == "shadow":
        for iteration in range(4, 20):
            assert not guard.observe(sandbox.workspace, action, observation(iteration))
        assert len(guard.receipts) == 2 and not guard.notice
        assert len(guard.history) <= 12


def test_distinct_output_or_source_is_not_identical_progress(sandbox):
    guard = AgentLoopGuard(SandboxPolicy(loop_guard_mode="enforce"), "repair")
    action = CodeAction(kind="read", path="app.py")
    for iteration in range(1, 8):
        sandbox.workspace.write_file("app.py", f"VALUE = {iteration}\n")
        assert not guard.observe(sandbox.workspace, action, observation(iteration))
    assert not guard.receipts
    for iteration in range(8, 15):
        assert not guard.observe(sandbox.workspace, action, observation(iteration, str(iteration)))
    assert not guard.receipts


def test_source_hash_failure_clears_prior_cycle_and_abstains(sandbox, monkeypatch):
    guard = AgentLoopGuard(SandboxPolicy(loop_guard_mode="enforce"), "repair")
    action = CodeAction(kind="read", path="app.py")
    guard.observe(sandbox.workspace, action, observation(1))
    guard.observe(sandbox.workspace, action, observation(2))
    assert guard.notice
    def fail(*args):
        raise ValueError("private path or source must not enter a receipt")
    monkeypatch.setattr("code_agent.loop_guard.file_manifest", fail)
    assert not guard.observe(sandbox.workspace, action, observation(3))
    assert not guard.history and not guard.notice
    assert guard.receipts[-1].disposition == "unavailable"
    assert valid_loop_receipt(guard.receipts[-1])
    assert "private" not in guard.receipts[-1].model_dump_json()


def test_event_identity_excludes_rationale_duration_and_unused_arguments():
    first = CodeAction(kind="read", path="app.py", content="ignored", rationale="one")
    second = CodeAction(kind="read", path="app.py", pattern="ignored", rationale="two")
    left = observation()
    right = left.model_copy(update={"duration_ms": 500, "summary": "different", "iteration": 9})
    assert event_digest(first, left, "a" * 64) == event_digest(second, right, "a" * 64)
    assert event_digest(first, left, "a" * 64) != event_digest(first, left, "b" * 64)
    assert event_digest(first, left, "a" * 64) != event_digest(first, left.model_copy(update={"ok": False}), "a" * 64)
    for kind in ("write", "search", "delete", "list"):
        assert event_digest(CodeAction(kind=kind, path="app.py", pattern="value", content="one"), left, "a" * 64)
    assert event_digest(CodeAction(kind="write", path="app.py", content="one"), left, "a" * 64) != event_digest(CodeAction(kind="write", path="app.py", content="two"), left, "a" * 64)
    assert event_digest(CodeAction(kind="search", pattern="one"), left, "a" * 64) != event_digest(CodeAction(kind="search", pattern="two"), left, "a" * 64)


def test_pytest_timing_only_normalization_preserves_failures_and_timeout():
    action = CodeAction(kind="test")
    left = observation(kind="test", ok=False, output="FAILED test_app.py::test_value\n=== 1 failed, 2 passed in 0.12s ===")
    right = left.model_copy(update={"output": left.output.replace("0.12", "99.45")})
    result = SandboxCommandResult(command=["pytest", "-q"], exit_code=1, duration_ms=1)
    assert event_digest(action, left, "a" * 64, result) == event_digest(action, right, "a" * 64, result)
    assertion = left.model_copy(update={"output": left.output.replace("test_value", "test_other")})
    assert event_digest(action, left, "a" * 64, result) != event_digest(action, assertion, "a" * 64, result)
    assert event_digest(action, left, "a" * 64, result) != event_digest(action, left, "a" * 64, result.model_copy(update={"timed_out": True}))


@pytest.mark.parametrize("field,value", [
    ("loop_guard_mode", "guess"), ("loop_guard_repetition_limit", 2),
    ("loop_guard_repetition_limit", 6), ("loop_guard_max_cycle_length", 0), ("loop_guard_max_cycle_length", 5),
])
def test_invalid_operator_policy_is_rejected(field, value):
    with pytest.raises(ValidationError):
        SandboxPolicy(**{field: value})


@pytest.mark.parametrize("change", ["seal", "witness", "reason", "mode", "iterations", "source", "length", "repeats"])
def test_malformed_or_resealed_inconsistent_receipt_rejected(sandbox, change):
    guard = AgentLoopGuard(SandboxPolicy(loop_guard_mode="enforce"), "implementation")
    for iteration in range(1, 4):
        guard.observe(sandbox.workspace, CodeAction(kind="read", path="app.py"), observation(iteration))
    row = guard.receipts[-1].model_copy(deep=True)
    if change == "seal":
        row.fingerprint = "a" * 64
    elif change == "witness":
        row.witness_sha256[0] = "a" * 64
    elif change == "reason":
        row.reason = "source_manifest_unavailable"
    elif change == "mode":
        row.mode = "shadow"
    elif change == "iterations":
        row.iteration = 1
    elif change == "source":
        row.source_sha256 = "not-a-digest"
    elif change == "length":
        row.cycle_length = 2
    else:
        row.repetitions = 1
    if change != "seal":
        row.fingerprint = fingerprint(row.model_dump(mode="json", exclude={"fingerprint"}))
    assert not valid_loop_receipt(row)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "shadow", "enforce"])
async def test_agent_loop_warnings_are_trusted_and_shadow_is_observational(sandbox, mode):
    actions = [CodeAction(kind="read", path="app.py")] * 6
    model = ScriptedModel(actions)
    policy = SandboxPolicy(loop_guard_mode=mode, max_iterations=6)
    result = await CodingAgent(model, capture_telemetry=False).solve(
        CodeTask(repository="fixture", issue="Fix the current value to satisfy owner tests.", policy=policy), sandbox)
    assert result.model_calls == (3 if mode == "enforce" else 6)
    assert result.status == ("failed" if mode == "enforce" else "budget_exhausted")
    assert result.changed_files == [] and result.final_test is None
    assert any(NOTICE in str(messages[0].content) for messages in model.messages) == (mode == "enforce")
    assert all(valid_loop_receipt(row) for row in result.loop_guard)


@pytest.mark.asyncio
async def test_shadow_prompt_is_byte_identical_to_off(sandbox):
    models = []
    for mode in ("off", "shadow"):
        sandbox.runs = 0
        model = ScriptedModel([CodeAction(kind="read", path="app.py")] * 5)
        await CodingAgent(model, capture_telemetry=False).solve(CodeTask(repository="fixture",
            issue="Fix the current value to satisfy owner tests.", policy=SandboxPolicy(loop_guard_mode=mode, max_iterations=5)), sandbox)
        # Durations are independently measured by the existing agent, not this guard.
        payloads = []
        for messages in model.messages:
            payload = json.loads(messages[1].content)
            for item in payload["recent_observations"]:
                item["duration_ms"] = 0
            payloads.append([messages[0].content, payload])
        models.append(payloads)
    assert models[0] == models[1]


@pytest.mark.asyncio
async def test_authored_comparisons_are_not_live_performance():
    report = await evaluate()
    assert report["total"] == report["passed"] == 8
    assert report["provider_calls"] == 0 and not report["repository_code_executed"] and not report["live_quality_measured"]
    repeat = report["scenarios"][0]["arms"]
    assert repeat["off"]["scripted_model_calls"] == 12 and repeat["enforce"]["scripted_model_calls"] == 3


def test_frozen_matrix_changes_only_guard_mode():
    from code_agent.benchmark_release import BenchmarkMatrix, DatasetLock
    from code_agent.evaluation import dataset_fingerprint, load_code_cases
    path = Path("evals/experiments/loop_guard_matrix.json")
    matrix = BenchmarkMatrix.model_validate_json(path.read_text())
    assert [row.loop_guard_mode for row in matrix.variants] == ["off", "shadow", "enforce"]
    assert all(row.model_dump(exclude={"name", "loop_guard_mode"}) == matrix.variants[0].model_dump(exclude={"name", "loop_guard_mode"}) for row in matrix.variants)
    cases = load_code_cases((path.parent / matrix.dataset).resolve())[:matrix.max_cases]
    lock = DatasetLock.model_validate_json((path.parent / matrix.dataset_lock).read_text())
    assert len(cases) == 5 and dataset_fingerprint(cases) == lock.dataset_fingerprint


@pytest.mark.asyncio
async def test_verified_pr_retains_warning_and_reverifies_after_stage_stop(tmp_path):
    from test_verified_pr import FakeSandbox, SequenceModel, analysis, approval, repository, task

    from code_agent.verified_pr import VerifiedPRAgent
    policy = SandboxPolicy(loop_guard_mode="enforce", max_repair_rounds=0)
    sandbox = FakeSandbox(repository(tmp_path), policy)
    actions = SequenceModel([CodeAction(kind="write", path="app.py", content="VALUE = 2\n"),
                             *[CodeAction(kind="read", path="app.py")] * 3,
                             CodeAction(kind="test")])
    try:
        result = await VerifiedPRAgent(SequenceModel([analysis()]), actions, SequenceModel([approval()])).solve(task(policy), sandbox)
        assert result.status == "completed"  # Ordinary gates decide; a stage stop is not a veto or approval.
        assert [row.disposition for row in result.loop_guard] == ["warning", "stopped"]
        assert result.final_test.exit_code == 0 and result.verification.final_decision == "verified"
        assert actions.calls == 4  # Test author skipped after failed implementation stage.
    finally:
        sandbox.close()


@pytest.mark.asyncio
async def test_tournament_retains_content_free_losing_team_receipts(tmp_path, monkeypatch):
    from test_repair_tournament import ControlledTeam, code_task, fixture_repository, tournament
    original = ControlledTeam.solve
    async def solve(self, task, sandbox):
        result = await original(self, task, sandbox)
        guard = AgentLoopGuard(task.policy.model_copy(update={"loop_guard_mode": "enforce"}), "implementation")
        for iteration in range(1, 4):
            guard.observe(sandbox.workspace, CodeAction(kind="read", path="app.py"), observation(iteration))
        result.loop_guard = guard.receipts
        return result
    monkeypatch.setattr(ControlledTeam, "solve", solve)
    result = await tournament(options={"candidate-2": {"extra": True}}).solve(code_task(loop_guard_mode="enforce"), fixture_repository(tmp_path))
    assert result.tournament.winner_id == "candidate-1"
    assert len(result.loop_guard) == 4
    assert {row.candidate_id for row in result.loop_guard} == {"candidate-1", "candidate-2"}
    assert all(valid_loop_receipt(row) for row in result.loop_guard)


def test_custom_repeat_limit_and_cycle_cap(sandbox):
    guard = AgentLoopGuard(SandboxPolicy(loop_guard_mode="enforce", loop_guard_repetition_limit=5,
                                        loop_guard_max_cycle_length=1), "repair")
    for iteration in range(1, 6):
        assert guard.observe(sandbox.workspace, CodeAction(kind="read", path="app.py"), observation(iteration)) == (iteration == 5)
    assert guard.receipts[-1].repetition_limit == 5 and all(valid_loop_receipt(row) for row in guard.receipts)


def test_dossier_preserves_sealed_controller_evidence(sandbox):
    from code_agent.job_models import ArtifactReference, CodeTaskRequest, JobRecord
    from code_agent.models import CodeAgentResult
    from code_agent.verification import build_pr_dossier
    guard = AgentLoopGuard(SandboxPolicy(loop_guard_mode="enforce"), "repair")
    for iteration in range(1, 4):
        guard.observe(sandbox.workspace, CodeAction(kind="read", path="app.py"), observation(iteration))
    result = CodeAgentResult(status="failed", summary="Held", loop_guard=guard.receipts)
    job = JobRecord(task_id="authored", user_id="user", status="failed", created_at="now", updated_at="now",
        request=CodeTaskRequest(repository="fixture", issue="Fix the authored current value."))
    payload, markdown = build_pr_dossier(job, result, ArtifactReference(sha256="a" * 64, size_bytes=0))
    assert json.loads(payload)["loop_guard"] == [row.model_dump(mode="json") for row in guard.receipts]
    assert "Agent cycle detection" in markdown and "enforce/stopped" in markdown
    assert "VALUE = 0" not in payload and "VALUE = 0" not in markdown


def test_release_card_preserves_cycle_counts(tmp_path):
    from test_benchmark_release import _report, _write_report

    from code_agent.benchmark_release import benchmark_card, build_release
    baseline = _report("baseline", [False, True], cost=0.1, p95=100)
    candidate = _report("candidate", [False, True], cost=0.1, p95=100)
    candidate.loop_guard_status_counts = {"warning": 1, "stopped": 1}
    _write_report(tmp_path / "base.json", baseline)
    _write_report(tmp_path / "candidate.json", candidate)
    release = build_release([("baseline", tmp_path / "base.json", baseline), ("candidate", tmp_path / "candidate.json", candidate)],
                            title="authored paired controls", baseline_name="baseline")
    assert next(row for row in release.variants if row.name == "candidate").loop_guard_status_counts == candidate.loop_guard_status_counts
    assert "Agent cycle detection" in benchmark_card(release) and "stopped=1" in benchmark_card(release)
