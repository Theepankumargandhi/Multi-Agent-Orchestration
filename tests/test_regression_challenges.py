"""Authored controls do not execute repository functions or generated code on the host."""

import ast
import asyncio
import json

import pytest

from code_agent.models import RegressionChallengePolicy, RepairTournamentPolicy, SandboxCommandResult
from code_agent.regression_challenges import (
    PROBE_COMMAND,
    RECEIPT_PATH,
    RUNNER_PATH,
    BehavioralProbe,
    BehavioralProbeSuite,
    candidate_matches,
    execute_suite,
    fingerprint,
    generate_suite,
    runner_source,
    validate_suite,
)
from code_agent.repair_tournament import RepairTournament, report_fingerprint, verify_tournament_report
from tests.test_repair_tournament import (
    ControlledModel,
    ControlledSandbox,
    ControlledTeam,
    code_task,
    fixture_repository,
)


def suite(expected=3):
    return BehavioralProbeSuite(probes=[BehavioralProbe(target="app.value", expected=expected,
        rationale="Authored expected output for the requested public contract.")])


class ProbeSandbox(ControlledSandbox):
    """Scripted command receipts; NOT execution of the trusted harness or repository."""

    def __init__(self, *args, behavior="normal", **kwargs):
        super().__init__(*args, **kwargs)
        self.behavior, self.probe_runs = behavior, 0

    def run(self, command):
        if command != PROBE_COMMAND:
            return super().run(command)
        self.probe_runs += 1
        assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in
            ast.parse(self.workspace.read_file(RUNNER_PATH)).body[:3]}
        probes = json.loads(assignments["PAYLOAD"])["probes"]
        value = int(self.workspace.read_file("app.py").split("=")[1].strip())
        statuses = ["matched" if value == probe["expected"] else "mismatched" for probe in probes]
        if self.behavior == "error":
            statuses = ["error"] * len(probes)
        elif self.behavior == "unstable" and self.probe_runs == 2:
            statuses = ["mismatched" if status == "matched" else "matched" for status in statuses]
        elif self.behavior == "mutation":
            self.workspace.write_file("unapproved.py", "VALUE = 0\n")
        elif self.behavior == "harness_tamper":
            self.workspace.write_file(RUNNER_PATH, "# replaced\n")
        receipt = {"suite_sha256": assignments["SUITE_HASH"], "statuses": statuses}
        if self.behavior == "forged_hash":
            receipt["suite_sha256"] = "different-suite"
        elif self.behavior == "bad_shape":
            receipt["statuses"] = []
        self.workspace.write_file(RECEIPT_PATH, json.dumps(receipt))
        return SandboxCommandResult(command=command, exit_code=124 if self.behavior == "timeout" else 0,
            timed_out=self.behavior == "timeout", duration_ms=1)


def solver(*, designer=None, behaviors=None, expected=3, max_calls=64):
    sandboxes = []
    behaviors = behaviors or {}

    def factory(repository, policy):
        sandbox = ProbeSandbox(repository, policy, behavior=behaviors.get(len(sandboxes), "normal"))
        sandboxes.append(sandbox)
        return sandbox

    from code_agent.models import ReviewVerdict
    designer = designer or ControlledModel(suite(expected))
    policy = RepairTournamentPolicy(candidates=2, max_parallel=2, max_model_calls=max_calls,
        regression_challenges=RegressionChallengePolicy(allowed_targets=["app.value"]))
    repair = RepairTournament(lambda cid, _, budget: ControlledTeam(cid, budget, value=2 if cid == "candidate-1" else 3),
        ControlledModel(ReviewVerdict(approved=True, summary="Authored approval.")),
        policy=policy, sandbox_factory=factory, regression_model=designer)
    return repair, designer, sandboxes


@pytest.mark.asyncio
async def test_independent_common_suite_exposes_weak_patch_and_preserves_source(tmp_path):
    root = fixture_repository(tmp_path)
    repair, designer, sandboxes = solver()
    result = await repair.solve(code_task(), root)
    assert result.status == "completed" and result.tournament.winner_id == "candidate-2"
    assert verify_tournament_report(result.tournament)
    assert result.model_calls == 5 and result.tournament.shared_model_calls == 1
    assert result.tournament.regression_baseline.statuses == ["mismatched"]
    first, second = result.tournament.candidates
    assert first.regression_probes.statuses == ["mismatched"] and not first.eligible
    assert candidate_matches(second.regression_probes) and second.eligible
    assert first.regression_probes.suite_sha256 == second.regression_probes.suite_sha256 == result.tournament.regression_suite_sha256
    payload = json.loads(designer.messages[0][-1].content)
    assert payload["baseline_sources"]["app.py"].replace("\r\n", "\n") == "VALUE = 1\n"
    assert set(payload) == {"issue", "allowed_targets", "max_probes", "baseline_sources"}
    assert "regression-probe" not in result.patch and "regression-result" not in result.patch
    assert root.joinpath("app.py").read_text() == "VALUE = 1\n"
    assert len(sandboxes) == 3 and all(s.closed and s.probe_runs == 2 for s in sandboxes)
    assert result.prompt_tokens == 100 and result.completion_tokens == 50
    from code_agent.execution import JobStateError
    from code_agent.job_models import CodeTaskRequest
    from tests.test_code_execution import make_store

    store = make_store(tmp_path)
    job, _ = store.enqueue("owner", CodeTaskRequest(repository="fixture", issue=code_task().issue))
    store.claim("worker")
    completed = store.complete(job.task_id, "worker", result)
    assert completed.status == "awaiting_approval" and completed.result.patch == ""
    dossier = json.loads(store.dossier(job.task_id, "owner", "json"))
    assert dossier["repair_tournament"]["regression_suite"] == suite().model_dump(mode="json")
    assert "Pre-patch regression probes" in store.dossier(job.task_id, "owner", "markdown")
    with pytest.raises(JobStateError):
        store.approved_patch(job.task_id, "owner")


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["error", "unstable", "mutation", "harness_tamper", "forged_hash", "bad_shape", "timeout"])
async def test_invalid_baseline_holds_before_repair_calls_and_cleans(tmp_path, behavior):
    repair, _, sandboxes = solver(behaviors={0: behavior})
    result = await repair.solve(code_task(), fixture_repository(tmp_path))
    assert result.status == "failed" and not result.patch
    assert result.model_calls == result.tournament.shared_model_calls == 1
    assert verify_tournament_report(result.tournament)
    assert len(sandboxes) == 1 and sandboxes[0].closed
    assert all(not row.eligible for row in result.tournament.candidates)


@pytest.mark.asyncio
async def test_baseline_passing_probes_are_not_claimed_as_discriminating(tmp_path):
    repair, _, sandboxes = solver(expected=1)
    result = await repair.solve(code_task(), fixture_repository(tmp_path))
    assert result.status == "failed" and not result.patch
    assert "regression_baseline_invalid_or_nondiscriminating" in result.tournament.blocking_reasons
    assert len(sandboxes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["error", "unstable", "mutation", "harness_tamper", "forged_hash", "bad_shape", "timeout"])
async def test_candidate_probe_failure_cannot_publish_patch(tmp_path, behavior):
    repair, _, sandboxes = solver(behaviors={1: behavior, 2: behavior})
    result = await repair.solve(code_task(), fixture_repository(tmp_path))
    assert result.status == "failed" and not result.patch and verify_tournament_report(result.tournament)
    assert all(s.closed for s in sandboxes)


@pytest.mark.asyncio
async def test_invalid_generation_and_shared_budget_fail_closed(tmp_path):
    root = fixture_repository(tmp_path)
    for designer in (ControlledModel(error=True), ControlledModel({"probes": []}), ControlledModel(suite().model_copy(
        update={"probes": [BehavioralProbe(target="os.system", expected=0, rationale="Not allowed.")]}))):
        repair, _, sandboxes = solver(designer=designer)
        result = await repair.solve(code_task(), root)
        assert result.status == "failed" and not result.patch and not sandboxes
        assert result.model_calls == 1 and verify_tournament_report(result.tournament)
    repair, _, sandboxes = solver(max_calls=1)
    result = await repair.solve(code_task(), root)
    assert result.status == "failed" and result.model_calls == 1 and result.tournament.denied_model_calls == 2
    assert all(s.closed for s in sandboxes)


@pytest.mark.asyncio
async def test_generation_deadline_and_cancellation_do_not_start_repairs(tmp_path):
    repair, _, sandboxes = solver(designer=ControlledModel(suite(), delay=0.2))
    repair.policy.regression_challenges.generation_timeout_seconds = 0.01
    result = await repair.solve(code_task(), fixture_repository(tmp_path))
    assert result.status == "failed" and not sandboxes
    repair, _, sandboxes = solver(designer=ControlledModel(suite(), delay=1))
    (tmp_path / "second").mkdir()
    pending = asyncio.create_task(repair.solve(code_task(), fixture_repository(tmp_path / "second")))
    await asyncio.sleep(0.05)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not sandboxes


def test_public_targets_and_json_limits_are_operator_owned():
    for targets in (["os._exit"], ["a.b", "a.b"], ["../app.value"], ["__builtins__.eval"]):
        with pytest.raises(ValueError):
            RegressionChallengePolicy(allowed_targets=targets)
    for values in ({"expected": float("nan")}, {"args": ["x" * 7000]}, {"kwargs": {"__secret": 1}},
                   {"relation": "same_result", "expected": 3}, {"second_args": [1]}):
        with pytest.raises(ValueError):
            BehavioralProbe(target="app.value", rationale="Authored case", **values)
    nested = None
    for _ in range(10):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting"):
        BehavioralProbe(target="app.value", expected=nested, rationale="Deep JSON")
    duplicate = BehavioralProbeSuite(probes=[suite().probes[0], suite().probes[0]])
    with pytest.raises(ValueError, match="duplicate"):
        validate_suite(duplicate, RegressionChallengePolicy(allowed_targets=["app.value"]))
    with pytest.raises(ValueError, match="too many"):
        validate_suite(BehavioralProbeSuite(probes=[suite(2).probes[0], suite(3).probes[0]]),
                       RegressionChallengePolicy(allowed_targets=["app.value"], max_probes=1))


def test_runner_embeds_only_data_and_supports_paired_relations():
    probe = BehavioralProbe(target="app.value", args=["'; __import__('os').system('bad') #"],
        relation="different_result", second_args=[None], rationale="Untrusted input is only JSON data.")
    source = runner_source(BehavioralProbeSuite(probes=[probe]))
    compile(source, RUNNER_PATH, "exec")  # Syntax check only; NEVER executes this source on the host.
    assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(source).body[:3]}
    assert json.loads(assignments["PAYLOAD"])["probes"][0]["args"] == probe.args
    assert "eval(" not in source and "exec(" not in source


@pytest.mark.asyncio
async def test_reserved_paths_are_preserved_and_python_must_be_allowed(tmp_path):
    task = code_task()
    sandbox = ProbeSandbox(fixture_repository(tmp_path), task.policy)
    sandbox.start()
    try:
        sandbox.workspace.write_file(RUNNER_PATH, "# owner file\n")
        with pytest.raises(ValueError, match="already exist"):
            await execute_suite(task, sandbox, suite())
        assert sandbox.workspace.read_file(RUNNER_PATH) == "# owner file\n"
        task.policy.allowed_test_executables = {"pytest"}
        with pytest.raises(ValueError, match="allow python"):
            await execute_suite(task, sandbox, suite())
    finally:
        sandbox.close()


@pytest.mark.asyncio
async def test_receipt_semantics_reject_forged_winner_after_rehash(tmp_path):
    repair, _, _ = solver()
    result = await repair.solve(code_task(), fixture_repository(tmp_path))
    report = result.tournament.model_copy(deep=True)
    report.candidates[1].regression_probes.statuses = ["mismatched"]
    report.fingerprint = report_fingerprint(report)
    assert not verify_tournament_report(report)
    report = result.tournament.model_copy(deep=True)
    report.regression_suite["probes"][0]["expected"] = 99
    report.fingerprint = report_fingerprint(report)
    assert not verify_tournament_report(report)
    report = result.tournament.model_copy(deep=True)
    report.shared_model_calls = 0
    report.model_calls -= 1
    report.fingerprint = report_fingerprint(report)
    assert not verify_tournament_report(report)


@pytest.mark.asyncio
async def test_context_bound_fails_before_dispatch_and_package_fallback(tmp_path):
    task = code_task()
    root = fixture_repository(tmp_path)
    root.joinpath("app.py").write_text("# " + "a" * 41000)
    sandbox = ProbeSandbox(root, task.policy)
    sandbox.start()
    model = ControlledModel(suite())
    try:
        with pytest.raises(ValueError, match="bounded input"):
            await generate_suite(task, sandbox.workspace, RegressionChallengePolicy(allowed_targets=["app.value"]), model)
        assert model.calls == 0
        sandbox.workspace.delete_file("app.py")
        sandbox.workspace.write_file("app/__init__.py", "def value(): return 1\n")
        await generate_suite(task, sandbox.workspace, RegressionChallengePolicy(allowed_targets=["app.value"]), model)
        assert "app/__init__.py" in model.messages[0][-1].content
    finally:
        sandbox.close()


@pytest.mark.asyncio
async def test_real_docker_harness_executes_exact_and_metamorphic_probes(tmp_path):
    from code_agent.sandbox import DockerSandbox
    task = code_task()
    if not DockerSandbox.available() or not DockerSandbox.image_available(task.policy.image):
        pytest.skip("Docker engine and allowlisted image are required; never substitute host execution")
    root = fixture_repository(tmp_path)
    root.joinpath("app.py").write_text("def value(x):\n    return x * 2\n")
    probes = BehavioralProbeSuite(probes=[
        BehavioralProbe(target="app.value", args=[3], expected=6, rationale="Exact expected output"),
        BehavioralProbe(target="app.value", args=[0], second_args=[0], relation="same_result", rationale="Determinism"),
        BehavioralProbe(target="app.value", args=[1], second_args=[2], relation="different_result", rationale="Distinct outputs"),
    ])
    sandbox = DockerSandbox(root, task.policy)
    try:
        sandbox.start()
        evidence = await execute_suite(task, sandbox, probes)
        assert candidate_matches(evidence)
        assert evidence.suite_sha256 == fingerprint(probes.model_dump(mode="json"))
        assert sandbox.workspace.changed_files() == []
    finally:
        sandbox.close()
