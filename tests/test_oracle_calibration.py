"""Authored control receipts and AST inspection, never host execution of repository/mutant code."""

import ast
import asyncio
import json
import time
from pathlib import Path

import pytest

from code_agent.models import (
    AnalysisPlan,
    CodeAgentResult,
    OracleCalibrationPolicy,
    RegressionChallengePolicy,
    RepairTournamentPolicy,
    ReviewVerdict,
    SandboxCommandResult,
    VerificationReport,
    VerificationRound,
)
from code_agent.oracle_calibration import (
    calibrate_suite,
    generate_mutants,
    reference_fingerprint,
    refresh_reference_binding,
    report_digest,
    verify_calibration_report,
)
from code_agent.regression_challenges import PROBE_COMMAND, RECEIPT_PATH, RUNNER_PATH, BehavioralProbeSuite
from code_agent.repair_tournament import (
    RepairTournament,
    report_fingerprint,
    verify_tournament_report,
)
from code_agent.security_policy import summarize_security
from code_agent.verification import repository_fingerprint, run_quality_gates
from evals.regression_challenge_evaluation import FIXTURE, authored_suite
from tests.test_repair_tournament import ControlledModel, ControlledSandbox, code_task

REFERENCE = Path("evals/fixtures/oracle_reference")


class CalibrationSandbox(ControlledSandbox):
    """Scripted clamp outcomes based on AST fault class; NOT a Python interpreter."""

    def __init__(self, *args, behavior="normal", **kwargs):
        super().__init__(*args, **kwargs)
        self.behavior, self.probe_runs = behavior, 0

    def run(self, command):
        if command != PROBE_COMMAND:
            if self.behavior == "owner_failure":
                return SandboxCommandResult(command=command, exit_code=1, duration_ms=0)
            if self.behavior == "owner_mutation":
                self.workspace.write_file("changed.py", "VALUE = 99\n")
            return super().run(command)
        self.probe_runs += 1
        if self.behavior == "slow":
            time.sleep(0.1)
        assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in
                       ast.parse(self.workspace.read_file(RUNNER_PATH)).body[:3]}
        probes = json.loads(assignments["PAYLOAD"])["probes"]
        source = self.workspace.read_file("app.py")
        functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
        guards = [node for function in functions for node in function.body if isinstance(node, ast.If)]
        lower_broken = upper_broken = False
        if guards:
            lower_broken = isinstance(guards[0].test, ast.Constant) or isinstance(guards[0].test.ops[0], ast.Gt)
            upper_broken = isinstance(guards[1].test, ast.Constant) or isinstance(guards[1].test.ops[0], ast.Lt)
            upper_broken = upper_broken or (isinstance(guards[0].test, ast.Compare) and isinstance(guards[0].test.ops[0], ast.Gt))
        elif "max(lower" not in source:
            lower_broken = True
        statuses = []
        for probe in probes:
            # Explicit authored output classification, not executing a function.
            mismatched = (probe["relation"] == "same_result" and lower_broken
                          or probe["relation"] == "equals" and probe["args"][0] < 0 and lower_broken
                          or probe["relation"] == "equals" and probe["args"][0] > 10 and upper_broken
                          or probe["relation"] == "equals" and probe["expected"] not in {0, 10})
            statuses.append("mismatched" if mismatched else "matched")
        mutant = bool(guards and (lower_broken or upper_broken))
        if self.behavior == "mutant_error" and mutant:
            statuses = ["error"] * len(probes)
        if self.behavior == "unstable" and self.probe_runs % 2 == 0:
            statuses = ["mismatched" if value == "matched" else "matched" for value in statuses]
        if self.behavior == "mutation":
            self.workspace.write_file("changed.py", "VALUE = 99\n")
        if self.behavior == "invalid_receipt":
            statuses = []
        self.workspace.write_file(RECEIPT_PATH, json.dumps({"suite_sha256": assignments["SUITE_HASH"], "statuses": statuses}))
        return SandboxCommandResult(command=command, exit_code=124 if self.behavior == "timeout" else 0,
                                    timed_out=self.behavior == "timeout", duration_ms=0)


def policy(reference=REFERENCE, **kwargs):
    return OracleCalibrationPolicy(reference_sha256=reference_fingerprint(reference), **kwargs)


def factory_with_receipts(sandboxes, behavior="normal"):
    def factory(repository, policy):
        sandbox = CalibrationSandbox(repository, policy, behavior=behavior)
        sandboxes.append(sandbox)
        return sandbox
    return factory


@pytest.mark.asyncio
async def test_reference_agreement_and_mutation_kill_matrix_are_source_bound():
    sandboxes = []
    report = await calibrate_suite(code_task(), FIXTURE.resolve(), authored_suite(), policy(), REFERENCE.resolve(),
                                   sandbox_factory=factory_with_receipts(sandboxes))
    assert report.eligible and verify_calibration_report(report)
    assert report.killed_mutants == 4 and report.mutation_score == 1 and report.command_runs == 11
    assert report.reference_unchanged and report.reference_owner_tests_passed
    assert set(report.minimal_cover_indices) == {0, 2}
    assert len(report.mutants) == 4 and all(row.killing_probe_indices for row in report.mutants)
    assert "Authored operator reference" not in report.model_dump_json()
    assert all(s.closed and s.workspace.root is None for s in sandboxes)


@pytest.mark.asyncio
async def test_weak_suite_and_wrong_oracle_are_distinct_holds():
    weak = BehavioralProbeSuite(probes=[authored_suite().probes[0]])
    report = await calibrate_suite(code_task(), FIXTURE.resolve(), weak, policy(), REFERENCE.resolve(), sandbox_factory=CalibrationSandbox)
    assert not report.eligible and verify_calibration_report(report)
    assert report.mutation_score == 0.5 and report.killed_mutants == 2
    assert "mutation_sensitivity_below_threshold" in report.blocking_reasons
    wrong = BehavioralProbeSuite(probes=[authored_suite().probes[0].model_copy(update={"expected": 99})])
    report = await calibrate_suite(code_task(), FIXTURE.resolve(), wrong, policy(), REFERENCE.resolve(), sandbox_factory=CalibrationSandbox)
    assert not report.eligible and verify_calibration_report(report)
    assert "probe_expectations_disagree_with_reference" in report.blocking_reasons
    assert not report.mutants and report.command_runs == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior", ["owner_failure", "owner_mutation", "mutant_error", "unstable", "mutation", "invalid_receipt", "timeout"])
async def test_execution_failures_do_not_inflate_detection_score(behavior):
    sandboxes = []
    report = await calibrate_suite(code_task(), FIXTURE.resolve(), authored_suite(), policy(), REFERENCE.resolve(),
                                   sandbox_factory=factory_with_receipts(sandboxes, behavior))
    assert not report.eligible and verify_calibration_report(report)
    assert report.killed_mutants == 0 and report.mutation_score == 0
    assert all(s.closed for s in sandboxes)
    if behavior == "mutant_error":
        assert len(report.mutants) == 4 and all(row.status == "invalid" for row in report.mutants)


@pytest.mark.asyncio
async def test_missing_unpinned_and_relative_references_never_start_a_container(tmp_path):
    for reference, configured in ((None, policy()), (REFERENCE, policy()), (REFERENCE.resolve(), policy().model_copy(
            update={"reference_sha256": "0" * 64})), (tmp_path / "absent", policy())):
        sandboxes = []
        report = await calibrate_suite(code_task(), FIXTURE.resolve(), authored_suite(), configured, reference,
                                       sandbox_factory=factory_with_receipts(sandboxes))
        assert not report.eligible and verify_calibration_report(report) and not sandboxes


@pytest.mark.asyncio
async def test_phase_deadline_cleans_in_flight_commands_and_external_cancellation():
    sandboxes = []
    configured = policy(timeout_seconds=0.05)
    report = await calibrate_suite(code_task(), FIXTURE.resolve(), authored_suite(), configured, REFERENCE.resolve(),
                                   sandbox_factory=factory_with_receipts(sandboxes, "slow"))
    assert not report.eligible and "oracle_calibration_deadline_exceeded" in report.blocking_reasons
    assert verify_calibration_report(report) and all(s.closed for s in sandboxes)
    sandboxes = []
    pending = asyncio.create_task(calibrate_suite(code_task(), FIXTURE.resolve(), authored_suite(), policy(), REFERENCE.resolve(),
                                                  sandbox_factory=factory_with_receipts(sandboxes, "slow")))
    await asyncio.sleep(0.05)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert all(s.closed and s.workspace.root is None for s in sandboxes)


@pytest.mark.asyncio
async def test_reference_revocation_and_frozen_snapshot_mismatch_hold(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    reference.mkdir()
    reference.joinpath("app.py").write_bytes(REFERENCE.joinpath("app.py").read_bytes())
    configured = policy(reference)
    sandboxes = []

    class RevokingSandbox(CalibrationSandbox):
        def run(self, command):
            result = super().run(command)
            reference.joinpath("app.py").write_text("def clamp(value): return value\n")
            return result

    report = await calibrate_suite(code_task(), FIXTURE.resolve(), authored_suite(), configured, reference,
                                   sandbox_factory=RevokingSandbox)
    assert not report.reference_unchanged and not report.eligible and verify_calibration_report(report)
    original = reference_fingerprint
    counter = 0
    def changed_snapshot(root, *limits):
        nonlocal counter
        counter += 1
        return "0" * 64 if counter == 2 else original(root, *limits)
    configured = policy(reference)
    monkeypatch.setattr("code_agent.oracle_calibration.reference_fingerprint", changed_snapshot)
    report = await calibrate_suite(code_task(), FIXTURE.resolve(), authored_suite(), configured, reference,
                                   sandbox_factory=factory_with_receipts(sandboxes))
    assert not report.eligible and not sandboxes and verify_calibration_report(report)


def test_mutation_engine_is_bounded_deterministic_and_scope_limited():
    source = "def selected(x):\n    return min(x + 1, 10)\n\ndef untouched(x):\n    return max(x - 1, 0)\n"
    mutations = generate_mutants({"module.py": source}, {"module.py": {"selected"}}, 8)
    assert [row.operator for row in mutations] == ["min_max_swap", "arithmetic_swap"]
    assert mutations == generate_mutants({"module.py": source}, {"module.py": {"selected"}}, 8)
    original = ast.parse(source).body[1]
    for row in mutations:
        assert ast.dump(ast.parse(row.source).body[1]) == ast.dump(original)
        compile(row.source, row.path, "exec")  # Syntax check only, never execute code on host.
    assert len(generate_mutants({"module.py": source}, {"module.py": {"selected"}}, 1)) == 1
    for limit in (0, 9):
        with pytest.raises(ValueError):
            generate_mutants({}, {}, limit)
    with pytest.raises(ValueError, match="bounded"):
        generate_mutants({"module.py": "#" * 100001}, {"module.py": {"selected"}}, 4)
    with pytest.raises(ValueError, match="node"):
        generate_mutants({"module.py": "x=1\n" * 6000}, {"module.py": {"selected"}}, 4)


def test_reference_identity_is_portable_bounded_and_secret_filtered(tmp_path):
    root = tmp_path / "reference"
    root.mkdir()
    root.joinpath("app.py").write_bytes(b"def clamp(x):\n    return x\n")
    first = reference_fingerprint(root)
    root.joinpath("app.py").write_bytes(b"def clamp(x):\r\n    return x\r\n")
    root.joinpath(".env").write_text("placeholder-only\n")
    root.joinpath("ignored.key").write_text("placeholder-only\n")
    assert reference_fingerprint(root) == first
    root.joinpath("binary.bin").write_bytes(b"\0\r\n")
    binary = reference_fingerprint(root)
    root.joinpath("binary.bin").write_bytes(b"\0\n")
    assert reference_fingerprint(root) != binary
    with pytest.raises(ValueError, match="limits"):
        reference_fingerprint(root, max_files=1)
    with pytest.raises(ValueError, match="limits"):
        reference_fingerprint(root, max_bytes=1)
    with pytest.raises(ValueError, match="exist"):
        reference_fingerprint(root / "absent")


@pytest.mark.asyncio
async def test_semantic_receipt_validation_rejects_rehashed_score_manipulation():
    report = await calibrate_suite(code_task(), FIXTURE.resolve(), authored_suite(), policy(), REFERENCE.resolve(), sandbox_factory=CalibrationSandbox)
    for field, value in (("mutation_score", 0.5), ("minimal_cover_indices", [0]), ("command_runs", 0), ("eligible", False)):
        changed = report.model_copy(deep=True)
        setattr(changed, field, value)
        changed.fingerprint = report_digest(changed)
        assert not verify_calibration_report(changed)
    changed = report.model_copy(deep=True)
    changed.mutants[0].execution.statuses = ["matched"] * 4
    changed.fingerprint = report_digest(changed)
    assert not verify_calibration_report(changed)


def test_policy_caps_and_hash_are_operator_validated():
    for values in ({"reference_sha256": "wrong"}, {"reference_sha256": "0" * 64, "min_mutants": 8, "max_mutants": 2},
                   {"reference_sha256": "0" * 64, "min_mutation_score": float("nan")},
                   {"reference_sha256": "0" * 64, "extra": True}):
        with pytest.raises(ValueError):
            OracleCalibrationPolicy(**values)


class AuthoredFunctionalTeam:
    async def solve(self, task, sandbox):
        before = repository_fingerprint(sandbox.repository, task.policy.max_files, task.policy.max_repository_bytes)
        sandbox.workspace.write_file("app.py", "def clamp(value, lower=0, upper=10):\n    return max(lower, min(value, upper))\n")
        patch, changed = sandbox.workspace.unified_diff(), sandbox.workspace.changed_files()
        gates, final = await run_quality_gates(task, sandbox, patch=patch, changed_files=changed,
                                              source_fingerprint_before=before, baseline_lint=None)
        return CodeAgentResult(status="completed", summary="Authored independent fix", patch=patch, changed_files=changed,
            final_test=final, writes=1, verification=VerificationReport(analysis=AnalysisPlan(summary="Bounded fix"),
                final_decision="verified", rounds=[VerificationRound(round=0, verdict=ReviewVerdict(approved=True, summary="Authored review"),
                                                                    quality_gates=gates)]),
            security=summarize_security([], task.policy.security_policy_version))


@pytest.mark.asyncio
async def test_tournament_holds_wrong_oracles_and_keeps_reference_out_of_prompts_and_patches():
    configured = RepairTournamentPolicy(candidates=1, max_parallel=1, regression_challenges=RegressionChallengePolicy(
        allowed_targets=["app.clamp"], oracle_calibration=policy()))
    designer = ControlledModel(authored_suite())
    repair = RepairTournament(lambda *_: AuthoredFunctionalTeam(), ControlledModel(ReviewVerdict(approved=True, summary="Authored challenge")),
        policy=configured, regression_model=designer, reference_repository=REFERENCE.resolve(), sandbox_factory=CalibrationSandbox)
    result = await repair.solve(code_task(), FIXTURE.resolve())
    assert result.status == "completed" and verify_tournament_report(result.tournament)
    assert result.tournament.oracle_calibration.eligible and result.model_calls == 2
    assert "Authored operator reference" not in result.patch
    assert "Authored operator reference" not in str(designer.messages)
    assert "clamp(value" in result.patch and result.tournament.oracle_calibration.reference_unchanged
    result.tournament.oracle_calibration = None
    result.tournament.fingerprint = report_fingerprint(result.tournament)
    assert not verify_tournament_report(result.tournament)
    repair.regression_model = ControlledModel(BehavioralProbeSuite(probes=[authored_suite().probes[0].model_copy(update={"expected": 99})]))
    result = await repair.solve(code_task(), FIXTURE.resolve())
    assert result.status == "failed" and result.model_calls == 1 and not result.patch
    assert verify_tournament_report(result.tournament)
    assert "oracle_calibration_not_passed" in result.tournament.blocking_reasons


@pytest.mark.asyncio
async def test_real_docker_reference_and_mutant_execution():
    from code_agent.sandbox import DockerSandbox
    if not DockerSandbox.available() or not DockerSandbox.image_available(code_task().policy.image):
        pytest.skip("Docker/image required; never execute references or mutants on host")
    report = await calibrate_suite(code_task(), FIXTURE.resolve(), authored_suite(), policy(), REFERENCE.resolve())
    assert report.eligible and report.mutation_score == 1 and verify_calibration_report(report)


@pytest.mark.asyncio
async def test_private_reference_cannot_overlap_task_source(tmp_path):
    configured = RepairTournamentPolicy(candidates=1, max_parallel=1, regression_challenges=RegressionChallengePolicy(
        allowed_targets=["app.clamp"], oracle_calibration=policy()))
    designer = ControlledModel(authored_suite())
    repair = RepairTournament(lambda *_: AuthoredFunctionalTeam(), ControlledModel(), policy=configured,
        regression_model=designer, reference_repository=FIXTURE.resolve(), sandbox_factory=CalibrationSandbox)
    with pytest.raises(ValueError, match="separate"):
        await repair.solve(code_task(), FIXTURE.resolve())
    repair.reference_repository = Path("relative")
    with pytest.raises(ValueError, match="absolute"):
        await repair.solve(code_task(), FIXTURE.resolve())
    assert designer.calls == 0


@pytest.mark.asyncio
async def test_private_snapshot_is_cleaned_even_when_sandbox_close_fails(monkeypatch):
    from code_agent.workspace import EphemeralWorkspace
    snapshots = []
    def private_snapshot(source, policy):
        workspace = EphemeralWorkspace(source, policy)
        snapshots.append(workspace)
        return workspace
    class FailedCleanup(CalibrationSandbox):
        def close(self):
            super().close()
            raise OSError("authored close failure")
    monkeypatch.setattr("code_agent.oracle_calibration.EphemeralWorkspace", private_snapshot)
    report = await calibrate_suite(code_task(), FIXTURE.resolve(), authored_suite(), policy(), REFERENCE.resolve(), sandbox_factory=FailedCleanup)
    assert not report.eligible and verify_calibration_report(report)
    assert "oracle_calibration_cleanup_error:OSError" in report.blocking_reasons
    assert len(snapshots) == 1 and snapshots[0].root is None


@pytest.mark.asyncio
async def test_reference_revocation_is_rechecked_and_cannot_be_cleared(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    original = REFERENCE.joinpath("app.py").read_bytes()
    reference.joinpath("app.py").write_bytes(original)
    report = await calibrate_suite(code_task(), FIXTURE.resolve(), authored_suite(), policy(reference), reference,
                                   sandbox_factory=CalibrationSandbox)
    assert report.eligible
    reference.joinpath("app.py").write_text("def clamp(x): return x\n")
    await refresh_reference_binding(report, reference, code_task())
    assert not report.eligible and not report.reference_unchanged and verify_calibration_report(report)
    reference.joinpath("app.py").write_bytes(original)
    await refresh_reference_binding(report, reference, code_task())
    assert not report.eligible and verify_calibration_report(report)


@pytest.mark.asyncio
async def test_insufficient_mutants_and_operator_threshold_are_explicit():
    report = await calibrate_suite(code_task(), FIXTURE.resolve(), authored_suite(), policy(min_mutants=8, max_mutants=8),
                                   REFERENCE.resolve(), sandbox_factory=CalibrationSandbox)
    assert not report.eligible and "insufficient_reference_mutants" in report.blocking_reasons
    assert verify_calibration_report(report)
    weak = BehavioralProbeSuite(probes=[authored_suite().probes[0]])
    report = await calibrate_suite(code_task(), FIXTURE.resolve(), weak, policy(min_mutation_score=0.5), REFERENCE.resolve(),
                                   sandbox_factory=CalibrationSandbox)
    assert report.eligible and report.mutation_score == 0.5 and verify_calibration_report(report)
