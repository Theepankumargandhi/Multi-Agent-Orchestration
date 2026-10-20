import ast
import json
import os
import subprocess
import sys
import time

import pytest

from code_agent.counterfactual import (
    COMMAND,
    RECEIPT_PATH,
    RUNNER_PATH,
    assess_trials,
    counterfactual_gate,
    decode_receipt,
    fingerprint,
    runner_source,
    select_tests,
    valid_report,
    verify_counterfactual,
)
from code_agent.models import (
    CodeTask,
    CounterfactualReport,
    CounterfactualTrial,
    SandboxCommandResult,
    SandboxPolicy,
)
from code_agent.sandbox import DockerSandbox, DockerUnavailableError
from code_agent.workspace import EphemeralWorkspace

TEST = "from app import value\ndef test_value():\n    assert value() == 2\n"


def fixture(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "app.py").write_text("def value():\n    return 1\n")
    (root / ".env").write_text("PRIVATE_DATA=not-for-clones\n")
    policy = SandboxPolicy(counterfactual_verification_enabled=True, max_repair_rounds=0)
    workspace = EphemeralWorkspace(root, policy)
    workspace.prepare()
    workspace.write_file("app.py", "def value():\n    return 2\n")
    workspace.write_file("tests/test_regression.py", TEST)

    class Sandbox:
        repository = root

    sandbox = Sandbox()
    sandbox.workspace = workspace
    return sandbox, CodeTask(repository="fixture", issue="Fix value from one to two with regression coverage.", policy=policy)


class ReceiptSandbox:
    """Authored simulation: parses harness constants, never executes repository code."""
    mode = "normal"
    instances = []

    def __init__(self, source, policy):
        self.workspace = EphemeralWorkspace(source, policy)
        self.policy = policy
        self.closed = False
        self.instances.append(self)

    def start(self):
        self.workspace.prepare()
        assert not self.workspace.resolve(".env", allow_missing=True).exists()

    def run(self, command):
        assert command == COMMAND
        constants = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(self.workspace.read_file(RUNNER_PATH)).body[:5]}
        nodes = sorted(constants["ELIGIBLE"])
        patched = "return 2" in self.workspace.read_file("app.py")
        status = "passed" if patched or self.mode == "tautology" else "assertion"
        if not patched and self.mode in {"error", "skipped"}:
            status = self.mode
        if self.mode == "tamper":
            self.workspace.write_file("app.py", "VALUE = 0\n")
        exit_code = 1 if status in {"assertion", "error"} else 0
        self.workspace.write_file(RECEIPT_PATH, json.dumps({"manifest_sha256": constants["MANIFEST_HASH"],
            "collected": nodes, "outcomes": {node: status for node in nodes}, "collection_errors": 0,
            "overflow": False, "exit_code": exit_code}))
        return SandboxCommandResult(command=command, exit_code=exit_code, stdout="PRIVATE_FAILURE_OUTPUT", duration_ms=1)

    def close(self):
        self.closed = True
        self.workspace.cleanup()


def trials(original="assertion", patched="passed"):
    return [CounterfactualTrial(arm=arm, repeat=repeat, outcomes={fingerprint("test"): status},
        workspace_unchanged=True, source_sha256=fingerprint(arm), manifest_sha256=fingerprint("manifest"), exit_code=0 if status == "passed" else 1)
        for arm, status in [("patched", patched), ("original", original)] for repeat in range(2)]


def passing_report():
    rows = trials()
    for row in rows:
        row.receipt_sha256 = fingerprint((row.arm, row.repeat))
    report = CounterfactualReport(status="passed", reason="stable_discrimination", sources_unchanged=True,
        original_sha256=fingerprint("original"), candidate_sha256=fingerprint("candidate"),
        test_file_sha256={"tests/test_regression.py": fingerprint(TEST)}, eligible_test_sha256=[fingerprint("test")],
        discriminating_test_sha256=[fingerprint("test")], trials=rows)
    report.fingerprint = fingerprint(report.model_dump(mode="json", exclude={"fingerprint"}))
    return report


def test_ast_selection_ignores_unchanged_tests_and_fixture_only_changes(tmp_path):
    sandbox, task = fixture(tmp_path)
    try:
        overlays, eligible = select_tests(sandbox.workspace, sandbox.repository, sandbox.workspace.changed_files())
        assert overlays == {"tests/test_regression.py": TEST}
        assert eligible == {"tests/test_regression.py::test_value": [2, 3]}
        (sandbox.repository / "tests").mkdir()
        (sandbox.repository / "tests/test_regression.py").write_text(TEST)
        sandbox.workspace.write_file("tests/test_regression.py", "# formatting only\n" + TEST)
        assert select_tests(sandbox.workspace, sandbox.repository, sandbox.workspace.changed_files()) == ({}, {})
        sandbox.workspace.write_file("tests/test_class.py", "class TestBoundary:\n    def test_amount(self):\n        assert 1 == 1\n")
        assert "tests/test_class.py::TestBoundary::test_amount" in select_tests(sandbox.workspace, sandbox.repository, sandbox.workspace.changed_files())[1]
    finally:
        sandbox.workspace.cleanup()


@pytest.mark.parametrize("mode", ["deleted", "unsupported-path", "oversized-original", "too-many-files", "too-many-nodes"])
def test_selection_rejects_additional_read_and_capacity_hazards(tmp_path, mode):
    sandbox, _ = fixture(tmp_path)
    try:
        changed = sandbox.workspace.changed_files()
        if mode == "deleted":
            sandbox.workspace.delete_file("tests/test_regression.py")
        elif mode == "unsupported-path":
            changed.append("tests/test unsupported.py")
        elif mode == "oversized-original":
            (sandbox.repository / "tests").mkdir()
            (sandbox.repository / "tests/test_regression.py").write_text("x" * 100001)
        elif mode == "too-many-files":
            for index in range(7):
                path = f"tests/test_added_{index}.py"
                sandbox.workspace.write_file(path, TEST)
                changed.append(path)
        else:
            sandbox.workspace.write_file("tests/test_regression.py", "x=0\n" * 6000)
        with pytest.raises(ValueError):
            select_tests(sandbox.workspace, sandbox.repository, changed)
    finally:
        sandbox.workspace.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["network-enabled", "python-disallowed", "no-production-change", "timeout"])
async def test_unsupported_policy_and_command_timeout_hold_verification(tmp_path, mode):
    sandbox, task = fixture(tmp_path)
    class Timeout(ReceiptSandbox):
        def run(self, command):
            return super().run(command).model_copy(update={"timed_out": True})

    try:
        changed = sandbox.workspace.changed_files()
        updates = {}
        if mode == "network-enabled":
            updates["network_enabled"] = True
        elif mode == "python-disallowed":
            updates["allowed_test_executables"] = {"pytest"}
        elif mode == "no-production-change":
            changed = ["tests/test_regression.py"]
        selected = task.model_copy(update={"policy": task.policy.model_copy(update=updates)})
        report = await verify_counterfactual(selected, sandbox, changed, sandbox_factory=Timeout)
        assert report.status == "failed"
        assert report.reason == ("unsupported_sandbox_policy" if updates else "no_production_change"
                                 if mode == "no-production-change" else "counterfactual_execution_invalid")
    finally:
        sandbox.workspace.cleanup()


def test_report_validator_and_receipt_exit_contradiction_fail_closed():
    assert not valid_report("not-a-report")
    proof = passing_report()
    proof.trials[0].receipt_sha256 = "bad"
    proof.fingerprint = fingerprint(proof.model_dump(mode="json", exclude={"fingerprint"}))
    assert not valid_report(proof)
    proof = passing_report()
    proof.eligible_test_sha256 = ["bad"]
    proof.fingerprint = fingerprint(proof.model_dump(mode="json", exclude={"fingerprint"}))
    assert not valid_report(proof)
    with pytest.raises(ValueError, match="read bound"):
        decode_receipt("x" * 100001, "", {}, 0)
    with pytest.raises(ValueError, match="contradicts"):
        decode_receipt(json.dumps({"manifest_sha256": "expected", "exit_code": 0, "collected": ["tests/test.py::test_a"],
            "outcomes": {"tests/test.py::test_a": "assertion"}, "collection_errors": 0, "overflow": False}),
            "expected", {"tests/test.py::test_a": [1, 2]}, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,reason", [("normal", "stable_discrimination"), ("tautology", "no_discriminating_test"),
    ("error", "original_errors_or_skips"), ("skipped", "original_errors_or_skips"), ("tamper", "workspace_or_receipt_invalid")])
async def test_counterfactual_copies_same_tests_and_preserves_sources(tmp_path, mode, reason):
    sandbox, task = fixture(tmp_path)
    class Clone(ReceiptSandbox):
        instances = []

    Clone.mode = mode
    try:
        report = await verify_counterfactual(task, sandbox, sandbox.workspace.changed_files(), sandbox_factory=Clone)
        assert report.reason == reason
        assert report.sources_unchanged
        assert len(Clone.instances) == len(report.trials) == 4
        assert all(clone.closed and clone.workspace.root is None for clone in Clone.instances)
        assert report.status == ("passed" if mode == "normal" else "failed")
        assert counterfactual_gate(report).status == ("passed" if mode == "normal" else "failed")
        assert "PRIVATE_FAILURE_OUTPUT" not in report.model_dump_json()
        assert "test_value" not in report.model_dump_json()
        assert report.fingerprint == fingerprint(report.model_dump(mode="json", exclude={"fingerprint"}))
        assert (sandbox.repository / "app.py").read_text().endswith("return 1\n")
        assert sandbox.workspace.read_file("app.py").endswith("return 2\n")
    finally:
        sandbox.workspace.cleanup()


@pytest.mark.parametrize("update", [{"manifest_sha256": "bad"}, {"overflow": True}, {"collection_errors": 1},
    {"exit_code": True}, {"collected": []}, {"collected": ["test.py::test_a", "test.py::test_a"]},
    {"outcomes": {"test.py::test_other": "passed"}}, {"outcomes": {"test.py::test_a": "unknown"}}])
def test_receipt_parser_rejects_drift_and_unsafe_outcomes(update):
    manifest = fingerprint("manifest")
    raw = {"manifest_sha256": manifest, "exit_code": 0, "collected": ["test.py::test_a"],
           "outcomes": {"test.py::test_a": "passed"}, "collection_errors": 0, "overflow": False}
    raw.update(update)
    with pytest.raises(ValueError):
        decode_receipt(json.dumps(raw), manifest, {"test.py::test_a": [1, 2]}, 0)


def test_assessment_requires_repeatability_matching_nodes_and_successful_candidate():
    assert assess_trials(trials())[0] == "stable_discrimination"
    assert assess_trials(trials()[:3])[0] == "incomplete_execution"
    assert assess_trials(trials(patched="assertion"))[0] == "patched_tests_not_all_passed"
    changed = trials()
    changed[-1].outcomes = {fingerprint("test"): "passed"}
    assert assess_trials(changed)[0] == "unstable_or_mismatched_execution"
    changed = trials()
    changed[0].outcomes = changed[1].outcomes = {fingerprint("other"): "passed"}
    assert assess_trials(changed)[0] == "unstable_or_mismatched_execution"


@pytest.mark.asyncio
async def test_budget_and_unavailable_docker_fail_closed_without_host_fallback(tmp_path):
    sandbox, task = fixture(tmp_path)
    class Unavailable(ReceiptSandbox):
        instances = []

        def start(self):
            raise DockerUnavailableError("PRIVATE_ERROR")

    try:
        report = await verify_counterfactual(task, sandbox, sandbox.workspace.changed_files(),
            deadline=time.perf_counter() + 1, sandbox_factory=Unavailable)
        assert report.status == "budget_exhausted" and not Unavailable.instances
        assert report.sources_unchanged
        report = await verify_counterfactual(task, sandbox, sandbox.workspace.changed_files(), sandbox_factory=Unavailable)
        assert report.status == "unavailable" and report.reason == "docker_unavailable"
        assert Unavailable.instances[0].closed
        assert "PRIVATE_ERROR" not in report.model_dump_json()
    finally:
        sandbox.workspace.cleanup()


@pytest.mark.asyncio
async def test_unsupported_owner_command_and_no_assertions_are_rejected(tmp_path):
    sandbox, task = fixture(tmp_path)
    try:
        report = await verify_counterfactual(task.model_copy(update={"test_command": ["python", "custom.py"]}), sandbox,
                                            sandbox.workspace.changed_files())
        assert report.reason == "unsupported_owner_command"
        sandbox.workspace.write_file("tests/test_regression.py", "def test_value():\n    pass\n")
        report = await verify_counterfactual(task, sandbox, sandbox.workspace.changed_files())
        assert report.reason == "no_changed_assertion_tests"
    finally:
        sandbox.workspace.cleanup()


@pytest.mark.asyncio
async def test_real_docker_counterfactual_executes_trusted_harness(tmp_path):
    if not DockerSandbox.available() or not DockerSandbox.image_available("agentforge-code-sandbox:local"):
        pytest.skip("Docker engine or sandbox image is unavailable")
    sandbox, task = fixture(tmp_path)
    try:
        report = await verify_counterfactual(task, sandbox, sandbox.workspace.changed_files())
        assert report.status == "passed", report.reason
        assert report.discriminating_test_sha256
        assert report.sources_unchanged
    finally:
        sandbox.workspace.cleanup()


@pytest.mark.parametrize("production,expected", [("return 1", "assertion"), ("return 2", "passed"),
    ("raise ValueError('fixture error')", "error"), ("assert False", "error")])
def test_fixed_pytest_harness_on_authored_trusted_fixture_only(tmp_path, production, expected):
    # This host unit test executes only these literal fixture sources. Production
    # verification never falls back to a host process for user repositories.
    (tmp_path / "app.py").write_text("def value():\n    " + production + "\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_regression.py").write_text(TEST)
    eligible = {"tests/test_regression.py::test_value": [2, 3]}
    source, manifest = runner_source({"tests/test_regression.py": TEST}, eligible)
    (tmp_path / RUNNER_PATH).write_text(source)
    result = subprocess.run([sys.executable, "-I", "-B", RUNNER_PATH], cwd=tmp_path,
        env={**os.environ, "TMP": str(tmp_path), "TEMP": str(tmp_path)}, capture_output=True, text=True, timeout=30)
    assert result.returncode == (0 if expected == "passed" else 1), result.stderr
    raw = (tmp_path / RECEIPT_PATH).read_text()
    receipt = json.loads(raw)
    assert receipt["exit_code"] == result.returncode
    outcomes, _ = decode_receipt(raw, manifest, eligible, result.returncode)
    assert set(outcomes.values()) == {expected}


@pytest.mark.asyncio
async def test_single_agent_benchmark_rejects_gate_before_creating_checkpoint(tmp_path):
    from code_agent.evaluation import run_benchmark

    checkpoint = tmp_path / "checkpoint"
    with pytest.raises(ValueError, match="verified-PR"):
        await run_benchmark([], checkpoint_dir=checkpoint, policy=SandboxPolicy(counterfactual_verification_enabled=True))
    assert not checkpoint.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("passed", [True, False])
async def test_verified_workflow_counterfactual_report_blocks_approval_when_required(tmp_path, monkeypatch, passed):
    from test_verified_pr import FakeSandbox, SequenceModel, analysis, approval, repository, task

    from code_agent.models import CodeAction, CounterfactualReport
    from code_agent.verified_pr import VerifiedPRAgent

    async def verify(selected, sandbox, changed, **kwargs):
        assert selected.policy.counterfactual_verification_enabled
        assert kwargs["original_sha256"] and kwargs["deadline"] > time.perf_counter()
        assert "app.py" in changed
        return passing_report() if passed else CounterfactualReport(status="failed", reason="no_discriminating_test")

    monkeypatch.setattr("code_agent.verified_pr.verify_counterfactual", verify)
    policy = SandboxPolicy(counterfactual_verification_enabled=True, max_repair_rounds=0)
    sandbox = FakeSandbox(repository(tmp_path), policy)
    actions = SequenceModel([CodeAction(kind="write", path="app.py", content="VALUE = 2\n"),
                             CodeAction(kind="test"), CodeAction(kind="test")])
    agent = VerifiedPRAgent(SequenceModel([analysis()]), actions, SequenceModel([approval()]))
    try:
        result = await agent.solve(task(policy), sandbox)
    finally:
        sandbox.close()
    assert result.status == ("completed" if passed else "failed")
    assert result.verification.rounds[-1].counterfactual.reason == ("stable_discrimination" if passed else "no_discriminating_test")
    assert any(gate.name == "counterfactual_regression" and gate.status == ("passed" if passed else "failed")
               for gate in result.verification.rounds[-1].quality_gates)


@pytest.mark.asyncio
async def test_counterfactual_blocker_reaches_bounded_repair_then_is_reverified(tmp_path, monkeypatch):
    from test_verified_pr import FakeSandbox, SequenceModel, analysis, approval, repository, task

    from code_agent.models import CodeAction, CounterfactualReport
    from code_agent.verified_pr import VerifiedPRAgent

    attempts = []
    async def verify(selected, sandbox, changed, **kwargs):
        attempts.append(1)
        return CounterfactualReport(status="failed", reason="no_discriminating_test") if len(attempts) == 1 else passing_report()

    monkeypatch.setattr("code_agent.verified_pr.verify_counterfactual", verify)
    policy = SandboxPolicy(counterfactual_verification_enabled=True, max_repair_rounds=1)
    sandbox = FakeSandbox(repository(tmp_path), policy)
    actions = SequenceModel([CodeAction(kind="write", path="app.py", content="VALUE = 2\n"),
        CodeAction(kind="test"), CodeAction(kind="test"),
        CodeAction(kind="write", path="tests/test_regression.py", content="from app import VALUE\ndef test_fixed_value():\n    assert VALUE == 2\n"),
        CodeAction(kind="test")])
    agent = VerifiedPRAgent(SequenceModel([analysis()]), actions, SequenceModel([approval(), approval()]))
    try:
        result = await agent.solve(task(policy), sandbox)
    finally:
        sandbox.close()
    assert result.status == "completed" and result.verification.repair_rounds == 1
    assert [row.counterfactual.status for row in result.verification.rounds] == ["failed", "passed"]


def test_complete_report_binding_is_required_for_success():
    proof = passing_report()
    assert valid_report(proof)
    assert not valid_report(proof.model_copy(update={"fingerprint": "wrong"}))
    assert not valid_report(proof.model_copy(update={"sources_unchanged": False}))
    assert not valid_report(CounterfactualReport(status="passed"))
    assert not valid_report(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [True, False])
async def test_benchmark_holds_missing_counterfactual_success_evidence(tmp_path, monkeypatch, complete):
    from code_agent.evaluation import run_benchmark
    from code_agent.evaluation_models import CodeBenchmarkCase
    from evals.repair_memory_evaluation import _scripted_evidence

    (tmp_path / "repo").mkdir()
    class Sandbox:
        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def close(self):
            pass

    class Agent:
        async def solve(self, *_args):
            result = _scripted_evidence()
            if complete:
                result.verification.rounds[-1].counterfactual = passing_report()
            return result

    monkeypatch.setattr("code_agent.evaluation.DockerSandbox", Sandbox)
    monkeypatch.setattr("code_agent.evaluation.build_verified_pr_agent", lambda *_args, **_kwargs: Agent())
    case = CodeBenchmarkCase(id="authored", repository="repo", issue="Fix the authored decimal precision boundary.",
                             test_command=["python", "-m", "pytest", "-q"])
    report = await run_benchmark([case], repository_root=tmp_path, default_model="scripted", workflow="verified_pr",
        policy=SandboxPolicy(counterfactual_verification_enabled=True))
    assert report.quality_metrics_valid == complete
    assert (report.outcomes[0].failure_category == "counterfactual_evidence_invalid") != complete
    assert report.resolved == int(complete)


@pytest.mark.parametrize("source", ["def test_value():\n    assert (\n", "x" * 100001,
    "def test_value():\n    assert True\ndef test_value():\n    assert False\n"],
    ids=["invalid-python", "oversized-source", "duplicate-function"])
def test_unsafe_or_ambiguous_test_parse_is_rejected(tmp_path, source):
    sandbox, _ = fixture(tmp_path)
    try:
        sandbox.workspace.write_file("tests/test_regression.py", source)
        with pytest.raises(ValueError):
            select_tests(sandbox.workspace, sandbox.repository, sandbox.workspace.changed_files())
    finally:
        sandbox.workspace.cleanup()


@pytest.mark.asyncio
async def test_reserved_paths_and_source_drift_cannot_be_silently_ignored(tmp_path):
    sandbox, task = fixture(tmp_path)
    try:
        report = await verify_counterfactual(task, sandbox, sandbox.workspace.changed_files(), original_sha256="a" * 64,
                                            sandbox_factory=ReceiptSandbox)
        assert report.reason == "original_source_drift"
        sandbox.workspace.write_file(RUNNER_PATH, "# existing owner content\n")
        report = await verify_counterfactual(task, sandbox, sandbox.workspace.changed_files(), sandbox_factory=ReceiptSandbox)
        assert report.status == "failed" and report.reason == "counterfactual_execution_invalid"
        assert sandbox.workspace.read_file(RUNNER_PATH) == "# existing owner content\n"
    finally:
        sandbox.workspace.cleanup()
