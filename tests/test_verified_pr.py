from __future__ import annotations

from pathlib import Path

import pytest

from code_agent.models import (
    AnalysisPlan,
    CodeAction,
    CodeTask,
    ReviewFinding,
    ReviewVerdict,
    SandboxCommandResult,
    SandboxPolicy,
)
from code_agent.verification import (
    changed_lines_by_file,
    is_test_path,
    repository_fingerprint,
    run_quality_gates,
)
from code_agent.verified_pr import VerifiedPRAgent
from code_agent.workspace import EphemeralWorkspace


class SequenceModel:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    async def ainvoke(self, _messages, config=None):
        value = self.values[self.calls]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return value


class FakeSandbox:
    def __init__(self, repository: Path, policy: SandboxPolicy):
        self.repository = repository
        self.policy = policy
        self.workspace = EphemeralWorkspace(repository, policy)
        self.workspace.prepare()

    def run(self, command: list[str]) -> SandboxCommandResult:
        if ".agentforge-coverage.json" in command:
            self.workspace.write_file(
                ".agentforge-coverage.json",
                '{"files":{"app.py":{"executed_lines":[1]}}}',
            )
        return SandboxCommandResult(
            command=command,
            exit_code=0,
            stdout="quality command passed",
            duration_ms=1.0,
        )

    def close(self):
        self.workspace.cleanup()


def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    (root / "tests").mkdir(parents=True)
    (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "tests" / "test_app.py").write_text(
        "def test_value():\n    assert True\n", encoding="utf-8"
    )
    return root


def analysis() -> AnalysisPlan:
    return AnalysisPlan(
        summary="Update the value and protect it with a regression test.",
        relevant_paths=["app.py", "tests/test_app.py"],
        risks=["Changing unrelated behavior"],
        acceptance_criteria=["Tests pass"],
        test_strategy=["Add a focused unit test"],
    )


def approval(findings=None) -> ReviewVerdict:
    return ReviewVerdict(
        approved=True,
        summary="Patch is focused and supported by tests.",
        findings=findings or [],
    )


def task(policy: SandboxPolicy | None = None) -> CodeTask:
    return CodeTask(
        repository="fixture",
        issue="Change the value from one to two and add regression coverage.",
        model="fake-model",
        policy=policy or SandboxPolicy(max_repair_rounds=1),
    )


@pytest.mark.asyncio
async def test_verified_pr_runs_specialists_and_produces_typed_evidence(tmp_path: Path):
    sandbox = FakeSandbox(repository(tmp_path), SandboxPolicy(max_repair_rounds=1))
    actions = SequenceModel(
        [
            CodeAction(kind="write", path="app.py", content="VALUE = 2\n", rationale="Fix value"),
            CodeAction(kind="test", rationale="Verify implementation"),
            CodeAction(
                kind="write",
                path="tests/test_app.py",
                content="from app import VALUE\n\ndef test_value():\n    assert VALUE == 2\n",
                rationale="Add regression test",
            ),
            CodeAction(kind="test", rationale="Verify regression test"),
        ]
    )
    agent = VerifiedPRAgent(SequenceModel([analysis()]), actions, SequenceModel([approval()]))
    try:
        result = await agent.solve(task(sandbox.policy), sandbox)
    finally:
        sandbox.close()

    assert result.status == "completed"
    assert result.verification is not None
    assert result.verification.final_decision == "verified"
    assert result.verification.context is not None
    assert result.verification.context.selected_files
    assert result.verification.context.strategy == "hybrid_rerank"
    assert result.verification.context.embedding_backend.startswith("hashing-semantic")
    assert result.verification.context.reranker_backend == "query-facet-reranker-v1"
    assert result.verification.context.query_plan["concepts"]
    assert result.verification.context.estimated_tokens > 0
    assert result.verification.context.parser_backends
    assert result.sandbox["context_pack_sha256"] == result.verification.context.fingerprint
    assert result.verification.test_files_changed == ["tests/test_app.py"]
    assert result.verification.repair_rounds == 0
    assert {item.stage for item in result.observations} >= {"implementation", "test_author"}
    assert all(gate.status != "failed" for gate in result.verification.rounds[-1].quality_gates)
    assert set(result.changed_files) == {"app.py", "tests/test_app.py"}


@pytest.mark.asyncio
async def test_unsafe_patch_triggers_bounded_repair_and_second_review(tmp_path: Path):
    # Disable pre-tool mediation here to exercise the independent post-patch
    # scanner and bounded repair layer in isolation (defense in depth).
    policy = SandboxPolicy(
        max_repair_rounds=1,
        max_workflow_writes=8,
        security_policy_enabled=False,
    )
    sandbox = FakeSandbox(repository(tmp_path), policy)
    actions = SequenceModel(
        [
            CodeAction(
                kind="write",
                path="app.py",
                content="import os\nVALUE = os.system('echo unsafe')\n",
            ),
            CodeAction(kind="test"),
            CodeAction(kind="test"),
            CodeAction(kind="write", path="app.py", content="VALUE = 2\n"),
            CodeAction(kind="test"),
        ]
    )
    reviewers = SequenceModel([approval(), approval()])
    agent = VerifiedPRAgent(SequenceModel([analysis()]), actions, reviewers)
    try:
        result = await agent.solve(task(policy), sandbox)
    finally:
        sandbox.close()

    assert result.status == "completed"
    assert result.verification is not None
    assert result.verification.repair_rounds == 1
    assert len(result.verification.rounds) == 2
    first_unsafe = next(
        gate
        for gate in result.verification.rounds[0].quality_gates
        if gate.name == "unsafe_code_scan"
    )
    assert first_unsafe.status == "failed"
    assert result.verification.rounds[1].verdict.approved is True
    assert "os.system" not in result.patch


@pytest.mark.asyncio
async def test_high_reviewer_finding_blocks_even_if_model_says_approved(tmp_path: Path):
    policy = SandboxPolicy(max_repair_rounds=0)
    sandbox = FakeSandbox(repository(tmp_path), policy)
    actions = SequenceModel(
        [
            CodeAction(kind="write", path="app.py", content="VALUE = 2\n"),
            CodeAction(kind="test"),
            CodeAction(kind="test"),
        ]
    )
    high_finding = ReviewFinding(
        severity="high",
        category="correctness",
        message="Boundary behavior remains incorrect.",
    )
    agent = VerifiedPRAgent(
        SequenceModel([analysis()]),
        actions,
        SequenceModel([approval([high_finding])]),
    )
    try:
        result = await agent.solve(task(policy), sandbox)
    finally:
        sandbox.close()

    assert result.status == "failed"
    assert result.verification is not None
    assert result.verification.final_decision == "blocked"
    assert "Boundary behavior remains incorrect." in result.verification.blocking_reasons
    assert result.verification.rounds[0].verdict.approved is False


@pytest.mark.asyncio
async def test_test_author_cannot_modify_production_file(tmp_path: Path):
    policy = SandboxPolicy(max_repair_rounds=0)
    sandbox = FakeSandbox(repository(tmp_path), policy)
    actions = SequenceModel(
        [
            CodeAction(kind="write", path="app.py", content="VALUE = 2\n"),
            CodeAction(kind="test"),
            CodeAction(kind="write", path="app.py", content="VALUE = 999\n"),
            CodeAction(kind="test"),
        ]
    )
    agent = VerifiedPRAgent(SequenceModel([analysis()]), actions, SequenceModel([approval()]))
    try:
        result = await agent.solve(task(policy), sandbox)
    finally:
        sandbox.close()

    rejected_write = [
        item for item in result.observations if item.stage == "test_author" and item.action == "write"
    ]
    assert rejected_write and rejected_write[0].ok is False
    assert "VALUE = 999" not in result.patch


@pytest.mark.asyncio
async def test_deterministic_gates_detect_secret_and_source_mutation(tmp_path: Path):
    policy = SandboxPolicy()
    root = repository(tmp_path)
    sandbox = FakeSandbox(root, policy)
    before = repository_fingerprint(root, policy.max_files, policy.max_repository_bytes)
    root.joinpath("app.py").write_text("VALUE = 3\n", encoding="utf-8")
    patch = (
        "--- /dev/null\n+++ b/app.py\n@@ -0,0 +1 @@\n"
        "+api_key = 'sk-abcdefghijklmnopqrstuvwxyz123456'\n"
    )
    coverage_task = task(policy).model_copy(
        update={"test_command": ["python", "-m", "pytest", "--cov=app"]}
    )
    try:
        gates, _ = await run_quality_gates(
            coverage_task,
            sandbox,
            patch=patch,
            changed_files=["app.py"],
            source_fingerprint_before=before,
            baseline_lint=SandboxCommandResult(command=["ruff"], exit_code=0, duration_ms=1),
        )
        coverage_artifact_removed = ".agentforge-coverage.json" not in sandbox.workspace.list_files()
    finally:
        sandbox.close()

    by_name = {gate.name: gate for gate in gates}
    assert by_name["secret_scan"].status == "failed"
    assert by_name["source_repository_integrity"].status == "failed"
    assert by_name["changed_line_coverage"].status == "passed"
    assert coverage_artifact_removed
    assert is_test_path("tests/test_api.py")
    assert not is_test_path("service/api.py")
    diff = "--- a/app.py\n+++ b/app.py\n@@ -2,2 +2,3 @@\n same\n+added\n-old\n+replacement\n"
    assert changed_lines_by_file(diff) == {"app.py": {3, 4}}
