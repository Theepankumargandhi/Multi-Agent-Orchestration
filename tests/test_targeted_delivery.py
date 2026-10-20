"""Exact-edit and fresh-delivery controls; Docker checks never fall back to host execution."""
import asyncio
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from code_agent.agent import CodingAgent
from code_agent.delivery import (
    DeliveryReceipt,
    delivery_gate,
    digest,
    result_delivery_valid,
    safe_patch_paths,
    valid_delivery,
    verify_delivery,
)
from code_agent.models import CodeAction, CodeAgentResult, CodeTask, SandboxPolicy
from code_agent.providers import ProviderAdapter, StructuredOutputError, resolve_model
from code_agent.security_policy import SecurityPolicyEngine
from code_agent.verification import repository_fingerprint
from code_agent.workspace import EphemeralWorkspace, WorkspacePolicyError


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "app.py").write_bytes(b"def value():\r\n    return 1\r\n")
    result = EphemeralWorkspace(root, SandboxPolicy())
    result.prepare()
    yield result
    result.cleanup()


def test_exact_edit_preserves_bytes_and_rejects_stale_hash(workspace):
    read = workspace.read_lines("app.py", 1, 2)
    assert "\r\n" in read["text"]
    result = workspace.edit_file("app.py", read["file_sha256"], "return 1", "return 2")
    assert result == hashlib.sha256(b"def value():\r\n    return 2\r\n").hexdigest()
    assert workspace.resolve("app.py").read_bytes() == b"def value():\r\n    return 2\r\n"
    assert (workspace.source / "app.py").read_bytes() == b"def value():\r\n    return 1\r\n"
    with pytest.raises(WorkspacePolicyError, match="stale"):
        workspace.edit_file("app.py", read["file_sha256"], "return 2", "return 3")


@pytest.mark.parametrize("before,old,new", [("aaa", "aa", "x"), ("a a", "a", "x"), ("abc", "missing", "x"), ("abc", "abc", "abc")])
def test_ambiguous_overlapping_missing_and_noop_edits_rejected(workspace, before, old, new):
    workspace.write_file("app.py", before)
    read = workspace.read_lines("app.py", 1, 1)
    with pytest.raises(WorkspacePolicyError):
        workspace.edit_file("app.py", read["file_sha256"], old, new)
    assert workspace.read_file("app.py") == before


@pytest.mark.parametrize("name", ["../outside.py", ".env", "pytest.ini", "tests/conftest.py", "data/private.py"])
def test_protected_and_escaping_edit_paths_are_denied(workspace, name):
    with pytest.raises(WorkspacePolicyError):
        workspace.check_agent_edit_path(name)


def test_bounded_read_and_atomic_failure_leave_source_unchanged(workspace, monkeypatch):
    with pytest.raises(WorkspacePolicyError):
        workspace.read_lines("app.py", 1, 201)
    receipt = workspace.read_lines("app.py", 1, 2)
    monkeypatch.setattr("code_agent.workspace.os.replace", lambda *_args: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(OSError):
        workspace.edit_file("app.py", receipt["file_sha256"], "return 1", "return 2")
    assert "return 1" in workspace.read_file("app.py")
    assert not list(workspace.root.glob(".agentforge-edit-*"))


@pytest.mark.asyncio
async def test_actual_coding_loop_uses_current_read_receipt_and_write_budget(workspace):
    from tests.test_verified_pr import FakeSandbox
    sandbox = FakeSandbox(workspace.source, SandboxPolicy(targeted_edits_only=True, max_iterations=4))
    class Model:
        calls = 0
        async def ainvoke(self, messages, config=None):
            self.calls += 1
            if self.calls == 1:
                return CodeAction(kind="read_lines", path="app.py", end_line=2)
            if self.calls == 2:
                payload = json.loads(messages[1].content)
                read = json.loads(payload["recent_observations"][0]["output"])
                return CodeAction(kind="edit", path="app.py", expected_sha256=read["file_sha256"], old_text="return 1", new_text="return 2")
            return CodeAction(kind="test")
    try:
        result = await CodingAgent(Model()).solve(CodeTask(repository="repo", issue="Change the return value safely", policy=sandbox.policy), sandbox)
        assert result.status == "completed" and result.writes == 1
        assert [row.action for row in result.observations] == ["test", "read_lines", "edit", "test"]
    finally:
        sandbox.close()


def test_edit_security_has_the_same_dangerous_execution_checks_as_write():
    task = CodeTask(repository="repo", issue="Repair a bounded arithmetic bug")
    action = CodeAction(kind="edit", path="app.py", expected_sha256="a" * 64, old_text="return 1", new_text="return eval('2')")
    assert not SecurityPolicyEngine().evaluate(action, task).allowed


@pytest.mark.parametrize("identifier,expected", [("groq:openai/gpt-oss-20b", ("groq", "openai/gpt-oss-20b")),
    ("openai:gpt-example", ("openai", "gpt-example")), ("llama-example", ("groq", "llama-example"))])
def test_explicit_provider_resolution(identifier, expected):
    assert resolve_model(identifier) == expected


@pytest.mark.parametrize("identifier", ["", "groq:", "unknown:example", " openai:example", "groq:a:b"])
def test_bad_provider_identifiers_never_fall_back(identifier):
    with pytest.raises(ValueError):
        resolve_model(identifier)


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments,valid", [({"kind": "test"}, True), ({"kind": "read_lines", "start_line": True}, False),
    ({"kind": "edit", "path": "app.py"}, False), ({"kind": "test", "unknown": "ignored?"}, False)])
async def test_adapter_validates_original_arguments_without_retries(arguments, valid):
    class Base:
        calls = 0
        def with_structured_output(self, schema, **kwargs):
            assert kwargs == {"method": "function_calling", "include_raw": True}
            return self
        async def ainvoke(self, messages, config=None):
            self.calls += 1
            return {"parsed": {"kind": "test"}, "raw": SimpleNamespace(tool_calls=[{"name": "CodeAction", "args": arguments}]), "parsing_error": None}
    base = Base()
    model = ProviderAdapter(base, "groq", "example").with_structured_output(CodeAction)
    if valid:
        assert (await model.ainvoke([])).kind == "test"
    else:
        with pytest.raises(StructuredOutputError):
            await model.ainvoke([])
    assert base.calls == 1


@pytest.mark.parametrize("patch", ["--- a/../../escape.py\n+++ b/../../escape.py\n", "--- a/.env\n+++ b/.env\n", "diff --git a/x b/x\nnew file mode 120000\n", "not a patch"])
def test_delivery_rejects_unsafe_and_unsupported_patch_forms(patch):
    with pytest.raises(ValueError):
        safe_patch_paths(patch, 20)


def test_integrity_and_patch_binding_are_required_for_delivery_pass():
    proof = DeliveryReceipt(status="passed", reason="fresh_candidate_tests_passed", patch_sha256="a" * 64,
        source_sha256="b" * 64, candidate_sha256="c" * 64, command_sha256="d" * 64, image="safe",
        image_id="sha256:" + "e" * 64, application_exit_code=0, test_exit_code=0,
        candidate_matches=True, source_unchanged=True, protected_files_unchanged=True)
    proof.fingerprint = digest(proof.model_dump(mode="json", exclude={"fingerprint"}))
    assert valid_delivery(proof) and delivery_gate(proof).status == "passed"
    assert not valid_delivery(proof, patch="different artifact")
    proof.test_exit_code = 1
    assert not valid_delivery(proof)
    assert not result_delivery_valid(CodeAgentResult(status="completed", summary="No proof"))


@pytest.mark.parametrize("provider", ["openai", "groq"])
def test_factory_routes_explicit_models_and_disables_sdk_retries(monkeypatch, provider):
    from code_agent.providers import build_provider
    calls = []
    def constructor(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()
    module = SimpleNamespace(**{"ChatOpenAI" if provider == "openai" else "ChatGroq": constructor})
    monkeypatch.setitem(sys.modules, "langchain_openai" if provider == "openai" else "langchain_groq", module)
    built = build_provider(provider + ":example", max_completion_tokens=512, timeout_seconds=20)
    assert built.provider == provider and built.model == "example"
    assert calls == [{"model": "example", "temperature": 0, "max_tokens": 512, "timeout": 20, "max_retries": 0}]
    with pytest.raises(ValueError):
        build_provider(provider + ":example", max_completion_tokens=100000)


@pytest.mark.asyncio
async def test_incomplete_specialist_stage_cannot_be_reported_as_completed_team(tmp_path):
    from code_agent.verified_pr import VerifiedPRAgent
    from tests.test_verified_pr import FakeSandbox, SequenceModel, analysis, approval, repository
    policy = SandboxPolicy(require_specialist_completion=True, max_iterations=1, max_repair_rounds=0)
    sandbox = FakeSandbox(repository(tmp_path), policy)
    task = CodeTask(repository="repo", issue="Fix one value and require completed specialist stages", policy=policy)
    agent = VerifiedPRAgent(SequenceModel([analysis()]), SequenceModel([CodeAction(kind="write", path="app.py", content="VALUE = 2\n")]), SequenceModel([approval()]))
    try:
        result = await agent.solve(task, sandbox)
        assert result.status == "failed"
        assert result.verification.stage_outcomes["implementation"] == "budget_exhausted"
        assert result.verification.stage_outcomes["test_author"].startswith("not_run")
        assert any(row.name == "specialist_stage_completion" and row.status == "failed" for row in result.verification.rounds[-1].quality_gates)
    finally:
        sandbox.close()


@pytest.mark.asyncio
async def test_real_docker_fresh_patch_replay_does_not_run_repository_code_on_host(tmp_path):
    from code_agent.sandbox import DockerSandbox
    image = "agentforge-code-sandbox:local"
    if not DockerSandbox.available() or not DockerSandbox.image_available(image):
        pytest.skip("Real Docker image required; no host fallback")
    root = tmp_path / "original"
    root.mkdir()
    (root / "app.py").write_bytes(b"VALUE = 1\n")
    (root / "test_app.py").write_bytes(b"from app import VALUE\ndef test_value():\n    assert VALUE == 2\n")
    policy = SandboxPolicy(delivery_verification_enabled=True, command_timeout_seconds=30)
    task = CodeTask(repository="fixture", issue="Change one value and verify fresh delivery", policy=policy)
    before = repository_fingerprint(root, policy.max_files, policy.max_repository_bytes)
    candidate = DockerSandbox(root, policy)
    try:
        await asyncio.to_thread(candidate.start)
        candidate.workspace.edit_file("app.py", candidate.workspace.read_lines("app.py", 1, 1)["file_sha256"], "VALUE = 1", "VALUE = 2")
        proof = await verify_delivery(task, candidate, candidate.workspace.unified_diff(), before)
        assert valid_delivery(proof, patch=candidate.workspace.unified_diff(), command=task.test_command)
    finally:
        await asyncio.to_thread(candidate.close)
    assert (root / "app.py").read_bytes() == b"VALUE = 1\n"
