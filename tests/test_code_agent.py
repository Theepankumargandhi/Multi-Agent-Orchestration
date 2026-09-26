from pathlib import Path
from types import SimpleNamespace

import pytest

from code_agent.agent import CodingAgent
from code_agent.api import CodeTaskDecision, CodeTaskManager, JobRecord
from code_agent.evaluation import score_code_result
from code_agent.models import (
    CodeAction,
    CodeAgentResult,
    CodeTask,
    SandboxCommandResult,
    SandboxPolicy,
)
from code_agent.repository_map import build_repository_map
from code_agent.sandbox import DockerSandbox
from code_agent.workspace import EphemeralWorkspace, WorkspacePolicyError


def test_ephemeral_workspace_filters_secrets_confines_paths_and_builds_diff(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
    (source / "private.key").write_text("secret\n", encoding="utf-8")
    (source / "data").mkdir()
    (source / "data" / "messages.db").write_text("private", encoding="utf-8")

    workspace = EphemeralWorkspace(source, SandboxPolicy())
    root = workspace.prepare()
    try:
        assert workspace.list_files() == ["app.py"]
        with pytest.raises(WorkspacePolicyError, match="escapes"):
            workspace.read_file("../.env")
        with pytest.raises(WorkspacePolicyError, match="secret"):
            workspace.write_file(".env", "NEW_SECRET=value")
        workspace.write_file("app.py", "VALUE = 2\n")
        workspace.write_file("tests/test_app.py", "def test_value():\n    assert 2 == 2\n")
        (root / ".pytest_cache").mkdir()
        (root / ".pytest_cache" / "runtime.txt").write_text("generated", encoding="utf-8")
        patch = workspace.unified_diff()
        assert "-VALUE = 1" in patch and "+VALUE = 2" in patch
        assert "tests/test_app.py" in patch
        assert ".pytest_cache" not in patch
        assert (source / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    finally:
        workspace.cleanup()
    assert not root.exists()


def test_sandbox_rejects_shell_and_non_allowlisted_images(tmp_path: Path):
    policy = SandboxPolicy()
    sandbox = DockerSandbox(tmp_path, policy)
    sandbox._validate_command(["python", "-m", "pytest", "-q"])
    with pytest.raises(WorkspacePolicyError, match="executable"):
        sandbox._validate_command(["sh", "-c", "cat /etc/passwd"])
    with pytest.raises(ValueError, match="allowlisted"):
        SandboxPolicy(image="untrusted:latest")


def test_sandbox_container_command_enforces_hardening(tmp_path: Path, monkeypatch):
    repository = tmp_path / "source"
    repository.mkdir()
    (repository / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:2] == ["docker", "info"]:
            return SimpleNamespace(returncode=0, stdout="27.0", stderr="")
        if command[:2] == ["docker", "create"]:
            return SimpleNamespace(returncode=0, stdout="sandbox-id\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("code_agent.sandbox.subprocess.run", fake_run)
    sandbox = DockerSandbox(repository, SandboxPolicy())
    try:
        sandbox.start()
        create = next(command for command in commands if command[:2] == ["docker", "create"])
        assert create[create.index("--network") + 1] == "none"
        assert "--read-only" in create
        assert create[create.index("--cap-drop") + 1] == "ALL"
        assert create[create.index("--security-opt") + 1] == "no-new-privileges"
        assert create[create.index("--user") + 1] == "65532:65532"
        mount = create[create.index("--mount") + 1]
        assert "target=/workspace" in mount
        assert str(repository) not in mount
    finally:
        sandbox.close()
    assert ["docker", "rm", "--force", "sandbox-id"] in commands
    cleanup = next(command for command in commands if command[:2] == ["docker", "run"])
    assert cleanup[cleanup.index("--network") + 1] == "none"
    assert "--read-only" in cleanup
    assert cleanup[cleanup.index("--cap-drop") + 1] == "ALL"
    assert cleanup[cleanup.index("--security-opt") + 1] == "no-new-privileges"
    assert cleanup[cleanup.index("--user") + 1] == "65532:65532"
    assert cleanup[cleanup.index("--entrypoint") + 1] == "python"
    assert cleanup[-3:-1] == ["-I", "-c"]


@pytest.mark.integration
def test_real_docker_sandbox_is_non_root_offline_read_only_and_testable(tmp_path: Path):
    if not DockerSandbox.available() or not DockerSandbox.image_available(SandboxPolicy().image):
        pytest.skip("Docker engine or sandbox image is unavailable")
    repository = tmp_path / "source"
    repository.mkdir()
    (repository / "test_smoke.py").write_text("def test_smoke():\n    assert 2 + 2 == 4\n", encoding="utf-8")
    sandbox = DockerSandbox(repository, SandboxPolicy(command_timeout_seconds=30))
    try:
        sandbox.start()
        identity = sandbox.run(["python", "-c", "import os; print(os.getuid(), os.getgid())"])
        assert identity.exit_code == 0
        assert identity.stdout.strip() == "65532 65532"

        root_write = sandbox.run(
            [
                "python",
                "-c",
                "from pathlib import Path; Path('/blocked-by-read-only-root').write_text('x')",
            ]
        )
        assert root_write.exit_code != 0

        network = sandbox.run(
            [
                "python",
                "-c",
                "import socket; socket.create_connection(('1.1.1.1', 53), timeout=1)",
            ]
        )
        assert network.exit_code != 0

        tests = sandbox.run(["python", "-m", "pytest", "-q"])
        assert tests.exit_code == 0
        assert "1 passed" in tests.stdout
    finally:
        sandbox.close()
    assert not (repository / "blocked-by-read-only-root").exists()


class FakeWorkspace:
    def __init__(self):
        self.files = {"app.py": "VALUE = 1\n"}
        self.original = dict(self.files)

    def list_files(self, limit=500):
        return sorted(self.files)[:limit]

    def read_file(self, path):
        return self.files[path]

    def search(self, pattern, limit=100):
        return ["app.py:1: VALUE = 1"] if "VALUE" in pattern else []

    def write_file(self, path, content):
        self.files[path] = content

    def delete_file(self, path):
        del self.files[path]

    def changed_files(self):
        return sorted(name for name in set(self.files) | set(self.original) if self.files.get(name) != self.original.get(name))

    def unified_diff(self):
        return "--- a/app.py\n+++ b/app.py\n-VALUE = 1\n+VALUE = 2\n"


def test_repository_map_extracts_python_symbols_without_executing_code():
    workspace = FakeWorkspace()
    workspace.files["worker.py"] = (
        "import os\n\nclass Worker:\n    def run(self):\n        return 1\n\ndef helper():\n    return 2\n"
    )
    repository_map = build_repository_map(workspace)
    assert "worker.py" in repository_map
    assert "class: Worker" in repository_map
    assert "method: Worker.run" in repository_map
    assert "function: helper" in repository_map


class FakeSandbox:
    def __init__(self):
        self.workspace = FakeWorkspace()
        self.test_runs = 0

    def run(self, command):
        self.test_runs += 1
        passed = self.workspace.files["app.py"] == "VALUE = 2\n"
        return SandboxCommandResult(
            command=command,
            exit_code=0 if passed else 1,
            stdout="1 passed" if passed else "1 failed",
            duration_ms=5,
        )


class FakeActionModel:
    def __init__(self):
        self.actions = iter(
            [
                CodeAction(kind="read", path="app.py", rationale="Inspect the failing value."),
                CodeAction(kind="write", path="app.py", content="VALUE = 2\n", rationale="Fix it."),
                CodeAction(kind="test", rationale="Verify the change."),
            ]
        )

    async def ainvoke(self, messages, config=None):
        assert messages
        assert config and config.get("callbacks")
        return next(self.actions)


class OpenAIRateLimitError(RuntimeError):
    pass


class FlakyActionModel(FakeActionModel):
    def __init__(self):
        super().__init__()
        self.failed_once = False

    async def ainvoke(self, messages, config=None):
        if not self.failed_once:
            self.failed_once = True
            raise OpenAIRateLimitError("retry this transient failure")
        return await super().ainvoke(messages, config=config)


@pytest.mark.asyncio
async def test_coding_agent_requires_changed_files_and_passing_tests():
    task = CodeTask(repository="fixture", issue="Change VALUE from one to two and verify tests.")
    result = await CodingAgent(FakeActionModel()).solve(task, FakeSandbox())
    assert result.status == "completed"
    assert result.baseline_test.exit_code == 1
    assert result.final_test.exit_code == 0
    assert result.changed_files == ["app.py"]
    assert result.writes == 1
    assert "VALUE = 2" in result.patch
    assert result.sandbox["original_repository_unchanged"] is True
    assert result.telemetry is not None
    assert result.telemetry.spans[0].attributes["gen_ai.operation.name"] == "invoke_agent"
    score = score_code_result(result)
    assert score.resolved is True
    assert score.tests_passed is True
    assert score.patch_nonempty is True


@pytest.mark.asyncio
async def test_coding_agent_retries_transient_provider_errors(monkeypatch):
    async def no_wait(delay):
        return None

    monkeypatch.setattr("code_agent.agent.asyncio.sleep", no_wait)
    task = CodeTask(repository="fixture", issue="Change VALUE from one to two and verify tests.")
    result = await CodingAgent(FlakyActionModel()).solve(task, FakeSandbox())
    assert result.status == "completed"
    assert result.model_calls == 4


@pytest.mark.integration
@pytest.mark.asyncio
async def test_coding_agent_fixes_failure_in_real_sandbox_without_touching_source(tmp_path: Path):
    if not DockerSandbox.available() or not DockerSandbox.image_available(SandboxPolicy().image):
        pytest.skip("Docker engine or sandbox image is unavailable")
    repository = tmp_path / "source"
    repository.mkdir()
    (repository / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "test_app.py").write_text(
        "from app import VALUE\n\ndef test_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )
    task = CodeTask(
        repository="fixture",
        issue="Change VALUE from one to two so the repository test passes.",
        policy=SandboxPolicy(command_timeout_seconds=60),
    )
    sandbox = DockerSandbox(repository, task.policy)
    try:
        sandbox.start()
        result = await CodingAgent(FakeActionModel()).solve(task, sandbox)
    finally:
        sandbox.close()
    assert result.status == "completed"
    assert result.baseline_test.exit_code != 0
    assert result.final_test.exit_code == 0
    assert "VALUE = 2" in result.patch
    assert (repository / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_public_job_record_hides_patch_until_approval():
    request = {
        "repository": "fixture",
        "issue": "Fix a reproducible defect in the fixture repository.",
        "model": "test-model",
        "test_command": ["python", "-m", "pytest"],
    }
    result = CodeAgentResult(status="completed", summary="passed", patch="SECRET PATCH")
    record = JobRecord(
        task_id="task",
        user_id="user",
        status="awaiting_approval",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        request=request,
        result=result,
    )
    assert record.public()["result"]["patch"] == ""
    assert record.result.patch == "SECRET PATCH"


@pytest.mark.asyncio
async def test_patch_download_is_user_scoped_and_approval_gated(tmp_path: Path):
    manager = CodeTaskManager(tmp_path, SandboxPolicy())
    record = JobRecord(
        task_id="task",
        user_id="owner",
        status="awaiting_approval",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        request={
            "repository": "fixture",
            "issue": "Fix a reproducible defect in the fixture repository.",
            "model": "test-model",
            "test_command": ["python", "-m", "pytest"],
        },
        result=CodeAgentResult(status="completed", summary="passed", patch="approved patch"),
    )
    manager.jobs[record.task_id] = record
    with pytest.raises(Exception) as not_approved:
        await manager.patch("task", "owner")
    assert getattr(not_approved.value, "status_code", None) == 403
    with pytest.raises(Exception) as wrong_user:
        await manager.get("task", "other-user")
    assert getattr(wrong_user.value, "status_code", None) == 404
    await manager.decide("task", "owner", CodeTaskDecision(approve=True))
    assert await manager.patch("task", "owner") == "approved patch"
