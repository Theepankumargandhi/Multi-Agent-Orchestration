"""Docker execution boundary for tests run against an ephemeral workspace."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from uuid import uuid4

from code_agent.models import SandboxCommandResult, SandboxPolicy
from code_agent.workspace import EphemeralWorkspace, WorkspacePolicyError


class DockerUnavailableError(RuntimeError):
    pass


class DockerSandbox:
    def __init__(self, repository: Path, policy: SandboxPolicy):
        self.repository = repository.resolve()
        self.policy = policy
        self.workspace = EphemeralWorkspace(self.repository, policy)
        self.container_id: str | None = None

    @staticmethod
    def available(timeout_seconds: int = 5) -> bool:
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def image_available(image: str, timeout_seconds: int = 5) -> bool:
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def start(self) -> "DockerSandbox":
        if self.policy.image not in self.policy.allowed_images:
            raise WorkspacePolicyError("sandbox image is not allowlisted")
        if not self.available():
            raise DockerUnavailableError("Docker engine is unavailable")
        if not self.image_available(self.policy.image):
            raise DockerUnavailableError("Allowlisted sandbox image is not available locally")
        workspace = self.workspace.prepare()
        if "," in str(workspace):
            self.workspace.cleanup()
            raise WorkspacePolicyError("workspace path contains an unsupported comma")
        name = f"agentforge-code-{uuid4().hex[:12]}"
        network = "bridge" if self.policy.network_enabled else "none"
        command = [
            "docker",
            "create",
            "--name",
            name,
            "--network",
            network,
            "--cpus",
            str(self.policy.cpus),
            "--memory",
            self.policy.memory,
            "--pids-limit",
            str(self.policy.pids_limit),
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=128m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65532:65532",
            "--workdir",
            "/workspace",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONUNBUFFERED=1",
            "--mount",
            f"type=bind,source={workspace},target=/workspace",
            self.policy.image,
            "python",
            "-c",
            "import time; time.sleep(10**9)",
        ]
        try:
            created = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if created.returncode != 0:
                raise DockerUnavailableError(created.stderr.strip() or "failed to create sandbox")
            self.container_id = created.stdout.strip()
            started = subprocess.run(
                ["docker", "start", self.container_id],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if started.returncode != 0:
                raise DockerUnavailableError(started.stderr.strip() or "failed to start sandbox")
            return self
        except Exception:
            self.close()
            raise

    def _validate_command(self, command: list[str]) -> None:
        if not command or command[0] not in self.policy.allowed_test_executables:
            raise WorkspacePolicyError("test executable is not allowlisted")
        if len(command) > 30 or any(not arg or len(arg) > 500 or "\x00" in arg for arg in command):
            raise WorkspacePolicyError("test command exceeds configured limits")

    def run(self, command: list[str]) -> SandboxCommandResult:
        self._validate_command(command)
        if not self.container_id:
            raise RuntimeError("sandbox is not running")
        started = time.perf_counter()
        try:
            result = subprocess.run(
                ["docker", "exec", "--user", "65532:65532", self.container_id, *command],
                capture_output=True,
                text=True,
                timeout=self.policy.command_timeout_seconds,
                check=False,
            )
            duration = (time.perf_counter() - started) * 1000
            return SandboxCommandResult(
                command=command,
                exit_code=result.returncode,
                stdout=result.stdout[-self.policy.max_output_chars :],
                stderr=result.stderr[-self.policy.max_output_chars :],
                duration_ms=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = (time.perf_counter() - started) * 1000
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return SandboxCommandResult(
                command=command,
                exit_code=124,
                stdout=stdout[-self.policy.max_output_chars :],
                stderr=stderr[-self.policy.max_output_chars :],
                duration_ms=duration,
                timed_out=True,
            )

    def close(self) -> None:
        container_id, self.container_id = self.container_id, None
        try:
            if container_id:
                try:
                    subprocess.run(
                        ["docker", "rm", "--force", container_id],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
        finally:
            self.workspace.cleanup()

    def __enter__(self) -> "DockerSandbox":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
