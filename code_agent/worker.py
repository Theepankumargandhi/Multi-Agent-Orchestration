"""Lease-based worker for durable sandbox coding jobs."""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from code_agent.agent import CodingAgent, build_coding_model
from code_agent.execution import LocalArtifactStore, SQLiteJobStore
from code_agent.job_models import JobRecord
from code_agent.models import CodeAgentResult, CodeTask, SandboxPolicy
from code_agent.sandbox import DockerSandbox
from code_agent.verified_pr import build_verified_pr_agent

Solver = Callable[[JobRecord], Awaitable[CodeAgentResult]]


def repository_path(root: Path, relative: str, policy: SandboxPolicy) -> Path:
    validated = CodeTask(
        repository=relative,
        issue="Validate repository path for a sandbox coding task.",
        policy=policy,
    ).repository
    candidate = (root.resolve() / validated).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("repository escapes configured root") from exc
    if not candidate.is_dir():
        raise ValueError("repository does not exist below configured root")
    return candidate


class CodeAgentWorker:
    def __init__(
        self,
        store: SQLiteJobStore,
        repository_root: Path,
        policy: SandboxPolicy,
        *,
        worker_id: str | None = None,
        lease_seconds: int = 90,
        solver: Solver | None = None,
        workflow: str = "verified_pr",
    ):
        self.store = store
        self.repository_root = repository_root.resolve()
        self.policy = policy
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}"
        self.lease_seconds = max(15, lease_seconds)
        if workflow not in {"verified_pr", "single_agent"}:
            raise ValueError("workflow must be verified_pr or single_agent")
        self.workflow = workflow
        self.solver = solver or self._solve

    async def _solve(self, job: JobRecord) -> CodeAgentResult:
        task = CodeTask(
            repository=job.request.repository,
            issue=job.request.issue,
            model=job.request.model,
            test_command=job.request.test_command,
            policy=self.policy,
        )
        if task.test_command[0] not in self.policy.allowed_test_executables:
            raise ValueError("test executable is not allowlisted")
        sandbox: DockerSandbox | None = None
        try:
            repository = repository_path(self.repository_root, task.repository, self.policy)
            sandbox = DockerSandbox(repository, task.policy)
            await asyncio.to_thread(sandbox.start)
            agent = (
                build_verified_pr_agent(task.model)
                if self.workflow == "verified_pr"
                else CodingAgent(build_coding_model(task.model))
            )
            return await asyncio.wait_for(
                agent.solve(task, sandbox), timeout=task.policy.task_timeout_seconds + 30
            )
        finally:
            if sandbox is not None:
                await asyncio.to_thread(sandbox.close)

    async def _heartbeat(self, task_id: str, stopped: asyncio.Event) -> None:
        interval = max(5, self.lease_seconds // 3)
        while not stopped.is_set():
            try:
                await asyncio.wait_for(stopped.wait(), timeout=interval)
            except TimeoutError:
                owns_lease = await asyncio.to_thread(
                    self.store.heartbeat, task_id, self.worker_id, self.lease_seconds
                )
                if not owns_lease:
                    return

    async def run_once(self) -> JobRecord | None:
        job = await asyncio.to_thread(self.store.claim, self.worker_id, self.lease_seconds)
        if job is None:
            return None
        stopped = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(job.task_id, stopped))
        try:
            result = await self.solver(job)
            return await asyncio.to_thread(self.store.complete, job.task_id, self.worker_id, result)
        except asyncio.CancelledError:
            await asyncio.to_thread(self.store.release, job.task_id, self.worker_id)
            raise
        except Exception as exc:
            retryable = not isinstance(exc, ValueError)
            message = f"Sandbox coding task failed: {type(exc).__name__}"
            return await asyncio.to_thread(
                self.store.fail, job.task_id, self.worker_id, message, retryable
            )
        finally:
            stopped.set()
            await heartbeat

    async def run_forever(self, poll_seconds: float = 1.0) -> None:
        while True:
            result = await self.run_once()
            if result is None:
                await asyncio.sleep(max(0.1, poll_seconds))


def build_worker_from_env() -> CodeAgentWorker:
    load_dotenv()
    image = os.getenv("CODE_AGENT_IMAGE", "agentforge-code-sandbox:local").strip()
    policy = SandboxPolicy(
        image=image,
        allowed_images={image},
        network_enabled=os.getenv("CODE_AGENT_NETWORK_ENABLED", "false").strip().lower()
        in {"1", "true", "yes", "on"},
        security_policy_enabled=os.getenv("CODE_AGENT_SECURITY_POLICY_ENABLED", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        security_policy_version=os.getenv(
            "CODE_AGENT_SECURITY_POLICY_VERSION", "agentforge-security-v1"
        ).strip(),
        max_iterations=int(os.getenv("CODE_AGENT_MAX_ITERATIONS", "20")),
        max_writes=int(os.getenv("CODE_AGENT_MAX_WRITES", "8")),
        max_changed_files=int(os.getenv("CODE_AGENT_MAX_CHANGED_FILES", "20")),
        max_repair_rounds=int(os.getenv("CODE_AGENT_MAX_REPAIR_ROUNDS", "2")),
        max_workflow_writes=int(os.getenv("CODE_AGENT_MAX_WORKFLOW_WRITES", "16")),
        min_changed_line_coverage=float(
            os.getenv("CODE_AGENT_MIN_CHANGED_LINE_COVERAGE", "0.8")
        ),
        task_timeout_seconds=int(os.getenv("CODE_AGENT_TASK_TIMEOUT_SECONDS", "900")),
    )
    artifacts = LocalArtifactStore(
        Path(os.getenv("CODE_AGENT_ARTIFACT_ROOT", "data/code-agent/artifacts")),
        max_bytes=int(os.getenv("CODE_AGENT_MAX_ARTIFACT_BYTES", "4000000")),
    )
    store = SQLiteJobStore(
        Path(os.getenv("CODE_AGENT_DB_PATH", "data/code-agent/jobs.db")),
        artifacts,
        max_records=int(os.getenv("CODE_AGENT_MAX_RECORDS", "5000")),
    )
    return CodeAgentWorker(
        store,
        Path(os.getenv("CODE_AGENT_REPOSITORY_ROOT", "repositories")),
        policy,
        lease_seconds=int(os.getenv("CODE_AGENT_LEASE_SECONDS", "90")),
        workflow=os.getenv("CODE_AGENT_WORKFLOW", "verified_pr").strip().lower(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the durable sandbox coding-agent worker")
    parser.add_argument("--once", action="store_true", help="Claim at most one job and exit")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    worker = build_worker_from_env()
    if args.once:
        asyncio.run(worker.run_once())
    else:
        try:
            asyncio.run(worker.run_forever(args.poll_seconds))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
