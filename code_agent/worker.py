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
from code_agent.repository_access import RepositoryAccess
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
        repository_access: RepositoryAccess | None = None,
    ):
        self.store = store
        self.repository_root = repository_root.resolve()
        self.repository_access = repository_access or RepositoryAccess()
        self.policy = policy
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}"
        self.lease_seconds = max(15, lease_seconds)
        if workflow not in {"verified_pr", "single_agent", "repair_tournament", "adaptive_supervisor"}:
            raise ValueError("workflow must be verified_pr, single_agent, repair_tournament, or adaptive_supervisor")
        if policy.counterfactual_verification_enabled and workflow == "single_agent":
            raise ValueError("counterfactual verification requires a verified-PR workflow")
        if policy.patch_minimization_enabled and workflow == "single_agent":
            raise ValueError("patch minimization requires a verified-PR workflow")
        if policy.mutation_testing_mode != "off" and workflow == "single_agent":
            raise ValueError("patch mutation testing requires a verified-PR workflow")
        self.workflow = workflow
        self.solver = solver or self._solve

    async def _solve(self, job: JobRecord) -> CodeAgentResult:
        self.repository_access.require(job.user_id, job.request.repository, self.repository_root)
        task = CodeTask(
            repository=job.request.repository,
            repair_memory_namespace=job.user_id,
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
            if self.workflow in {"repair_tournament", "adaptive_supervisor"}:
                from code_agent.repair_tournament import build_adaptive_supervisor, build_repair_tournament

                builder = build_adaptive_supervisor if self.workflow == "adaptive_supervisor" else build_repair_tournament
                return await builder(task.model).solve(task, repository)
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
            retryable = not isinstance(exc, (ValueError, PermissionError))
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
        targeted_edits_only=os.getenv("CODE_AGENT_TARGETED_EDITS_ONLY", "true").lower() == "true",
        delivery_verification_enabled=os.getenv("CODE_AGENT_DELIVERY_VERIFICATION_ENABLED", "true").lower() == "true",
        require_specialist_completion=os.getenv("CODE_AGENT_REQUIRE_SPECIALISTS", "true").lower() == "true",
        execution_retrieval_enabled=os.getenv("CODE_AGENT_EXECUTION_RETRIEVAL_ENABLED", "false").strip().lower()
        in {"1", "true", "yes", "on"},
        repair_memory_enabled=os.getenv("CODE_AGENT_REPAIR_MEMORY_ENABLED", "false").strip().lower()
        in {"1", "true", "yes", "on"},
        counterfactual_verification_enabled=os.getenv("CODE_AGENT_COUNTERFACTUAL_VERIFICATION_ENABLED", "false").strip().lower()
        in {"1", "true", "yes", "on"},
        counterfactual_timeout_seconds=int(os.getenv("CODE_AGENT_COUNTERFACTUAL_TIMEOUT_SECONDS", "180")),
        execution_retrieval_max_refreshes=int(os.getenv("CODE_AGENT_EXECUTION_RETRIEVAL_MAX_REFRESHES", "4")),
        execution_retrieval_max_tokens=int(os.getenv("CODE_AGENT_EXECUTION_RETRIEVAL_MAX_TOKENS", "1024")),
        fault_localization_enabled=os.getenv("CODE_AGENT_FAULT_LOCALIZATION_ENABLED", "false").strip().lower()
        in {"1", "true", "yes", "on"},
        fault_localization_max_refreshes=int(os.getenv("CODE_AGENT_FAULT_LOCALIZATION_MAX_REFRESHES", "2")),
        fault_localization_timeout_seconds=int(os.getenv("CODE_AGENT_FAULT_LOCALIZATION_TIMEOUT_SECONDS", "120")),
        fault_localization_max_tokens=int(os.getenv("CODE_AGENT_FAULT_LOCALIZATION_MAX_TOKENS", "768")),
        patch_minimization_enabled=os.getenv("CODE_AGENT_PATCH_MINIMIZATION_ENABLED", "false").strip().lower()
        in {"1", "true", "yes", "on"},
        patch_minimization_max_files=int(os.getenv("CODE_AGENT_PATCH_MINIMIZATION_MAX_FILES", "3")),
        patch_minimization_timeout_seconds=int(os.getenv("CODE_AGENT_PATCH_MINIMIZATION_TIMEOUT_SECONDS", "180")),
        loop_guard_mode=os.getenv("CODE_AGENT_LOOP_GUARD_MODE", "off").strip().lower(),
        loop_guard_repetition_limit=int(os.getenv("CODE_AGENT_LOOP_GUARD_REPETITION_LIMIT", "3")),
        loop_guard_max_cycle_length=int(os.getenv("CODE_AGENT_LOOP_GUARD_MAX_CYCLE_LENGTH", "4")),
        mutation_testing_mode=os.getenv("CODE_AGENT_MUTATION_TESTING_MODE", "off").strip().lower(),
        mutation_testing_max_mutants=int(os.getenv("CODE_AGENT_MUTATION_TESTING_MAX_MUTANTS", "3")),
        mutation_testing_min_mutants=int(os.getenv("CODE_AGENT_MUTATION_TESTING_MIN_MUTANTS", "1")),
        mutation_testing_min_score=float(os.getenv("CODE_AGENT_MUTATION_TESTING_MIN_SCORE", "1.0")),
        mutation_testing_timeout_seconds=int(os.getenv("CODE_AGENT_MUTATION_TESTING_TIMEOUT_SECONDS", "180")),
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
