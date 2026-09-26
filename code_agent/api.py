"""Authenticated asynchronous API for approval-gated sandbox coding jobs."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request, Response
from prometheus_client import REGISTRY
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

from code_agent.agent import CodingAgent, build_coding_model
from code_agent.execution import (
    ArtifactIntegrityError,
    IdempotencyConflictError,
    JobStateError,
    LocalArtifactStore,
    SQLiteJobStore,
)
from code_agent.job_models import CodeTaskDecision, CodeTaskRequest, JobRecord
from code_agent.models import CodeTask, SandboxPolicy
from code_agent.sandbox import DockerSandbox


class CodeTaskManager:
    """Legacy in-memory manager retained for backwards-compatible library use."""

    def __init__(
        self,
        repository_root: Path,
        policy: SandboxPolicy,
        max_concurrency: int = 1,
        max_records: int = 200,
    ):
        self.repository_root = repository_root.resolve()
        self.policy = policy
        self.jobs: dict[str, JobRecord] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(max(1, min(max_concurrency, 4)))
        self.max_records = max(20, min(max_records, 5000))

    def _prune_terminal_jobs(self) -> None:
        overflow = len(self.jobs) - self.max_records + 1
        if overflow <= 0:
            return
        terminal = sorted(
            (
                record
                for record in self.jobs.values()
                if record.status in {"approved", "rejected", "failed"}
            ),
            key=lambda record: record.updated_at,
        )
        for record in terminal[:overflow]:
            self.jobs.pop(record.task_id, None)

    def _repository_path(self, relative: str) -> Path:
        validated = CodeTask(
            repository=relative,
            issue="Validate repository path for a sandbox coding task.",
            policy=self.policy,
        ).repository
        candidate = (self.repository_root / validated).resolve()
        try:
            candidate.relative_to(self.repository_root)
        except ValueError as exc:
            raise ValueError("repository escapes configured root") from exc
        if not candidate.is_dir():
            raise ValueError("repository does not exist below configured root")
        return candidate

    async def submit(self, user_id: str, request: CodeTaskRequest) -> JobRecord:
        try:
            self._repository_path(request.repository)
            task = CodeTask(
                repository=request.repository,
                issue=request.issue,
                model=request.model,
                test_command=request.test_command,
                policy=self.policy,
            )
            if task.test_command[0] not in self.policy.allowed_test_executables:
                raise ValueError("test executable is not allowlisted")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        now = datetime.now(UTC).isoformat()
        record = JobRecord(
            task_id=str(uuid4()),
            user_id=user_id,
            status="queued",
            created_at=now,
            updated_at=now,
            request=request,
        )
        async with self.lock:
            active = sum(not task.done() for task in self.tasks.values())
            if active >= 20:
                raise HTTPException(status_code=429, detail="Too many queued coding tasks")
            self._prune_terminal_jobs()
            if len(self.jobs) >= self.max_records:
                raise HTTPException(status_code=429, detail="Coding task record capacity reached")
            self.jobs[record.task_id] = record
            self.tasks[record.task_id] = asyncio.create_task(self._run(record.task_id, task))
        return record

    async def _run(self, task_id: str, task: CodeTask) -> None:
        sandbox: DockerSandbox | None = None
        async with self.semaphore:
            record = self.jobs[task_id]
            record.status = "running"
            record.updated_at = datetime.now(UTC).isoformat()
            try:
                repository = self._repository_path(task.repository)
                sandbox = DockerSandbox(repository, task.policy)
                await asyncio.to_thread(sandbox.start)
                agent = CodingAgent(build_coding_model(task.model))
                result = await asyncio.wait_for(
                    agent.solve(task, sandbox), timeout=task.policy.task_timeout_seconds + 30
                )
                record.result = result
                record.status = "awaiting_approval" if result.status == "completed" else "failed"
                if result.status != "completed":
                    record.error = result.summary
            except asyncio.CancelledError:
                record.status = "failed"
                record.error = "Coding task cancelled during shutdown."
                raise
            except Exception as exc:
                record.status = "failed"
                record.error = f"Sandbox coding task failed: {type(exc).__name__}"
            finally:
                if sandbox is not None:
                    await asyncio.to_thread(sandbox.close)
                record.updated_at = datetime.now(UTC).isoformat()
                self.tasks.pop(task_id, None)

    async def get(self, task_id: str, user_id: str) -> JobRecord:
        record = self.jobs.get(task_id)
        if record is None or record.user_id != user_id:
            raise HTTPException(status_code=404, detail="Coding task not found")
        return record

    async def decide(self, task_id: str, user_id: str, decision: CodeTaskDecision) -> JobRecord:
        record = await self.get(task_id, user_id)
        if record.status != "awaiting_approval":
            raise HTTPException(status_code=409, detail="Coding task is not awaiting approval")
        record.status = "approved" if decision.approve else "rejected"
        record.decision_reason = decision.reason.strip()
        record.updated_at = datetime.now(UTC).isoformat()
        if not decision.approve and record.result is not None:
            record.result.patch = ""
        return record

    async def patch(self, task_id: str, user_id: str) -> str:
        record = await self.get(task_id, user_id)
        if record.status != "approved" or record.result is None:
            raise HTTPException(status_code=403, detail="Patch requires explicit approval")
        return record.result.patch

    async def cancel_all(self) -> None:
        tasks = list(self.tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class DurableCodeTaskManager:
    """API facade over the durable queue, with an optional embedded worker pool."""

    def __init__(
        self,
        store: SQLiteJobStore,
        repository_root: Path,
        policy: SandboxPolicy,
        *,
        execution_mode: str = "embedded",
        max_concurrency: int = 1,
        max_attempts: int = 3,
        lease_seconds: int = 90,
        workflow: str = "verified_pr",
    ):
        if execution_mode not in {"embedded", "external"}:
            raise ValueError("CODE_AGENT_EXECUTION_MODE must be embedded or external")
        self.store = store
        self.repository_root = repository_root.resolve()
        self.policy = policy
        self.execution_mode = execution_mode
        self.max_attempts = max(1, min(max_attempts, 20))
        self.lease_seconds = max(15, lease_seconds)
        if workflow not in {"verified_pr", "single_agent"}:
            raise ValueError("CODE_AGENT_WORKFLOW must be verified_pr or single_agent")
        self.workflow = workflow
        self.max_concurrency = max(1, min(max_concurrency, 4))
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.tasks: set[asyncio.Task] = set()
        self._started = False

    def _validate(self, request: CodeTaskRequest) -> None:
        task = CodeTask(
            repository=request.repository,
            issue=request.issue,
            model=request.model,
            test_command=request.test_command,
            policy=self.policy,
        )
        if task.test_command[0] not in self.policy.allowed_test_executables:
            raise ValueError("test executable is not allowlisted")
        candidate = (self.repository_root / task.repository).resolve()
        try:
            candidate.relative_to(self.repository_root)
        except ValueError as exc:
            raise ValueError("repository escapes configured root") from exc
        if not candidate.is_dir():
            raise ValueError("repository does not exist below configured root")

    async def submit(
        self, user_id: str, request: CodeTaskRequest, idempotency_key: str = ""
    ) -> JobRecord:
        try:
            self._validate(request)
            record, created = await asyncio.to_thread(
                self.store.enqueue, user_id, request, idempotency_key, self.max_attempts
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except JobStateError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if created and self.execution_mode == "embedded" and not self._started:
            task = asyncio.create_task(self._run_embedded())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        return record

    async def _run_embedded(self) -> None:
        from code_agent.worker import CodeAgentWorker

        async with self.semaphore:
            worker = CodeAgentWorker(
                self.store,
                self.repository_root,
                self.policy,
                lease_seconds=self.lease_seconds,
                workflow=self.workflow,
            )
            await worker.run_once()

    async def _worker_loop(self, slot: int) -> None:
        from code_agent.worker import CodeAgentWorker

        worker = CodeAgentWorker(
            self.store,
            self.repository_root,
            self.policy,
            worker_id=f"embedded-{os.getpid()}-{slot}",
            lease_seconds=self.lease_seconds,
            workflow=self.workflow,
        )
        await worker.run_forever()

    async def start(self) -> None:
        if self.execution_mode != "embedded" or self._started:
            return
        self._started = True
        for slot in range(self.max_concurrency):
            task = asyncio.create_task(self._worker_loop(slot))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def get(self, task_id: str, user_id: str) -> JobRecord:
        record = await asyncio.to_thread(self.store.get, task_id, user_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Coding task not found")
        return record

    async def decide(self, task_id: str, user_id: str, decision: CodeTaskDecision) -> JobRecord:
        try:
            return await asyncio.to_thread(
                self.store.decide, task_id, user_id, decision.approve, decision.reason
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Coding task not found") from exc
        except JobStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def patch(self, task_id: str, user_id: str) -> str:
        try:
            return await asyncio.to_thread(self.store.approved_patch, task_id, user_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Coding task not found") from exc
        except JobStateError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ArtifactIntegrityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def events(self, task_id: str, user_id: str) -> list[dict]:
        try:
            return await asyncio.to_thread(self.store.events, task_id, user_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Coding task not found") from exc

    async def dossier(self, task_id: str, user_id: str, format: str) -> str:
        try:
            return await asyncio.to_thread(self.store.dossier, task_id, user_id, format)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Coding task not found") from exc
        except JobStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ArtifactIntegrityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def stats(self, user_id: str) -> dict[str, int]:
        return await asyncio.to_thread(self.store.stats, user_id)

    async def cancel_all(self) -> None:
        tasks = list(self.tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._started = False


load_dotenv()

CODE_AGENT_ENABLED = os.getenv("CODE_AGENT_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_sandbox_image = os.getenv("CODE_AGENT_IMAGE", "agentforge-code-sandbox:local").strip()
_policy = SandboxPolicy(
    image=_sandbox_image,
    allowed_images={_sandbox_image},
    network_enabled=os.getenv("CODE_AGENT_NETWORK_ENABLED", "false").strip().lower()
    in {"1", "true", "yes", "on"},
    security_policy_enabled=os.getenv("CODE_AGENT_SECURITY_POLICY_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"},
    security_policy_version=os.getenv("CODE_AGENT_SECURITY_POLICY_VERSION", "agentforge-security-v1").strip(),
    max_iterations=int(os.getenv("CODE_AGENT_MAX_ITERATIONS", "20")),
    max_writes=int(os.getenv("CODE_AGENT_MAX_WRITES", "8")),
    max_changed_files=int(os.getenv("CODE_AGENT_MAX_CHANGED_FILES", "20")),
    max_repair_rounds=int(os.getenv("CODE_AGENT_MAX_REPAIR_ROUNDS", "2")),
    max_workflow_writes=int(os.getenv("CODE_AGENT_MAX_WORKFLOW_WRITES", "16")),
    min_changed_line_coverage=float(os.getenv("CODE_AGENT_MIN_CHANGED_LINE_COVERAGE", "0.8")),
    task_timeout_seconds=int(os.getenv("CODE_AGENT_TASK_TIMEOUT_SECONDS", "900")),
)
_artifact_store = LocalArtifactStore(
    Path(os.getenv("CODE_AGENT_ARTIFACT_ROOT", "data/code-agent/artifacts")),
    max_bytes=int(os.getenv("CODE_AGENT_MAX_ARTIFACT_BYTES", "4000000")),
)
_job_store = SQLiteJobStore(
    Path(os.getenv("CODE_AGENT_DB_PATH", "data/code-agent/jobs.db")),
    _artifact_store,
    max_records=int(os.getenv("CODE_AGENT_MAX_RECORDS", "5000")),
)


class _CodeJobCollector:
    def collect(self):
        metrics = _job_store.operational_metrics()
        jobs = GaugeMetricFamily(
            "agentforge_code_jobs",
            "Durable sandbox coding jobs by lifecycle state",
            labels=["status"],
        )
        for status in (
            "queued", "running", "retry_wait", "awaiting_approval",
            "approved", "rejected", "failed", "dead_letter",
        ):
            jobs.add_metric([status], metrics["counts"].get(status, 0))
        yield jobs
        artifacts = GaugeMetricFamily(
            "agentforge_code_artifact_bytes",
            "Bytes held in integrity-checked coding patch artifacts",
        )
        artifacts.add_metric([], metrics["artifact_bytes"])
        yield artifacts
        recoveries = CounterMetricFamily(
            "agentforge_code_lease_recoveries",
            "Coding jobs recovered after an expired worker lease",
        )
        recoveries.add_metric([], metrics["lease_recoveries"])
        yield recoveries
        verification = GaugeMetricFamily(
            "agentforge_code_verification_outcomes",
            "Verified PR workflow outcomes by final decision",
            labels=["decision"],
        )
        for decision in ("verified", "blocked"):
            verification.add_metric([decision], metrics["verification_counts"][decision])
        yield verification
        context_files = GaugeMetricFamily(
            "agentforge_code_context_files",
            "Files considered or selected across persisted graph-aware context runs",
            labels=["scope"],
        )
        context_files.add_metric(["candidate"], metrics["context_candidates"])
        context_files.add_metric(["selected"], metrics["context_selected"])
        yield context_files
        context_runs = GaugeMetricFamily(
            "agentforge_code_context_runs",
            "Persisted coding jobs containing a context-selection receipt",
        )
        context_runs.add_metric([], metrics["context_runs"])
        yield context_runs
        context_tokens = GaugeMetricFamily(
            "agentforge_code_context_tokens",
            "Estimated model-input tokens selected across persisted coding jobs",
        )
        context_tokens.add_metric([], metrics["context_tokens"])
        yield context_tokens
        context_saved = GaugeMetricFamily(
            "agentforge_code_context_chars_saved",
            "Context characters removed through focused compression and deduplication",
        )
        context_saved.add_metric([], metrics["context_chars_saved"])
        yield context_saved
        incremental = GaugeMetricFamily(
            "agentforge_code_context_incremental_files",
            "Changed files updated through incremental parse-tree reuse",
        )
        incremental.add_metric([], metrics["context_incremental_files"])
        yield incremental


REGISTRY.register(_CodeJobCollector())
CODE_AGENT_MANAGER = DurableCodeTaskManager(
    store=_job_store,
    repository_root=Path(os.getenv("CODE_AGENT_REPOSITORY_ROOT", "repositories")),
    policy=_policy,
    execution_mode=os.getenv("CODE_AGENT_EXECUTION_MODE", "embedded").strip().lower(),
    max_concurrency=int(os.getenv("CODE_AGENT_MAX_CONCURRENCY", "1")),
    max_attempts=int(os.getenv("CODE_AGENT_MAX_ATTEMPTS", "3")),
    lease_seconds=int(os.getenv("CODE_AGENT_LEASE_SECONDS", "90")),
    workflow=os.getenv("CODE_AGENT_WORKFLOW", "verified_pr").strip().lower(),
)

router = APIRouter(prefix="/code", tags=["sandboxed-code-agent"])


def _enabled() -> None:
    if not CODE_AGENT_ENABLED:
        raise HTTPException(status_code=404, detail="Sandboxed coding agent is disabled")


def _user_id(request: Request) -> str:
    user_id = str(getattr(request.state, "user_id", "") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Coding tasks require authenticated user mode")
    return user_id


@router.post("/tasks", status_code=202)
async def create_code_task(payload: CodeTaskRequest, request: Request):
    _enabled()
    idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    if idempotency_key and (
        not 8 <= len(idempotency_key) <= 128
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-" for character in idempotency_key)
    ):
        raise HTTPException(status_code=400, detail="Invalid Idempotency-Key header")
    record = await CODE_AGENT_MANAGER.submit(_user_id(request), payload, idempotency_key)
    return record.public()


@router.get("/tasks/{task_id}")
async def get_code_task(task_id: str, request: Request):
    _enabled()
    return (await CODE_AGENT_MANAGER.get(task_id, _user_id(request))).public()


@router.post("/tasks/{task_id}/decision")
async def decide_code_task(task_id: str, payload: CodeTaskDecision, request: Request):
    _enabled()
    return (await CODE_AGENT_MANAGER.decide(task_id, _user_id(request), payload)).public()


@router.get("/tasks/{task_id}/patch")
async def get_code_patch(task_id: str, request: Request):
    _enabled()
    return {"task_id": task_id, "patch": await CODE_AGENT_MANAGER.patch(task_id, _user_id(request))}


@router.get("/tasks/{task_id}/events")
async def get_code_task_events(task_id: str, request: Request):
    _enabled()
    events = await CODE_AGENT_MANAGER.events(task_id, _user_id(request))
    return {"task_id": task_id, "events": events}


@router.get("/tasks/{task_id}/dossier")
async def get_code_task_dossier(
    task_id: str,
    request: Request,
    format: Literal["json", "markdown"] = "json",
):
    _enabled()
    content = await CODE_AGENT_MANAGER.dossier(task_id, _user_id(request), format)
    media_type = "application/json" if format == "json" else "text/markdown"
    return Response(content=content, media_type=media_type)


@router.get("/queue")
async def get_code_queue(request: Request):
    _enabled()
    user_id = _user_id(request)
    return {
        "backend": "sqlite",
        "execution_mode": CODE_AGENT_MANAGER.execution_mode,
        "workflow": CODE_AGENT_MANAGER.workflow,
        "counts": await CODE_AGENT_MANAGER.stats(user_id),
    }
