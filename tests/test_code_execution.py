from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

import pytest

from code_agent.api import DurableCodeTaskManager
from code_agent.execution import (
    ArtifactIntegrityError,
    IdempotencyConflictError,
    JobStateError,
    LocalArtifactStore,
    SQLiteJobStore,
)
from code_agent.job_models import CodeTaskDecision, CodeTaskRequest
from code_agent.models import (
    AnalysisPlan,
    CodeAgentResult,
    CodeContextReceipt,
    SandboxPolicy,
    VerificationReport,
)
from code_agent.worker import CodeAgentWorker


def make_store(tmp_path: Path, max_records: int = 100) -> SQLiteJobStore:
    return SQLiteJobStore(
        tmp_path / "queue" / "jobs.db",
        LocalArtifactStore(tmp_path / "artifacts"),
        max_records=max_records,
    )


def request() -> CodeTaskRequest:
    return CodeTaskRequest(
        repository="fixture",
        issue="Fix the reproducible parser boundary defect.",
        model="test-model",
        test_command=["python", "-m", "pytest", "-q"],
    )


def completed_result(patch: str = "--- a/app.py\n+++ b/app.py\n") -> CodeAgentResult:
    return CodeAgentResult(
        status="completed",
        summary="Tests pass in the sandbox.",
        patch=patch,
        changed_files=["app.py"],
    )


def test_enqueue_is_durable_and_idempotent_per_user(tmp_path: Path):
    store = make_store(tmp_path)
    first, created = store.enqueue("alice", request(), "submission-123")
    duplicate, duplicate_created = store.enqueue("alice", request(), "submission-123")

    assert created is True
    assert duplicate_created is False
    assert duplicate.task_id == first.task_id
    reopened = make_store(tmp_path)
    assert reopened.get(first.task_id, "alice") == first
    assert reopened.get(first.task_id, "bob") is None

    changed = request().model_copy(update={"issue": "Fix a different reproducible parser defect."})
    with pytest.raises(IdempotencyConflictError):
        reopened.enqueue("alice", changed, "submission-123")
    other_user, _ = reopened.enqueue("bob", changed, "submission-123")
    assert other_user.task_id != first.task_id


def test_claim_is_exclusive_and_expired_lease_is_recovered(tmp_path: Path):
    store = make_store(tmp_path)
    queued, _ = store.enqueue("alice", request(), max_attempts=3)
    first_claim = store.claim("worker-one", lease_seconds=15)

    assert first_claim is not None
    assert first_claim.task_id == queued.task_id
    assert first_claim.status == "running"
    assert first_claim.attempt == 1
    assert store.claim("worker-two") is None
    previous_expiry = first_claim.lease_expires_at
    assert store.heartbeat(queued.task_id, "wrong-worker") is False
    assert store.heartbeat(queued.task_id, "worker-one", lease_seconds=30) is True
    heartbeat_record = store.get(queued.task_id, "alice")
    assert heartbeat_record is not None
    assert heartbeat_record.lease_expires_at > previous_expiry

    with closing(store._connect()) as connection:
        connection.execute(
            "UPDATE code_jobs SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE task_id=?",
            (queued.task_id,),
        )
    recovered = store.claim("worker-two")
    assert recovered is not None
    assert recovered.task_id == queued.task_id
    assert recovered.attempt == 2
    assert recovered.lease_owner == "worker-two"
    assert any(event["actor"] == "lease-reaper" for event in store.events(queued.task_id, "alice"))


def test_retry_budget_moves_job_to_dead_letter(tmp_path: Path):
    store = make_store(tmp_path)
    retrying, _ = store.enqueue("alice", request(), max_attempts=2)
    first_attempt = store.claim("worker")
    assert first_attempt is not None and first_attempt.task_id == retrying.task_id
    waiting = store.fail(first_attempt.task_id, "worker", "temporary runtime outage", retryable=True)
    assert waiting.status == "retry_wait"
    with closing(store._connect()) as connection:
        connection.execute(
            "UPDATE code_jobs SET available_at='2000-01-01T00:00:00+00:00' WHERE task_id=?",
            (retrying.task_id,),
        )
    second_attempt = store.claim("worker-two")
    assert second_attempt is not None and second_attempt.attempt == 2
    dead_after_retry = store.fail(
        second_attempt.task_id, "worker-two", "runtime still unavailable", retryable=True
    )
    assert dead_after_retry.status == "dead_letter"

    queued, _ = store.enqueue("alice", request(), max_attempts=1)
    claimed = store.claim("worker")
    assert claimed is not None

    failed = store.fail(claimed.task_id, "worker", "temporary runtime outage", retryable=True)
    assert failed.status == "dead_letter"
    assert store.claim("another-worker") is None
    assert store.stats() == {"dead_letter": 2}


def test_patch_is_separate_approval_gated_and_integrity_checked(tmp_path: Path):
    store = make_store(tmp_path)
    queued, _ = store.enqueue("alice", request())
    claimed = store.claim("worker")
    assert claimed is not None
    result = completed_result().model_copy(
        update={
            "verification": VerificationReport(
                analysis=AnalysisPlan(summary="Inspect the parser boundary."),
                context=CodeContextReceipt(
                    query="parser boundary",
                    candidate_files=10,
                    context_chars=800,
                    original_context_chars=1000,
                    estimated_tokens=200,
                    index_incremental_files=2,
                    fingerprint="context-fingerprint",
                ),
                final_decision="verified",
            )
        }
    )
    completed = store.complete(queued.task_id, "worker", result)

    assert completed.status == "awaiting_approval"
    assert completed.result is not None and completed.result.patch == ""
    assert completed.artifact is not None
    assert completed.dossier_json is not None
    assert completed.dossier_markdown is not None
    assert completed.public()["result"]["patch"] == ""
    dossier = json.loads(store.dossier(queued.task_id, "alice", "json"))
    assert dossier["task"]["task_id"] == queued.task_id
    assert dossier["patch_artifact"]["sha256"] == completed.artifact.sha256
    assert "--- a/app.py" not in json.dumps(dossier)
    assert "Verified PR evidence" in store.dossier(queued.task_id, "alice", "markdown")
    metrics = store.operational_metrics()
    assert metrics["context_runs"] == 1
    assert metrics["context_candidates"] == 10
    assert metrics["context_tokens"] == 200
    assert metrics["context_chars_saved"] == 200
    assert metrics["context_incremental_files"] == 2
    with pytest.raises(JobStateError):
        store.approved_patch(queued.task_id, "alice")

    approved = store.decide(queued.task_id, "alice", True, "Reviewed tests and diff")
    assert approved.status == "approved"
    assert store.approved_patch(queued.task_id, "alice").startswith("--- a/app.py")

    patch_path = tmp_path / "artifacts" / queued.task_id / "patch.diff"
    patch_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        store.approved_patch(queued.task_id, "alice")
    dossier_path = tmp_path / "artifacts" / queued.task_id / "dossier.json"
    dossier_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        store.dossier(queued.task_id, "alice", "json")


def test_rejection_erases_patch_artifact(tmp_path: Path):
    store = make_store(tmp_path)
    queued, _ = store.enqueue("alice", request())
    store.claim("worker")
    store.complete(queued.task_id, "worker", completed_result())
    rejected = store.decide(queued.task_id, "alice", False, "Diff is too broad")

    assert rejected.status == "rejected"
    assert not (tmp_path / "artifacts" / queued.task_id / "patch.diff").exists()
    assert not (tmp_path / "artifacts" / queued.task_id / "dossier.json").exists()
    with pytest.raises(JobStateError):
        store.approved_patch(queued.task_id, "alice")
    with pytest.raises(JobStateError):
        store.dossier(queued.task_id, "alice")


def test_blocked_job_retains_evidence_but_not_patch_body(tmp_path: Path):
    store = make_store(tmp_path)
    queued, _ = store.enqueue("alice", request())
    store.claim("worker")
    blocked = CodeAgentResult(
        status="failed",
        summary="Independent verification blocked the patch.",
        patch="--- a/app.py\n+++ b/app.py\n+unsafe = True\n",
        changed_files=["app.py"],
    )
    completed = store.complete(queued.task_id, "worker", blocked)

    assert completed.status == "failed"
    assert completed.artifact is None
    assert completed.dossier_json is not None
    assert not (tmp_path / "artifacts" / queued.task_id / "patch.diff").exists()
    dossier = json.loads(store.dossier(queued.task_id, "alice"))
    assert dossier["outcome"]["status"] == "failed"
    assert "unsafe = True" not in json.dumps(dossier)
    with pytest.raises(JobStateError):
        store.decide(queued.task_id, "alice", True, "Cannot override verification")


@pytest.mark.asyncio
async def test_worker_executes_claim_and_manager_survives_restart(tmp_path: Path):
    repository_root = tmp_path / "repositories"
    (repository_root / "fixture").mkdir(parents=True)
    store = make_store(tmp_path)
    manager = DurableCodeTaskManager(
        store,
        repository_root,
        SandboxPolicy(),
        execution_mode="external",
    )
    submitted = await manager.submit("alice", request(), "durable-job-123")

    async def fake_solver(_job):
        return completed_result()

    worker = CodeAgentWorker(
        store,
        repository_root,
        SandboxPolicy(),
        worker_id="test-worker",
        solver=fake_solver,
    )
    result = await worker.run_once()
    assert result is not None and result.status == "awaiting_approval"

    restarted = DurableCodeTaskManager(
        make_store(tmp_path),
        repository_root,
        SandboxPolicy(),
        execution_mode="external",
    )
    restored = await restarted.get(submitted.task_id, "alice")
    assert restored.status == "awaiting_approval"
    assert submitted.task_id in await restarted.dossier(
        submitted.task_id, "alice", "markdown"
    )
    with pytest.raises(Exception) as wrong_user:
        await restarted.dossier(submitted.task_id, "bob", "json")
    assert getattr(wrong_user.value, "status_code", None) == 404
    await restarted.decide(
        submitted.task_id,
        "alice",
        CodeTaskDecision(approve=True, reason="Verified"),
    )
    assert (await restarted.patch(submitted.task_id, "alice")).startswith("--- a/app.py")
