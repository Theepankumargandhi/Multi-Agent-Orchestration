"""Transactional execution plane for sandbox coding jobs.

SQLite provides a dependency-free durable reference backend with atomic claims,
leases, retry scheduling, and an append-only transition log. Multiple worker
processes can safely claim jobs from the same database on one host.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from code_agent.job_models import ArtifactReference, CodeTaskRequest, JobRecord
from code_agent.models import CodeAgentResult
from code_agent.verification import build_pr_dossier


class IdempotencyConflictError(ValueError):
    """An idempotency key was reused for a different request."""


class JobStateError(RuntimeError):
    """A requested job transition is not valid."""


class ArtifactIntegrityError(RuntimeError):
    """A persisted artifact no longer matches its recorded digest."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def request_fingerprint(request: CodeTaskRequest) -> str:
    canonical = json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LocalArtifactStore:
    """Atomic, content-address-verified storage for generated patches."""

    def __init__(self, root: Path, max_bytes: int = 4_000_000):
        self.root = root.resolve()
        self.max_bytes = max(1_000, min(max_bytes, 20_000_000))
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, task_id: str) -> tuple[Path, Path]:
        if not task_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for character in task_id):
            raise ValueError("invalid task id")
        directory = (self.root / task_id).resolve()
        directory.relative_to(self.root)
        return directory / "patch.diff", directory / "manifest.json"

    def _named_path(self, task_id: str, name: str) -> Path:
        patch_path, _ = self._paths(task_id)
        if name not in {"patch.diff", "dossier.json", "dossier.md"}:
            raise ValueError("artifact name is not allowlisted")
        return patch_path.parent / name

    def _write_named(self, task_id: str, name: str, content: str) -> ArtifactReference:
        path = self._named_path(task_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = content.encode("utf-8")
        if len(payload) > self.max_bytes:
            raise ValueError(f"{name} artifact exceeds configured byte limit")
        digest = hashlib.sha256(payload).hexdigest()
        temporary = path.with_suffix(f".tmp-{uuid4().hex}")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
        return ArtifactReference(sha256=digest, size_bytes=len(payload))

    def _read_named(self, task_id: str, name: str, expected: ArtifactReference) -> str:
        path = self._named_path(task_id, name)
        try:
            payload = path.read_bytes()
        except FileNotFoundError as exc:
            raise ArtifactIntegrityError(f"{name} artifact is missing") from exc
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected.sha256 or len(payload) != expected.size_bytes:
            raise ArtifactIntegrityError(f"{name} artifact failed integrity verification")
        return payload.decode("utf-8")

    def write_patch(self, task_id: str, patch: str) -> ArtifactReference:
        patch_path, manifest_path = self._paths(task_id)
        reference = self._write_named(task_id, "patch.diff", patch)
        manifest = json.dumps(reference.model_dump(), sort_keys=True)
        temporary_manifest = manifest_path.with_suffix(f".tmp-{uuid4().hex}")
        temporary_manifest.write_text(manifest, encoding="utf-8")
        os.replace(temporary_manifest, manifest_path)
        return reference

    def read_patch(self, task_id: str, expected: ArtifactReference) -> str:
        return self._read_named(task_id, "patch.diff", expected)

    def write_dossier(self, task_id: str, json_payload: str, markdown: str) -> tuple[ArtifactReference, ArtifactReference]:
        json_reference = self._write_named(task_id, "dossier.json", json_payload)
        try:
            markdown_reference = self._write_named(task_id, "dossier.md", markdown)
        except Exception:
            try:
                self._named_path(task_id, "dossier.json").unlink()
            except FileNotFoundError:
                pass
            raise
        return json_reference, markdown_reference

    def read_dossier(self, task_id: str, expected: ArtifactReference, format: str) -> str:
        name = "dossier.json" if format == "json" else "dossier.md"
        return self._read_named(task_id, name, expected)

    def delete_patch(self, task_id: str) -> None:
        patch_path, manifest_path = self._paths(task_id)
        for path in (patch_path, manifest_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            patch_path.parent.rmdir()
        except (FileNotFoundError, OSError):
            pass

    def delete_all(self, task_id: str) -> None:
        patch_path, manifest_path = self._paths(task_id)
        for path in (
            patch_path,
            manifest_path,
            self._named_path(task_id, "dossier.json"),
            self._named_path(task_id, "dossier.md"),
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            patch_path.parent.rmdir()
        except (FileNotFoundError, OSError):
            pass


class SQLiteJobStore:
    """Durable job queue using short SQLite transactions and worker leases."""

    TERMINAL = {"approved", "rejected", "failed", "dead_letter"}

    def __init__(self, path: Path, artifact_store: LocalArtifactStore, max_records: int = 5000):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts = artifact_store
        self.max_records = max(20, min(max_records, 100_000))
        self._setup()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    def _setup(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS code_jobs (
                    task_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT NOT NULL DEFAULT '',
                    decision_reason TEXT NOT NULL DEFAULT '',
                    attempt INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    lease_owner TEXT NOT NULL DEFAULT '',
                    lease_expires_at TEXT NOT NULL DEFAULT '',
                    heartbeat_at TEXT NOT NULL DEFAULT '',
                    idempotency_key TEXT NOT NULL DEFAULT '',
                    request_fingerprint TEXT NOT NULL,
                    artifact_sha256 TEXT NOT NULL DEFAULT '',
                    artifact_size_bytes INTEGER NOT NULL DEFAULT 0,
                    dossier_json_sha256 TEXT NOT NULL DEFAULT '',
                    dossier_json_size_bytes INTEGER NOT NULL DEFAULT 0,
                    dossier_markdown_sha256 TEXT NOT NULL DEFAULT '',
                    dossier_markdown_size_bytes INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 0
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_code_jobs_idempotency
                    ON code_jobs(user_id, idempotency_key) WHERE idempotency_key <> '';
                CREATE INDEX IF NOT EXISTS idx_code_jobs_claim
                    ON code_jobs(status, available_at, created_at);
                CREATE TABLE IF NOT EXISTS code_job_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES code_jobs(task_id) ON DELETE CASCADE,
                    occurred_at TEXT NOT NULL,
                    from_status TEXT NOT NULL,
                    to_status TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_code_job_events_task
                    ON code_job_events(task_id, event_id);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(code_jobs)").fetchall()
            }
            migrations = {
                "dossier_json_sha256": "TEXT NOT NULL DEFAULT ''",
                "dossier_json_size_bytes": "INTEGER NOT NULL DEFAULT 0",
                "dossier_markdown_sha256": "TEXT NOT NULL DEFAULT ''",
                "dossier_markdown_size_bytes": "INTEGER NOT NULL DEFAULT 0",
            }
            for name, definition in migrations.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE code_jobs ADD COLUMN {name} {definition}")

    @staticmethod
    def _record(row: sqlite3.Row) -> JobRecord:
        result = CodeAgentResult.model_validate_json(row["result_json"]) if row["result_json"] else None
        artifact = None
        if row["artifact_sha256"]:
            artifact = ArtifactReference(
                sha256=row["artifact_sha256"], size_bytes=row["artifact_size_bytes"]
            )
        dossier_json = None
        if row["dossier_json_sha256"]:
            dossier_json = ArtifactReference(
                sha256=row["dossier_json_sha256"], size_bytes=row["dossier_json_size_bytes"]
            )
        dossier_markdown = None
        if row["dossier_markdown_sha256"]:
            dossier_markdown = ArtifactReference(
                sha256=row["dossier_markdown_sha256"],
                size_bytes=row["dossier_markdown_size_bytes"],
            )
        return JobRecord(
            task_id=row["task_id"], user_id=row["user_id"], status=row["status"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            available_at=row["available_at"], request=CodeTaskRequest.model_validate_json(row["request_json"]),
            result=result, error=row["error"], decision_reason=row["decision_reason"],
            attempt=row["attempt"], max_attempts=row["max_attempts"],
            lease_owner=row["lease_owner"], lease_expires_at=row["lease_expires_at"],
            heartbeat_at=row["heartbeat_at"], idempotency_key=row["idempotency_key"],
            request_fingerprint=row["request_fingerprint"], artifact=artifact,
            dossier_json=dossier_json, dossier_markdown=dossier_markdown,
        )

    @staticmethod
    def _event(connection: sqlite3.Connection, task_id: str, before: str, after: str, actor: str, detail: dict[str, Any] | None = None) -> None:
        connection.execute(
            "INSERT INTO code_job_events(task_id,occurred_at,from_status,to_status,actor,detail_json) VALUES(?,?,?,?,?,?)",
            (task_id, utc_now(), before, after, actor[:200], json.dumps(detail or {}, sort_keys=True)),
        )

    def enqueue(self, user_id: str, request: CodeTaskRequest, idempotency_key: str = "", max_attempts: int = 3) -> tuple[JobRecord, bool]:
        now = utc_now()
        fingerprint = request_fingerprint(request)
        task_id = str(uuid4())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                existing = connection.execute(
                    "SELECT * FROM code_jobs WHERE user_id=? AND idempotency_key=?",
                    (user_id, idempotency_key),
                ).fetchone()
                if existing:
                    if existing["request_fingerprint"] != fingerprint:
                        raise IdempotencyConflictError("idempotency key was already used for a different request")
                    connection.commit()
                    return self._record(existing), False
            count = connection.execute("SELECT COUNT(*) FROM code_jobs").fetchone()[0]
            if count >= self.max_records:
                raise JobStateError("coding task record capacity reached")
            connection.execute(
                """INSERT INTO code_jobs(
                    task_id,user_id,status,created_at,updated_at,available_at,request_json,
                    max_attempts,idempotency_key,request_fingerprint
                ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (task_id, user_id, "queued", now, now, now, request.model_dump_json(),
                 max(1, min(max_attempts, 20)), idempotency_key, fingerprint),
            )
            self._event(connection, task_id, "", "queued", user_id, {"idempotent": bool(idempotency_key)})
            row = connection.execute("SELECT * FROM code_jobs WHERE task_id=?", (task_id,)).fetchone()
            connection.commit()
            return self._record(row), True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(self, task_id: str, user_id: str | None = None) -> JobRecord | None:
        query = "SELECT * FROM code_jobs WHERE task_id=?"
        params: tuple[Any, ...] = (task_id,)
        if user_id is not None:
            query += " AND user_id=?"
            params += (user_id,)
        with closing(self._connect()) as connection:
            row = connection.execute(query, params).fetchone()
        return self._record(row) if row else None

    def _recover_expired(self, connection: sqlite3.Connection, now: str) -> int:
        rows = connection.execute(
            "SELECT * FROM code_jobs WHERE status='running' AND lease_expires_at<>'' AND lease_expires_at<=?",
            (now,),
        ).fetchall()
        for row in rows:
            after = "dead_letter" if row["attempt"] >= row["max_attempts"] else "queued"
            error = "worker lease expired; retry budget exhausted" if after == "dead_letter" else "worker lease expired; recovered"
            connection.execute(
                """UPDATE code_jobs SET status=?,updated_at=?,available_at=?,error=?,
                    lease_owner='',lease_expires_at='',heartbeat_at='',version=version+1 WHERE task_id=?""",
                (after, now, now, error, row["task_id"]),
            )
            self._event(connection, row["task_id"], "running", after, "lease-reaper", {"attempt": row["attempt"]})
        return len(rows)

    def claim(self, worker_id: str, lease_seconds: int = 90) -> JobRecord | None:
        now = utc_now()
        expires = (datetime.now(UTC) + timedelta(seconds=max(15, lease_seconds))).isoformat()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_expired(connection, now)
            row = connection.execute(
                """SELECT * FROM code_jobs
                   WHERE status IN ('queued','retry_wait') AND available_at<=?
                   ORDER BY available_at,created_at LIMIT 1""",
                (now,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            changed = connection.execute(
                """UPDATE code_jobs SET status='running',updated_at=?,attempt=attempt+1,
                    lease_owner=?,lease_expires_at=?,heartbeat_at=?,version=version+1
                    WHERE task_id=? AND version=?""",
                (now, worker_id, expires, now, row["task_id"], row["version"]),
            ).rowcount
            if changed != 1:
                connection.rollback()
                return None
            self._event(connection, row["task_id"], row["status"], "running", worker_id)
            claimed = connection.execute("SELECT * FROM code_jobs WHERE task_id=?", (row["task_id"],)).fetchone()
            connection.commit()
            return self._record(claimed)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def heartbeat(self, task_id: str, worker_id: str, lease_seconds: int = 90) -> bool:
        now = utc_now()
        expires = (datetime.now(UTC) + timedelta(seconds=max(15, lease_seconds))).isoformat()
        with closing(self._connect()) as connection:
            changed = connection.execute(
                """UPDATE code_jobs SET heartbeat_at=?,lease_expires_at=?,updated_at=?,version=version+1
                   WHERE task_id=? AND status='running' AND lease_owner=?""",
                (now, expires, now, task_id, worker_id),
            ).rowcount
        return changed == 1

    def complete(self, task_id: str, worker_id: str, result: CodeAgentResult) -> JobRecord:
        clean_result = result.model_copy(update={"patch": ""})
        now = utc_now()
        connection = self._connect()
        artifact = None
        dossier_references = None
        artifacts_started = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM code_jobs WHERE task_id=?", (task_id,)).fetchone()
            if row is None or row["status"] != "running" or row["lease_owner"] != worker_id:
                raise JobStateError("worker no longer owns this job lease")
            patch_digest = ArtifactReference(
                sha256=hashlib.sha256(result.patch.encode("utf-8")).hexdigest(),
                size_bytes=len(result.patch.encode("utf-8")),
            )
            artifacts_started = True
            if result.status == "completed" and result.patch:
                artifact = self.artifacts.write_patch(task_id, result.patch)
                patch_digest = artifact
            dossier_payloads = build_pr_dossier(self._record(row), result, patch_digest)
            dossier_references = self.artifacts.write_dossier(task_id, *dossier_payloads)
            after = "awaiting_approval" if result.status == "completed" and artifact else "failed"
            error = "" if after == "awaiting_approval" else (result.summary or "coding agent produced no patch")
            connection.execute(
                """UPDATE code_jobs SET status=?,updated_at=?,result_json=?,error=?,
                    artifact_sha256=?,artifact_size_bytes=?,dossier_json_sha256=?,dossier_json_size_bytes=?,
                    dossier_markdown_sha256=?,dossier_markdown_size_bytes=?,lease_owner='',lease_expires_at='',
                    heartbeat_at='',version=version+1 WHERE task_id=?""",
                (after, now, clean_result.model_dump_json(), error,
                 artifact.sha256 if artifact else "", artifact.size_bytes if artifact else 0,
                 dossier_references[0].sha256, dossier_references[0].size_bytes,
                 dossier_references[1].sha256, dossier_references[1].size_bytes, task_id),
            )
            self._event(connection, task_id, "running", after, worker_id, {"result_status": result.status})
            updated = connection.execute("SELECT * FROM code_jobs WHERE task_id=?", (task_id,)).fetchone()
            connection.commit()
            return self._record(updated)
        except Exception:
            connection.rollback()
            if artifacts_started:
                self.artifacts.delete_all(task_id)
            raise
        finally:
            connection.close()

    def fail(self, task_id: str, worker_id: str, error: str, retryable: bool = True, base_delay_seconds: int = 2) -> JobRecord:
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM code_jobs WHERE task_id=?", (task_id,)).fetchone()
            if row is None or row["status"] != "running" or row["lease_owner"] != worker_id:
                raise JobStateError("worker no longer owns this job lease")
            retry = retryable and row["attempt"] < row["max_attempts"]
            after = "retry_wait" if retry else ("dead_letter" if retryable else "failed")
            delay = min(300, max(1, base_delay_seconds) * (2 ** max(0, row["attempt"] - 1))) if retry else 0
            available = (now_dt + timedelta(seconds=delay)).isoformat()
            connection.execute(
                """UPDATE code_jobs SET status=?,updated_at=?,available_at=?,error=?,
                    lease_owner='',lease_expires_at='',heartbeat_at='',version=version+1 WHERE task_id=?""",
                (after, now, available, error[:2000], task_id),
            )
            self._event(connection, task_id, "running", after, worker_id, {"retryable": retryable, "delay_seconds": delay})
            updated = connection.execute("SELECT * FROM code_jobs WHERE task_id=?", (task_id,)).fetchone()
            connection.commit()
            return self._record(updated)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def release(self, task_id: str, worker_id: str) -> None:
        now = utc_now()
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT status,lease_owner FROM code_jobs WHERE task_id=?", (task_id,)).fetchone()
            if row and row["status"] == "running" and row["lease_owner"] == worker_id:
                connection.execute(
                    """UPDATE code_jobs SET status='queued',updated_at=?,available_at=?,error='worker shutdown; job requeued',
                       lease_owner='',lease_expires_at='',heartbeat_at='',version=version+1 WHERE task_id=?""",
                    (now, now, task_id),
                )
                self._event(connection, task_id, "running", "queued", worker_id, {"reason": "shutdown"})

    def decide(self, task_id: str, user_id: str, approve: bool, reason: str = "") -> JobRecord:
        now = utc_now()
        after = "approved" if approve else "rejected"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM code_jobs WHERE task_id=? AND user_id=?", (task_id, user_id)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            if row["status"] != "awaiting_approval":
                raise JobStateError("coding task is not awaiting approval")
            if not approve:
                self.artifacts.delete_all(task_id)
            artifact_reset = "" if not approve else row["artifact_sha256"]
            artifact_size_reset = 0 if not approve else row["artifact_size_bytes"]
            dossier_json_reset = "" if not approve else row["dossier_json_sha256"]
            dossier_json_size_reset = 0 if not approve else row["dossier_json_size_bytes"]
            dossier_md_reset = "" if not approve else row["dossier_markdown_sha256"]
            dossier_md_size_reset = 0 if not approve else row["dossier_markdown_size_bytes"]
            connection.execute(
                """UPDATE code_jobs SET status=?,updated_at=?,decision_reason=?,artifact_sha256=?,
                    artifact_size_bytes=?,dossier_json_sha256=?,dossier_json_size_bytes=?,
                    dossier_markdown_sha256=?,dossier_markdown_size_bytes=?,version=version+1 WHERE task_id=?""",
                (after, now, reason.strip(), artifact_reset, artifact_size_reset,
                 dossier_json_reset, dossier_json_size_reset, dossier_md_reset, dossier_md_size_reset, task_id),
            )
            self._event(connection, task_id, "awaiting_approval", after, user_id, {"reason": reason.strip()})
            updated = connection.execute("SELECT * FROM code_jobs WHERE task_id=?", (task_id,)).fetchone()
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self._record(updated)

    def approved_patch(self, task_id: str, user_id: str) -> str:
        record = self.get(task_id, user_id)
        if record is None:
            raise KeyError(task_id)
        if record.status != "approved" or record.artifact is None:
            raise JobStateError("patch requires explicit approval")
        return self.artifacts.read_patch(task_id, record.artifact)

    def dossier(self, task_id: str, user_id: str, format: str = "json") -> str:
        if format not in {"json", "markdown"}:
            raise ValueError("dossier format must be json or markdown")
        record = self.get(task_id, user_id)
        if record is None:
            raise KeyError(task_id)
        reference = record.dossier_json if format == "json" else record.dossier_markdown
        if reference is None:
            raise JobStateError("PR evidence dossier is not available")
        return self.artifacts.read_dossier(task_id, reference, format)

    def events(self, task_id: str, user_id: str) -> list[dict[str, Any]]:
        if self.get(task_id, user_id) is None:
            raise KeyError(task_id)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT occurred_at,from_status,to_status,actor,detail_json FROM code_job_events WHERE task_id=? ORDER BY event_id",
                (task_id,),
            ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event.pop("detail_json")
            event["detail"] = json.loads(row["detail_json"])
            event["actor"] = "owner" if row["actor"] == user_id else (
                "lease-reaper" if row["actor"] == "lease-reaper" else "worker"
            )
            events.append(event)
        return events

    def stats(self, user_id: str | None = None) -> dict[str, int]:
        query = "SELECT status,COUNT(*) AS count FROM code_jobs"
        params: tuple[Any, ...] = ()
        if user_id is not None:
            query += " WHERE user_id=?"
            params = (user_id,)
        query += " GROUP BY status"
        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return {row["status"]: row["count"] for row in rows}

    def operational_metrics(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            artifact_bytes = connection.execute(
                "SELECT COALESCE(SUM(artifact_size_bytes),0) FROM code_jobs WHERE artifact_sha256<>''"
            ).fetchone()[0]
            recoveries = connection.execute(
                "SELECT COUNT(*) FROM code_job_events WHERE actor='lease-reaper'"
            ).fetchone()[0]
            results = connection.execute(
                "SELECT result_json FROM code_jobs WHERE result_json IS NOT NULL"
            ).fetchall()
        verification_counts = {"verified": 0, "blocked": 0}
        context_candidates = 0
        context_selected = 0
        context_runs = 0
        context_tokens = 0
        context_chars_saved = 0
        context_incremental_files = 0
        for row in results:
            try:
                result = json.loads(row["result_json"])
                decision = (result.get("verification") or {}).get("final_decision")
                if decision in verification_counts:
                    verification_counts[decision] += 1
                context = (result.get("verification") or {}).get("context") or {}
                if context:
                    context_runs += 1
                    context_candidates += int(context.get("candidate_files") or 0)
                    context_selected += len(context.get("selected_files") or [])
                    context_tokens += int(context.get("estimated_tokens") or 0)
                    context_chars_saved += max(
                        0,
                        int(context.get("original_context_chars") or 0)
                        - int(context.get("context_chars") or 0),
                    )
                    context_incremental_files += int(context.get("index_incremental_files") or 0)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return {
            "counts": self.stats(),
            "artifact_bytes": int(artifact_bytes),
            "lease_recoveries": int(recoveries),
            "verification_counts": verification_counts,
            "context_candidates": context_candidates,
            "context_selected": context_selected,
            "context_runs": context_runs,
            "context_tokens": context_tokens,
            "context_chars_saved": context_chars_saved,
            "context_incremental_files": context_incremental_files,
        }
