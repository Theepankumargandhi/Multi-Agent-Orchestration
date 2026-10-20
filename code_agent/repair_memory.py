"""Human-reviewed repair lessons with local hybrid retrieval and tenant isolation.

Only the operator's management CLI writes this store. Runtime connections are
read-only and check expiry, scope, content integrity, and taints every planning step.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator

from code_agent.models import CodeAgentResult, CodeTask, RepairMemoryReceipt
from code_agent.retrieval_backends import HashingSemanticEmbedder, cosine, tokenize
from code_agent.security_policy import classify_untrusted_content
from code_agent.verification import blocking_reasons

_REQUIRED = {"mandatory_tests", "patch_scope", "secret_scan", "unsafe_code_scan", "source_repository_integrity"}
_STOP = {"fix", "bug", "issue", "test", "tests", "the", "and", "with", "for", "from", "that", "this", "should", "when"}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def scope_hash(project: str, namespace: str, repository: str) -> str:
    if any(not item.strip() or len(item) > 200 for item in (project, namespace, repository)):
        raise ValueError("repair memory requires explicit project, tenant, and repository scope")
    return digest([project, namespace, repository])


class RepairLesson(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    problem: str = Field(min_length=10, max_length=400)
    lesson: str = Field(min_length=10, max_length=800)
    applicable_paths: list[str] = Field(min_length=1, max_length=8)
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewer_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    created_at: float = Field(ge=0)
    expires_at: float = Field(gt=0)

    @field_validator("applicable_paths")
    @classmethod
    def safe_paths(cls, paths):
        if len(set(paths)) != len(paths) or any(
            not path or len(path) > 300 or path.startswith("/") or ":" in path or "\\" in path
            or any(ord(char) < 32 for char in path)
            or ".." in PurePosixPath(path).parts for path in paths
        ):
            raise ValueError("lesson paths must be unique repository-relative paths")
        return paths

    @property
    def identifier(self):
        return digest(self.model_dump(mode="json"))

    @property
    def text(self):
        return self.problem + "\n" + self.lesson + "\n" + " ".join(self.applicable_paths)


def validate_evidence(result: CodeAgentResult) -> str:
    """Accept a verified patch, not mere passing tests or a model's completion claim."""
    report = result.verification
    if (result.status != "completed" or not result.patch.strip() or not report
            or report.final_decision != "verified" or report.blocking_reasons
            or blocking_reasons(report) or not result.final_test
            or result.final_test.exit_code != 0 or result.final_test.timed_out
            or result.sandbox.get("original_repository_unchanged") is not True
            or not result.security or result.security.blocked_actions
            or any(failure.terminal for failure in result.provider_failures)):
        raise ValueError("lesson requires a verified patch with passing final execution and no terminal hold")
    passed = {gate.name for gate in report.rounds[-1].quality_gates if gate.status == "passed"}
    if not _REQUIRED <= passed:
        raise ValueError("lesson evidence is missing mandatory passed quality gates")
    if result.tournament:
        from code_agent.repair_tournament import verify_tournament_report
        if not result.tournament.winner_id or not verify_tournament_report(result.tournament):
            raise ValueError("lesson requires valid winning tournament evidence")
    # Store only the digest, never the patch, issue transcript, or raw execution output.
    return digest(result.model_dump(mode="json"))


class RepairMemoryStore:
    def __init__(self, path: Path):
        self.path = path.absolute()

    def _connect(self, *, write=False):
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=2)
        else:
            connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=1)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self):
        with closing(self._connect(write=True)) as connection, connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1}:
                raise ValueError("unsupported repair memory schema")
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )}
            if (version == 0 and tables) or (version == 1 and tables != {"lessons"}):
                raise ValueError("refusing to initialize an unrelated database as repair memory")
            connection.execute("""CREATE TABLE IF NOT EXISTS lessons (
                id TEXT PRIMARY KEY, scope TEXT NOT NULL, content_key TEXT UNIQUE NOT NULL,
                body TEXT NOT NULL, expires REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS lesson_scope ON lessons(scope, revoked, expires)")
            connection.execute("PRAGMA user_version=1")

    def approve(self, result: CodeAgentResult, *, project: str, namespace: str, repository: str,
                problem: str, lesson: str, applicable_paths: list[str], reviewer: str,
                approve: bool = False, ttl_days: int = 30, now: float | None = None) -> str:
        if not approve or not reviewer.strip() or len(reviewer) > 200:
            raise ValueError("explicit human lesson approval and reviewer identity are required")
        if not 1 <= ttl_days <= 90:
            raise ValueError("lesson TTL must be between 1 and 90 days")
        clock = time.time() if now is None else now
        record = RepairLesson(scope_sha256=scope_hash(project, namespace, repository),
            problem=problem, lesson=lesson, applicable_paths=applicable_paths,
            evidence_sha256=validate_evidence(result), reviewer_sha256=digest(reviewer),
            created_at=clock, expires_at=clock + ttl_days * 86400)
        if not set(record.applicable_paths) <= set(result.changed_files):
            raise ValueError("lesson paths must be supported by the verified patch's changed files")
        if set(classify_untrusted_content(record.text)) & {"instruction_injection", "secret_material"}:
            raise ValueError("lesson contains detected unsafe instructions or secret material")
        self.initialize()
        with closing(self._connect(write=True)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            if (connection.execute("SELECT COUNT(*) FROM lessons").fetchone()[0] >= 5000
                    or connection.execute("SELECT COUNT(*) FROM lessons WHERE scope=?", (record.scope_sha256,)).fetchone()[0] >= 200):
                raise ValueError("repair memory capacity reached; explicitly delete unused lessons")
            content_key = digest([record.scope_sha256, problem, lesson, applicable_paths])
            try:
                connection.execute("INSERT INTO lessons(id,scope,content_key,body,expires) VALUES (?,?,?,?,?)",
                    (record.identifier, record.scope_sha256, content_key, record.model_dump_json(), record.expires_at))
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate reviewed lesson") from exc
        return record.identifier

    def withdraw(self, identifier: str, scope: str, *, delete: bool = False) -> bool:
        """Scoped operator action; revocation also removes the stored lesson text."""
        if not self.path.is_file():
            raise ValueError("repair memory store does not exist")
        with closing(self._connect(write=True)) as connection, connection:
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError("unsupported repair memory schema")
            connection.execute("PRAGMA secure_delete=ON")
            if delete:
                cursor = connection.execute("DELETE FROM lessons WHERE id=? AND scope=?", (identifier, scope))
            else:
                cursor = connection.execute("UPDATE lessons SET revoked=1, body='' WHERE id=? AND scope=?", (identifier, scope))
            return cursor.rowcount > 0

    def records(self, *, scope: str, now: float | None = None):
        clock = time.time() if now is None else now
        if self.path.stat().st_size > 50_000_000:
            raise ValueError("repair memory database exceeds the read bound")
        with closing(self._connect()) as connection:
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError("unsupported repair memory schema")
            rows = connection.execute("SELECT id,CASE WHEN length(body)<=5000 THEN body ELSE '' END AS body,expires FROM lessons WHERE scope=? AND revoked=0 AND expires>? ORDER BY id LIMIT 200",
                                      (scope, clock)).fetchall()
        valid, invalid = [], 0
        for row in rows:
            try:
                record = RepairLesson.model_validate_json(row["body"])
                if (record.identifier != row["id"] or record.scope_sha256 != scope
                        or record.expires_at != row["expires"] or not record.created_at <= clock < record.expires_at
                        or record.expires_at - record.created_at > 90 * 86400
                        or set(classify_untrusted_content(record.text)) & {"instruction_injection", "secret_material"}):
                    raise ValueError("invalid lesson")
                valid.append(record)
            except (ValueError, TypeError):
                invalid += 1
        return valid, invalid

    def retrieve(self, *, scope: str, query: str, known_paths: set[str], now: float | None = None):
        valid, invalid = self.records(scope=scope, now=now)
        context, scores = pack_lessons(valid, query, known_paths)
        return context, scores, digest(sorted(record.identifier for record in valid)), invalid


def pack_lessons(valid: list[RepairLesson], query: str, known_paths: set[str]):
    terms = set(tokenize(query)) - _STOP
    if not terms:
        return "", {}
    embedder = HashingSemanticEmbedder()
    query_vector = embedder.encode_query(query[:4000])
    ranked = []
    for record in valid:
        if not set(record.applicable_paths) <= known_paths:
            continue
        doc_terms = set(tokenize(record.text)) - _STOP
        overlap = terms & doc_terms
        if not overlap:
            continue
        lexical = len(overlap) / max(1, min(len(terms), len(doc_terms)))
        vector = embedder.encode_documents([record.text])[0]
        score = 0.7 * lexical + 0.3 * max(0.0, cosine(query_vector, vector))
        if score >= 0.20 and math.isfinite(score):
            ranked.append((score, record.identifier, record))
    sections, scores, chars = [], {}, 0
    for score, identifier, record in sorted(ranked, key=lambda item: (-item[0], item[1]))[:3]:
        section = json.dumps({"lesson_id": identifier, "problem": record.problem,
                              "reviewed_lesson": record.lesson, "applicable_paths": record.applicable_paths}, ensure_ascii=False)
        if chars + len(section) + 1 > 3000:
            continue
        sections.append(section)
        scores[identifier] = score
        chars += len(section) + 1
    return "\n".join(sections), scores


def retrieve_for_task(task: CodeTask, workspace, *, stage: str, iteration: int):
    receipt = RepairMemoryReceipt(stage=stage, iteration=iteration, status="unscoped")
    project = os.getenv("CODE_AGENT_REPAIR_MEMORY_PROJECT", "").strip()
    if not project or not task.repair_memory_namespace:
        return "", receipt
    try:
        receipt.scope_sha256 = scope_hash(project, task.repair_memory_namespace, task.repository)
        receipt.query_sha256 = digest(task.issue[:4000])
        store = RepairMemoryStore(Path(os.getenv("CODE_AGENT_REPAIR_MEMORY_PATH", "data/code-agent/repair-memory.db")))
        context, scores, snapshot, invalid = store.retrieve(scope=receipt.scope_sha256,
            query=task.issue[:4000], known_paths=set(workspace.list_files(task.policy.max_files)))
        receipt.snapshot_sha256, receipt.invalid_records = snapshot, invalid
        receipt.lesson_scores = scores
        receipt.status = "selected" if context else "no_match"
        if context:
            receipt.context_sha256 = digest(context)
            receipt.approximate_tokens = (len(context) + 3) // 4
        return context, receipt
    except (OSError, sqlite3.Error, ValueError, TypeError, RuntimeError):
        receipt.status = "unavailable"
        return "", receipt
