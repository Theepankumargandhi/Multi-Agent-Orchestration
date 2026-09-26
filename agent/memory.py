"""Tenant-isolated, provenance-aware long-term memory for AgentForge."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent.model_gateway import estimate_tokens

MemoryType = Literal["episodic", "semantic", "preference", "procedural"]
MemoryStatus = Literal["active", "quarantined", "superseded", "tombstoned"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_POISONING_RE = re.compile(
    r"(?i)(ignore\s+(all\s+)?previous|system\s+prompt|developer\s+message|"
    r"execute\s+(this|the)\s+(command|tool)|call\s+the\s+tool|override\s+(safety|policy)|"
    r"reveal\s+(a\s+)?secret|bypass\s+(the\s+)?guard)"
)
_SENSITIVE_RE = re.compile(
    r"(?i)(-----BEGIN [A-Z ]*PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{16,}|"
    r"\b(?:api[_ -]?key|password|access[_ -]?token)\s*[:=])"
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()-]{8,}\d)(?!\d)")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _fingerprint(value: object, key: bytes | None = None) -> str:
    payload = _canonical(value)
    if key:
        return hmac.new(key, payload, hashlib.sha256).hexdigest()
    return hashlib.sha256(payload).hexdigest()


def _normalize(value: str) -> str:
    return " ".join(_TOKEN_RE.findall(value.casefold()))[:240]


def _vector(text: str, dimensions: int = 256) -> dict[int, float]:
    tokens = _TOKEN_RE.findall(text.casefold())
    features = tokens + [f"{left}:{right}" for left, right in zip(tokens, tokens[1:])]
    counts: Counter[int] = Counter()
    for feature in features:
        digest = hashlib.sha256(feature.encode()).digest()
        counts[int.from_bytes(digest[:4], "big") % dimensions] += 1
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1
    return {index: value / norm for index, value in counts.items()}


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    return sum(value * right.get(index, 0) for index, value in left.items())


def _redact_pii(value: str) -> tuple[str, bool]:
    redacted, email_count = _EMAIL_RE.subn("[redacted-email]", value)
    redacted, phone_count = _PHONE_RE.subn("[redacted-phone]", redacted)
    return redacted, bool(email_count or phone_count)


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_type: MemoryType
    subject: str = Field(min_length=2, max_length=160)
    content: str = Field(min_length=2, max_length=1200)
    confidence: float = Field(default=0.8, ge=0, le=1)
    importance: float = Field(default=0.5, ge=0, le=1)
    trust_score: float = Field(default=0.8, ge=0, le=1)
    provenance: str = Field(default="explicit_user_statement", min_length=2, max_length=120)
    ttl_seconds: int | None = Field(default=None, ge=60, le=31_536_000)


class MemoryRecord(BaseModel):
    memory_id: str
    tenant_id: str
    memory_type: MemoryType
    subject: str
    content: str
    normalized_key: str
    confidence: float
    importance: float
    trust_score: float
    status: MemoryStatus
    provenance: str
    created_at: float
    updated_at: float
    expires_at: float | None = None
    use_count: int = 0
    usefulness: float = 0.5
    version: int = 1
    supersedes: str = ""
    fingerprint: str = ""


class MemoryScore(BaseModel):
    memory_id: str
    score: float
    semantic: float
    lexical: float
    recency: float
    confidence: float
    trust: float
    importance: float
    usefulness: float


class MemoryRetrievalReceipt(BaseModel):
    tenant_fingerprint: str
    query_fingerprint: str
    selected: list[MemoryScore]
    considered: int
    rejected: int
    token_budget: int
    tokens_used: int
    policy_version: str = "memory-policy-v1"
    receipt_fingerprint: str = ""


class MemorySearchResult(BaseModel):
    context: str
    records: list[MemoryRecord]
    receipt: MemoryRetrievalReceipt


class MemoryWriteReceipt(BaseModel):
    action: Literal["created", "deduplicated", "superseded", "quarantined", "ignored"]
    memory_id: str = ""
    superseded_id: str = ""
    reason: str = ""
    receipt_fingerprint: str = ""


class MemoryCorrection(BaseModel):
    content: str = Field(min_length=2, max_length=1200)
    confidence: float = Field(default=1.0, ge=0, le=1)
    reason: str = Field(default="user_correction", max_length=160)


class MemoryOutcome(BaseModel):
    helpful: bool
    reason: str = Field(default="agent_outcome_feedback", max_length=160)


class MemoryPolicy(BaseModel):
    version: str = "memory-policy-v1"
    min_write_confidence: float = Field(default=0.65, ge=0, le=1)
    min_retrieval_score: float = Field(default=0.16, ge=0, le=1)
    max_candidates: int = Field(default=500, ge=10, le=10_000)
    default_token_budget: int = Field(default=384, ge=32, le=4096)
    max_results: int = Field(default=8, ge=1, le=50)
    recency_half_life_days: float = Field(default=90, gt=0, le=3650)


class AgentMemoryStore:
    """SQLite reference backend with strict tenant filtering and tombstone deletion."""

    def __init__(
        self,
        path: str | Path,
        *,
        policy: MemoryPolicy | None = None,
        integrity_key: bytes | None = None,
        clock=time.time,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.policy = policy or MemoryPolicy()
        self.integrity_key = integrity_key
        self.clock = clock
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL").close()
        connection.execute("PRAGMA foreign_keys=ON").close()
        return connection

    @contextmanager
    def _connection(self):
        """Commit or roll back and always release SQLite/WAL file handles."""
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_memories (
                    memory_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    normalized_key TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    importance REAL NOT NULL,
                    trust_score REAL NOT NULL,
                    status TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    usefulness REAL NOT NULL DEFAULT 0.5,
                    version INTEGER NOT NULL DEFAULT 1,
                    supersedes TEXT NOT NULL DEFAULT '',
                    fingerprint TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_tenant_status
                    ON agent_memories(tenant_id, status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_key
                    ON agent_memories(tenant_id, memory_type, normalized_key, status);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_one_active_key
                    ON agent_memories(tenant_id, memory_type, normalized_key)
                    WHERE status = 'active';
                CREATE TABLE IF NOT EXISTS memory_audit_events (
                    event_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    fingerprint TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_events_tenant
                    ON memory_audit_events(tenant_id, created_at DESC);
                """
            )

    def _seal_record(self, record: MemoryRecord) -> MemoryRecord:
        record.fingerprint = _fingerprint(
            record.model_dump(mode="json", exclude={"fingerprint"}), self.integrity_key
        )
        return record

    def verify_record(self, record: MemoryRecord) -> bool:
        expected = _fingerprint(
            record.model_dump(mode="json", exclude={"fingerprint"}), self.integrity_key
        )
        return hmac.compare_digest(record.fingerprint, expected)

    def _audit(self, connection, tenant_id: str, memory_id: str, action: str, metadata: dict) -> None:
        created_at = self.clock()
        event_id = f"mev_{uuid4().hex}"
        payload = {
            "event_id": event_id,
            "tenant_id": tenant_id,
            "memory_id": memory_id,
            "action": action,
            "metadata": metadata,
            "created_at": created_at,
        }
        connection.execute(
            "INSERT INTO memory_audit_events VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                tenant_id,
                memory_id,
                action,
                json.dumps(metadata, sort_keys=True),
                created_at,
                _fingerprint(payload, self.integrity_key),
            ),
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(**dict(row))

    def _save(self, connection, record: MemoryRecord) -> None:
        sealed = self._seal_record(record)
        connection.execute(
            """INSERT INTO agent_memories VALUES (
                :memory_id, :tenant_id, :memory_type, :subject, :content, :normalized_key,
                :confidence, :importance, :trust_score, :status, :provenance, :created_at,
                :updated_at, :expires_at, :use_count, :usefulness, :version, :supersedes,
                :fingerprint
            )""",
            sealed.model_dump(),
        )

    def remember(self, tenant_id: str, candidate: MemoryCandidate) -> MemoryWriteReceipt:
        tenant = tenant_id.strip()
        if not tenant:
            raise ValueError("tenant_id is required")
        content, pii_redacted = _redact_pii(candidate.content.strip())
        subject = candidate.subject.strip()
        normalized_key = _normalize(subject)
        secret_detected = bool(_SENSITIVE_RE.search(content))
        poisoning_detected = bool(_POISONING_RE.search(content))
        suspicious = poisoning_detected or secret_detected
        if secret_detected:
            content = "[quarantined-sensitive-content]"
        status: MemoryStatus = "quarantined" if suspicious else "active"
        if candidate.confidence < self.policy.min_write_confidence:
            receipt = MemoryWriteReceipt(action="ignored", reason="confidence_below_write_policy")
            receipt.receipt_fingerprint = _fingerprint(
                receipt.model_dump(exclude={"receipt_fingerprint"}), self.integrity_key
            )
            return receipt
        now = self.clock()
        with self._connection() as connection:
            existing_row = connection.execute(
                """SELECT * FROM agent_memories
                WHERE tenant_id = ? AND memory_type = ? AND normalized_key = ? AND status = 'active'
                ORDER BY version DESC LIMIT 1""",
                (tenant, candidate.memory_type, normalized_key),
            ).fetchone()
            existing = self._from_row(existing_row) if existing_row else None
            if existing and _normalize(existing.content) == _normalize(content):
                receipt = MemoryWriteReceipt(
                    action="deduplicated", memory_id=existing.memory_id, reason="same_active_memory"
                )
                self._audit(connection, tenant, existing.memory_id, "deduplicated", {})
            else:
                version = (existing.version + 1) if existing else 1
                memory_id = f"mem_{uuid4().hex}"
                record = MemoryRecord(
                    memory_id=memory_id,
                    tenant_id=tenant,
                    memory_type=candidate.memory_type,
                    subject=subject,
                    content=content,
                    normalized_key=normalized_key,
                    confidence=candidate.confidence,
                    importance=candidate.importance,
                    trust_score=candidate.trust_score,
                    status=status,
                    provenance=candidate.provenance + (":pii_redacted" if pii_redacted else ""),
                    created_at=now,
                    updated_at=now,
                    expires_at=now + candidate.ttl_seconds if candidate.ttl_seconds else None,
                    version=version,
                    supersedes=existing.memory_id if existing else "",
                )
                if existing and status == "active":
                    existing.status = "superseded"
                    existing.updated_at = now
                    existing = self._seal_record(existing)
                    connection.execute(
                        "UPDATE agent_memories SET status=?, updated_at=?, fingerprint=? WHERE memory_id=?",
                        (existing.status, existing.updated_at, existing.fingerprint, existing.memory_id),
                    )
                self._save(connection, record)
                action = "quarantined" if status == "quarantined" else (
                    "superseded" if existing else "created"
                )
                receipt = MemoryWriteReceipt(
                    action=action,
                    memory_id=memory_id,
                    superseded_id=existing.memory_id if existing and status == "active" else "",
                    reason="memory_poisoning_or_secret_pattern" if suspicious else "write_policy_passed",
                )
                self._audit(
                    connection,
                    tenant,
                    memory_id,
                    action,
                    {
                        "type": candidate.memory_type,
                        "pii_redacted": pii_redacted,
                        "poisoning_detected": poisoning_detected,
                        "secret_detected": secret_detected,
                    },
                )
        receipt.receipt_fingerprint = _fingerprint(
            receipt.model_dump(exclude={"receipt_fingerprint"}), self.integrity_key
        )
        return receipt

    def list_memories(
        self,
        tenant_id: str,
        *,
        memory_type: MemoryType | None = None,
        include_inactive: bool = False,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        clauses = ["tenant_id = ?"]
        params: list[object] = [tenant_id]
        if memory_type:
            clauses.append("memory_type = ?")
            params.append(memory_type)
        if not include_inactive:
            clauses.append("status = 'active'")
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(self.clock())
        params.append(max(1, min(limit, 500)))
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM agent_memories WHERE {' AND '.join(clauses)} "
                "ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def search(
        self,
        tenant_id: str,
        query: str,
        *,
        limit: int | None = None,
        token_budget: int | None = None,
    ) -> MemorySearchResult:
        clean_query = query.strip()
        limit = max(1, min(limit or self.policy.max_results, self.policy.max_results))
        budget = max(32, min(token_budget or self.policy.default_token_budget, 4096))
        records = self.list_memories(
            tenant_id, include_inactive=False, limit=self.policy.max_candidates
        )
        query_vector = _vector(clean_query)
        query_tokens = set(_TOKEN_RE.findall(clean_query.casefold()))
        now = self.clock()
        ranked: list[tuple[MemoryRecord, MemoryScore]] = []
        for record in records:
            if not self.verify_record(record):
                continue
            memory_text = f"{record.subject} {record.content}"
            semantic = max(0, _cosine(query_vector, _vector(memory_text)))
            memory_tokens = set(_TOKEN_RE.findall(memory_text.casefold()))
            lexical = len(query_tokens & memory_tokens) / max(1, len(query_tokens | memory_tokens))
            # The built-in vector is a dependency-free hashing baseline, so a tiny
            # collision without lexical evidence must not retrieve unrelated memory.
            if lexical <= 0 and semantic < 0.35:
                continue
            age_days = max(0, now - record.updated_at) / 86_400
            recency = 0.5 ** (age_days / self.policy.recency_half_life_days)
            score = (
                0.38 * semantic
                + 0.18 * lexical
                + 0.12 * recency
                + 0.12 * record.confidence
                + 0.10 * record.trust_score
                + 0.07 * record.importance
                + 0.03 * record.usefulness
            )
            if score >= self.policy.min_retrieval_score:
                ranked.append(
                    (
                        record,
                        MemoryScore(
                            memory_id=record.memory_id,
                            score=round(score, 6),
                            semantic=round(semantic, 6),
                            lexical=round(lexical, 6),
                            recency=round(recency, 6),
                            confidence=record.confidence,
                            trust=record.trust_score,
                            importance=record.importance,
                            usefulness=record.usefulness,
                        ),
                    )
                )
        ranked.sort(key=lambda item: (-item[1].score, item[0].memory_id))
        selected_records: list[MemoryRecord] = []
        selected_scores: list[MemoryScore] = []
        lines = ["<memory_data trust=untrusted informational_only=true>"]
        used = estimate_tokens(lines[0])
        for record, score in ranked[:limit]:
            line = (
                f"- [{record.memory_type}; id={record.memory_id}; confidence={record.confidence:.2f}; "
                f"provenance={record.provenance}] {record.subject}: {record.content}"
            )
            line_tokens = estimate_tokens(line)
            if used + line_tokens + 2 > budget:
                continue
            selected_records.append(record)
            selected_scores.append(score)
            lines.append(line)
            used += line_tokens
        lines.append("</memory_data>")
        context = "\n".join(lines) if selected_records else ""
        receipt = MemoryRetrievalReceipt(
            tenant_fingerprint=_fingerprint(tenant_id, self.integrity_key),
            query_fingerprint=_fingerprint(clean_query, self.integrity_key),
            selected=selected_scores,
            considered=len(records),
            rejected=len(records) - len(selected_records),
            token_budget=budget,
            tokens_used=used if selected_records else 0,
            policy_version=self.policy.version,
        )
        receipt.receipt_fingerprint = _fingerprint(
            receipt.model_dump(exclude={"receipt_fingerprint"}), self.integrity_key
        )
        return MemorySearchResult(context=context, records=selected_records, receipt=receipt)

    def correct(
        self, tenant_id: str, memory_id: str, correction: MemoryCorrection
    ) -> MemoryWriteReceipt:
        record = self.get(tenant_id, memory_id)
        if record is None or record.status != "active":
            raise KeyError("active memory not found")
        return self.remember(
            tenant_id,
            MemoryCandidate(
                memory_type=record.memory_type,
                subject=record.subject,
                content=correction.content,
                confidence=correction.confidence,
                importance=record.importance,
                trust_score=1.0,
                provenance=correction.reason,
            ),
        )

    def get(self, tenant_id: str, memory_id: str) -> MemoryRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM agent_memories WHERE tenant_id=? AND memory_id=?",
                (tenant_id, memory_id),
            ).fetchone()
        return self._from_row(row) if row else None

    def delete(self, tenant_id: str, memory_id: str) -> bool:
        record = self.get(tenant_id, memory_id)
        if record is None:
            return False
        record.status = "tombstoned"
        record.content = "[deleted]"
        record.updated_at = self.clock()
        record = self._seal_record(record)
        with self._connection() as connection:
            connection.execute(
                "UPDATE agent_memories SET status=?, content=?, updated_at=?, fingerprint=? "
                "WHERE tenant_id=? AND memory_id=?",
                (
                    record.status,
                    record.content,
                    record.updated_at,
                    record.fingerprint,
                    tenant_id,
                    memory_id,
                ),
            )
            self._audit(connection, tenant_id, memory_id, "tombstoned", {})
        return True

    def record_outcome(
        self, tenant_id: str, memory_id: str, outcome: MemoryOutcome
    ) -> MemoryRecord:
        """Update usefulness from downstream evidence without changing memory content."""
        record = self.get(tenant_id, memory_id)
        if record is None or record.status != "active":
            raise KeyError("active memory not found")
        observation = 1.0 if outcome.helpful else 0.0
        record.usefulness = (
            (record.usefulness * record.use_count) + observation
        ) / (record.use_count + 1)
        record.use_count += 1
        record.updated_at = self.clock()
        record = self._seal_record(record)
        with self._connection() as connection:
            connection.execute(
                "UPDATE agent_memories SET usefulness=?, use_count=?, updated_at=?, fingerprint=? "
                "WHERE tenant_id=? AND memory_id=?",
                (
                    record.usefulness,
                    record.use_count,
                    record.updated_at,
                    record.fingerprint,
                    tenant_id,
                    memory_id,
                ),
            )
            self._audit(
                connection,
                tenant_id,
                memory_id,
                "outcome_recorded",
                {"helpful": outcome.helpful, "reason": outcome.reason},
            )
        return record

    def forget_tenant(self, tenant_id: str) -> int:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM agent_memories WHERE tenant_id=? AND status != 'tombstoned'",
                (tenant_id,),
            ).fetchall()
            for row in rows:
                record = self._from_row(row)
                record.status = "tombstoned"
                record.content = "[deleted]"
                record.updated_at = self.clock()
                record = self._seal_record(record)
                connection.execute(
                    "UPDATE agent_memories SET status=?, content=?, updated_at=?, fingerprint=? "
                    "WHERE tenant_id=? AND memory_id=?",
                    (
                        record.status,
                        record.content,
                        record.updated_at,
                        record.fingerprint,
                        tenant_id,
                        record.memory_id,
                    ),
                )
                self._audit(connection, tenant_id, record.memory_id, "tombstoned", {})
        return len(rows)

    def export(self, tenant_id: str) -> dict:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM agent_memories WHERE tenant_id=? ORDER BY updated_at DESC",
                (tenant_id,),
            ).fetchall()
        records = [self._from_row(row) for row in rows]
        return {
            "schema_version": "1.0",
            "tenant_fingerprint": _fingerprint(tenant_id, self.integrity_key),
            "exported_at": datetime.now(UTC).isoformat(),
            "memories": [item.model_dump(mode="json") for item in records],
            "export_fingerprint": _fingerprint(
                [item.model_dump(mode="json") for item in records], self.integrity_key
            ),
        }

    def audit_events(self, tenant_id: str, limit: int = 100) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_audit_events WHERE tenant_id=? "
                "ORDER BY created_at DESC LIMIT ?",
                (tenant_id, max(1, min(limit, 500))),
            ).fetchall()
        events = []
        for row in rows:
            metadata = json.loads(row["metadata"])
            payload = {
                "event_id": row["event_id"],
                "tenant_id": row["tenant_id"],
                "memory_id": row["memory_id"],
                "action": row["action"],
                "metadata": metadata,
                "created_at": row["created_at"],
            }
            events.append(
                {
                    "event_id": row["event_id"],
                    "memory_id": row["memory_id"],
                    "action": row["action"],
                    "metadata": metadata,
                    "created_at": row["created_at"],
                    "fingerprint": row["fingerprint"],
                    "integrity_verified": hmac.compare_digest(
                        row["fingerprint"], _fingerprint(payload, self.integrity_key)
                    ),
                }
            )
        return events

    def consolidate(self, tenant_id: str) -> dict[str, int]:
        records = self.list_memories(tenant_id, include_inactive=False, limit=500)
        groups: dict[tuple[str, str], list[MemoryRecord]] = {}
        for record in records:
            groups.setdefault((record.memory_type, record.normalized_key), []).append(record)
        superseded = 0
        for items in groups.values():
            if len(items) < 2:
                continue
            keep = max(items, key=lambda item: (item.version, item.trust_score, item.updated_at))
            for item in items:
                if item.memory_id != keep.memory_id:
                    item.status = "superseded"
                    item.updated_at = self.clock()
                    item = self._seal_record(item)
                    with self._connection() as connection:
                        connection.execute(
                            "UPDATE agent_memories SET status=?, updated_at=?, fingerprint=? "
                            "WHERE tenant_id=? AND memory_id=?",
                            (
                                item.status,
                                item.updated_at,
                                item.fingerprint,
                                tenant_id,
                                item.memory_id,
                            ),
                        )
                    superseded += 1
        promoted = 0
        for episode in records:
            if (
                episode.memory_type != "episodic"
                or episode.status != "active"
                or episode.use_count < 3
                or episode.usefulness < 0.7
            ):
                continue
            receipt = self.remember(
                tenant_id,
                MemoryCandidate(
                    memory_type="semantic",
                    subject=f"learned:{episode.subject}",
                    content=episode.content,
                    confidence=min(1.0, episode.confidence * episode.usefulness),
                    importance=episode.importance,
                    trust_score=episode.trust_score,
                    provenance=f"episodic_consolidation:{episode.memory_id}",
                ),
            )
            if receipt.action not in {"created", "superseded", "deduplicated"}:
                continue
            episode.status = "superseded"
            episode.updated_at = self.clock()
            episode = self._seal_record(episode)
            with self._connection() as connection:
                connection.execute(
                    "UPDATE agent_memories SET status=?, updated_at=?, fingerprint=? "
                    "WHERE tenant_id=? AND memory_id=?",
                    (
                        episode.status,
                        episode.updated_at,
                        episode.fingerprint,
                        tenant_id,
                        episode.memory_id,
                    ),
                )
                self._audit(
                    connection,
                    tenant_id,
                    episode.memory_id,
                    "consolidated",
                    {"semantic_memory_id": receipt.memory_id},
                )
            promoted += 1
        return {
            "active_examined": len(records),
            "superseded": superseded,
            "promoted": promoted,
        }


def extract_memory_candidates(user_text: str) -> list[MemoryCandidate]:
    """Conservative explicit-consent extractor; no full conversation is persisted."""
    text = " ".join((user_text or "").strip().split())
    if not text or len(text) > 1200:
        return []
    patterns: list[tuple[re.Pattern, MemoryType, str]] = [
        (re.compile(r"(?i)^remember\s+this\s+workflow\s*:\s*(.+)$"), "procedural", "workflow"),
        (re.compile(r"(?i)^remember\s+this\s+interaction\s*:\s*(.+)$"), "episodic", "interaction"),
        (re.compile(r"(?i)^remember\s+that\s+(.+)$"), "semantic", "user_fact"),
        (re.compile(r"(?i)^my\s+preferred\s+([\w -]{2,40})\s+is\s+(.+)$"), "preference", ""),
        (re.compile(r"(?i)^call\s+me\s+(.+)$"), "preference", "display_name"),
        (re.compile(r"(?i)^i\s+prefer\s+(.+)$"), "preference", "general_preference"),
    ]
    for pattern, memory_type, fixed_subject in patterns:
        match = pattern.match(text)
        if not match:
            continue
        if memory_type == "preference" and not fixed_subject and len(match.groups()) == 2:
            subject, content = match.group(1), match.group(2)
        else:
            subject, content = fixed_subject, match.group(match.lastindex or 1)
        return [
            MemoryCandidate(
                memory_type=memory_type,
                subject=subject,
                content=content,
                confidence=0.95,
                importance=0.7,
                trust_score=0.9,
                provenance="explicit_user_memory_request",
            )
        ]
    return []


_STORE: AgentMemoryStore | None = None
_STORE_CONFIG: tuple[str, str] | None = None


def get_memory_store() -> AgentMemoryStore:
    global _STORE, _STORE_CONFIG
    path = os.getenv("AGENT_MEMORY_DB_PATH", "data/memory/agent_memory.db")
    key_text = os.getenv("AGENT_MEMORY_INTEGRITY_KEY", "")
    enabled = os.getenv("AGENT_MEMORY_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if enabled and len(key_text.encode()) < 16:
        raise RuntimeError(
            "AGENT_MEMORY_ENABLED requires AGENT_MEMORY_INTEGRITY_KEY with at least 16 bytes"
        )
    config = (path, key_text)
    if _STORE is None or _STORE_CONFIG != config:
        _STORE = AgentMemoryStore(path, integrity_key=key_text.encode() or None)
        _STORE_CONFIG = config
    return _STORE
