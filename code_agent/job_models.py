"""Public and persisted contracts for durable coding-agent jobs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from code_agent.models import CodeAgentResult

JobStatus = Literal[
    "queued",
    "running",
    "retry_wait",
    "awaiting_approval",
    "approved",
    "rejected",
    "failed",
    "dead_letter",
]


class CodeTaskRequest(BaseModel):
    repository: str = Field(min_length=1, max_length=200)
    issue: str = Field(min_length=10, max_length=12000)
    model: str = Field(default="gpt-4o-mini", min_length=1, max_length=100)
    test_command: list[str] = Field(
        default_factory=lambda: ["python", "-m", "pytest", "-q"], min_length=1, max_length=30
    )


class CodeTaskDecision(BaseModel):
    approve: bool
    reason: str = Field(default="", max_length=1000)


class ArtifactReference(BaseModel):
    sha256: str
    size_bytes: int = Field(ge=0)


class JobRecord(BaseModel):
    task_id: str
    user_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    request: CodeTaskRequest
    result: CodeAgentResult | None = None
    error: str = ""
    decision_reason: str = ""
    attempt: int = 0
    max_attempts: int = 3
    available_at: str = ""
    lease_owner: str = ""
    lease_expires_at: str = ""
    heartbeat_at: str = ""
    idempotency_key: str = ""
    request_fingerprint: str = ""
    artifact: ArtifactReference | None = None
    dossier_json: ArtifactReference | None = None
    dossier_markdown: ArtifactReference | None = None

    def public(self) -> dict:
        payload = self.model_dump()
        if payload.get("result"):
            payload["result"]["patch"] = ""
        # Worker identity and request fingerprints are control-plane internals.
        payload.pop("lease_owner", None)
        payload.pop("request_fingerprint", None)
        return payload
