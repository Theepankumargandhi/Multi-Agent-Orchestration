"""Small, versioned text-evidence contracts; no service, model or environment imports."""

from __future__ import annotations

import hashlib
import json
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROTOCOL = "agentforge-document-sdk-v1"


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False, hide_input_in_errors=True
    )


class DocumentPage(Contract):
    document_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$")
    text: str = Field(min_length=1, max_length=200_000, repr=False)
    source: str = Field(min_length=1, max_length=500)
    title: str = Field(default="", max_length=200)
    version: str = Field(default="unversioned", min_length=1, max_length=100)
    page: int | None = Field(default=None, ge=1, le=100_000)

    @field_validator("text")
    @classmethod
    def meaningful_text(cls, value):
        if not value.strip() or "\x00" in value:
            raise ValueError("page text must be nonempty text without NUL bytes")
        return value

    @field_validator("source", "title", "version")
    @classmethod
    def safe_metadata(cls, value):
        if any(ord(char) < 32 for char in value):
            raise ValueError("metadata cannot contain control characters")
        return value

    @field_validator("source")
    @classmethod
    def no_credential_urls(cls, value):
        parsed = urlsplit(value)
        if parsed.username or parsed.password or parsed.query:
            raise ValueError("source labels cannot contain URL credentials or query strings")
        return value


class RAGConfig(Contract):
    chunk_size: int = Field(default=1000, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    max_pages: int = Field(default=200, ge=1, le=1000)
    max_corpus_bytes: int = Field(default=5_000_000, ge=1000, le=20_000_000)
    max_chunks: int = Field(default=2000, ge=1, le=5000)
    candidate_top_k: int = Field(default=12, ge=1, le=40)
    max_context_chars: int = Field(default=6000, ge=256, le=32_000)
    max_quote_chars: int = Field(default=360, ge=80, le=600)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    allow_external_processing: bool = False
    allow_lexical_fallback: bool = True

    @model_validator(mode="after")
    def valid_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class Evidence(Contract):
    evidence_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    document_id: str
    source: str
    title: str
    version: str
    page: int | None
    input_text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    chunk_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    # Shared sentence-aware chunking may retain overlap plus a complete sentence.
    text: str = Field(min_length=1, max_length=5001, repr=False)
    lexical_score: float = Field(ge=0)
    vector_score: float | None = None
    fused_score: float = Field(ge=0)


class SearchResult(Contract):
    protocol: Literal["agentforge-document-sdk-v1"] = PROTOCOL
    query_sha256: str
    corpus_sha256: str
    config_sha256: str
    status: Literal["ok", "empty", "degraded"]
    reason: Literal["ranked", "no_lexical_hits", "embedding_fallback", "context_budget_empty"]
    evidence: tuple[Evidence, ...]
    backend: str
    candidate_count: int = Field(ge=0)
    context_chars: int = Field(ge=0)
    embedding_invocations: int = Field(default=0, ge=0)
    external_processing_authorized: bool = False
    fingerprint: str


class Selection(Contract):
    evidence_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    quote: str = Field(min_length=3, max_length=600, repr=False)


class Citation(Contract):
    evidence_id: str
    document_id: str
    source: str
    version: str
    page: int | None
    input_text_sha256: str
    chunk_sha256: str
    quote: str = Field(repr=False)
    # Offsets are into the returned Evidence.text, NOT page coordinates or PDF geometry.
    chunk_start: int = Field(ge=0)
    chunk_end: int = Field(gt=0)


class AnswerResult(Contract):
    protocol: Literal["agentforge-document-sdk-v1"] = PROTOCOL
    status: Literal["evidence_found", "abstained"]
    reason: Literal[
        "literal_evidence_quotes",
        "no_evidence",
        "selection_rejected",
        "selector_failed",
        "selector_abstained",
    ]
    answer: str = Field(repr=False)
    citations: tuple[Citation, ...]
    retrieval: SearchResult = Field(repr=False)
    answer_mode: Literal["extractive", "extractive_selector"]
    selector_backend: str
    selector_invocations: int = Field(default=0, ge=0)
    # Unknown adapter/provider billing is null, never assumed free.
    provider_cost_usd: float | None = Field(default=None, ge=0)
    fingerprint: str


class SDKError(RuntimeError):
    """Content-free SDK error. Submitted text/provider errors are not interpolated."""


class ExternalProcessingDenied(SDKError):
    pass


class BackendError(SDKError):
    pass


class DocumentInputError(SDKError):
    pass
