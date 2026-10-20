"""Authenticated browser adapter; binary parsing occurs only in a bounded container."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentforge_sdk.claim_audit import NumericClaim
from agentforge_sdk.revalidation import CheckpointError
from code_agent.workbench import Owner, _PrivateValidationRoute
from service.document_claim_audits import DocumentClaimAudits
from service.document_claim_replay import DocumentClaimReplay
from service.document_claim_stress import DocumentClaimStress
from service.document_reconciliation import DocumentReconciliation
from service.document_revalidation import DocumentRevalidation
from service.document_runtime import DocumentRuntime
from service.document_source_replay import DocumentSourceReplay
from service.document_store import DocumentLibrary, LibraryError

MAX_UPLOAD = 3_000_000


class DocumentQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    document_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    kind: Literal["question", "calculation"]
    question: str = Field(default="", max_length=2000)
    operation: Literal["sum", "difference", "ratio", "percent_change"] = "sum"
    region_ids: list[str] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def meaningful(self):
        if self.kind == "question":
            if not self.question.strip() or self.region_ids:
                raise ValueError("invalid question")
        elif (
            self.question
            or not self.region_ids
            or len(set(self.region_ids)) != len(self.region_ids)
            or any(not re.fullmatch(r"[a-f0-9]{64}", r) for r in self.region_ids)
            or (self.operation != "sum" and len(self.region_ids) != 2)
        ):
            raise ValueError("invalid calculation")
        return self


class DocumentReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: Literal["supported", "unsupported", "needs_review"]
    note: str = Field(max_length=1000)


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    before_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    before_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    after_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    result_ids: list[str] = Field(min_length=1, max_length=10)
    refresh_local: bool = False
    max_answer_attempts: int = Field(default=10, ge=0, le=10)

    @model_validator(mode="after")
    def unique_ids(self):
        if len(set(self.result_ids)) != len(self.result_ids) or any(
            not re.fullmatch(r"[a-f0-9]{32}", r) for r in self.result_ids
        ):
            raise ValueError("invalid result selection")
        return self


class ComparisonStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    checkpoint_fingerprint: str = Field(pattern=r"^(?:initial|[a-f0-9]{64})$")


class _EvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


def _canonical(value: str) -> str:
    if value != " ".join(value.split()) or any(ord(c) < 32 for c in value):
        raise ValueError("use canonical single-line labels")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ValueError("labels must be valid UTF-8") from None
    return value


class EvidenceScope(_EvidenceRequest):
    entity: str = Field(min_length=1, max_length=60)
    period: str = Field(min_length=1, max_length=40)
    basis: str = Field(min_length=1, max_length=80)
    _labels = field_validator("entity", "period", "basis")(_canonical)


class EvidenceSourceRequest(_EvidenceRequest):
    document_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class EvidenceSetRequest(_EvidenceRequest):
    name: str = Field(min_length=3, max_length=80)
    scope: EvidenceScope
    scope_confirmed: Literal[True]
    sources: list[EvidenceSourceRequest] = Field(min_length=2, max_length=5)
    _name = field_validator("name")(_canonical)

    @field_validator("scope_confirmed", mode="before")
    @classmethod
    def explicit_confirmation(cls, value):
        if value is not True:
            raise ValueError("explicit scope confirmation required")
        return value

    @model_validator(mode="after")
    def distinct_source_ids(self):
        if len({s.document_id for s in self.sources}) != len(self.sources):
            raise ValueError("source identifiers must be unique")
        return self


class EvidenceLookup(_EvidenceRequest):
    collection_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    row_label: str = Field(min_length=1, max_length=200)
    column_label: str = Field(min_length=1, max_length=200)
    _labels = field_validator("row_label", "column_label")(_canonical)


class EvidenceCalculation(_EvidenceRequest):
    collection_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    operation: Literal["sum", "difference", "ratio", "percent_change"]
    fact_ids: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def valid_operands(self):
        if (
            len(set(self.fact_ids)) != len(self.fact_ids)
            or any(not re.fullmatch(r"[a-f0-9]{32}", i) for i in self.fact_ids)
            or (self.operation != "sum" and len(self.fact_ids) != 2)
        ):
            raise ValueError("invalid saved fact operands")
        return self


class EvidenceReview(_EvidenceRequest):
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: Literal["inspected", "needs_review", "flagged"]
    note: str = Field(max_length=1000)


class AuditClaim(_EvidenceRequest):
    result_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    claim: NumericClaim


class ClaimAuditRequest(_EvidenceRequest):
    name: str = Field(min_length=3, max_length=80)
    collection_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    claims: list[AuditClaim] = Field(min_length=1, max_length=10)
    _name = field_validator("name")(_canonical)

    @model_validator(mode="after")
    def unique_claims(self):
        if len({item.claim.claim_id for item in self.claims}) != len(self.claims):
            raise ValueError("claim identifiers must be unique")
        return self


class ClaimAuditStep(_EvidenceRequest):
    checkpoint_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class ClaimBundleRequest(ClaimAuditStep):
    include_extraction_text: Literal[True]

    @field_validator("include_extraction_text", mode="before")
    @classmethod
    def explicit_private_export(cls, value):
        if value is not True:
            raise ValueError("explicit private snapshot acknowledgement required")
        return value


class SourceReplayRequest(ClaimAuditStep):
    confirm_reextract: Literal[True]
    allow_original_ocr: bool = False

    @field_validator("confirm_reextract", mode="before")
    @classmethod
    def explicit_reextraction(cls, value):
        if value is not True:
            raise ValueError("explicit local re-extraction confirmation required")
        return value


class CellConcordanceRequest(ClaimAuditStep):
    confirm_crosscheck: Literal[True]

    @field_validator("confirm_crosscheck", mode="before")
    @classmethod
    def explicit_crosscheck(cls, value):
        if value is not True:
            raise ValueError("explicit local PDF text cross-check confirmation required")
        return value


class PixelConcordanceRequest(ClaimAuditStep):
    confirm_local_ocr: Literal[True]

    @field_validator("confirm_local_ocr", mode="before")
    @classmethod
    def explicit_pixel_ocr(cls, value):
        if value is not True:
            raise ValueError("explicit rendered-pixel local OCR consent required")
        return value


class ClaimReplayRequest(_EvidenceRequest):
    name: str = Field(min_length=3, max_length=80)
    target_set_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    source_checkpoint_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    target_collection_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    scope_confirmed: Literal[True]
    _name = field_validator("name")(_canonical)

    @field_validator("scope_confirmed", mode="before")
    @classmethod
    def explicit_confirmation(cls, value):
        if value is not True:
            raise ValueError("explicit scope confirmation required")
        return value


class ClaimStressRequest(_EvidenceRequest):
    name: str = Field(min_length=3, max_length=80)
    source_checkpoint_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    _name = field_validator("name")(_canonical)


def documents_enabled():
    return os.getenv("WORKBENCH_DOCUMENTS_ENABLED", "false").lower() == "true"


def create_document_router(library: DocumentLibrary | None = None) -> APIRouter:
    router = APIRouter(
        prefix="/workbench/documents", tags=["Document evidence"], route_class=_PrivateValidationRoute
    )
    instance = library

    def store():
        nonlocal instance
        if not documents_enabled():
            raise HTTPException(404, "Document Workbench is disabled")
        if instance is None:
            instance = DocumentLibrary(
                Path(os.getenv("WORKBENCH_DOCUMENT_DB", "data/workbench-documents/library.db")),
                DocumentRuntime(),
            )
        return instance

    async def call(method, *args, **kwargs):
        try:
            return await asyncio.to_thread(method, *args, **kwargs)
        except LibraryError as exc:
            raise HTTPException(exc.status, exc.code) from None
        except CheckpointError:
            raise HTTPException(409, "comparison_checkpoint_invalid") from None
        except Exception:
            raise HTTPException(503, "document_library_unavailable") from None

    @router.get("")
    async def listing(user: Owner):
        return {"documents": await call(store().list, user)}

    @router.post("", status_code=201)
    async def upload(
        request: Request,
        user: Owner,
        name: str = Query(min_length=1, max_length=120),
        ocr: bool = False,
        parent_id: str | None = Query(default=None, pattern=r"^[a-f0-9]{32}$"),
    ):
        library = store()
        # Names are display metadata only, never host paths; no HTML/SVG/archive input.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _().-]{0,119}", name) or ".." in name:
            raise HTTPException(422, "invalid_document_name")
        suffix = Path(name).suffix.lower()
        types = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
        if suffix not in types or request.headers.get("content-type") != types[suffix]:
            raise HTTPException(415, "unsupported_document_type")
        raw = bytearray()
        try:
            async with asyncio.timeout(20):
                async for block in request.stream():
                    if len(raw) + len(block) > MAX_UPLOAD:
                        raise HTTPException(413, "document_upload_budget")
                    raw.extend(block)
        except TimeoutError:
            raise HTTPException(408, "document_upload_timeout") from None
        valid = (
            suffix == ".pdf"
            and raw.startswith(b"%PDF-")
            or suffix == ".png"
            and raw.startswith(b"\x89PNG\r\n\x1a\n")
            or suffix in {".jpg", ".jpeg"}
            and raw.startswith(b"\xff\xd8\xff")
        )
        if not valid:
            raise HTTPException(415, "document_signature_mismatch")
        return await call(
            library.ingest, user, bytes(raw), name=name, suffix=suffix, ocr=ocr, parent_id=parent_id
        )

    # Literal collection routes must precede /{document_id}; otherwise history would
    # be interpreted as a document ID by Starlette's ordered route matching.
    @router.get("/reconciliations")
    async def evidence_sets(user: Owner):
        return {"sets": await call(DocumentReconciliation(store()).list, user)}

    @router.post("/reconciliations", status_code=201)
    async def create_evidence_set(payload: EvidenceSetRequest, user: Owner):
        return await call(DocumentReconciliation(store()).create, user, payload.model_dump())

    @router.get("/reconciliations/{set_id}")
    async def evidence_set(set_id: str, user: Owner):
        return await call(DocumentReconciliation(store()).detail, user, set_id)

    @router.post("/reconciliations/{set_id}/facts")
    async def lookup_evidence(set_id: str, payload: EvidenceLookup, user: Owner):
        return await call(DocumentReconciliation(store()).lookup, user, set_id, payload.model_dump())

    @router.post("/reconciliations/{set_id}/calculations")
    async def calculate_evidence(set_id: str, payload: EvidenceCalculation, user: Owner):
        return await call(DocumentReconciliation(store()).calculate, user, set_id, payload.model_dump())

    @router.post("/reconciliations/{set_id}/results/{result_id}/reviews")
    async def review_evidence(set_id: str, result_id: str, payload: EvidenceReview, user: Owner):
        return await call(
            DocumentReconciliation(store()).review, user, set_id, result_id, payload.model_dump()
        )

    @router.delete("/reconciliations/{set_id}")
    async def delete_evidence_set(set_id: str, user: Owner):
        await call(DocumentReconciliation(store()).delete, user, set_id)
        return {"deleted": True}

    @router.get("/reconciliations/{set_id}/audits")
    async def claim_audits(set_id: str, user: Owner):
        return {"audits": await call(DocumentClaimAudits(store()).list, user, set_id)}

    @router.post("/reconciliations/{set_id}/audits", status_code=201)
    async def create_claim_audit(set_id: str, payload: ClaimAuditRequest, user: Owner):
        return await call(DocumentClaimAudits(store()).create, user, set_id, payload.model_dump())

    @router.get("/reconciliations/{set_id}/audits/{audit_id}")
    async def claim_audit(set_id: str, audit_id: str, user: Owner):
        return await call(DocumentClaimAudits(store()).detail, user, set_id, audit_id)

    @router.post("/reconciliations/{set_id}/audits/{audit_id}/advance")
    async def advance_claim_audit(set_id: str, audit_id: str, payload: ClaimAuditStep, user: Owner):
        return await call(
            DocumentClaimAudits(store()).advance, user, set_id, audit_id, payload.checkpoint_fingerprint
        )

    @router.get("/reconciliations/{set_id}/audits/{audit_id}/export")
    async def export_claim_audit(set_id: str, audit_id: str, user: Owner):
        return await call(DocumentClaimAudits(store()).detail, user, set_id, audit_id, export=True)

    @router.delete("/reconciliations/{set_id}/audits/{audit_id}")
    async def delete_claim_audit(set_id: str, audit_id: str, user: Owner):
        await call(DocumentClaimAudits(store()).delete, user, set_id, audit_id)
        return {"deleted": True}

    @router.get("/reconciliations/{set_id}/audits/{audit_id}/replays")
    async def claim_replays(set_id: str, audit_id: str, user: Owner):
        return {"replays": await call(DocumentClaimReplay(store()).list, user, set_id, audit_id)}

    @router.post("/reconciliations/{set_id}/audits/{audit_id}/replays", status_code=201)
    async def create_claim_replay(set_id: str, audit_id: str, payload: ClaimReplayRequest, user: Owner):
        return await call(DocumentClaimReplay(store()).create, user, set_id, audit_id, payload.model_dump())

    @router.get("/reconciliations/{set_id}/audits/{audit_id}/replays/{replay_id}")
    async def claim_replay(set_id: str, audit_id: str, replay_id: str, user: Owner):
        return await call(DocumentClaimReplay(store()).detail, user, set_id, audit_id, replay_id)

    @router.post("/reconciliations/{set_id}/audits/{audit_id}/replays/{replay_id}/advance")
    async def advance_claim_replay(
        set_id: str, audit_id: str, replay_id: str, payload: ClaimAuditStep, user: Owner
    ):
        return await call(
            DocumentClaimReplay(store()).advance,
            user,
            set_id,
            audit_id,
            replay_id,
            payload.checkpoint_fingerprint,
        )

    @router.delete("/reconciliations/{set_id}/audits/{audit_id}/replays/{replay_id}")
    async def delete_claim_replay(set_id: str, audit_id: str, replay_id: str, user: Owner):
        await call(DocumentClaimReplay(store()).delete, user, set_id, audit_id, replay_id)
        return {"deleted": True}

    @router.get("/{document_id}")
    async def detail(document_id: str, user: Owner):
        return await call(store().detail, user, document_id)

    @router.post("/reconciliations/{set_id}/audits/{audit_id}/sources/{source_id}/replay-original")
    async def replay_original_source(
        set_id: str, audit_id: str, source_id: str, payload: SourceReplayRequest, user: Owner
    ):
        return await call(
            DocumentSourceReplay(store()).replay, user, set_id, audit_id, source_id, payload.model_dump()
        )

    @router.post("/reconciliations/{set_id}/audits/{audit_id}/replay-bundle")
    async def export_claim_bundle(set_id: str, audit_id: str, payload: ClaimBundleRequest, user: Owner):
        return await call(
            DocumentClaimAudits(store()).replay_bundle,
            user,
            set_id,
            audit_id,
            payload.checkpoint_fingerprint,
        )

    @router.post("/reconciliations/{set_id}/audits/{audit_id}/sources/{source_id}/crosscheck-cells")
    async def crosscheck_source_cells(
        set_id: str, audit_id: str, source_id: str, payload: CellConcordanceRequest, user: Owner
    ):
        return await call(
            DocumentSourceReplay(store()).crosscheck, user, set_id, audit_id, source_id, payload.model_dump()
        )

    @router.post("/reconciliations/{set_id}/audits/{audit_id}/sources/{source_id}/crosscheck-pixels")
    async def crosscheck_source_pixels(
        set_id: str, audit_id: str, source_id: str, payload: PixelConcordanceRequest, user: Owner
    ):
        return await call(
            DocumentSourceReplay(store()).crosscheck_pixels,
            user,
            set_id,
            audit_id,
            source_id,
            payload.model_dump(),
        )

    @router.get("/reconciliations/{set_id}/audits/{audit_id}/stress-tests")
    async def claim_stress_tests(set_id: str, audit_id: str, user: Owner):
        return {"stress_tests": await call(DocumentClaimStress(store()).list, user, set_id, audit_id)}

    @router.post("/reconciliations/{set_id}/audits/{audit_id}/stress-tests", status_code=201)
    async def create_claim_stress(set_id: str, audit_id: str, payload: ClaimStressRequest, user: Owner):
        return await call(DocumentClaimStress(store()).create, user, set_id, audit_id, payload.model_dump())

    @router.get("/reconciliations/{set_id}/audits/{audit_id}/stress-tests/{run_id}")
    async def claim_stress(set_id: str, audit_id: str, run_id: str, user: Owner):
        return await call(DocumentClaimStress(store()).detail, user, set_id, audit_id, run_id)

    @router.post("/reconciliations/{set_id}/audits/{audit_id}/stress-tests/{run_id}/advance")
    async def advance_claim_stress(
        set_id: str, audit_id: str, run_id: str, payload: ClaimAuditStep, user: Owner
    ):
        return await call(
            DocumentClaimStress(store()).advance,
            user,
            set_id,
            audit_id,
            run_id,
            payload.checkpoint_fingerprint,
        )

    @router.delete("/reconciliations/{set_id}/audits/{audit_id}/stress-tests/{run_id}")
    async def delete_claim_stress(set_id: str, audit_id: str, run_id: str, user: Owner):
        await call(DocumentClaimStress(store()).delete, user, set_id, audit_id, run_id)
        return {"deleted": True}

    @router.get("/{document_id}/pages/{page}")
    async def preview(document_id: str, page: int, user: Owner):
        return await call(store().preview, user, document_id, page)

    @router.post("/{document_id}/queries")
    async def query(document_id: str, payload: DocumentQuery, user: Owner):
        return await call(store().query, user, document_id, payload.model_dump())

    @router.post("/{document_id}/results/{result_id}/reviews")
    async def review(document_id: str, result_id: str, payload: DocumentReview, user: Owner):
        return await call(store().review, user, document_id, result_id, **payload.model_dump())

    @router.delete("/{document_id}")
    async def delete(document_id: str, user: Owner):
        await call(store().delete, user, document_id)
        return {"deleted": True}

    @router.post("/{document_id}/comparisons", status_code=201)
    async def create_comparison(document_id: str, payload: ComparisonRequest, user: Owner):
        return await call(DocumentRevalidation(store()).create, user, document_id, payload.model_dump())

    @router.get("/{document_id}/comparisons")
    async def comparisons(document_id: str, user: Owner):
        return {"comparisons": await call(DocumentRevalidation(store()).list, user, document_id)}

    @router.get("/{document_id}/comparisons/{comparison_id}")
    async def comparison(document_id: str, comparison_id: str, user: Owner):
        return await call(DocumentRevalidation(store()).detail, user, document_id, comparison_id)

    @router.post("/{document_id}/comparisons/{comparison_id}/advance")
    async def advance_comparison(document_id: str, comparison_id: str, payload: ComparisonStep, user: Owner):
        return await call(
            DocumentRevalidation(store()).advance,
            user,
            document_id,
            comparison_id,
            payload.checkpoint_fingerprint,
        )

    return router
