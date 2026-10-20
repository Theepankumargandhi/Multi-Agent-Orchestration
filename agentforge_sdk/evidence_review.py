"""Explicit review overlays; original extraction is never rewritten or auto-approved."""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from threading import RLock
from typing import Literal

from pydantic import Field, model_validator

from agentforge_sdk.inputs import _file_bytes
from agentforge_sdk.models import Contract, RAGConfig, SDKError, fingerprint
from agentforge_sdk.multimodal import _REVIEW_CAPABILITY, MultimodalRAG, _seal
from agentforge_sdk.rag import EmbeddingBackend, EvidenceSelector
from agentforge_sdk.spatial import (
    Box,
    Calculation,
    MultimodalDocument,
    Region,
    SpatialAnswer,
    SpatialCitation,
    region_fingerprint,
)

_DIGEST = r"^[a-f0-9]{64}$"
_ACTOR = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$"


class ReviewError(SDKError):
    pass


class ReviewedRegionInput(Contract):
    """An explicit transcription/annotation, not a model confidence override."""

    kind: Literal["text", "table_cell"] = "text"
    text: str = Field(min_length=1, max_length=4000, repr=False)
    box: Box
    table_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
    row: int | None = Field(default=None, ge=0, le=999)
    column: int | None = Field(default=None, ge=0, le=999)

    @model_validator(mode="after")
    def region_contract(self):
        Region(region_id="0" * 64, **self.model_dump())
        return self


class PageCorrection(Contract):
    page: int = Field(ge=1, le=200)
    original_page_sha256: str = Field(pattern=_DIGEST)
    regions: tuple[ReviewedRegionInput, ...] = Field(min_length=1, max_length=1000, repr=False)
    # Required, with no default: the application must request a full-page review.
    attestation: Literal["complete_page_review"]

    @model_validator(mode="after")
    def unambiguous_grids(self):
        tables = defaultdict(list)
        for region in self.regions:
            if region.kind == "table_cell":
                tables[region.table_id].append(region)
        for cells in tables.values():
            rows, columns = max(c.row for c in cells) + 1, max(c.column for c in cells) + 1
            slots = {(c.row, c.column) for c in cells}
            if (rows < 2 or columns < 2 or len(slots) != len(cells)
                    or rows * columns != len(cells)):
                raise ValueError("reviewed grids require every address in a rectangular table")
            xs, ys = {}, {}
            for cell in cells:
                x, y = (cell.box.x0, cell.box.x1), (cell.box.y0, cell.box.y1)
                if xs.setdefault(cell.column, x) != x or ys.setdefault(cell.row, y) != y:
                    raise ValueError("reviewed table geometry must agree by row and column")
            if (any(xs[c][1] > xs[c + 1][0] for c in range(columns - 1))
                    or any(ys[r][1] > ys[r + 1][0] for r in range(rows - 1))):
                raise ValueError("reviewed table cells cannot overlap or reverse grid order")
        return self


class ReviewDraft(Contract):
    protocol: Literal["agentforge-evidence-review-draft-v1"] = "agentforge-evidence-review-draft-v1"
    original_document_fingerprint: str = Field(pattern=_DIGEST)
    file_sha256: str = Field(pattern=_DIGEST)
    corrections: tuple[PageCorrection, ...] = Field(min_length=1, max_length=200, repr=False)
    rationale: str = Field(min_length=1, max_length=500, repr=False)
    fingerprint: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def valid_draft(self):
        pages = [c.page for c in self.corrections]
        if pages != sorted(set(pages)):
            raise ValueError("reviewed pages must be unique and ordered")
        if not self.rationale.strip() or any(ord(c) < 32 for c in self.rationale):
            raise ValueError("review rationale must be meaningful single-line text")
        if self.fingerprint != fingerprint(self.model_dump(mode="json", exclude={"fingerprint"})):
            raise ValueError("review draft binding is invalid")
        return self


class ReviewApproval(Contract):
    protocol: Literal["agentforge-evidence-review-approval-v1"] = "agentforge-evidence-review-approval-v1"
    draft_fingerprint: str = Field(pattern=_DIGEST)
    reviewer_id: str = Field(pattern=_ACTOR)
    decision: Literal["approve", "reject"]
    issued_at: int = Field(ge=0, le=4_102_444_800)
    expires_at: int = Field(ge=1, le=4_102_444_800)
    signature: str = Field(pattern=_DIGEST, repr=False)
    fingerprint: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def valid_record(self):
        if not 0 < self.expires_at - self.issued_at <= 86_400:
            raise ValueError("review approval lifetime must be between 1 second and 1 day")
        if self.fingerprint != fingerprint(self.model_dump(mode="json", exclude={"fingerprint"})):
            raise ValueError("approval receipt binding is invalid")
        return self


def _signed_payload(approval: ReviewApproval) -> bytes:
    return fingerprint(approval.model_dump(mode="json", exclude={"signature", "fingerprint"})).encode("ascii")


class ReviewAuthority:
    """Trusted application capability, not an identity service or proof of human review.

    Possession permits issuing decisions. Never expose this object to an agent tool
    or unauthenticated endpoint. Revocations are local to this instance/process.
    """

    def __init__(self, reviewer_keys: Mapping[str, bytes], *, clock: Callable[[], int] | None = None):
        import re

        if (not isinstance(reviewer_keys, Mapping) or not 1 <= len(reviewer_keys) <= 100
                or any(not isinstance(name, str) or not re.fullmatch(_ACTOR, name)
                       or type(key) is not bytes or not 32 <= len(key) <= 64 for name, key in reviewer_keys.items())
                or (clock is not None and not callable(clock))):
            raise ValueError("review authority requires bounded reviewer IDs and 32-64-byte keys")
        self._keys = dict(reviewer_keys)
        self._clock = clock if clock is not None else (lambda: int(time.time()))
        self._revoked: set[str] = set()
        self._lock = RLock()
        self._last_time: int | None = None
        self._disabled = False

    def _now(self):
        try:
            now = self._clock()
        except Exception:
            raise ReviewError("review clock is unavailable") from None
        if type(now) is not int or not 0 <= now <= 4_102_444_800:
            raise ReviewError("review clock is outside the supported range")
        with self._lock:
            if self._last_time is not None and now < self._last_time:
                raise ReviewError("review clock moved backwards")
            self._last_time = now
        return now

    def decide(self, draft: ReviewDraft, *, reviewer_id: str, decision: Literal["approve", "reject"],
               lifetime_seconds: int = 3600) -> ReviewApproval:
        """Call only after the authenticated review action; this method does not perform review."""
        draft = ReviewDraft.model_validate(draft.model_dump())
        if (not isinstance(reviewer_id, str) or reviewer_id not in self._keys
                or not isinstance(decision, str) or decision not in {"approve", "reject"}
                or type(lifetime_seconds) is not int or not 1 <= lifetime_seconds <= 86_400):
            raise ReviewError("review decision is outside the configured authority")
        with self._lock:
            if self._disabled:
                raise ReviewError("review authority is disabled")
        now = self._now()
        payload = dict(draft_fingerprint=draft.fingerprint, reviewer_id=reviewer_id, decision=decision,
                       issued_at=now, expires_at=now + lifetime_seconds, signature="0" * 64)
        # Seal before model validation: these models intentionally reject partial receipts.
        payload["protocol"] = "agentforge-evidence-review-approval-v1"
        signature_payload = fingerprint({k: v for k, v in payload.items() if k != "signature"}).encode("ascii")
        payload["signature"] = hmac.new(self._keys[reviewer_id], signature_payload, hashlib.sha256).hexdigest()
        return ReviewApproval(**payload, fingerprint=fingerprint(payload))

    def revoke(self, approval: ReviewApproval) -> None:
        approval = ReviewApproval.model_validate(approval.model_dump())
        with self._lock:
            if len(self._revoked) >= 10_000 and approval.fingerprint not in self._revoked:
                self._disabled = True
                raise ReviewError("local revocation capacity reached; replace the authority")
            self._revoked.add(approval.fingerprint)

    def permits(self, approval: ReviewApproval, draft: ReviewDraft) -> bool:
        try:
            approval = ReviewApproval.model_validate(approval.model_dump())
            draft = ReviewDraft.model_validate(draft.model_dump())
            key = self._keys.get(approval.reviewer_id)
            now = self._now()
            with self._lock:
                revoked = self._disabled or approval.fingerprint in self._revoked
            return (key is not None and not revoked and approval.decision == "approve"
                    and approval.draft_fingerprint == draft.fingerprint
                    and approval.issued_at <= now < approval.expires_at
                    and hmac.compare_digest(approval.signature, hmac.new(key, _signed_payload(approval), hashlib.sha256).hexdigest()))
        except (ValueError, AttributeError, TypeError, ReviewError):
            return False


def _derive(document: MultimodalDocument, draft: ReviewDraft) -> MultimodalDocument:
    if document.fingerprint != draft.original_document_fingerprint or document.file_sha256 != draft.file_sha256:
        raise ReviewError("review draft belongs to another extraction")
    if any(p.method == "reviewed" for p in document.pages):
        raise ReviewError("reviews must target the original machine extraction")
    pages = list(document.pages)
    for correction in draft.corrections:
        if correction.page > len(pages):
            raise ReviewError("review page does not exist")
        page = pages[correction.page - 1]
        if fingerprint(page.model_dump(mode="json")) != correction.original_page_sha256:
            raise ReviewError("review page is stale")
        if page.reason in {"unsupported_geometry", "budget_exceeded"}:
            raise ReviewError("review cannot bypass unsupported geometry or resource limits")
        regions = []
        for supplied in correction.regions:
            region = Region(region_id="0" * 64, **supplied.model_dump())
            regions.append(region.model_copy(update={"region_id": region_fingerprint(document.file_sha256, page.page, region)}))
        pages[correction.page - 1] = page.model_copy(update={"method": "reviewed", "status": "ready",
                                                            "reason": "extracted", "regions": tuple(regions)})
    payload = document.model_dump(mode="json", exclude={"fingerprint"})
    # model_dump(mode='python') supplies strict tuples to the immutable contract.
    payload["pages"] = tuple(p.model_dump() for p in pages)
    payload["extractor"] += ";page-review-v1"
    return MultimodalDocument(**payload, fingerprint=fingerprint(payload))


def prepare_review(document: MultimodalDocument, corrections: Sequence[PageCorrection], *, rationale: str) -> ReviewDraft:
    document = MultimodalDocument.model_validate(document.model_dump())
    if not isinstance(corrections, (tuple, list)) or not 1 <= len(corrections) <= 200:
        raise ValueError("review requires bounded page corrections")
    payload = dict(protocol="agentforge-evidence-review-draft-v1", original_document_fingerprint=document.fingerprint,
                   file_sha256=document.file_sha256, corrections=tuple(c.model_dump() for c in corrections), rationale=rationale)
    draft = ReviewDraft(**payload, fingerprint=fingerprint(payload))
    _derive(document, draft)  # Reject impossible/budget-exceeding proposals before requesting approval.
    return draft


class ReviewedEvidence(Contract):
    protocol: Literal["agentforge-reviewed-evidence-v1"] = "agentforge-reviewed-evidence-v1"
    original: MultimodalDocument = Field(repr=False)
    draft: ReviewDraft = Field(repr=False)
    approval: ReviewApproval
    document: MultimodalDocument = Field(repr=False)
    fingerprint: str = Field(pattern=_DIGEST)


def apply_review(document: MultimodalDocument, draft: ReviewDraft, approval: ReviewApproval, *,
                 authority: ReviewAuthority) -> ReviewedEvidence:
    document = MultimodalDocument.model_validate(document.model_dump())
    draft = ReviewDraft.model_validate(draft.model_dump())
    approval = ReviewApproval.model_validate(approval.model_dump())
    if not authority.permits(approval, draft):
        raise ReviewError("review approval is invalid, rejected, expired or revoked")
    derived = _derive(document, draft)
    return _seal(ReviewedEvidence, dict(original=document, draft=draft, approval=approval, document=derived))


class ReviewedCitation(Contract):
    spatial: SpatialCitation
    origin: Literal["machine_extracted", "reviewer_supplied"]
    original_page_sha256: str = Field(pattern=_DIGEST)
    reviewer_id: str | None = None
    approval_fingerprint: str | None = None


class ReviewedAnswer(Contract):
    protocol: Literal["agentforge-reviewed-answer-v1"] = "agentforge-reviewed-answer-v1"
    bundle_fingerprint: str
    original_document_fingerprint: str
    approval_fingerprint: str
    result: SpatialAnswer = Field(repr=False)
    citations: tuple[ReviewedCitation, ...] = Field(max_length=60)
    fingerprint: str


class ReviewedCalculation(Contract):
    protocol: Literal["agentforge-reviewed-calculation-v1"] = "agentforge-reviewed-calculation-v1"
    bundle_fingerprint: str
    original_document_fingerprint: str
    approval_fingerprint: str
    result: Calculation
    citations: tuple[ReviewedCitation, ...] = Field(max_length=32)
    fingerprint: str


class ReviewedRAG:
    """Source-checked reviewed corpus with live in-process expiry/revocation checks."""

    def __init__(self, evidence: ReviewedEvidence, *, authority: ReviewAuthority, source_path: Path,
                 config: RAGConfig | None = None, embedder: EmbeddingBackend | None = None,
                 selector: EvidenceSelector | None = None):
        evidence = ReviewedEvidence.model_validate(evidence.model_dump())
        expected = apply_review(evidence.original, evidence.draft, evidence.approval, authority=authority)
        if expected != evidence:
            raise ReviewError("reviewed bundle does not match the approved transformation")
        self._evidence, self._authority, self._source_path = evidence, authority, Path(source_path)
        self._reviewed_pages = {p.page for p in evidence.draft.corrections}
        self._page_hashes = {p.page: fingerprint(p.model_dump(mode="json")) for p in evidence.original.pages}
        self._check()
        self._rag = MultimodalRAG(evidence.document, config=config, embedder=embedder, selector=selector,
                                  _review_capability=_REVIEW_CAPABILITY)

    @property
    def evidence(self) -> ReviewedEvidence:
        return self._evidence

    def _check(self):
        evidence = self._evidence
        if not self._authority.permits(evidence.approval, evidence.draft):
            raise ReviewError("review approval is invalid, rejected, expired or revoked")
        try:
            raw = _file_bytes(self._source_path, evidence.original.config.max_file_bytes)
            if hashlib.sha256(raw).hexdigest() != evidence.original.file_sha256:
                raise ValueError("source changed")
        except (OSError, ValueError, SDKError):
            raise ReviewError("reviewed source is unavailable, changed or outside its bounds") from None

    def is_authorized(self) -> bool:
        """Check current local approval/source state without calling a text backend."""
        try:
            self._check()
            return True
        except ReviewError:
            return False

    def _citations(self, citations):
        return tuple(ReviewedCitation(spatial=c, original_page_sha256=self._page_hashes[c.page],
                                     origin="reviewer_supplied" if c.page in self._reviewed_pages else "machine_extracted",
                                     reviewer_id=self._evidence.approval.reviewer_id if c.page in self._reviewed_pages else None,
                                     approval_fingerprint=self._evidence.approval.fingerprint if c.page in self._reviewed_pages else None)
                     for c in citations)

    def _wrap(self, model, result):
        return _seal(model, dict(bundle_fingerprint=self._evidence.fingerprint,
                                 original_document_fingerprint=self._evidence.original.fingerprint,
                                 approval_fingerprint=self._evidence.approval.fingerprint,
                                 result=result, citations=self._citations(result.citations)))

    def citation(self, region_id: str, *, quote: str | None = None) -> ReviewedCitation:
        """Expose a literal region citation without dropping its current review lineage."""
        self._check()
        citation = self._citations((self._rag.citation(region_id, quote=quote),))[0]
        self._check()
        return citation

    def answer(self, question: str, *, top_k: int = 3) -> ReviewedAnswer:
        self._check()
        result = self._rag.answer(question, top_k=top_k)
        self._check()  # A custom text backend could outlive/revoke approval during its call.
        return self._wrap(ReviewedAnswer, result)

    def calculate(self, operation: str, region_ids: Sequence[str]) -> ReviewedCalculation:
        self._check()
        result = self._rag.calculate(operation, region_ids)
        self._check()
        return self._wrap(ReviewedCalculation, result)

    def verify_citation(self, citation: ReviewedCitation) -> bool:
        try:
            self._check()
            citation = ReviewedCitation.model_validate(citation.model_dump())
            return self._rag.verify_citation(citation.spatial) and self._citations((citation.spatial,))[0] == citation
        except (ValueError, KeyError, AttributeError, TypeError, ReviewError):
            return False

    def verify_answer(self, answer: ReviewedAnswer, *, question: str | None = None) -> bool:
        try:
            self._check()
            answer = ReviewedAnswer.model_validate(answer.model_dump())
            valid = self._rag.verify_answer(answer.result, question=question) and self._wrap(ReviewedAnswer, answer.result) == answer
            self._check()
            return valid
        except (ValueError, KeyError, AttributeError, TypeError, ReviewError):
            return False

    def verify_calculation(self, calculation: ReviewedCalculation) -> bool:
        try:
            self._check()
            calculation = ReviewedCalculation.model_validate(calculation.model_dump())
            valid = (self._rag.verify_calculation(calculation.result)
                     and self._wrap(ReviewedCalculation, calculation.result) == calculation)
            self._check()
            return valid
        except (ValueError, AttributeError, TypeError, ReviewError):
            return False
