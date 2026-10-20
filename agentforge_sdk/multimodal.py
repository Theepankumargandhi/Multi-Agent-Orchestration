"""Spatial retrieval and deterministic table arithmetic over one frozen document."""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

from agentforge_sdk._numbers import calculate_numbers, parse_number
from agentforge_sdk.models import DocumentPage, RAGConfig, fingerprint
from agentforge_sdk.rag import DocumentRAG, EmbeddingBackend, EvidenceSelector, _normalize_question
from agentforge_sdk.spatial import Calculation, MultimodalDocument, SpatialAnswer, SpatialCitation

_ABSTAIN = "No validated spatial evidence answer is available."
_REVIEW_CAPABILITY = object()


def _seal(model, payload):
    # Include protocol defaults in the hashed representation.
    item = model(**payload, fingerprint="0" * 64)
    return item.model_copy(update={"fingerprint": fingerprint(item.model_dump(mode="json", exclude={"fingerprint"}))})


class MultimodalRAG:
    """Reuse the text retrieval core, but never treat normalized chunk offsets as geometry.

    Each text page is an internal region-keyed record. Public citations are mapped
    back to the immutable extraction manifest and refer to whole-region boxes.
    Any incomplete physical page disables all answers and calculations by default.
    """

    def __init__(self, document: MultimodalDocument, *, config: RAGConfig | None = None,
                 embedder: EmbeddingBackend | None = None, selector: EvidenceSelector | None = None,
                 _review_capability: object | None = None):
        self._document = MultimodalDocument.model_validate(document.model_dump())
        if any(p.method == "reviewed" for p in self._document.pages) and _review_capability is not _REVIEW_CAPABILITY:
            raise ValueError("reviewed evidence must be used through ReviewedRAG with an authorized bundle")
        self._regions = {r.region_id: (p.page, r) for p in self._document.pages for r in p.regions}
        # 4000-char regions may each be kept whole. DocumentRAG still enforces caller budgets.
        config = config or RAGConfig(chunk_size=4000, chunk_overlap=0, max_pages=1000, max_chunks=1000)
        self._cell_context = {}
        grid = {(page, r.table_id, r.row, r.column): r for page, r in self._regions.values()
                if r.kind == "table_cell"}
        pages = []
        if document.ready:
            for rid, (page, region) in self._regions.items():
                text = region.text
                if region.kind == "table_cell" and region.row and region.column:
                    labels = [grid.get((page, region.table_id, region.row, 0)),
                              grid.get((page, region.table_id, 0, region.column))]
                    if all(label is not None and len(label.text) <= config.max_quote_chars for label in labels):
                        contextual = " | ".join([label.text for label in labels] + [text])
                        if len(text) <= config.max_quote_chars and len(contextual) <= config.chunk_size:
                            # First row/column are retrieval hints, never inferred factual labels.
                            # Public output cites each original cell separately, not this joined text.
                            self._cell_context[rid] = tuple(label.region_id for label in labels)
                            text = contextual
                pages.append(DocumentPage(document_id="r:" + rid, source=document.source, version=document.version,
                                          page=page, text=text))
        self._rag = DocumentRAG(pages, config=config, embedder=embedder, selector=selector)

    @property
    def document(self) -> MultimodalDocument:
        return self._document

    def citation(self, region_id: str, *, quote: str | None = None) -> SpatialCitation:
        """Get a whole-region or literal-substring citation from this ready document."""
        if not self._document.ready:
            raise ValueError("document extraction is incomplete")
        page, region = self._regions[region_id]
        quote = region.text if quote is None else quote
        if not isinstance(quote, str) or not quote or (start := region.text.find(quote)) < 0:
            raise ValueError("citation quote must occur in its region")
        doc = self._document
        return SpatialCitation(document_fingerprint=doc.fingerprint, file_sha256=doc.file_sha256,
                               document_id=doc.document_id, source=doc.source, version=doc.version,
                               page=page, region_id=region_id, box=region.box, quote=quote,
                               region_start=start, region_end=start + len(quote),
                               table_id=region.table_id, row=region.row, column=region.column)

    def verify_citation(self, citation: SpatialCitation) -> bool:
        """Verify geometry/text lineage, NOT OCR correctness, source authenticity or truth."""
        try:
            item = SpatialCitation.model_validate(citation.model_dump())
            return self.citation(item.region_id, quote=item.quote) == item
        except (KeyError, ValueError, AttributeError, TypeError):
            return False

    def _citations(self, receipt):
        citations, seen = [], set()
        for item in receipt.citations:
            region_id = item.document_id[2:]
            ids = (*self._cell_context[region_id], region_id) if region_id in self._cell_context else (region_id,)
            for rid in ids:
                cite = self.citation(rid, quote=None if region_id in self._cell_context else item.quote)
                key = (cite.region_id, cite.quote)
                if key not in seen:
                    citations.append(cite)
                    seen.add(key)
        return tuple(citations)

    @staticmethod
    def _answer_text(citations):
        return "\n".join(f'"{cite.quote}" [{i}]' for i, cite in enumerate(citations, 1)) if citations else _ABSTAIN

    def answer(self, question: str, *, top_k: int = 3) -> SpatialAnswer:
        clean = _normalize_question(question)
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer between 1 and 20")
        payload = dict(document_fingerprint=self._document.fingerprint,
                       query_sha256=hashlib.sha256(clean.encode()).hexdigest(),
                       status="abstained", reason="incomplete_extraction", answer=_ABSTAIN, citations=())
        if self._document.ready:
            result = self._rag.answer(clean, top_k=top_k)
            citations = self._citations(result)
            payload.update(status="evidence_found" if citations else "abstained",
                           reason="literal_evidence" if citations else "no_evidence",
                           answer=self._answer_text(citations),
                           citations=citations, text_receipt=result)
        return _seal(SpatialAnswer, payload)

    def verify_answer(self, answer: SpatialAnswer, *, question: str | None = None) -> bool:
        try:
            answer = SpatialAnswer.model_validate(answer.model_dump())
            if (answer.document_fingerprint != self._document.fingerprint
                    or answer.fingerprint != fingerprint(answer.model_dump(mode="json", exclude={"fingerprint"}))
                    or (question is not None and answer.query_sha256 != hashlib.sha256(_normalize_question(question).encode()).hexdigest())):
                return False
            if not self._document.ready:
                return (answer.status == "abstained" and answer.reason == "incomplete_extraction"
                        and not answer.citations and answer.text_receipt is None and answer.answer == _ABSTAIN)
            receipt = answer.text_receipt
            if (receipt is None or not self._rag.verify_answer(receipt, question=question)
                    or receipt.retrieval.query_sha256 != answer.query_sha256):
                return False
            citations = self._citations(receipt)
            return (citations == answer.citations and answer.status == receipt.status
                    and answer.reason == ("literal_evidence" if citations else "no_evidence")
                    and answer.answer == self._answer_text(citations))
        except (KeyError, ValueError, AttributeError, TypeError):
            return False

    def calculate(self, operation: str, region_ids: Sequence[str]) -> Calculation:
        """Compute explicitly selected native/reviewed cells; never eval model expressions.

        Decimal arithmetic, 50-digit precision, ROUND_HALF_EVEN. Percent change is
        (new - old) / abs(old) * 100, with region IDs ordered [old, new].
        Units must match; dates, commas, locale guessing and OCR numbers are unsupported.
        """
        if operation not in {"sum", "difference", "ratio", "percent_change"}:
            raise ValueError("unsupported cell calculation operation")
        if (not isinstance(region_ids, (tuple, list)) or not 1 <= len(region_ids) <= 32
                or any(not isinstance(r, str) or not re.fullmatch(r"[a-f0-9]{64}", r) for r in region_ids)
                or len(set(region_ids)) != len(region_ids)
                or (operation != "sum" and len(region_ids) != 2)):
            raise ValueError("calculation requires bounded, unique cell IDs and valid arity")
        payload = dict(document_fingerprint=self._document.fingerprint, operation=operation,
                       region_ids=tuple(region_ids), status="abstained", reason="incomplete_extraction", citations=())
        if not self._document.ready:
            return _seal(Calculation, payload)
        try:
            cells = [self._regions[r][1] for r in region_ids]
        except KeyError:
            return _seal(Calculation, {**payload, "reason": "invalid_cells"})
        if any(c.kind != "table_cell" or c.confidence is not None for c in cells):
            return _seal(Calculation, {**payload, "reason": "invalid_cells"})
        parsed = [parse_number(c.text) for c in cells]
        if any(m is None for m in parsed):
            return _seal(Calculation, {**payload, "reason": "ambiguous_number"})
        units = [m[1] for m in parsed]
        if len(set(units)) != 1:
            return _seal(Calculation, {**payload, "reason": "ambiguous_number"})
        values = [m[0] for m in parsed]
        if ((operation == "ratio" and values[1] == 0) or (operation == "percent_change" and values[0] == 0)):
            return _seal(Calculation, {**payload, "reason": "zero_denominator"})
        formatted, unit = calculate_numbers(operation, values, units[0])
        return _seal(Calculation, {**payload, "status": "calculated", "reason": "verified_cells",
                                  "value": formatted, "unit": unit,
                                  "citations": tuple(self.citation(r) for r in region_ids)})

    def verify_calculation(self, calculation: Calculation) -> bool:
        """Re-run the bounded tool against this corpus, including abstention receipts."""
        try:
            item = Calculation.model_validate(calculation.model_dump())
            return item == self.calculate(item.operation, item.region_ids)
        except (ValueError, TypeError, AttributeError):
            return False
