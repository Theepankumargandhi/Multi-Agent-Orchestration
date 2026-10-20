"""Bounded, deterministic extraction diffs and receipt impact; no semantic-truth claims."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from decimal import Decimal
from typing import Literal

from pydantic import Field

from agentforge_sdk._numbers import NUMBER as _NUMBER
from agentforge_sdk.evidence_review import ReviewedAnswer, ReviewedCalculation, ReviewedRAG
from agentforge_sdk.models import Contract, SDKError, fingerprint
from agentforge_sdk.multimodal import MultimodalRAG, _seal
from agentforge_sdk.spatial import Box, Calculation, EvidencePage, MultimodalDocument, Region, SpatialAnswer

_HASH = r"^[a-f0-9]{64}$"
ChangeFlag = Literal["text", "numeric", "unit", "geometry", "address", "page", "method", "confidence"]
Reader = MultimodalRAG | ReviewedRAG
Receipt = SpatialAnswer | Calculation | ReviewedAnswer | ReviewedCalculation


class ChangeConfig(Contract):
    min_line_overlap: float = Field(default=0.6, ge=0.5, le=1)
    max_spatial_comparisons: int = Field(default=100_000, ge=1, le=1_000_000)


class RegionReference(Contract):
    region_id: str = Field(pattern=_HASH)
    page: int = Field(ge=1, le=200)
    kind: Literal["text", "table_cell"]
    box: Box
    text_sha256: str = Field(pattern=_HASH)


class RegionChange(Contract):
    before: RegionReference | None
    after: RegionReference | None
    relation: Literal["unchanged", "relocated", "modified", "added", "removed", "ambiguous", "unmatched"]
    basis: Literal["same_manifest", "literal", "table_labels", "spatial", "unpaired_page", "unpaired_table", "none", "budget"]
    flags: tuple[ChangeFlag, ...] = ()


class PageAlignment(Contract):
    before_page: int | None
    after_page: int | None
    basis: Literal["same_manifest", "unique_content", "unique_anchor", "unmatched"]


class DocumentChanges(Contract):
    protocol: Literal["agentforge-document-changes-v1"] = "agentforge-document-changes-v1"
    before_fingerprint: str = Field(pattern=_HASH)
    after_fingerprint: str = Field(pattern=_HASH)
    config: ChangeConfig
    status: Literal["complete", "held"]
    reason: Literal["compared", "incomplete_extraction", "comparison_budget"]
    metadata_changes: tuple[str, ...]
    pages: tuple[PageAlignment, ...] = Field(max_length=400)
    regions: tuple[RegionChange, ...] = Field(max_length=2000)
    spatial_comparisons: int = Field(ge=0, le=1_000_000)
    fingerprint: str = Field(pattern=_HASH)


class ReceiptImpact(Contract):
    protocol: Literal["agentforge-receipt-impact-v1"] = "agentforge-receipt-impact-v1"
    changes_fingerprint: str
    source_receipt_fingerprint: str = Field(pattern=_HASH)
    before_context_fingerprint: str
    after_context_fingerprint: str
    receipt_kind: Literal["answer", "calculation"]
    status: Literal["dependencies_unchanged", "dependencies_changed", "needs_review", "invalid_receipt"]
    reasons: tuple[str, ...]
    dependencies: tuple[RegionChange, ...] = Field(max_length=180)
    requires_reissue: bool
    fingerprint: str


class CalculationReplay(Contract):
    protocol: Literal["agentforge-calculation-replay-v1"] = "agentforge-calculation-replay-v1"
    impact: ReceiptImpact
    status: Literal["recalculated", "abstained"]
    reason: Literal["tool_replayed", "dependency_alignment_unavailable", "source_receipt_unverifiable",
                    "target_incomplete", "extraction_profile_changed", "target_not_authorized", "target_tool_abstained"]
    result: Calculation | ReviewedCalculation | None = Field(default=None, repr=False)
    value_changed: bool | None = None
    fingerprint: str


def _ref(page: EvidencePage, region: Region):
    return RegionReference(region_id=region.region_id, page=page.page, kind=region.kind,
                           box=region.box, text_sha256=hashlib.sha256(region.text.encode("utf-8")).hexdigest())


def _page_key(page):
    return fingerprint(sorted((r.kind, r.text, r.row, r.column) for r in page.regions)) if page.regions else None


def _anchors(page):
    return {r.text for r in page.regions if r.kind == "text" or r.row == 0 or r.column == 0}


def _unique_pairs(left, right, key: Callable):
    a, b = defaultdict(list), defaultdict(list)
    for item in left:
        a[key(item)].append(item)
    for item in right:
        b[key(item)].append(item)
    return [(items[0], b[k][0]) for k, items in a.items() if k is not None and len(items) == len(b[k]) == 1]


def _align_pages(before, after):
    pairs = [(a, b, "same_manifest") for a, b in zip(before.pages, after.pages, strict=True)] if before.fingerprint == after.fingerprint else []
    if not pairs:
        exact = _unique_pairs(before.pages, after.pages, _page_key)
        pairs = [(a, b, "unique_content") for a, b in exact]
        used_a, used_b = {a.page for a, _ in exact}, {b.page for _, b in exact}
        left = [p for p in before.pages if p.page not in used_a and p.status == "ready"]
        right = [p for p in after.pages if p.page not in used_b and p.status == "ready"]
        # Only anchors occurring on ONE page of each entire document can establish identity.
        counts_a = Counter(t for p in before.pages for t in _anchors(p))
        counts_b = Counter(t for p in after.pages for t in _anchors(p))
        anchors_a = {p.page: {t for t in _anchors(p) if counts_a[t] == counts_b[t] == 1} for p in left}
        anchors_b = {p.page: _anchors(p) for p in right}
        candidates = {(a.page, b.page): len(anchors_a[a.page] & anchors_b[b.page]) for a in left for b in right}
        for a in left:
            ranked = sorted(((score, j) for (i, j), score in candidates.items() if i == a.page and score), reverse=True)
            if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
                continue
            score, page = ranked[0]
            reverse = [(s, i) for (i, j), s in candidates.items() if j == page and s]
            if sum(s >= score for s, _ in reverse) == 1:
                pairs.append((a, next(b for b in right if b.page == page), "unique_anchor"))
    return sorted(pairs, key=lambda pair: pair[0].page)


def _unit(match):
    return match[1] or {"$": "USD", "€": "EUR", "£": "GBP"}.get(match[2]) or match[4] or "scalar"


def _flags(a_page, a, b_page, b):
    flags = []
    if a.text != b.text:
        flags.append("text")
        x, y = _NUMBER.fullmatch(a.text), _NUMBER.fullmatch(b.text)
        if x and y:
            if Decimal(x[3]) != Decimal(y[3]):
                flags.append("numeric")
            if _unit(x) != _unit(y):
                flags.append("unit")
    if a.box != b.box or (a_page.width, a_page.height) != (b_page.width, b_page.height):
        flags.append("geometry")
    if (a.table_id, a.row, a.column) != (b.table_id, b.row, b.column):
        flags.append("address")
    if a_page.page != b_page.page:
        flags.append("page")
    if a_page.method != b_page.method:
        flags.append("method")
    if a.confidence != b.confidence:
        flags.append("confidence")
    return tuple(sorted(flags))


def _change(ap, a, bp, b, basis, *, relation=None):
    flags = _flags(ap, a, bp, b) if a is not None and b is not None else ()
    relation = relation or ("modified" if any(f in flags for f in ("text", "method", "confidence"))
                            else "relocated" if flags else "unchanged")
    return RegionChange(before=_ref(ap, a) if a is not None else None, after=_ref(bp, b) if b is not None else None,
                        relation=relation, basis=basis, flags=flags)


def _tables(page):
    tables = defaultdict(list)
    for region in page.regions:
        if region.kind == "table_cell":
            tables[region.table_id].append(region)
    return list(tables.values())


def _table_key(cells):
    headers = [r.text for r in cells if r.row == 0]
    # Duplicate headers/row labels have no automatic semantic alignment.
    rows = [r.text for r in cells if r.column == 0]
    if not headers or len(headers) != len(set(headers)) or len(rows) != len(set(rows)):
        return None
    return tuple(sorted(headers))


def _cell_keys(cells):
    headers = {c.column: c.text for c in cells if c.row == 0}
    labels = {c.row: c.text for c in cells if c.column == 0}
    return {(labels[c.row], headers[c.column]): c for c in cells if c.row in labels and c.column in headers}


def _overlap(a, b):
    intersection = max(0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
    union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - intersection
    return intersection / union


def compare_documents(before: MultimodalDocument, after: MultimodalDocument, *, config: ChangeConfig | None = None) -> DocumentChanges:
    """Compare extraction manifests, not unseen pixels or semantic truth. Does not authorize reuse."""
    before, after = (MultimodalDocument.model_validate(d.model_dump()) for d in (before, after))
    if before.document_id != after.document_id:
        raise ValueError("version comparison requires the same explicit logical document ID")
    config = ChangeConfig.model_validate((config or ChangeConfig()).model_dump())
    paired = _align_pages(before, after)
    changes, alignments, comparisons, exhausted = [], [], 0, False
    for ap, bp, basis in paired:
        alignments.append(PageAlignment(before_page=ap.page, after_page=bp.page, basis=basis))
        if basis == "same_manifest":
            right = {r.region_id: r for r in bp.regions}
            changes.extend(_change(ap, r, bp, right[r.region_id], "same_manifest") for r in ap.regions)
            continue
        table_pairs = _unique_pairs(_tables(ap), _tables(bp), _table_key)
        used_a, used_b = set(), set()
        for a_cells, b_cells in table_pairs:
            a_keys, b_keys = _cell_keys(a_cells), _cell_keys(b_cells)
            for key in sorted(a_keys.keys() & b_keys.keys()):
                a, b = a_keys[key], b_keys[key]
                changes.append(_change(ap, a, bp, b, "table_labels"))
                used_a.add(a.region_id)
                used_b.add(b.region_id)
        a_text = [r for r in ap.regions if r.kind == "text"]
        b_text = [r for r in bp.regions if r.kind == "text"]
        for a, b in _unique_pairs(a_text, b_text, lambda r: r.text):
            changes.append(_change(ap, a, bp, b, "literal"))
            used_a.add(a.region_id)
            used_b.add(b.region_id)
        left, right = [r for r in a_text if r.region_id not in used_a], [r for r in b_text if r.region_id not in used_b]
        # Duplicate text is not disambiguated by arbitrary order or proximity.
        counts_a, counts_b = Counter(r.text for r in a_text), Counter(r.text for r in b_text)
        candidates = {}
        for a in left:
            for b in right:
                if comparisons == config.max_spatial_comparisons:
                    exhausted = True
                    break
                comparisons += 1
                if counts_a[a.text] == counts_b[b.text] == 1:
                    score = _overlap(a.box, b.box)
                    if score >= config.min_line_overlap:
                        candidates[a.region_id, b.region_id] = score
            if exhausted:
                break
        if not exhausted:
            for a in left:
                choices = sorted(((s, j) for (i, j), s in candidates.items() if i == a.region_id), reverse=True)
                if choices and (len(choices) == 1 or choices[0][0] > choices[1][0]):
                    score, target = choices[0]
                    if sum(s >= score for (i, j), s in candidates.items() if j == target) == 1:
                        b = next(r for r in right if r.region_id == target)
                        changes.append(_change(ap, a, bp, b, "spatial"))
                        used_a.add(a.region_id)
                        used_b.add(b.region_id)
        for a in ap.regions:
            if a.region_id not in used_a:
                ambiguous = a.kind == "table_cell" or counts_a[a.text] > 1 or counts_b[a.text] > 1
                changes.append(_change(ap, a, bp, None, "budget" if exhausted else "unpaired_table" if a.kind == "table_cell" else "none",
                                       relation="ambiguous" if ambiguous else "unmatched" if exhausted else "removed"))
        for b in bp.regions:
            if b.region_id not in used_b:
                changes.append(_change(ap, None, bp, b, "budget" if exhausted else "unpaired_table" if b.kind == "table_cell" else "none",
                                       relation="unmatched" if exhausted or b.kind == "table_cell" else "added"))
    used_pages_a, used_pages_b = {a.page for a, _, _ in paired}, {b.page for _, b, _ in paired}
    for doc, used, old in ((before, used_pages_a, True), (after, used_pages_b, False)):
        for page in doc.pages:
            if page.page not in used:
                alignments.append(PageAlignment(before_page=page.page if old else None, after_page=None if old else page.page, basis="unmatched"))
                changes.extend(_change(page, r if old else None, page, None if old else r, "unpaired_page", relation="unmatched") for r in page.regions)
    metadata = tuple(name for name in ("source", "version", "file_sha256", "media_type", "config", "extractor", "ocr_backend") if getattr(before, name) != getattr(after, name))
    return _seal(DocumentChanges, dict(before_fingerprint=before.fingerprint, after_fingerprint=after.fingerprint,
                                       config=config, status="held" if exhausted or not before.ready or not after.ready else "complete",
                                       reason="comparison_budget" if exhausted else "incomplete_extraction" if not before.ready or not after.ready else "compared",
                                       metadata_changes=metadata, pages=tuple(alignments), regions=tuple(changes), spatial_comparisons=comparisons))


def _document(reader: Reader):
    if isinstance(reader, ReviewedRAG):
        return reader.evidence.document
    if isinstance(reader, MultimodalRAG):
        return reader.document
    raise ValueError("revision impact requires SDK evidence readers")


def _context(reader):
    return reader.evidence.fingerprint if isinstance(reader, ReviewedRAG) else reader.document.fingerprint


def _inner(receipt):
    return receipt.result if isinstance(receipt, (ReviewedAnswer, ReviewedCalculation)) else receipt


class DocumentRevision:
    """Assess verified old dependencies; never migrate answers or review approvals automatically."""

    def __init__(self, before: Reader, after: Reader, *, config: ChangeConfig | None = None):
        self._before, self._after = before, after
        self._old_doc, self._new_doc = _document(before), _document(after)
        self._changes = compare_documents(self._old_doc, self._new_doc, config=config)
        self._by_id = {c.before.region_id: c for c in self._changes.regions if c.before is not None}
        self._old_regions = {r.region_id: (p, r) for p in self._old_doc.pages for r in p.regions}
        self._old_grid = {(p.page, r.table_id, r.row, r.column): r for p, r in self._old_regions.values() if r.kind == "table_cell"}

    @property
    def changes(self) -> DocumentChanges:
        return self._changes

    def verify_changes(self, changes: DocumentChanges) -> bool:
        try:
            return DocumentChanges.model_validate(changes.model_dump()) == self._changes
        except (ValueError, TypeError, AttributeError):
            return False

    def _assess(self, receipt: Receipt, *, answer: bool, question=None):
        expected = (ReviewedAnswer if answer else ReviewedCalculation) if isinstance(self._before, ReviewedRAG) else (SpatialAnswer if answer else Calculation)
        valid = isinstance(receipt, expected)
        if valid:
            valid = self._before.verify_answer(receipt, question=question) if answer else self._before.verify_calculation(receipt)
        inner = _inner(receipt)
        dependencies, reasons = [], []
        if not valid:
            reasons, status = ["source_receipt_unverifiable"], "invalid_receipt"
        else:
            ids = set(c.region_id for c in inner.citations)
            # Numeric tools cite operands, not axes: include labels to detect changed meaning.
            for rid in tuple(ids):
                page, region = self._old_regions[rid]
                if region.kind == "table_cell":
                    for row, column in ((region.row, 0), (0, region.column)):
                        label = self._old_grid.get((page.page, region.table_id, row, column))
                        if label is not None:
                            ids.add(label.region_id)
            dependencies = [self._by_id[rid] for rid in sorted(ids)]
            if not dependencies:
                reasons.append("no_positive_dependencies")
            if self._changes.status == "held":
                reasons.append(self._changes.reason)
            if isinstance(self._after, ReviewedRAG) and not self._after.is_authorized():
                reasons.append("target_not_authorized")
            if any(c.after is None for c in dependencies):
                reasons.append("dependency_alignment_unavailable")
            if any(k in self._changes.metadata_changes for k in ("config", "extractor", "ocr_backend", "media_type")):
                reasons.append("extraction_profile_changed")
            if reasons:
                status = "needs_review"
            elif any(any(f in c.flags for f in ("text", "numeric", "unit", "method", "confidence")) for c in dependencies):
                status, reasons = "dependencies_changed", ["cited_evidence_changed"]
            else:
                status, reasons = "dependencies_unchanged", ["cited_content_unchanged"]
            if answer and any(c.relation not in {"unchanged", "relocated"} for c in self._changes.regions):
                reasons.append("rerun_retrieval_for_changed_corpus")
        receipt_hash = getattr(receipt, "fingerprint", None)
        if not isinstance(receipt_hash, str) or not re.fullmatch(_HASH, receipt_hash):
            receipt_hash = "0" * 64
        return _seal(ReceiptImpact, dict(changes_fingerprint=self._changes.fingerprint,
                                         source_receipt_fingerprint=receipt_hash,
                                         before_context_fingerprint=_context(self._before), after_context_fingerprint=_context(self._after),
                                         receipt_kind="answer" if answer else "calculation", status=status, reasons=tuple(reasons),
                                         dependencies=tuple(dependencies), requires_reissue=status in {"needs_review", "invalid_receipt"}
                                         or _context(self._before) != _context(self._after)))

    def assess_answer(self, answer: SpatialAnswer | ReviewedAnswer, *, question: str | None = None) -> ReceiptImpact:
        return self._assess(answer, answer=True, question=question)

    def assess_calculation(self, calculation: Calculation | ReviewedCalculation) -> ReceiptImpact:
        return self._assess(calculation, answer=False)

    def recalculate(self, calculation: Calculation | ReviewedCalculation) -> CalculationReplay:
        impact = self.assess_calculation(calculation)
        payload = dict(impact=impact, status="abstained", reason="dependency_alignment_unavailable")
        if impact.status == "invalid_receipt":
            return _seal(CalculationReplay, {**payload, "reason": "source_receipt_unverifiable"})
        if not self._new_doc.ready:
            return _seal(CalculationReplay, {**payload, "reason": "target_incomplete"})
        if "target_not_authorized" in impact.reasons:
            return _seal(CalculationReplay, {**payload, "reason": "target_not_authorized"})
        if "extraction_profile_changed" in impact.reasons:
            return _seal(CalculationReplay, {**payload, "reason": "extraction_profile_changed"})
        if impact.status == "needs_review":
            return _seal(CalculationReplay, payload)
        inner = _inner(calculation)
        ids = tuple(self._by_id[rid].after.region_id for rid in inner.region_ids)
        try:
            result = self._after.calculate(inner.operation, ids)
            if not self._after.verify_calculation(result):
                raise SDKError("target receipt is not currently authorized")
        except SDKError:
            return _seal(CalculationReplay, {**payload, "reason": "target_not_authorized"})
        new = _inner(result)
        return _seal(CalculationReplay, {**payload, "status": "recalculated" if new.status == "calculated" else "abstained",
                                         "reason": "tool_replayed" if new.status == "calculated" else "target_tool_abstained",
                                         "result": result, "value_changed": (inner.value, inner.unit) != (new.value, new.unit) if new.status == "calculated" else None})

    def verify_impact(self, impact: ReceiptImpact, receipt: Receipt, *, question=None) -> bool:
        try:
            impact = ReceiptImpact.model_validate(impact.model_dump())
            expected = self.assess_answer(receipt, question=question) if impact.receipt_kind == "answer" else self.assess_calculation(receipt)
            return impact == expected
        except (ValueError, TypeError, AttributeError, KeyError):
            return False

    def verify_recalculation(self, replay: CalculationReplay, source: Calculation | ReviewedCalculation) -> bool:
        try:
            return CalculationReplay.model_validate(replay.model_dump()) == self.recalculate(source)
        except (ValueError, TypeError, AttributeError, KeyError):
            return False
