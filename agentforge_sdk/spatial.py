"""Immutable spatial evidence contracts. Coordinates are normalized, top-left origin."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from agentforge_sdk.models import AnswerResult, Contract, DocumentPage, fingerprint


class Box(Contract):
    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def positive_area(self):
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError("region must have positive area")
        return self


class ExtractionConfig(Contract):
    max_file_bytes: int = Field(default=10_000_000, ge=1000, le=20_000_000)
    max_pages: int = Field(default=50, ge=1, le=200)
    max_pixels: int = Field(default=12_000_000, ge=10_000, le=25_000_000)
    max_regions: int = Field(default=1000, ge=1, le=1000)
    max_text_chars: int = Field(default=500_000, ge=1000, le=2_000_000)
    render_dpi: int = Field(default=150, ge=72, le=300)
    min_ocr_confidence: float = Field(default=0.6, ge=0, le=1)


class Region(Contract):
    region_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    kind: Literal["text", "table_cell"]
    box: Box
    text: str = Field(min_length=1, max_length=4000, repr=False)
    confidence: float | None = Field(default=None, ge=0, le=1)
    table_id: str | None = None
    row: int | None = Field(default=None, ge=0, le=999)
    column: int | None = Field(default=None, ge=0, le=999)

    @model_validator(mode="after")
    def canonical_text_and_cell(self):
        if self.text != " ".join(self.text.split()) or "\x00" in self.text:
            raise ValueError("region text must be canonical nonempty text")
        cell = self.table_id is not None and self.row is not None and self.column is not None
        if (self.kind == "table_cell") != cell or (
            self.kind == "text" and any(v is not None for v in (self.table_id, self.row, self.column))
        ):
            raise ValueError("table cells require a complete grid address")
        return self


class EvidencePage(Contract):
    page: int = Field(ge=1, le=200)
    width: float = Field(gt=0, le=100_000)
    height: float = Field(gt=0, le=100_000)
    method: Literal["native", "ocr", "reviewed", "none"]
    status: Literal["ready", "held"]
    reason: Literal[
        "extracted", "ocr_required", "ocr_failed", "low_confidence", "empty_page",
        "unsupported_geometry", "ambiguous_table", "budget_exceeded",
    ]
    regions: tuple[Region, ...] = Field(max_length=1000)

    @model_validator(mode="after")
    def consistent_status(self):
        if self.status == "ready":
            if not self.regions or self.reason != "extracted" or self.method == "none":
                raise ValueError("ready pages require extracted regions")
        elif self.regions or self.reason == "extracted":
            raise ValueError("held pages cannot expose answerable regions")
        return self


class MultimodalDocument(Contract):
    protocol: Literal["agentforge-spatial-evidence-v1"] = "agentforge-spatial-evidence-v1"
    document_id: str
    source: str
    version: str
    file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    media_type: Literal["application/pdf", "image/png", "image/jpeg"]
    config: ExtractionConfig
    extractor: str
    ocr_backend: str
    pages: tuple[EvidencePage, ...] = Field(min_length=1, max_length=200)
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def valid_manifest(self):
        DocumentPage(document_id=self.document_id, source=self.source, version=self.version, text="metadata")
        if [p.page for p in self.pages] != list(range(1, len(self.pages) + 1)):
            raise ValueError("manifest must include every physical page in order")
        regions = [r for p in self.pages for r in p.regions]
        if (len(self.pages) > self.config.max_pages or len(regions) > self.config.max_regions
                or sum(len(r.text) for r in regions) > self.config.max_text_chars
                or len({r.region_id for r in regions}) != len(regions)):
            raise ValueError("manifest exceeds configured bounds or has duplicate regions")
        for page in self.pages:
            cell_addresses = set()
            for region in page.regions:
                if region.region_id != region_fingerprint(self.file_sha256, page.page, region):
                    raise ValueError("region content binding is invalid")
                if page.method == "ocr" and (
                    region.kind != "text" or region.confidence is None
                    or region.confidence < self.config.min_ocr_confidence
                ):
                    raise ValueError("OCR regions must satisfy the declared confidence floor")
                if page.method in {"native", "reviewed"} and region.confidence is not None:
                    raise ValueError("native/reviewed evidence has no OCR confidence score")
                if region.kind == "table_cell":
                    address = (region.table_id, region.row, region.column)
                    if address in cell_addresses:
                        raise ValueError("duplicate table cell address")
                    cell_addresses.add(address)
        if self.fingerprint != fingerprint(self.model_dump(mode="json", exclude={"fingerprint"})):
            raise ValueError("document receipt binding is invalid")
        return self

    @property
    def ready(self) -> bool:
        return all(page.status == "ready" for page in self.pages)


def region_fingerprint(file_hash: str, page: int, region: Region) -> str:
    return fingerprint({"file_sha256": file_hash, "page": page,
                        **region.model_dump(mode="json", exclude={"region_id"})})


class SpatialCitation(Contract):
    document_fingerprint: str
    file_sha256: str
    document_id: str
    source: str
    version: str
    page: int = Field(ge=1)
    region_id: str
    box: Box
    quote: str = Field(min_length=1, max_length=4000, repr=False)
    # Offsets refer to Region.text; the box encloses the REGION, not just this quote.
    region_start: int = Field(ge=0)
    region_end: int = Field(gt=0)
    table_id: str | None = None
    row: int | None = None
    column: int | None = None


class SpatialAnswer(Contract):
    protocol: Literal["agentforge-spatial-answer-v1"] = "agentforge-spatial-answer-v1"
    document_fingerprint: str
    query_sha256: str
    status: Literal["evidence_found", "abstained"]
    reason: Literal["literal_evidence", "no_evidence", "incomplete_extraction"]
    answer: str = Field(repr=False)
    citations: tuple[SpatialCitation, ...] = Field(max_length=60)
    text_receipt: AnswerResult | None = Field(default=None, repr=False)
    fingerprint: str


class Calculation(Contract):
    protocol: Literal["agentforge-cell-calculation-v1"] = "agentforge-cell-calculation-v1"
    document_fingerprint: str
    operation: Literal["sum", "difference", "ratio", "percent_change"]
    region_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    status: Literal["calculated", "abstained"]
    reason: Literal["verified_cells", "incomplete_extraction", "invalid_cells", "ambiguous_number", "zero_denominator"]
    value: str | None = None
    unit: str | None = None
    citations: tuple[SpatialCitation, ...] = Field(max_length=32)
    fingerprint: str
