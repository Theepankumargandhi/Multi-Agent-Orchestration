"""Bounded, content-blind requests to the second PDF text engine."""

from typing import Literal

from pydantic import Field, model_validator

from agentforge_sdk.models import Contract, fingerprint
from agentforge_sdk.spatial import Box

MAX_CELLS = 60
MAX_PIXEL_CELLS = 12


class CellTarget(Contract):
    region_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    page: int = Field(ge=1, le=10)
    width: float = Field(gt=0, le=100_000)
    height: float = Field(gt=0, le=100_000)
    box: Box


class CellProbe(Contract):
    protocol: Literal["agentforge-pdf-cell-probe-v1"] = "agentforge-pdf-cell-probe-v1"
    file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_count: int = Field(ge=1, le=10)
    targets: tuple[CellTarget, ...] = Field(min_length=1, max_length=MAX_CELLS)

    @model_validator(mode="after")
    def unique_targets(self):
        if len({t.region_id for t in self.targets}) != len(self.targets):
            raise ValueError("cell targets must be unique")
        if any(t.page > self.page_count for t in self.targets):
            raise ValueError("cell page is outside the original")
        return self


class CellSample(Contract):
    region_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    page: int = Field(ge=1, le=10)
    status: Literal["observed", "empty", "unsupported_geometry", "budget_exceeded"]
    text: str | None = Field(default=None, max_length=4000, repr=False)

    @model_validator(mode="after")
    def text_only_when_observed(self):
        if self.status == "observed":
            if not self.text or "\x00" in self.text or self.text != " ".join(self.text.split()):
                raise ValueError("observed text must be canonical nonempty text")
        elif self.text is not None:
            raise ValueError("held cell cannot expose text")
        return self


class CellProbeResult(Contract):
    protocol: Literal["agentforge-pdf-cell-samples-v1"] = "agentforge-pdf-cell-samples-v1"
    file_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    request_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_count: int = Field(ge=1, le=10)
    engine: str = Field(pattern=r"^pypdfium2:[0-9]+\.[0-9]+\.[0-9]+$")
    samples: tuple[CellSample, ...] = Field(min_length=1, max_length=MAX_CELLS, repr=False)
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def bounded_binding(self):
        if len({s.region_id for s in self.samples}) != len(self.samples):
            raise ValueError("duplicate cell sample")
        if sum(len(s.text or "") for s in self.samples) > 100_000:
            raise ValueError("cell sample text budget")
        if self.fingerprint != fingerprint(self.model_dump(mode="json", exclude={"fingerprint"})):
            raise ValueError("cell sample binding")
        return self


class PixelProbe(CellProbe):
    protocol: Literal["agentforge-pixel-cell-probe-v1"] = "agentforge-pixel-cell-probe-v1"
    profile: Literal["pdfium216-edge3-tesseract6-eng-v1"] = "pdfium216-edge3-tesseract6-eng-v1"
    targets: tuple[CellTarget, ...] = Field(min_length=1, max_length=MAX_PIXEL_CELLS)

    @model_validator(mode="after")
    def bounded_selected_pages(self):
        if len({target.page for target in self.targets}) > 2:
            raise ValueError("pixel probe selected-page budget")
        return self


class PixelSample(CellSample):
    status: Literal[
        "observed",
        "empty",
        "unsupported_geometry",
        "budget_exceeded",
        "low_confidence",
        "ocr_failed",
        "boundary_ink",
    ]
    min_confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def confidence_is_a_filter(self):
        if self.status == "observed":
            if self.min_confidence is None or self.min_confidence < 0.6:
                raise ValueError("observed OCR words require confidence floor")
        elif self.status == "low_confidence":
            if self.min_confidence is None or self.min_confidence >= 0.6:
                raise ValueError("low confidence requires a below-floor score")
        elif self.min_confidence is not None:
            raise ValueError("unobserved cells cannot claim confidence")
        return self


class PixelProbeResult(CellProbeResult):
    protocol: Literal["agentforge-pixel-cell-samples-v1"] = "agentforge-pixel-cell-samples-v1"
    profile: Literal["pdfium216-edge3-tesseract6-eng-v1"] = "pdfium216-edge3-tesseract6-eng-v1"
    engine: Literal["tesseract-tsv-psm6:eng"] = "tesseract-tsv-psm6:eng"
    renderer: str = Field(pattern=r"^pypdfium2:[0-9]+\.[0-9]+\.[0-9]+$")
    samples: tuple[PixelSample, ...] = Field(min_length=1, max_length=MAX_PIXEL_CELLS, repr=False)
    raster_pages: int = Field(ge=0, le=2)
    rendered_pixels: int = Field(ge=0, le=12_000_000)
    ocr_calls: int = Field(ge=0, le=MAX_PIXEL_CELLS)

    @model_validator(mode="after")
    def raster_accounting(self):
        attempted = sum(
            sample.status in {"observed", "empty", "low_confidence", "ocr_failed"} for sample in self.samples
        )
        if not attempted <= self.ocr_calls <= len(self.samples):
            raise ValueError("pixel probe OCR accounting")
        if (self.raster_pages == 0) != (self.rendered_pixels == 0) or (
            self.ocr_calls and not self.raster_pages
        ):
            raise ValueError("pixel probe raster accounting")
        if self.raster_pages > len({sample.page for sample in self.samples}):
            raise ValueError("pixel probe page accounting")
        return self
