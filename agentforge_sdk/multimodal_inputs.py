"""Local PDF/image ingestion with complete-page accounting and conservative holds."""

from __future__ import annotations

import hashlib
import io
import math
import re
import warnings
from collections import defaultdict
from contextlib import ExitStack, closing
from importlib.metadata import version as package_version
from pathlib import Path
from threading import RLock

from agentforge_sdk.inputs import _file_bytes
from agentforge_sdk.models import DocumentInputError, DocumentPage, ExternalProcessingDenied, fingerprint
from agentforge_sdk.ocr import LocalOCR, OCRWord
from agentforge_sdk.spatial import (
    Box,
    EvidencePage,
    ExtractionConfig,
    MultimodalDocument,
    Region,
    region_fingerprint,
)

# PDFium is not thread-safe, even across separate document handles.
_PDFIUM_LOCK = RLock()


def _box(bounds, width, height):
    x0, y0, x1, y1 = bounds
    return Box(x0=float(x0 / width), y0=float(y0 / height), x1=float(x1 / width), y1=float(y1 / height))


def _region(file_hash, page, text, box, **fields):
    text = " ".join(text.split())
    draft = Region(region_id="0" * 64, text=text, box=box, **fields)
    return draft.model_copy(update={"region_id": region_fingerprint(file_hash, page, draft)})


def _held(page, width, height, reason, method="none"):
    return EvidencePage(page=page, width=float(width), height=float(height), method=method,
                        status="held", reason=reason, regions=())


def _ocr_page(png, width, height, number, file_hash, config, backend):
    if backend is None:
        return _held(number, width, height, "ocr_required")
    try:
        # Recheck consent boundary immediately before invocation; mutable custom adapters are untrusted.
        _ocr_name(backend)
        raw = backend.recognize(png, width=width, height=height)
        if not isinstance(raw, tuple) or len(raw) > 10_000:
            raise ValueError("OCR must return a bounded tuple")
        words = tuple(OCRWord.model_validate(word.model_dump()) for word in raw)
        if not words:
            return _held(number, width, height, "empty_page", "ocr")
        if any(word.confidence < config.min_ocr_confidence for word in words):
            return _held(number, width, height, "low_confidence", "ocr")
        lines = defaultdict(list)
        for word in words:
            lines[word.line].append(word)
        regions = []
        for group in lines.values():
            group.sort(key=lambda w: w.box.x0)
            bounds = Box(x0=min(w.box.x0 for w in group), y0=min(w.box.y0 for w in group),
                         x1=max(w.box.x1 for w in group), y1=max(w.box.y1 for w in group))
            regions.append(_region(file_hash, number, " ".join(w.text for w in group), bounds,
                                   kind="text", confidence=min(w.confidence for w in group)))
        return _ready(number, width, height, "ocr", regions, config)
    except ExternalProcessingDenied:
        raise
    except Exception:
        return _held(number, width, height, "ocr_failed", "ocr")


def _ocr_name(backend):
    if backend is None:
        return "none"
    if getattr(backend, "external_processing", None) is not False:
        raise ExternalProcessingDenied("this ingestion profile accepts explicit local OCR only")
    name = getattr(backend, "name", None)
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.:+/-]{1,120}", name):
        raise ValueError("local OCR requires a stable, content-free backend name")
    return name


def _ready(number, width, height, method, regions, config):
    if not regions:
        return _held(number, width, height, "empty_page", method)
    if len(regions) > config.max_regions or sum(len(r.text) for r in regions) > config.max_text_chars:
        return _held(number, width, height, "budget_exceeded", method)
    return EvidencePage(page=number, width=float(width), height=float(height), method=method,
                        status="ready", reason="extracted", regions=tuple(regions))


def _native_page(page, number, file_hash, config):
    """Ruled, rectangular native tables only; no inferred cells for OCR or merged grids."""
    width, height = page.width, page.height
    regions, table_bounds = [], []
    for index, table in enumerate(page.find_tables()):
        values = table.extract()
        table_id = f"p{number}:t{index}"
        if not values or len(values) != len(table.rows) or len({len(r) for r in values}) != 1:
            return _held(number, width, height, "ambiguous_table", "native")
        for row, (cells, texts) in enumerate(zip(table.rows, values, strict=True)):
            for column, (bounds, text) in enumerate(zip(cells.cells, texts, strict=True)):
                if bounds is None or text is None or not text.strip():
                    return _held(number, width, height, "ambiguous_table", "native")
                regions.append(_region(file_hash, number, text, _box(bounds, width, height),
                                       kind="table_cell", table_id=table_id, row=row, column=column))
        table_bounds.append(table.bbox)
    lines = []
    words = page.extract_words()
    if len(words) > 10_000:
        return _held(number, width, height, "budget_exceeded", "native")
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        cx, cy = (word["x0"] + word["x1"]) / 2, (word["top"] + word["bottom"]) / 2
        if any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in table_bounds):
            continue
        if not lines or abs(lines[-1][0]["top"] - word["top"]) > 3:
            lines.append([])
        lines[-1].append(word)
    for group in lines:
        group.sort(key=lambda w: w["x0"])
        bounds = (min(w["x0"] for w in group), min(w["top"] for w in group),
                  max(w["x1"] for w in group), max(w["bottom"] for w in group))
        regions.append(_region(file_hash, number, " ".join(w["text"] for w in group),
                               _box(bounds, width, height), kind="text"))
    return _ready(number, width, height, "native", regions, config)


def _png(image):
    output = io.BytesIO()
    with image.convert("RGB") as converted:
        converted.save(output, format="PNG")
    return output.getvalue()


def _pdf_pages(raw, file_hash, config, backend):
    import pdfplumber

    pages = []
    with ExitStack() as stack, pdfplumber.open(io.BytesIO(raw), strict_metadata=True) as pdf:
        rasterizer = None
        if pdf.doc.encryption or not 1 <= len(pdf.pages) <= config.max_pages:
            raise ValueError("encrypted or over-budget PDF")
        for number, page in enumerate(pdf.pages, 1):
            width, height = page.width, page.height
            # Avoid false geometry claims for rotated/cropped coordinate frames.
            if (page.rotation or tuple(page.cropbox) != tuple(page.mediabox)
                    or page.bbox[0:2] != (0, 0) or min(width, height) <= 0):
                pages.append(_held(number, width, height, "unsupported_geometry"))
            elif page.images:
                if backend is None:
                    pages.append(_held(number, width, height, "ocr_required"))
                else:
                    if rasterizer is None:
                        import pypdfium2

                        stack.enter_context(_PDFIUM_LOCK)
                        rasterizer = stack.enter_context(closing(pypdfium2.PdfDocument(raw)))
                    pages.append(_render_ocr(rasterizer, number, width, height, file_hash, config, backend))
            else:
                pages.append(_native_page(page, number, file_hash, config))
    return pages


def _render_ocr(pdf, number, width, height, file_hash, config, backend):
    scale = config.render_dpi / 72
    if math.ceil(width * scale) * math.ceil(height * scale) > config.max_pixels:
        return _held(number, width, height, "budget_exceeded")
    # Reuse the document handle within this import; release each native page/bitmap.
    with closing(pdf[number - 1]) as page, closing(page.render(scale=scale)) as bitmap:
        image = bitmap.to_pil()
        try:
            result = _ocr_page(_png(image), image.width, image.height, number, file_hash, config, backend)
            return result.model_copy(update={"width": float(width), "height": float(height)})
        finally:
            image.close()


def load_multimodal(
    path: Path, *, document_id: str, source: str | None = None, version="unversioned",
    config: ExtractionConfig | None = None, ocr: LocalOCR | None = None,
) -> MultimodalDocument:
    """Read one explicit trusted-local PDF/PNG/JPEG. No directory scan, cache or remote calls.

    This in-process parser is NOT a hostile-upload sandbox. Any held page prevents
    document-wide answering/calculations, rather than silently dropping missing evidence.
    """
    config = ExtractionConfig.model_validate((config or ExtractionConfig()).model_dump())
    path = Path(path)
    metadata = DocumentPage(document_id=document_id, source=source or path.name, version=version, text="metadata")
    backend_name = _ocr_name(ocr)
    if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise DocumentInputError("multimodal input requires one explicit PDF, PNG or JPEG file")
    try:
        raw = _file_bytes(path, config.max_file_bytes)
        file_hash = hashlib.sha256(raw).hexdigest()
        if path.suffix.lower() == ".pdf":
            if not raw.startswith(b"%PDF-"):
                raise ValueError("invalid PDF header")
            pages = _pdf_pages(raw, file_hash, config, ocr)
            media_type = "application/pdf"
        else:
            from PIL import Image

            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as image:
                    if (image.format not in {"PNG", "JPEG"} or getattr(image, "n_frames", 1) != 1
                            or image.width * image.height > config.max_pixels):
                        raise ValueError("unsupported or oversized raster")
                    media_type = "image/png" if image.format == "PNG" else "image/jpeg"
                    if image.getexif().get(274, 1) != 1:
                        pages = [_held(1, image.width, image.height, "unsupported_geometry")]
                    else:
                        pages = [_ocr_page(_png(image), image.width, image.height, 1, file_hash, config, ocr)]
        if _ocr_name(ocr) != backend_name:
            raise ExternalProcessingDenied("OCR backend identity changed during extraction")
        if (sum(len(p.regions) for p in pages) > config.max_regions
                or sum(len(r.text) for p in pages for r in p.regions) > config.max_text_chars):
            pages = [_held(p.page, p.width, p.height, "budget_exceeded", p.method) for p in pages]
        payload = dict(document_id=metadata.document_id, source=metadata.source, version=metadata.version,
                       file_sha256=file_hash, media_type=media_type, config=config.model_dump(),
                       extractor=f"pdfplumber:{package_version('pdfplumber')};pypdfium2:{package_version('pypdfium2')};Pillow:{package_version('Pillow')}",
                       ocr_backend=backend_name, pages=tuple(p.model_dump() for p in pages),
                       protocol="agentforge-spatial-evidence-v1")
        return MultimodalDocument(**payload, fingerprint=fingerprint(payload))
    except ExternalProcessingDenied:
        raise
    except ImportError:
        raise DocumentInputError("multimodal extraction requires optional sdk[multimodal] dependencies") from None
    except Exception:
        raise DocumentInputError("multimodal input could not be extracted within the declared profile") from None
