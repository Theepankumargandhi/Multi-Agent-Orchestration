"""Container-only rendered-cell OCR. Expected text and native PDF text are never read."""

import base64
import hashlib
import io
import json
import math
import time
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

from agentforge_sdk.models import fingerprint
from agentforge_sdk.ocr import TesseractOCR
from service.document_cell_contracts import PixelProbe, PixelProbeResult, PixelSample
from service.document_cell_geometry import page_geometry


def _prepare_crop(image):
    """Remove a fixed thin ruling fringe only when adjacent ink is unambiguous.

    Mixed ink in the outer band or any ink in the inner guard band is withheld.
    No expected text, adaptive trim, mask or retry participates in this policy.
    """
    width, height = image.size
    if min(width, height) <= 12:
        return None
    with image.convert("L") as gray:
        edges = []
        for offset in range(3):
            edges.extend(
                (
                    (3, offset, width - 3, offset + 1),
                    (3, height - offset - 1, width - 3, height - offset),
                    (offset, 3, offset + 1, height - 3),
                    (width - offset - 1, 3, width - offset, height - 3),
                )
            )
        for rectangle in edges:
            with gray.crop(rectangle) as band:
                low, high = band.getextrema()
            if low < 240 <= high:
                return None
        # After the three-pixel ruling fringe, insist on another three clear pixels.
        for rectangle in (
            (3, 3, width - 3, 6),
            (3, height - 6, width - 3, height - 3),
            (3, 3, 6, height - 3),
            (width - 6, 3, width - 3, height - 3),
        ):
            with gray.crop(rectangle) as band:
                if band.getextrema()[0] < 240:
                    return None
    return image.crop((3, 3, width - 3, height - 3))


def _recognize(image, backend):
    output = io.BytesIO()
    image.save(output, format="PNG")
    try:
        words = backend.recognize(output.getvalue(), width=image.width, height=image.height)
    except Exception:
        return dict(status="ocr_failed", text=None, min_confidence=None)
    if not words:
        return dict(status="empty", text=None, min_confidence=None)
    confidence = min(word.confidence for word in words)
    if confidence < 0.6:
        return dict(status="low_confidence", text=None, min_confidence=confidence)
    words = sorted(words, key=lambda word: (word.line, word.box.x0, word.box.y0))
    text = " ".join(" ".join(word.text for word in words).split())
    if not 0 < len(text) <= 4000 or "\x00" in text:
        return dict(status="budget_exceeded", text=None, min_confidence=None)
    return dict(status="observed", text=text, min_confidence=confidence)


def probe_pixels(raw: bytes, probe: PixelProbe) -> PixelProbeResult:
    import pypdfium2

    probe = PixelProbe.model_validate(probe.model_dump())
    if not 0 < len(raw) <= 3_000_000 or not raw.startswith(b"%PDF-"):
        raise ValueError("pixel probe PDF budget")
    if hashlib.sha256(raw).hexdigest() != probe.file_sha256:
        raise ValueError("pixel probe byte binding")
    deadline = time.monotonic() + 50
    backend = TesseractOCR(timeout_seconds=4)
    samples, raster_pages, rendered_pixels, ocr_calls = {}, 0, 0, 0

    def held(target, status):
        samples[target.region_id] = PixelSample(region_id=target.region_id, page=target.page, status=status)

    with closing(pypdfium2.PdfDocument(raw)) as pdf:
        if len(pdf) != probe.page_count:
            raise ValueError("pixel probe page binding")
        for number in sorted({target.page for target in probe.targets}):
            targets = [target for target in probe.targets if target.page == number]
            with closing(pdf[number - 1]) as page:
                eligible = []
                for target in targets:
                    if page_geometry(page, target):
                        eligible.append(target)
                    else:
                        held(target, "unsupported_geometry")
                if not eligible:
                    continue
                width, height = (math.ceil(side * 3) for side in page.get_size())
                pixels = width * height
                if (
                    pixels > 6_000_000
                    or rendered_pixels + pixels > 12_000_000
                    or time.monotonic() >= deadline - 4
                ):
                    for target in eligible:
                        held(target, "budget_exceeded")
                    continue
                # PDFium renders pixels only; do not create/read a native text-page handle here.
                with closing(page.render(scale=3, may_draw_forms=False)) as bitmap, bitmap.to_pil() as image:
                    if image.size != (width, height):
                        raise ValueError("pixel probe raster binding")
                    raster_pages += 1
                    rendered_pixels += pixels
                    for target in eligible:
                        b = target.box
                        rectangle = (
                            math.floor(b.x0 * width),
                            math.floor(b.y0 * height),
                            math.ceil(b.x1 * width),
                            math.ceil(b.y1 * height),
                        )
                        crop_pixels = (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1])
                        if not 0 < crop_pixels <= 500_000 or time.monotonic() >= deadline - 4:
                            held(target, "budget_exceeded")
                            continue
                        with image.crop(rectangle) as crop:
                            prepared = _prepare_crop(crop)
                            if prepared is None:
                                held(target, "boundary_ink")
                                continue
                            with prepared:
                                ocr_calls += 1
                                result = _recognize(prepared, backend)
                        samples[target.region_id] = PixelSample(
                            region_id=target.region_id, page=number, **result
                        )
    payload = dict(
        protocol="agentforge-pixel-cell-samples-v1",
        profile=probe.profile,
        file_sha256=probe.file_sha256,
        request_fingerprint=fingerprint(probe.model_dump(mode="json")),
        page_count=probe.page_count,
        engine=backend.name,
        renderer="pypdfium2:" + version("pypdfium2"),
        samples=[samples[target.region_id].model_dump(mode="json") for target in probe.targets],
        raster_pages=raster_pages,
        rendered_pixels=rendered_pixels,
        ocr_calls=ocr_calls,
    )
    return PixelProbeResult.model_validate_json(json.dumps({**payload, "fingerprint": fingerprint(payload)}))


def main():
    payload = json.loads(Path("/tmp/input.json").read_bytes())
    if set(payload) != {"data", "probe"}:
        raise ValueError("pixel probe envelope")
    result = probe_pixels(
        base64.b64decode(payload["data"], validate=True),
        PixelProbe.model_validate_json(json.dumps(payload["probe"])),
    ).model_dump_json()
    if len(result.encode()) > 600_000:
        raise ValueError("pixel probe output budget")
    Path("/tmp/pixels.json").write_text(result, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(2) from None
