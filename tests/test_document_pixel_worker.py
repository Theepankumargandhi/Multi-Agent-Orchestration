"""Authored raster/OCR handles test withholding and resource accounting, not OCR accuracy."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw
from test_document_cell_worker import RAW, Page, request

from agentforge_sdk.models import fingerprint
from agentforge_sdk.ocr import OCRWord
from agentforge_sdk.reconciliation import FactReconciliation
from agentforge_sdk.spatial import Box
from service.document_cell_concordance import check_cells
from service.document_cell_contracts import PixelProbe, PixelProbeResult, PixelSample
from service.document_pixel_worker import _prepare_crop, probe_pixels
from service.document_runtime import DocumentRuntime


def pixel_request():
    payload = request().model_dump(mode="json")
    payload["protocol"] = "agentforge-pixel-cell-probe-v1"
    return PixelProbe.model_validate_json(json.dumps(payload))


class Backend:
    name = "tesseract-tsv-psm6:eng"

    def __init__(self):
        self.calls, self.fail = [], False
        self.words = (
            OCRWord(text="125", box=Box(x0=0.1, y0=0.1, x1=0.5, y1=0.5), confidence=0.95, line=(1, 1, 1)),
        )

    def recognize(self, png, *, width, height):
        self.calls.append((width, height))
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        if self.fail:
            raise RuntimeError("PRIVATE_OCR_ERROR")
        return self.words


@pytest.fixture
def engine(monkeypatch):
    backend, state = Backend(), SimpleNamespace(closed=False, bitmaps=0, rendering=[])

    class Bitmap:
        def close(self):
            state.bitmaps += 1

        def to_pil(self):
            return Image.new("RGB", (1800, 2400), "white")

    class RasterPage(Page):
        def get_textpage(self):
            pytest.fail("pixel worker must not read native text")

        def render(self, **kwargs):
            state.rendering.append(kwargs)
            return Bitmap()

    page = RasterPage()

    class Document:
        def __init__(self, raw):
            assert raw == RAW

        def __len__(self):
            return 1

        def __getitem__(self, number):
            assert number == 0
            return page

        def close(self):
            state.closed = True

    monkeypatch.setitem(sys.modules, "pypdfium2", SimpleNamespace(PdfDocument=Document))
    monkeypatch.setattr("service.document_pixel_worker.version", lambda _: "5.14.0")
    monkeypatch.setattr("service.document_pixel_worker.TesseractOCR", lambda **kwargs: backend)
    return backend, page, state


def test_renders_pixels_only_and_crops_exact_region_with_valid_accounting(engine):
    backend, page, state = engine
    result = probe_pixels(RAW, pixel_request())
    assert result.samples[0].text == "125" and result.samples[0].min_confidence == 0.95
    assert result.raster_pages == 1 and result.rendered_pixels == 4_320_000 and result.ocr_calls == 1
    assert backend.calls and backend.calls[0][0] * backend.calls[0][1] <= 500_000
    assert state.rendering == [dict(scale=3, may_draw_forms=False)]
    assert page.closed and state.closed and state.bitmaps == 1


@pytest.mark.parametrize(
    "mode,status",
    [("low", "low_confidence"), ("empty", "empty"), ("fail", "ocr_failed"), ("long", "budget_exceeded")],
)
def test_ocr_failures_low_confidence_empty_and_overlong_outputs_withhold_text(engine, mode, status):
    backend, _, _ = engine
    if mode == "low":
        backend.words = (backend.words[0].model_copy(update={"confidence": 0.59}),)
    elif mode == "empty":
        backend.words = ()
    elif mode == "long":
        backend.words = (backend.words[0].model_copy(update={"text": "x" * 500}),) * 9
    else:
        backend.fail = True
    result = probe_pixels(RAW, pixel_request())
    assert result.samples[0].status == status and result.samples[0].text is None
    assert result.ocr_calls == 1 and "PRIVATE" not in result.model_dump_json()


def test_multiple_words_follow_line_then_horizontal_order_not_glyph_top(engine):
    backend, _, _ = engine
    backend.words = (
        OCRWord(text="total", box=Box(x0=0.5, y0=0.1, x1=0.8, y1=0.4), confidence=0.9, line=(1, 1, 1)),
        OCRWord(text="Requests", box=Box(x0=0.1, y0=0.2, x1=0.4, y1=0.5), confidence=0.95, line=(1, 1, 1)),
    )
    sample = probe_pixels(RAW, pixel_request()).samples[0]
    assert sample.text == "Requests total" and sample.min_confidence == 0.9


def test_fixed_frame_policy_removes_only_thin_uniform_ruling_and_preserves_interior():
    with Image.new("RGB", (100, 60), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 99, 59), outline="black", width=2)
        draw.rectangle((20, 20, 40, 40), fill="black")
        prepared = _prepare_crop(image)
        assert prepared is not None
        with prepared:
            assert prepared.size == (94, 54) and prepared.getpixel((20, 20)) == (0, 0, 0)


@pytest.mark.parametrize("kind", ["mixed_fringe", "inner_guard", "thick_rule", "tiny"])
def test_ambiguous_boundary_ink_is_withheld_without_trimming_for_a_match(kind):
    with Image.new("RGB", (10, 10) if kind == "tiny" else (100, 60), "white") as image:
        draw = ImageDraw.Draw(image)
        if kind == "mixed_fringe":
            draw.point((1, 10), fill="black")
        elif kind == "inner_guard":
            draw.point((4, 10), fill="black")
        elif kind == "thick_rule":
            draw.rectangle((0, 0, 99, 59), outline="black", width=4)
        assert _prepare_crop(image) is None


def test_rotated_geometry_does_not_render_or_ocr(engine):
    backend, page, state = engine
    page.rotation = 90
    result = probe_pixels(RAW, pixel_request())
    assert result.samples[0].status == "unsupported_geometry"
    assert result.ocr_calls == result.raster_pages == result.rendered_pixels == 0
    assert not backend.calls and not state.rendering


def test_boundary_withholding_precedes_ocr_and_is_accounted_as_uncheckable(engine, monkeypatch):
    backend, _, _ = engine
    monkeypatch.setattr("service.document_pixel_worker._prepare_crop", lambda _: None)
    result = probe_pixels(RAW, pixel_request())
    assert result.samples[0].status == "boundary_ink" and result.raster_pages == 1
    assert result.ocr_calls == 0 and not backend.calls


def test_valid_result_digest_does_not_bypass_inconsistent_ocr_accounting(engine):
    payload = probe_pixels(RAW, pixel_request()).model_dump(mode="json")
    payload["ocr_calls"] = 0
    payload["fingerprint"] = fingerprint({k: v for k, v in payload.items() if k != "fingerprint"})
    with pytest.raises(ValueError):
        PixelProbeResult.model_validate_json(json.dumps(payload))


def test_page_raster_budget_precedes_allocation(engine):
    backend, page, state = engine
    page.size = (100_000, 100_000)
    page.media = page.crop = page.bounds = (0, 0, *page.size)
    probe = pixel_request()
    probe = probe.model_copy(
        update={"targets": (probe.targets[0].model_copy(update={"width": 100_000.0, "height": 100_000.0}),)}
    )
    result = probe_pixels(RAW, probe)
    assert result.samples[0].status == "budget_exceeded" and not backend.calls and not state.rendering


def test_crop_budget_precedes_ocr(engine):
    backend, _, _ = engine
    probe = pixel_request()
    probe = probe.model_copy(
        update={
            "targets": (probe.targets[0].model_copy(update={"box": Box(x0=0.0, y0=0.0, x1=1.0, y1=1.0)}),)
        }
    )
    result = probe_pixels(RAW, probe)
    assert result.samples[0].status == "budget_exceeded" and result.raster_pages == 1
    assert result.ocr_calls == 0 and not backend.calls


def test_deadline_retains_first_result_and_holds_remaining_cells_without_retry(engine, monkeypatch):
    backend, _, _ = engine
    probe = pixel_request()
    target = probe.targets[0]
    probe = probe.model_copy(update={"targets": (target, target.model_copy(update={"region_id": "b" * 64}))})
    times = iter([0, 0, 0, 47])  # deadline creation, page schedule, first cell, second cell
    monkeypatch.setattr("service.document_pixel_worker.time.monotonic", lambda: next(times))
    result = probe_pixels(RAW, probe)
    assert [s.status for s in result.samples] == ["observed", "budget_exceeded"]
    assert result.ocr_calls == 1 and len(backend.calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        dict(status="observed", text="125", min_confidence=0.59),
        dict(status="low_confidence", text="125", min_confidence=0.4),
        dict(status="empty", text=None, min_confidence=0.9),
    ],
)
def test_confidence_floor_cannot_be_bypassed_by_worker_fields(payload):
    with pytest.raises(ValueError):
        PixelSample(region_id="a" * 64, page=1, **payload)


def test_pixel_selected_cell_and_page_caps_do_not_silently_truncate_or_execute():
    runtime = SimpleNamespace(probe_pixels=lambda *_: pytest.fail("must not execute"))
    citations = [SimpleNamespace(region_id=f"{n:064x}") for n in range(13)]
    receipt = FactReconciliation.model_construct(
        observations=(SimpleNamespace(source_id="source", citations=citations),)
    )
    result = check_cells(
        runtime,
        RAW,
        "source",
        SimpleNamespace(media_type="application/pdf"),
        {"input": (None, receipt, [])},
        pixels=True,
    )
    assert result["status"] == "cell_budget_exceeded" and result["requested_cells"] == 13
    citations = citations[:3]
    receipt = FactReconciliation.model_construct(
        observations=(SimpleNamespace(source_id="source", citations=citations),)
    )
    pages = tuple(
        SimpleNamespace(
            page=n + 1,
            width=600.0,
            height=800.0,
            method="native",
            status="ready",
            regions=(
                SimpleNamespace(
                    region_id=c.region_id,
                    text="125",
                    kind="table_cell",
                    box=Box(x0=0.4, y0=0.3, x1=0.6, y1=0.4),
                ),
            ),
        )
        for n, c in enumerate(citations)
    )
    result = check_cells(
        runtime,
        RAW,
        "source",
        SimpleNamespace(media_type="application/pdf", pages=pages),
        {"input": (None, receipt, [])},
        pixels=True,
    )
    assert result["status"] == "page_budget_exceeded"
    payload = pixel_request().model_dump(mode="json")
    payload["page_count"] = 3
    payload["targets"] = [
        {**payload["targets"][0], "region_id": c.region_id, "page": n + 1} for n, c in enumerate(citations)
    ]
    with pytest.raises(ValueError):
        PixelProbe.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("timeout", [True, False])
def test_runtime_uses_fixed_pixel_worker_and_owned_cleanup(monkeypatch, timeout):
    from service import document_runtime

    calls = []
    probe = pixel_request()
    payload = dict(
        protocol="agentforge-pixel-cell-samples-v1",
        profile=probe.profile,
        file_sha256=hashlib.sha256(RAW).hexdigest(),
        request_fingerprint=fingerprint(probe.model_dump(mode="json")),
        page_count=1,
        engine="tesseract-tsv-psm6:eng",
        renderer="pypdfium2:5.14.0",
        samples=[dict(region_id="a" * 64, page=1, status="observed", text="125", min_confidence=0.95)],
        raster_pages=1,
        rendered_pixels=4_320_000,
        ocr_calls=1,
    )
    serialized = json.dumps({**payload, "fingerprint": fingerprint(payload)})

    def run(argv, **kwargs):
        calls.append(argv)
        stdout = (
            "sha256:" + "a" * 64
            if argv[1] == "image"
            else serialized
            if "pixels.json" in " ".join(argv)
            else ""
        )
        failed = timeout and argv[-1] == "/app/document_pixel_worker.py"
        return SimpleNamespace(
            returncode=124 if failed else 0,
            timed_out=failed,
            stdout=stdout,
            stdout_bytes=len(stdout.encode()),
            stderr="PRIVATE",
        )

    monkeypatch.setattr(document_runtime, "run_bounded", run)
    if timeout:
        with pytest.raises(RuntimeError, match="isolated_document_exec_failed"):
            DocumentRuntime().probe_pixels(RAW, probe)
    else:
        assert DocumentRuntime().probe_pixels(RAW, probe)["result"]["ocr_calls"] == 1
    assert calls[-1][:3] == ["docker", "rm", "-f"]
    assert any(argv[-1] == "/app/document_pixel_worker.py" for argv in calls)
    create = next(argv for argv in calls if argv[1] == "create")
    assert {"--network=none", "--read-only", "--user=10001:10001", "--pull=never", "--memory=512m"} <= set(
        create
    )
