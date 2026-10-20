"""Explicit rendered-pixel checks expose native-text agreement with invisible PDF text."""

import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest
from test_document_cell_concordance import ProbeControl
from test_document_claim_audits import advance, item, report, start
from test_document_source_replay import assert_unleased, body, prepare, stored
from test_document_workbench_reconciliation import create, fact
from test_document_workbench_reconciliation import setup as setup
from test_document_workbench_versions import client_for, headers, upload

from agentforge_sdk.models import fingerprint
from service.document_runtime import DocumentRuntime
from service.document_store import DocumentLibrary


def endpoint(set_id, audit_id, source_id):
    return f"/workbench/documents/reconciliations/{set_id}/audits/{audit_id}/sources/{source_id}/crosscheck-pixels"


def crosscheck(client, set_id, audit_id, source_id, **extra):
    return client.post(
        endpoint(set_id, audit_id, source_id),
        headers=headers(),
        json={
            "checkpoint_fingerprint": body(client, set_id, audit_id)["checkpoint_fingerprint"],
            "confirm_local_ocr": True,
            **extra,
        },
    )


class PixelControl(ProbeControl):
    def probe_pixels(self, raw, probe):
        result = super().probe_cells(raw, probe)
        payload = result["result"]
        payload.update(
            protocol="agentforge-pixel-cell-samples-v1",
            profile=probe.profile,
            engine="tesseract-tsv-psm6:eng",
            renderer="pypdfium2:5.14.0",
            raster_pages=1,
            rendered_pixels=4_320_000,
            ocr_calls=len(payload["samples"]),
        )
        for sample in payload["samples"]:
            sample["min_confidence"] = 0.95 if sample["status"] == "observed" else None
            if self.mode == "low_confidence":
                sample.update(status="low_confidence", text=None, min_confidence=0.4)
            elif self.mode == "ocr_failed":
                sample.update(status="ocr_failed", text=None, min_confidence=None)
            elif self.mode == "boundary_ink":
                sample.update(status="boundary_ink", text=None, min_confidence=None)
            elif self.mode == "invalid_confidence":
                sample["min_confidence"] = 0.4
        if self.mode == "bad_profile":
            payload["profile"] = "PRIVATE_PROFILE"
        if self.mode == "boundary_ink":
            payload["ocr_calls"] = 0
        payload["fingerprint"] = fingerprint(
            {key: value for key, value in payload.items() if key != "fingerprint"}
        )
        return result


def test_pixel_probe_has_separate_consent_blind_request_and_preserves_verdicts(setup):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    runtime = library.runtime = PixelControl(library)
    original = stored(library)
    response = crosscheck(client, set_id, audit_id, ids[0])
    result = response.json()
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert result["protocol"] == "agentforge-pixel-concordance-v1"
    assert result["status"] == "all_cells_agree" and result["counts"]["agree"] == 5
    assert result["ocr_calls"] == 5 and result["raster_pages"] == 1
    assert (
        result["engine"] == "tesseract-tsv-psm6:eng"
        and result["profile"] == "pdfium216-edge3-tesseract6-eng-v1"
    )
    assert (
        '"text"' not in runtime.requests[0].model_dump_json()
        and "expected" not in runtime.requests[0].model_dump_json()
    )
    assert (
        "Requests" not in response.text and '"text"' not in response.text and "PRIVATE" not in response.text
    )
    assert not result["release_authorized"] and not result["publisher_identity_verified"]
    assert (
        not result["numeric_verdicts_rechecked"]
        and not result["other_sources_checked"]
        and result["model_calls"] == 0
    )
    assert stored(library) == original and report(client, set_id, audit_id).json()["counts"]["refuted"] == 1
    assert result["fingerprint"] == fingerprint({k: v for k, v in result.items() if k != "fingerprint"})
    assert_unleased(library)


@pytest.mark.parametrize(
    "mode,status",
    [
        ("different", "disagreement"),
        ("empty", "inconclusive"),
        ("low_confidence", "inconclusive"),
        ("ocr_failed", "inconclusive"),
        ("boundary_ink", "inconclusive"),
        ("unsupported_geometry", "inconclusive"),
        ("budget_exceeded", "inconclusive"),
        ("wrong_hash", "probe_binding_mismatch"),
        ("wrong_request", "probe_binding_mismatch"),
        ("missing", "probe_binding_mismatch"),
        ("invalid_confidence", "extraction_failed"),
        ("bad_profile", "extraction_failed"),
        ("fail", "extraction_failed"),
    ],
)
def test_low_quality_failure_and_binding_controls_cannot_become_agreement(setup, mode, status):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    library.runtime = PixelControl(library, mode)
    original = stored(library)
    response = crosscheck(client, set_id, audit_id, ids[0])
    assert response.status_code == 200 and response.json()["status"] == status
    assert "PRIVATE" not in response.text and not response.json()["release_authorized"]
    assert stored(library) == original
    assert_unleased(library)


@pytest.mark.parametrize(
    "extra",
    [
        {"confirm_local_ocr": False},
        {"confirm_local_ocr": 1},
        {"confirm_local_ocr": "true"},
        {"confirm_crosscheck": True},
        {"targets": "PRIVATE_TARGETS"},
        {"language": "PRIVATE_LANGUAGE"},
    ],
)
def test_local_ocr_consent_is_literal_and_tools_or_languages_are_not_accepted(setup, extra):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    runtime = library.runtime = PixelControl(library)
    response = crosscheck(client, set_id, audit_id, ids[0], **extra)
    assert response.status_code == 422 and runtime.calls == 0 and "PRIVATE" not in response.text


def test_owner_membership_stale_partial_and_operator_flag_precede_ocr(setup, monkeypatch):
    client, library = setup
    set_id, audit_id, ids = prepare(client)
    runtime = library.runtime = PixelControl(library)
    request = {
        "checkpoint_fingerprint": body(client, set_id, audit_id)["checkpoint_fingerprint"],
        "confirm_local_ocr": True,
    }
    assert (
        client.post(endpoint(set_id, audit_id, ids[0]), headers=headers("bob"), json=request).status_code
        == 404
    )
    assert crosscheck(client, set_id, audit_id, "f" * 32).status_code == 404
    assert crosscheck(client, set_id, audit_id, ids[0], checkpoint_fingerprint="0" * 64).status_code == 409
    partial = start(
        client,
        set_id,
        [item(report(client, set_id, audit_id).json()["items"][0]["result_id"])],
        "Partial pixel audit",
    )
    assert crosscheck(client, set_id, partial, ids[0]).status_code == 409
    monkeypatch.setenv("WORKBENCH_DOCUMENTS_ENABLED", "false")
    assert client.post(endpoint(set_id, audit_id, ids[0]), headers=headers(), json=request).status_code == 404
    assert runtime.calls == 0


def test_pixel_lease_excludes_upload_and_both_other_checks_and_rechecks_late_bytes(setup):
    from test_document_cell_concordance import crosscheck as check_text
    from test_document_source_replay import replay

    client, library = setup
    set_id, audit_id, ids = prepare(client)
    runtime = library.runtime = PixelControl(library, "wait")
    request = {
        "checkpoint_fingerprint": body(client, set_id, audit_id)["checkpoint_fingerprint"],
        "confirm_local_ocr": True,
    }
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.post, endpoint(set_id, audit_id, ids[0]), headers=headers(), json=request)
        try:
            assert runtime.entered.wait(5)
            assert crosscheck(client, set_id, audit_id, ids[1]).status_code == 429
            assert check_text(client, set_id, audit_id, ids[1]).status_code == 429
            assert replay(client, set_id, audit_id, ids[1]).status_code == 429
            assert upload(client, b"%PDF-extra", name="extra.pdf").status_code == 429
            assert report(client, set_id, audit_id).status_code == 200
            with closing(library.connect()) as db, db:
                db.execute("UPDATE documents SET original=? WHERE id=?", (b"%PDF-late", ids[0]))
        finally:
            runtime.release.set()
        assert future.result().status_code == 409
    assert_unleased(library)


def _vector_text(canvas, text, *, x, y, font):
    from fontTools.pens.reportLabPen import ReportLabPen
    from reportlab.graphics import renderPDF
    from reportlab.graphics.shapes import Drawing, Group
    from reportlab.lib.colors import black

    glyphs, cmap = font.getGlyphSet(), font.getBestCmap()
    scale, position = 14 / font["head"].unitsPerEm, 0
    drawing = Drawing(100, 40)
    for character in text:
        glyph = glyphs[cmap[ord(character)]]
        pen = ReportLabPen(glyphs)
        glyph.draw(pen)
        pen.path.fillColor, pen.path.strokeColor = black, None
        group = Group(pen.path)
        group.transform = (scale, 0, 0, scale, position, 0)
        drawing.add(group)
        position += glyph.width * scale
    renderPDF.draw(drawing, canvas, x, y)


@pytest.mark.skipif(
    os.getenv("AF_RUN_DOCUMENT_DOCKER") != "1",
    reason="explicit isolated rendered-cell OCR integration opt-in",
)
def test_actual_pixels_disagree_with_invisible_text_accepted_by_both_native_engines(tmp_path, monkeypatch):
    import reportlab
    from fontTools.ttLib import TTFont
    from reportlab.pdfgen import canvas
    from test_document_cell_concordance import crosscheck as check_text

    monkeypatch.setenv("WORKBENCH_DOCUMENTS_ENABLED", "true")
    library = DocumentLibrary(tmp_path / "pixels.db", DocumentRuntime())
    font_path = Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"
    with client_for(library) as client, closing(TTFont(str(font_path))) as font:
        ids = []
        for index in range(4):
            hidden = index >= 2
            output = io.BytesIO()
            pdf = canvas.Canvas(output, pagesize=(600, 800))
            if index % 2:
                pdf.drawString(60, 720, "Atlas cover: physical page two contains the table.")
                pdf.showPage()
            pdf.setFont("Helvetica", 14)
            pdf.drawString(60, 720, f"Atlas control {index}: visible Requests value is 125.")
            pdf.grid([60, 240, 360, 480], [430, 480, 530, 580])
            for row, values in enumerate(
                [["Metric", "Previous", "Current"], ["Requests", "100", "125"], ["Errors", "10", "20"]]
            ):
                for column, text in enumerate(values):
                    x, y = [70, 250, 370][column], 555 - row * 50
                    if hidden and row == 1 and column == 2:
                        pdf.saveState()
                        invisible = pdf.beginText(x, y)
                        invisible.setTextRenderMode(3)
                        invisible.textOut("999")
                        pdf.drawText(invisible)
                        pdf.restoreState()
                        _vector_text(pdf, "125", x=x, y=y, font=font)
                    else:
                        pdf.drawString(x, y, text)
            pdf.save()
            (tmp_path / f"pixel-control-{index}.pdf").write_bytes(output.getvalue())
            uploaded = upload(client, output.getvalue(), name=f"pixel-control-{index}.pdf").json()
            assert uploaded["status"] == "ready", uploaded
            ids.append(uploaded["id"])
        reports = []
        for selected, expected in [(ids[:2], "125"), (ids[2:], "999")]:
            set_id = create(client, selected)
            current = fact(client, set_id).json()["id"]
            audit_id = start(client, set_id, [item(current, value=expected)])
            advance(client, set_id, audit_id)
            assert report(client, set_id, audit_id).json()["counts"]["supported"] == 1
            original = stored(library)
            native = check_text(client, set_id, audit_id, selected[1]).json()
            assert native["status"] == "all_cells_agree", native
            pixel_response = crosscheck(client, set_id, audit_id, selected[1])
            assert pixel_response.status_code == 200, pixel_response.text
            pixels = pixel_response.json()
            (tmp_path / f"pixel-report-{expected}.json").write_text(
                json.dumps(pixels, indent=2), encoding="utf-8"
            )
            assert pixels["status"] == ("disagreement" if expected == "999" else "all_cells_agree"), (
                json.dumps(pixels, indent=2)
            )
            assert pixels["counts"] == dict(
                agree=2 if expected == "999" else 3, different=1 if expected == "999" else 0, uncheckable=0
            ), pixels
            assert pixels["ocr_calls"] == 3 and pixels["raster_pages"] == 1
            assert {cell["page"] for cell in pixels["cells"]} == {2}
            assert stored(library) == original
            reports.append({"native": native, "pixels": pixels})
        (tmp_path / "pixel-concordance-reports.json").write_text(
            json.dumps(reports, indent=2), encoding="utf-8"
        )
        image = client.get(f"/workbench/documents/{ids[3]}/pages/2", headers=headers()).json()["image"]
        (tmp_path / "visible-125-hidden-999.png").write_bytes(base64.b64decode(image.split(",")[1]))
        assert_unleased(library)
