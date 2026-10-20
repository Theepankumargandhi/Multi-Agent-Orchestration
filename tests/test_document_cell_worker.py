"""Authored PDFium handles exercise coordinate and budget boundaries without parsing uploads on host."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from agentforge_sdk.models import fingerprint
from agentforge_sdk.reconciliation import FactReconciliation
from agentforge_sdk.spatial import Box
from service.document_cell_concordance import check_cells
from service.document_cell_contracts import CellProbe, CellProbeResult, CellTarget
from service.document_cell_worker import probe_pdf
from service.document_runtime import DocumentRuntime

RAW = b"%PDF-authored-handle-control"


def request():
    return CellProbe(
        file_sha256=hashlib.sha256(RAW).hexdigest(),
        page_count=1,
        targets=(
            CellTarget(
                region_id="a" * 64, page=1, width=600.0, height=800.0, box=Box(x0=0.4, y0=0.3, x1=0.6, y1=0.4)
            ),
        ),
    )


class TextPage:
    count, value = 100, "125\r\n"

    def __init__(self):
        self.calls, self.closed = [], False

    def count_chars(self):
        return self.count

    def get_text_bounded(self, **kwargs):
        self.calls.append(kwargs)
        return self.value

    def close(self):
        self.closed = True


class Page:
    rotation, media, crop, bounds, size = 0, (0, 0, 600, 800), (0, 0, 600, 800), (0, 0, 600, 800), (600, 800)

    def __init__(self):
        self.text, self.closed = TextPage(), False

    def get_rotation(self):
        return self.rotation

    def get_mediabox(self, **kwargs):
        return self.media

    def get_cropbox(self):
        return self.crop

    def get_bbox(self):
        return self.bounds

    def get_size(self):
        return self.size

    def get_textpage(self):
        return self.text

    def close(self):
        self.closed = True


@pytest.fixture
def engine(monkeypatch):
    page = Page()
    pdf = SimpleNamespace(closed=False)

    class Document:
        def __init__(self, raw):
            assert raw == RAW

        def __len__(self):
            return 1

        def __getitem__(self, number):
            assert number == 0
            return page

        def close(self):
            pdf.closed = True

    monkeypatch.setitem(sys.modules, "pypdfium2", SimpleNamespace(PdfDocument=Document))
    monkeypatch.setattr("service.document_cell_worker.version", lambda name: "5.14.0")
    return page, pdf


def test_pdf_coordinate_transform_exact_text_and_handle_cleanup(engine):
    page, pdf = engine
    result = probe_pdf(RAW, request())
    assert result.samples[0].text == "125" and result.samples[0].status == "observed"
    assert page.text.calls == [dict(left=240.0, bottom=480.0, right=360.0, top=560.0, errors="strict")]
    assert page.closed and page.text.closed and pdf.closed
    assert result.request_fingerprint == fingerprint(request().model_dump(mode="json"))


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("rotation", 90),
        ("media", None),
        ("media", (1, 0, 600, 800)),
        ("crop", (0, 0, 590, 800)),
        ("bounds", (0, 1, 600, 800)),
        ("size", (599, 800)),
        ("size", (float("nan"), 800)),
    ],
)
def test_geometry_is_not_guessed_or_silently_normalized(engine, attribute, value):
    page, pdf = engine
    setattr(page, attribute, value)
    result = probe_pdf(RAW, request())
    assert result.samples[0].status == "unsupported_geometry" and result.samples[0].text is None
    assert not page.text.calls and page.closed and pdf.closed


@pytest.mark.parametrize(
    "count,text,status",
    [(20_001, "PRIVATE", "budget_exceeded"), (10, "", "empty"), (10, "x" * 4001, "budget_exceeded")],
)
def test_empty_and_over_budget_cells_never_become_observed(engine, count, text, status):
    page, _ = engine
    page.text.count, page.text.value = count, text
    sample = probe_pdf(RAW, request()).samples[0]
    assert sample.status == status and sample.text is None
    if count > 20_000:
        assert not page.text.calls


def test_bad_byte_binding_is_rejected_before_opening_and_result_is_sealed(engine):
    with pytest.raises(ValueError, match="byte binding"):
        probe_pdf(RAW + b"changed", request())
    payload = probe_pdf(RAW, request()).model_dump(mode="json")
    payload["samples"][0]["text"] = "999"
    with pytest.raises(ValueError):
        CellProbeResult.model_validate_json(json.dumps(payload))


def test_unsupported_or_absent_citations_do_not_call_an_engine():
    runtime = SimpleNamespace(probe_cells=lambda *_: pytest.fail("must not execute"))
    assert (
        check_cells(runtime, RAW, "source", SimpleNamespace(media_type="image/png"), {})["status"]
        == "unsupported_source"
    )
    assert (
        check_cells(runtime, RAW, "source", SimpleNamespace(media_type="application/pdf"), {})["status"]
        == "no_cited_cells"
    )
    citations = [SimpleNamespace(region_id=f"{number:064x}") for number in range(61)]
    receipt = FactReconciliation.model_construct(
        observations=(SimpleNamespace(source_id="source", citations=citations),)
    )
    result = check_cells(
        runtime, RAW, "source", SimpleNamespace(media_type="application/pdf"), {"input": (None, receipt, [])}
    )
    assert result["status"] == "cell_budget_exceeded" and result["requested_cells"] == 61


def test_non_native_citations_are_not_silently_checked_as_native_text():
    runtime = SimpleNamespace(probe_cells=lambda *_: pytest.fail("must not execute"))
    citation = SimpleNamespace(region_id="a" * 64)
    receipt = FactReconciliation.model_construct(
        observations=(SimpleNamespace(source_id="source", citations=(citation,)),)
    )
    document = SimpleNamespace(
        media_type="application/pdf",
        pages=(
            SimpleNamespace(
                method="ocr",
                status="ready",
                regions=(SimpleNamespace(region_id="a" * 64, kind="table_cell"),),
            ),
        ),
    )
    result = check_cells(runtime, RAW, "source", document, {"input": (None, receipt, [])})
    assert result["status"] == "unsupported_evidence"


def test_probe_contract_rejects_expected_text_duplicate_and_oversized_targets():
    payload = request().model_dump(mode="json")
    payload["targets"][0]["text"] = "PRIVATE_EXPECTED"
    with pytest.raises(ValueError):
        CellProbe.model_validate_json(json.dumps(payload))
    payload = request().model_dump(mode="json")
    payload["targets"] *= 2
    with pytest.raises(ValueError):
        CellProbe.model_validate_json(json.dumps(payload))
    payload["targets"] *= 31
    with pytest.raises(ValueError):
        CellProbe.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("timeout", [False, True])
def test_probe_uses_fixed_worker_and_boundary_and_cleans_owned_container(monkeypatch, timeout):
    from service import document_runtime

    calls, input_payloads = [], []
    payload = dict(
        protocol="agentforge-pdf-cell-samples-v1",
        file_sha256=request().file_sha256,
        request_fingerprint=fingerprint(request().model_dump(mode="json")),
        page_count=1,
        engine="pypdfium2:5.14.0",
        samples=[dict(region_id="a" * 64, page=1, status="observed", text="125")],
    )
    result_json = json.dumps({**payload, "fingerprint": fingerprint(payload)})

    def run(argv, **kwargs):
        calls.append(argv)
        if kwargs.get("stdin_file"):
            input_payloads.append(json.loads(kwargs["stdin_file"].read()))
        stdout = (
            "sha256:" + "a" * 64
            if argv[1] == "image"
            else result_json
            if "cells.json" in " ".join(argv)
            else ""
        )
        failed = timeout and argv[-1] == "/app/document_cell_worker.py"
        return SimpleNamespace(
            returncode=124 if failed else 0,
            timed_out=failed,
            stdout=stdout,
            stdout_bytes=len(stdout.encode()),
            stderr="PRIVATE_ERROR",
        )

    monkeypatch.setattr(document_runtime, "run_bounded", run)
    if timeout:
        with pytest.raises(RuntimeError, match="isolated_document_exec_failed") as error:
            DocumentRuntime().probe_cells(RAW, request())
        assert "PRIVATE" not in str(error.value)
    else:
        assert DocumentRuntime().probe_cells(RAW, request())["result"]["samples"][0]["text"] == "125"
    assert any(argv[-1] == "/app/document_cell_worker.py" for argv in calls)
    create = next(argv for argv in calls if argv[1] == "create")
    assert {
        "--network=none",
        "--read-only",
        "--user=10001:10001",
        "--memory=512m",
        "--pids-limit=64",
        "--pull=never",
    } <= set(create)
    assert set(input_payloads[0]) == {"data", "probe"} and '"text"' not in json.dumps(
        input_payloads[0]["probe"]
    )
    assert calls[-1][:3] == ["docker", "rm", "-f"] and calls[-1][3].startswith("agentforge-doc-")
