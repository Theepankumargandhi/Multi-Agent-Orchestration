from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import pytest

from agentforge_sdk import (
    Box,
    DocumentInputError,
    ExternalProcessingDenied,
    ExtractionConfig,
    MultimodalDocument,
    MultimodalRAG,
    OCRWord,
    TesseractOCR,
    load_multimodal,
)
from agentforge_sdk.models import fingerprint
from evals.multimodal_fixtures import create_fixtures


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    pytest.importorskip("reportlab")
    pytest.importorskip("pdfplumber")
    pytest.importorskip("pypdf")
    return create_fixtures(tmp_path_factory.mktemp("multimodal"))


def load(path, **kwargs):
    return load_multimodal(path, document_id="atlas", version="r1", **kwargs)


def cell(document, row, column):
    return next(r.region_id for r in document.pages[0].regions
                if r.kind == "table_cell" and r.row == row and r.column == column)


def test_real_native_pdf_table_geometry_literal_answer_and_json_roundtrip(fixtures):
    document = load(fixtures["native"])
    assert document.ready and len(document.pages) == 1
    cells = [r for r in document.pages[0].regions if r.kind == "table_cell"]
    assert len(cells) == 15
    region = next(r for r in cells if r.row == 1 and r.column == 1)
    assert region.text == "USD 120.50"
    assert region.box.x0 == pytest.approx(240 / 612)
    assert region.box.y0 == pytest.approx((792 - 618) / 792)
    restored = MultimodalDocument.model_validate_json(document.model_dump_json())
    rag = MultimodalRAG(restored)
    answer = rag.answer("Atlas p95 latency", top_k=1)
    assert "180 ms" in answer.answer and rag.verify_answer(answer, question="Atlas p95 latency")
    assert answer.citations[0].page == 1 and rag.verify_citation(answer.citations[0])
    assert answer.citations[0].file_sha256 == document.file_sha256
    assert not rag.verify_answer(answer, question="other")
    assert rag.answer("xylophonium").status == "abstained"


def test_decimal_cell_arithmetic_is_replayable_and_unit_checked(fixtures):
    document = load(fixtures["native"])
    rag = MultimodalRAG(document)
    ids = [cell(document, 1, c) for c in (1, 2)]
    result = rag.calculate("sum", ids)
    assert (result.status, result.value, result.unit) == ("calculated", "271.25", "USD")
    assert all(rag.verify_citation(c) for c in result.citations) and rag.verify_calculation(result)
    assert rag.calculate("difference", ids).value == "-30.25"
    requests = [cell(document, 2, c) for c in (1, 2)]
    assert rag.calculate("percent_change", requests).value == "25"
    assert rag.calculate("ratio", requests).value == "0.8"
    assert rag.calculate("ratio", [requests[0], cell(document, 3, 1)]).reason == "zero_denominator"
    assert rag.calculate("sum", [ids[0], requests[0]]).reason == "ambiguous_number"
    assert rag.calculate("sum", [cell(document, 4, 1)]).reason == "ambiguous_number"
    assert rag.calculate("sum", ["f" * 64]).reason == "invalid_cells"
    assert not rag.verify_calculation(result.model_copy(update={"value": "999"}))
    changed = result.model_copy(update={"value": "999"})
    changed = changed.model_copy(update={"fingerprint": fingerprint(changed.model_dump(mode="json", exclude={"fingerprint"}))})
    assert not rag.verify_calculation(changed)


def test_table_lookup_returns_value_with_separate_row_and_column_citations(fixtures):
    rag = MultimodalRAG(load(fixtures["native"]))
    answer = rag.answer("Revenue Q2", top_k=1)
    assert [c.quote for c in answer.citations] == ["Revenue", "Q2", "USD 150.75"]
    assert rag.verify_answer(answer, question="Revenue Q2")
    assert all(rag.verify_citation(c) for c in answer.citations)
    assert [(c.row, c.column) for c in answer.citations] == [(1, 0), (0, 2), (1, 2)]


@pytest.mark.parametrize("name,reason", [("scanned", "ocr_required"), ("mixed", "ocr_required"),
                                        ("rotated", "unsupported_geometry"), ("blank", "empty_page"),
                                        ("merged", "ambiguous_table")])
def test_all_pages_accounted_for_and_partial_documents_never_answer(fixtures, name, reason):
    document = load(fixtures[name])
    assert not document.ready and document.pages[-1].reason == reason
    if name == "mixed":
        assert [p.page for p in document.pages] == [1, 2] and document.pages[0].status == "ready"
    rag = MultimodalRAG(document)
    result = rag.answer("Atlas latency")
    assert result.reason == "incomplete_extraction" and not result.citations
    assert rag.verify_answer(result, question="Atlas latency")
    calc = rag.calculate("sum", ["f" * 64])
    assert calc.reason == "incomplete_extraction" and rag.verify_calculation(calc)


def test_spatial_provenance_rejects_coordinate_source_quote_and_cross_document_tampering(fixtures):
    document = load(fixtures["native"])
    rag = MultimodalRAG(document)
    result = rag.answer("Atlas latency", top_k=1)
    cite = result.citations[0]
    for change in ({"page": 2}, {"quote": "fabricated"}, {"source": "other.pdf"},
                   {"box": Box(x0=0.0, y0=0.0, x1=1.0, y1=1.0)}, {"file_sha256": "f" * 64}):
        assert not rag.verify_citation(cite.model_copy(update=change))
    assert not MultimodalRAG(load(fixtures["native"], source="other.pdf")).verify_citation(cite)
    changed = result.model_copy(update={"citations": (cite.model_copy(update={"page": 2}),)})
    changed = changed.model_copy(update={"fingerprint": fingerprint(changed.model_dump(mode="json", exclude={"fingerprint"}))})
    assert not rag.verify_answer(changed)
    payload = document.model_dump()
    payload["pages"][0]["regions"][0]["text"] = "not the extracted cell"
    payload["fingerprint"] = fingerprint({k: v for k, v in payload.items() if k != "fingerprint"})
    with pytest.raises(ValueError):
        MultimodalDocument.model_validate(payload)


class AuthoredOCR:
    name, external_processing = "authored-local-control", False

    def __init__(self, confidence=0.95):
        self.confidence, self.calls = confidence, 0

    def recognize(self, png, *, width, height):
        self.calls += 1
        assert png.startswith(b"\x89PNG") and width * height > 0
        return (OCRWord(text="Atlas", box=Box(x0=0.1, y0=0.1, x1=0.2, y1=0.2),
                        confidence=self.confidence, line=(1, 1, 1)),)


def test_explicit_ocr_quality_gate_and_no_backend_on_native_pages(fixtures):
    backend = AuthoredOCR()
    assert load(fixtures["native"], ocr=backend).ready and backend.calls == 0
    assert load(fixtures["scan"], ocr=backend).ready and backend.calls == 1
    weak = load(fixtures["scan"], ocr=AuthoredOCR(0.2))
    assert weak.pages[0].reason == "low_confidence" and not weak.pages[0].regions
    backend.external_processing = True
    with pytest.raises(ExternalProcessingDenied):
        load(fixtures["native"], ocr=backend)


def test_backend_failure_is_content_free_and_identity_changes_are_rejected(fixtures):
    class Broken(AuthoredOCR):
        def recognize(self, *_args, **_kwargs):
            raise RuntimeError("PRIVATE INPUT PROVIDER ERROR")

    document = load(fixtures["scan"], ocr=Broken())
    assert document.pages[0].reason == "ocr_failed" and "PRIVATE" not in document.model_dump_json()

    class Mutating(AuthoredOCR):
        def recognize(self, *args, **kwargs):
            self.external_processing = True
            return super().recognize(*args, **kwargs)

    with pytest.raises(ExternalProcessingDenied):
        load(fixtures["scan"], ocr=Mutating())


def test_resource_limits_and_hostile_input_errors(fixtures, tmp_path):
    assert load(fixtures["native"], config=ExtractionConfig(max_regions=1)).pages[0].reason == "budget_exceeded"
    with pytest.raises(DocumentInputError):
        load(fixtures["mixed"], config=ExtractionConfig(max_pages=1))
    with pytest.raises(DocumentInputError):
        load(fixtures["scan"], config=ExtractionConfig(max_pixels=10_000))
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-PRIVATE-MALFORMED")
    with pytest.raises(DocumentInputError) as exc:
        load(bad)
    assert "PRIVATE" not in str(exc.value)
    with pytest.raises(DocumentInputError):
        load(fixtures["native"], config=ExtractionConfig(max_file_bytes=1000))
    with pytest.raises(DocumentInputError):
        load(fixtures["native"].with_suffix(".svg"))


@pytest.mark.parametrize("options", [{"language": "eng; echo secret"}, {"timeout_seconds": True},
                                     {"timeout_seconds": 0}, {"external_processing": True}])
def test_tesseract_configuration_is_not_a_shell(options):
    with pytest.raises(ValueError):
        TesseractOCR(**options)


@pytest.mark.parametrize("op,ids", [("exec", []), ("sum", []), ("difference", ["a" * 64]),
                                   ("sum", ["a" * 64, "a" * 64]), ("sum", ["bad"]), ("sum", "string")])
def test_invalid_tools_fail_before_execution(fixtures, op, ids):
    with pytest.raises(ValueError):
        MultimodalRAG(load(fixtures["native"])).calculate(op, ids)


@pytest.mark.integration
@pytest.mark.parametrize("name", ["scan", "scanned", "mixed"])
def test_real_local_tesseract_image_pdf_and_mixed_document(fixtures, name):
    if not shutil.which("tesseract"):
        pytest.skip("local Tesseract executable and English data are required")
    document = load(fixtures[name], ocr=TesseractOCR())
    assert document.ready, [(p.page, p.reason) for p in document.pages]
    assert document.pages[-1].method == "ocr"
    rag = MultimodalRAG(document)
    answer = rag.answer("Recovery time seconds", top_k=1)
    assert answer.status == "evidence_found" and "42 seconds" in answer.answer
    assert rag.verify_answer(answer, question="Recovery time seconds")
    assert answer.citations[0].page == (2 if name == "mixed" else 1)
    assert all(r.confidence is not None for r in document.pages[-1].regions)
    # OCR text is NOT promoted to a fabricated table grid or numeric tool input.
    assert rag.calculate("sum", [document.pages[-1].regions[0].region_id]).reason == "invalid_cells"


def test_tesseract_tsv_success_is_bounded_and_shell_free(monkeypatch):
    tsv = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
           "5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t95.5\tAtlas\n")

    def run(command, **kwargs):
        assert command == ["tesseract", "stdin", "stdout", "-l", "eng", "--psm", "6", "tsv"]
        assert kwargs["shell"] is False and kwargs["timeout"] == 30
        kwargs["stdout"].write(tsv.encode())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    words = TesseractOCR().recognize(b"\x89PNG\r\n\x1a\n", width=100, height=100)
    assert words[0].text == "Atlas" and words[0].confidence == 0.955
    assert words[0].box == Box(x0=0.1, y0=0.2, x1=0.4, y1=0.3)


@pytest.mark.parametrize("failure", ["timeout", "exit", "large_output", "malformed"])
def test_tesseract_subprocess_failures_do_not_echo_inputs(monkeypatch, failure):
    from agentforge_sdk import BackendError

    def run(_command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("PRIVATE COMMAND", 30, output=b"PRIVATE TEXT")
        if failure == "large_output":
            kwargs["stdout"].seek(2_000_001)
            kwargs["stdout"].write(b"x")
        if failure == "malformed":
            kwargs["stdout"].write(b"level\ttext\n5\tPRIVATE-TEXT\n")
        return SimpleNamespace(returncode=1 if failure == "exit" else 0)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(BackendError) as exc:
        TesseractOCR().recognize(b"\x89PNG\r\n\x1a\n", width=100, height=100)
    assert "PRIVATE" not in str(exc.value)


@pytest.mark.parametrize("change", [{"x1": 1.1}, {"x1": 0.0}, {"y0": float("nan")}, {"x0": True}])
def test_invalid_geometry_is_rejected(change):
    with pytest.raises(ValueError):
        Box.model_validate({"x0": 0.1, "y0": 0.1, "x1": 0.5, "y1": 0.5, **change})


def test_encrypted_cropped_and_multi_frame_inputs(fixtures, tmp_path):
    from PIL import Image
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    writer.add_page(PdfReader(fixtures["native"]).pages[0])
    writer.encrypt("PRIVATE-PASSWORD")
    encrypted = tmp_path / "encrypted.pdf"
    with encrypted.open("wb") as stream:
        writer.write(stream)
    with pytest.raises(DocumentInputError) as exc:
        load(encrypted)
    assert "PRIVATE" not in str(exc.value)
    writer = PdfWriter()
    page = PdfReader(fixtures["native"]).pages[0]
    page.cropbox.upper_right = (300, 400)
    writer.add_page(page)
    cropped = tmp_path / "cropped.pdf"
    with cropped.open("wb") as stream:
        writer.write(stream)
    assert load(cropped).pages[0].reason == "unsupported_geometry"
    with Image.open(fixtures["scan"]) as image:
        animated = tmp_path / "animated.png"
        second = image.copy().rotate(180)
        image.save(animated, save_all=True, append_images=[second], duration=100)
        second.close()
    with pytest.raises(DocumentInputError):
        load(animated)


def test_exif_orientation_is_held_and_raster_budget_precedes_ocr(fixtures, tmp_path):
    from PIL import Image

    with Image.open(fixtures["scan"]) as image:
        rotated = tmp_path / "rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6
        image.save(rotated, exif=exif)
    backend = AuthoredOCR()
    assert load(rotated, ocr=backend).pages[0].reason == "unsupported_geometry"
    assert load(fixtures["scanned"], ocr=backend, config=ExtractionConfig(max_pixels=10_000)).pages[0].reason == "budget_exceeded"
    assert backend.calls == 0


def test_page_and_manifest_contracts_reject_false_completeness(fixtures):
    document = load(fixtures["mixed"])
    for change in ({"pages": tuple(reversed(document.pages))},
                   {"pages": (document.pages[0], document.pages[0])},
                   {"config": ExtractionConfig(max_regions=1)}):
        payload = document.model_dump()
        payload.update({k: v.model_dump() if hasattr(v, "model_dump") else tuple(p.model_dump() for p in v)
                        for k, v in change.items()})
        payload["fingerprint"] = fingerprint({k: v for k, v in payload.items() if k != "fingerprint"})
        with pytest.raises(ValueError):
            MultimodalDocument.model_validate(payload)


@pytest.mark.integration
def test_actual_jpeg_ocr(fixtures, tmp_path):
    from PIL import Image

    if not shutil.which("tesseract"):
        pytest.skip("real local Tesseract required")
    path = tmp_path / "scan.jpg"
    with Image.open(fixtures["scan"]) as image:
        image.save(path, quality=90)
    document = load(path, ocr=TesseractOCR())
    result = MultimodalRAG(document).answer("Recovery seconds", top_k=1)
    assert document.media_type == "image/jpeg" and "42 seconds" in result.answer


def test_sdk_and_evaluation_namespace_import_without_optional_platform_dependencies(tmp_path):
    import sys
    from pathlib import Path

    script = '''import sys
sys.path.insert(0, sys.argv[1])
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'langchain_core', 'langgraph', 'agent', 'code_agent', 'pdfplumber', 'pypdfium2', 'PIL'}:
            raise AssertionError('eager optional dependency')
sys.meta_path.insert(0, Block())
import agentforge_sdk, evals
from evals import multimodal_evaluation
assert agentforge_sdk.__version__ == '0.10.0'
assert not list(__import__('pathlib').Path.cwd().iterdir())
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(Path(__file__).resolve().parent.parent)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
