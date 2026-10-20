"""Real local OCR on authored documents; implementation controls, not a quality benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path

from agentforge_sdk import (
    Box,
    MultimodalRAG,
    OCRWord,
    TesseractOCR,
    load_multimodal,
)
from agentforge_sdk.models import fingerprint
from evals.multimodal_fixtures import create_fixtures


def evaluate(directory: Path) -> dict:
    directory = directory.absolute()
    root = (Path.cwd() / "data/evaluations").resolve()
    if directory.resolve() == root or not directory.resolve().is_relative_to(root):
        raise ValueError("multimodal evidence must be below data/evaluations/")
    for part in (directory, *directory.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("evaluation output links are unsupported")
    if not shutil.which("tesseract"):
        raise RuntimeError("this verification requires real local Tesseract and English data; no mock fallback")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / ".gitignore").write_text("*\n", encoding="utf-8")
    files = create_fixtures(directory / "fixtures")
    checks, receipts = [], {}

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    native = load_multimodal(files["native"], document_id="atlas", version="r1")
    rag = MultimodalRAG(native)
    answer = rag.answer("Atlas p95 latency", top_k=1)
    check("native_literal_answer", "180 ms" in answer.answer and rag.verify_answer(answer, question="Atlas p95 latency"))
    cells = {(r.row, r.column): r.region_id for r in native.pages[0].regions if r.kind == "table_cell"}
    check("native_table_grid", len(cells) == 15)
    lookup = rag.answer("Revenue Q2", top_k=1)
    check("table_value_and_separate_axis_citations", [c.quote for c in lookup.citations] == ["Revenue", "Q2", "USD 150.75"]
          and rag.verify_answer(lookup, question="Revenue Q2"))
    total = rag.calculate("sum", [cells[1, 1], cells[1, 2]])
    change = rag.calculate("percent_change", [cells[2, 1], cells[2, 2]])
    check("decimal_currency_sum", total.value == "271.25" and total.unit == "USD" and rag.verify_calculation(total))
    check("percent_change", change.value == "25" and change.unit == "%" and rag.verify_calculation(change))
    check("locale_ambiguity_held", rag.calculate("sum", [cells[4, 1]]).reason == "ambiguous_number")
    check("unit_mismatch_held", rag.calculate("sum", [cells[1, 1], cells[2, 1]]).reason == "ambiguous_number")
    check("divide_by_zero_held", rag.calculate("ratio", [cells[2, 1], cells[3, 1]]).reason == "zero_denominator")
    check("calculation_tampering_rejected", not rag.verify_calculation(total.model_copy(update={"value": "999"})))
    check("geometry_tampering_rejected", not rag.verify_citation(answer.citations[0].model_copy(update={"page": 2})))
    check("no_evidence_abstention", rag.answer("xylophonium").status == "abstained")
    receipts["native"] = {"document": native.model_dump(mode="json"), "answer": answer.model_dump(mode="json"),
                          "sum": total.model_dump(mode="json"), "percent_change": change.model_dump(mode="json"),
                          "table_lookup": lookup.model_dump(mode="json")}
    for name in ("scan", "scanned", "mixed"):
        document = load_multimodal(files[name], document_id=f"atlas-{name}", version="r1", ocr=TesseractOCR())
        engine = MultimodalRAG(document)
        result = engine.answer("Recovery time seconds", top_k=1)
        check(f"real_tesseract_{name}", document.ready and "42 seconds" in result.answer
              and engine.verify_answer(result, question="Recovery time seconds"))
        check(f"physical_page_{name}", bool(result.citations) and result.citations[0].page == (2 if name == "mixed" else 1))
        check(f"ocr_not_fabricated_grid_{name}", all(r.kind == "text" for r in document.pages[-1].regions))
        receipts[name] = {"document": document.model_dump(mode="json"), "answer": result.model_dump(mode="json")}
    for name, reason in (("mixed", "ocr_required"), ("merged", "ambiguous_table"),
                         ("rotated", "unsupported_geometry"), ("blank", "empty_page")):
        document = load_multimodal(files[name], document_id="held")
        result = MultimodalRAG(document).answer("Atlas latency")
        check(f"held_{name}", not document.ready and document.pages[-1].reason == reason
              and result.reason == "incomplete_extraction" and not result.citations)

    class LowConfidence:
        name, external_processing = "authored-low-confidence", False

        def recognize(self, *_args, **_kwargs):
            return (OCRWord(text="guess", box=Box(x0=0.1, y0=0.1, x1=0.5, y1=0.5),
                            confidence=0.1, line=(1, 1, 1)),)

    held = load_multimodal(files["scan"], document_id="weak", ocr=LowConfidence())
    check("authored_low_confidence_held", held.pages[0].reason == "low_confidence" and not held.ready)
    engine_version = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, check=True,
                                    timeout=10).stdout.splitlines()[0]
    report = {"protocol": "multimodal-sdk-controls-v1", "evidence_kind": "authored_documents_actual_local_ocr",
              "total": len(checks), "passed": sum(c["passed"] for c in checks), "checks": checks,
              "python": platform.python_version(), "ocr_engine": engine_version,
              "dependencies": {name: version(name) for name in ("pdfplumber", "pypdfium2", "Pillow", "reportlab")},
              "document_fingerprints": {name: receipt["document"]["fingerprint"] for name, receipt in receipts.items()},
              "live_provider_calls": 0, "real_world_quality_measured": False,
              "scope": "native ruled tables and raster text; no chart understanding or scanned-table reconstruction"}
    report["fingerprint"] = fingerprint(report)
    (directory / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = evaluate(parser.parse_args().output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
