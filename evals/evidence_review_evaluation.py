"""Actual local OCR plus authored review decisions; no claim that a human approved the fixture."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from agentforge_sdk import (
    MultimodalRAG,
    ReviewAuthority,
    ReviewedEvidence,
    ReviewedRAG,
    ReviewError,
    TesseractOCR,
    apply_review,
    load_multimodal,
    prepare_review,
)
from agentforge_sdk.models import fingerprint
from evals.evidence_review_fixtures import create_review_fixture


def evaluate(directory: Path) -> dict:
    directory = directory.absolute()
    root = (Path.cwd() / "data/evaluations").resolve()
    if directory.resolve() == root or not directory.resolve().is_relative_to(root):
        raise ValueError("review evidence must be below data/evaluations/")
    for part in (directory, *directory.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("review output links are unsupported")
    if not shutil.which("tesseract"):
        raise RuntimeError("review integration control requires actual local Tesseract")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / ".gitignore").write_text("*\n", encoding="utf-8")
    path, held, correction = create_review_fixture(directory / "fixtures")
    original = load_multimodal(path, document_id="review-control", version="r1", ocr=TesseractOCR())
    correction = correction.model_copy(update={"original_page_sha256": fingerprint(original.pages[0].model_dump(mode="json"))})
    now = [1000]
    # Public authored key for controls ONLY, never a production reviewer credential.
    authority = ReviewAuthority({"authored-reviewer": bytes(range(32))}, clock=lambda: now[0])
    draft = prepare_review(original, [correction], rationale="Authored review of the complete visible scan")
    approval = authority.decide(draft, reviewer_id="authored-reviewer", decision="approve", lifetime_seconds=60)
    evidence = apply_review(original, draft, approval, authority=authority)
    rag = ReviewedRAG(ReviewedEvidence.model_validate_json(evidence.model_dump_json()), authority=authority, source_path=path)
    answer = rag.answer("Requests After", top_k=1)
    cells = {(r.row, r.column): r.region_id for r in evidence.document.pages[0].regions if r.kind == "table_cell"}
    calculation = rag.calculate("percent_change", [cells[1, 1], cells[1, 2]])
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    check("missing_ocr_held", held.pages[0].reason == "ocr_required")
    check("actual_ocr_does_not_invent_table_grid", original.pages[0].method == "ocr"
          and not any(r.kind == "table_cell" for r in original.pages[0].regions))
    check("machine_extraction_unchanged", evidence.original == original)
    check("reviewed_method_not_native_or_high_confidence", evidence.document.pages[0].method == "reviewed"
          and all(r.confidence is None for r in evidence.document.pages[0].regions))
    check("manual_grid_lookup", [c.spatial.quote for c in answer.citations] == ["Requests", "After", "125"])
    check("reviewer_origin_on_each_citation", all(c.origin == "reviewer_supplied" and c.reviewer_id == "authored-reviewer" for c in answer.citations))
    check("answer_provenance_verified", rag.verify_answer(answer, question="Requests After"))
    check("reviewed_cell_calculation", calculation.result.value == "25" and rag.verify_calculation(calculation))
    check("tampered_origin_rejected", not rag.verify_citation(answer.citations[0].model_copy(update={"origin": "machine_extracted"})))
    foreign = ReviewAuthority({"authored-reviewer": b"z" * 32}, clock=lambda: now[0])
    check("wrong_authority_key_rejected", not foreign.permits(approval, draft))
    rejected = authority.decide(draft, reviewer_id="authored-reviewer", decision="reject")
    check("rejected_decision_not_authorized", not authority.permits(rejected, draft))
    authority.revoke(approval)
    check("revocation_invalidates_existing_answer", not rag.verify_answer(answer))
    check("revocation_invalidates_existing_calculation", not rag.verify_calculation(calculation))
    replacement = authority.decide(draft, reviewer_id="authored-reviewer", decision="approve", lifetime_seconds=30)
    fresh = ReviewedRAG(apply_review(original, draft, replacement, authority=authority), authority=authority, source_path=path)
    now[0] = 1030
    try:
        fresh.answer("Requests")
    except ReviewError:
        expired = True
    else:
        expired = False
    check("expiration_blocks_new_queries", expired)
    now[0] = 1000
    check("clock_rollback_does_not_revive_approval", not authority.permits(replacement, draft))
    check("machine_only_cannot_calculate_unextracted_grid", MultimodalRAG(original).calculate("sum", ["f" * 64]).status == "abstained")
    report = {"protocol": "evidence-review-controls-v1", "evidence_kind": "actual_ocr_authored_manual_annotations_and_approvals",
              "total": len(checks), "passed": sum(c["passed"] for c in checks), "checks": checks,
              "actual_ocr_page_status": original.pages[0].status, "actual_ocr_page_reason": original.pages[0].reason,
              "original_fingerprint": original.fingerprint, "reviewed_fingerprint": evidence.document.fingerprint,
              "review_bundle_fingerprint": evidence.fingerprint, "live_provider_calls": 0,
              "human_review_performed": False, "real_world_accuracy_measured": False,
              "scope": "SDK control and provenance behavior, not model-learned scanned table extraction"}
    report["fingerprint"] = fingerprint(report)
    (directory / "bundle.json").write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (directory / "answer.json").write_text(answer.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (directory / "calculation.json").write_text(calculation.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = evaluate(parser.parse_args().output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    if result["total"] != result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
