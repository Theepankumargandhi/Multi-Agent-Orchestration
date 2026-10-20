"""Authored manifest controls; real PDF extraction is exercised separately."""

from __future__ import annotations

import hashlib

import pytest

from agentforge_sdk import (
    Box,
    ChangeConfig,
    DocumentChanges,
    DocumentRevision,
    EvidencePage,
    ExtractionConfig,
    MultimodalDocument,
    MultimodalRAG,
    Region,
    compare_documents,
)
from agentforge_sdk.models import fingerprint
from agentforge_sdk.spatial import region_fingerprint


def document(*, version="v1", after="125", currency="USD", reverse=False, cover=False,
             duplicate=False, held=False, title="Atlas report", extractor="authored-control-v1"):
    file_hash = hashlib.sha256(version.encode()).hexdigest()
    pages = []

    def make_page(number, texts, table=()):
        regions = []
        for index, text in enumerate(texts):
            r = Region(region_id="0" * 64, kind="text", text=text,
                       box=Box(x0=0.1, y0=0.05 + index * 0.08, x1=0.9, y1=0.1 + index * 0.08))
            regions.append(r.model_copy(update={"region_id": region_fingerprint(file_hash, number, r)}))
        for row, values in enumerate(table):
            for col, text in enumerate(values):
                r = Region(region_id="0" * 64, kind="table_cell", text=text, table_id=f"p{number}:t0", row=row, column=col,
                           box=Box(x0=0.1 + col * 0.25, y0=0.3 + row * 0.1, x1=0.35 + col * 0.25, y1=0.4 + row * 0.1))
                regions.append(r.model_copy(update={"region_id": region_fingerprint(file_hash, number, r)}))
        return EvidencePage(page=number, width=612.0, height=792.0, method="native", status="ready", reason="extracted", regions=tuple(regions))

    if cover:
        pages.append(make_page(1, ["New cover page"]))
    rows = [("Requests", "100", after), ("Revenue", "USD 120.50", f"{currency} 150.75")]
    if reverse:
        rows.reverse()
    if duplicate:
        rows[1] = ("Requests", "80", "90")
    pages.append(make_page(len(pages) + 1, [title, "Atlas p95 latency is 180 ms."], [("Metric", "Before", "After"), *rows]))
    if held:
        pages[-1] = EvidencePage(page=len(pages), width=612.0, height=792.0, method="none", status="held", reason="ocr_required", regions=())
    payload = dict(protocol="agentforge-spatial-evidence-v1", document_id="atlas", source="atlas.pdf", version=version,
                   file_sha256=file_hash, media_type="application/pdf", config=ExtractionConfig().model_dump(),
                   extractor=extractor, ocr_backend="none", pages=tuple(p.model_dump() for p in pages))
    return MultimodalDocument(**payload, fingerprint=fingerprint(payload))


def row_cells(doc, label):
    page = doc.pages[-1]
    row = next(r.row for r in page.regions if r.kind == "table_cell" and r.column == 0 and r.text == label)
    return [next(r.region_id for r in page.regions if r.row == row and r.column == col) for col in (1, 2)]


def test_changed_numeric_cells_follow_page_insertion_and_reordered_rows():
    before, after = document(), document(version="v2", after="150", cover=True, reverse=True)
    old, new = MultimodalRAG(before), MultimodalRAG(after)
    revision = DocumentRevision(old, new)
    mapped = next(p for p in revision.changes.pages if p.before_page == 1)
    assert mapped.after_page == 2 and mapped.basis == "unique_anchor"
    calc = old.calculate("percent_change", row_cells(before, "Requests"))
    replay = revision.recalculate(calc)
    assert calc.value == "25" and replay.result.value == "50" and replay.value_changed
    assert replay.result.citations[0].page == 2 and new.verify_calculation(replay.result)
    assert revision.verify_recalculation(replay, calc)
    assert revision.assess_calculation(calc).status == "dependencies_changed"
    assert any("numeric" in c.flags and "page" in c.flags for c in revision.changes.regions)


def test_unchanged_content_requires_new_receipts_for_a_new_version():
    before, after = document(), document(version="v2", cover=True, reverse=True)
    old, new = MultimodalRAG(before), MultimodalRAG(after)
    answer = old.answer("Atlas p95 latency", top_k=1)
    revision = DocumentRevision(old, new)
    impact = revision.assess_answer(answer, question="Atlas p95 latency")
    assert impact.status == "dependencies_unchanged" and impact.requires_reissue
    assert revision.verify_impact(impact, answer, question="Atlas p95 latency")
    assert not new.verify_answer(answer)
    assert new.verify_answer(new.answer("Atlas p95 latency", top_k=1))


def test_unit_change_replays_the_tool_but_does_not_invent_currency_conversion():
    before, after = document(), document(version="v2", currency="EUR")
    old, new = MultimodalRAG(before), MultimodalRAG(after)
    calc = old.calculate("sum", row_cells(before, "Revenue"))
    revision = DocumentRevision(old, new)
    assert any("unit" in c.flags for c in revision.changes.regions)
    replay = revision.recalculate(calc)
    assert replay.status == "abstained" and replay.reason == "target_tool_abstained"
    assert replay.result.reason == "ambiguous_number" and replay.value_changed is None
    assert revision.verify_recalculation(replay, calc)


def test_numeric_formatting_change_is_not_reported_as_a_changed_numeric_value():
    before, after = document(), document(version="v2", after="125.0")
    old, new = MultimodalRAG(before), MultimodalRAG(after)
    revision = DocumentRevision(old, new)
    modified = [c for c in revision.changes.regions if "text" in c.flags]
    assert modified and all("numeric" not in c.flags for c in modified)
    replay = revision.recalculate(old.calculate("percent_change", row_cells(before, "Requests")))
    assert replay.status == "recalculated" and replay.value_changed is False


def test_column_reordering_preserves_operand_order_in_replay():
    before, after = document(), document(version="v2")
    payload = after.model_dump()
    for raw in payload["pages"][0]["regions"]:
        if raw["kind"] == "table_cell" and raw["column"]:
            column = 3 - raw["column"]
            raw["column"] = column
            raw["box"].update(x0=0.1 + column * 0.25, x1=0.35 + column * 0.25)
            raw["region_id"] = region_fingerprint(after.file_sha256, 1, Region.model_validate(raw))
    payload["fingerprint"] = fingerprint({k: v for k, v in payload.items() if k != "fingerprint"})
    after = MultimodalDocument.model_validate(payload)
    old, new = MultimodalRAG(before), MultimodalRAG(after)
    replay = DocumentRevision(old, new).recalculate(old.calculate("percent_change", row_cells(before, "Requests")))
    assert replay.result.value == "25" and [c.column for c in replay.result.citations] == [2, 1]


def test_duplicate_literals_are_not_aligned_by_nearest_box():
    def repeated(version):
        doc = document(version=version)
        payload = doc.model_dump()
        for region in payload["pages"][0]["regions"]:
            if region["kind"] == "text":
                region["text"] = "Repeated heading"
                region["region_id"] = region_fingerprint(doc.file_sha256, 1, Region.model_validate(region))
        payload["fingerprint"] = fingerprint({k: v for k, v in payload.items() if k != "fingerprint"})
        return MultimodalDocument.model_validate(payload)

    old, new = MultimodalRAG(repeated("v1")), MultimodalRAG(repeated("v2"))
    revision = DocumentRevision(old, new)
    assert revision.assess_answer(old.answer("Repeated heading", top_k=1)).status == "needs_review"


def test_duplicate_labels_never_choose_a_row_by_position():
    before, after = document(), document(version="v2", duplicate=True)
    old, new = MultimodalRAG(before), MultimodalRAG(after)
    revision = DocumentRevision(old, new)
    calc = old.calculate("sum", row_cells(before, "Requests"))
    assert revision.assess_calculation(calc).status == "needs_review"
    assert revision.recalculate(calc).reason == "dependency_alignment_unavailable"
    assert any(c.relation == "ambiguous" for c in revision.changes.regions)


def test_incomplete_target_is_not_a_clean_no_change_result():
    old = MultimodalRAG(document())
    revision = DocumentRevision(old, MultimodalRAG(document(version="v2", held=True)))
    assert revision.changes.status == "held" and revision.changes.reason == "incomplete_extraction"
    calc = old.calculate("sum", row_cells(old.document, "Requests"))
    assert revision.recalculate(calc).reason == "target_incomplete"


def test_same_manifest_and_roundtrip_preserve_all_region_bindings():
    doc = document()
    revision = DocumentRevision(MultimodalRAG(doc), MultimodalRAG(doc))
    changes = DocumentChanges.model_validate_json(revision.changes.model_dump_json())
    assert revision.verify_changes(changes)
    assert all(c.relation == "unchanged" for c in changes.regions)
    text = {r.region_id: r.text for p in doc.pages for r in p.regions}
    assert all(c.before.text_sha256 == hashlib.sha256(text[c.before.region_id].encode("utf-8")).hexdigest()
               for c in changes.regions)
    answer = revision._before.answer("Atlas latency", top_k=1)
    assert not revision.assess_answer(answer).requires_reissue
    assert not revision.verify_changes(changes.model_copy(update={"metadata_changes": ("fake",)}))


@pytest.mark.parametrize("kind", ["quote", "fingerprint", "foreign"])
def test_invalid_old_receipts_do_not_produce_a_recalculation(kind):
    old, new = MultimodalRAG(document()), MultimodalRAG(document(version="v2"))
    revision = DocumentRevision(old, new)
    calc = old.calculate("sum", row_cells(old.document, "Requests"))
    if kind == "fingerprint":
        calc = calc.model_copy(update={"fingerprint": "PRIVATE-INPUT"})
    elif kind == "quote":
        calc = calc.model_copy(update={"value": "9999"})
    else:
        calc = new.calculate("sum", row_cells(new.document, "Requests"))
    replay = revision.recalculate(calc)
    assert replay.reason == "source_receipt_unverifiable" and replay.result is None
    assert replay.impact.requires_reissue
    assert "PRIVATE" not in replay.model_dump_json()


def test_extraction_profile_drift_holds_replay_even_when_literal_values_match():
    before, after = document(), document(version="v2", extractor="other-parser")
    old, new = MultimodalRAG(before), MultimodalRAG(after)
    revision = DocumentRevision(old, new)
    calc = old.calculate("sum", row_cells(before, "Requests"))
    assert revision.recalculate(calc).reason == "extraction_profile_changed"


def test_comparison_budget_fails_closed_before_unbounded_spatial_pairing():
    before, after = document(), document(version="v2", title="Changed report")
    # Make both text lines change while retaining table-label anchors for the page.
    payload = after.model_dump()
    region = payload["pages"][0]["regions"][1]
    region["text"] = "Atlas p95 latency is 190 ms."
    draft = Region.model_validate({**region, "region_id": "0" * 64})
    region["region_id"] = region_fingerprint(after.file_sha256, 1, draft)
    payload["fingerprint"] = fingerprint({k: v for k, v in payload.items() if k != "fingerprint"})
    after = MultimodalDocument.model_validate(payload)
    diff = compare_documents(before, after, config=ChangeConfig(max_spatial_comparisons=1))
    assert diff.status == "held" and diff.reason == "comparison_budget" and diff.spatial_comparisons == 1


def test_comparison_rejects_unrelated_logical_document_ids():
    other = document().model_dump()
    other["document_id"] = "someone-else"
    other["fingerprint"] = fingerprint({k: v for k, v in other.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="logical"):
        compare_documents(document(), MultimodalDocument.model_validate(other))


def test_impact_tampering_is_rejected_even_if_rehashed():
    old, new = MultimodalRAG(document()), MultimodalRAG(document(version="v2"))
    revision = DocumentRevision(old, new)
    answer = old.answer("Atlas latency", top_k=1)
    impact = revision.assess_answer(answer)
    forged = impact.model_copy(update={"requires_reissue": False})
    forged = forged.model_copy(update={"fingerprint": fingerprint(forged.model_dump(mode="json", exclude={"fingerprint"}))})
    assert not revision.verify_impact(forged, answer)


def test_changed_uncited_content_does_not_claim_answer_reuse_is_safe():
    before, after = document(), document(version="v2", after="150")
    old, new = MultimodalRAG(before), MultimodalRAG(after)
    answer = old.answer("Atlas p95 latency", top_k=1)
    impact = DocumentRevision(old, new).assess_answer(answer)
    assert impact.status == "dependencies_unchanged" and impact.requires_reissue
    assert "rerun_retrieval_for_changed_corpus" in impact.reasons


def test_literal_line_change_uses_unique_geometry_not_a_language_model():
    before, after = document(), document(version="v2", title="Updated Atlas report")
    changes = compare_documents(before, after)
    changed = [c for c in changes.regions if c.relation == "modified"]
    assert len(changed) == 1 and changed[0].basis == "spatial" and changed[0].flags == ("text",)


def test_duplicate_pages_are_not_paired_by_order():
    def duplicated(version):
        payload = document(version=version).model_dump()
        second = payload["pages"][0].copy()
        second["page"] = 2
        second["regions"] = tuple({**r, "region_id": region_fingerprint(payload["file_sha256"], 2, Region.model_validate(r))}
                                  for r in second["regions"])
        payload["pages"] = (*payload["pages"], second)
        payload["fingerprint"] = fingerprint({k: v for k, v in payload.items() if k != "fingerprint"})
        return MultimodalDocument.model_validate(payload)

    changes = compare_documents(duplicated("v1"), duplicated("v2"))
    assert all(p.basis == "unmatched" for p in changes.pages)
    assert all(c.relation == "unmatched" for c in changes.regions)


def test_no_positive_dependencies_force_a_fresh_retrieval():
    old = MultimodalRAG(document())
    revision = DocumentRevision(old, MultimodalRAG(document(version="v2")))
    impact = revision.assess_answer(old.answer("xylophonium"))
    assert impact.status == "needs_review" and "no_positive_dependencies" in impact.reasons


@pytest.mark.parametrize("change", [{"max_spatial_comparisons": True}, {"max_spatial_comparisons": 0},
                                    {"min_line_overlap": float("nan")}, {"min_line_overlap": 0.1}])
def test_diff_limits_are_strict_and_finite(change):
    with pytest.raises(ValueError):
        ChangeConfig(**change)


def test_review_approvals_are_not_carried_forward_and_revocation_blocks_replay(tmp_path):
    pytest.importorskip("pdfplumber")
    from agentforge_sdk import ReviewAuthority, ReviewedRAG, apply_review, prepare_review
    from evals.evidence_review_fixtures import create_review_fixture

    path, original, correction = create_review_fixture(tmp_path)
    authority = ReviewAuthority({"authored-a": b"a" * 32, "authored-b": b"b" * 32}, clock=lambda: 1000)
    draft = prepare_review(original, [correction], rationale="Authored version control")
    first = authority.decide(draft, reviewer_id="authored-a", decision="approve")
    second = authority.decide(draft, reviewer_id="authored-b", decision="approve")
    old = ReviewedRAG(apply_review(original, draft, first, authority=authority), authority=authority, source_path=path)
    new = ReviewedRAG(apply_review(original, draft, second, authority=authority), authority=authority, source_path=path)
    revision = DocumentRevision(old, new)
    calc = old.calculate("percent_change", row_cells(old.evidence.document, "Requests"))
    replay = revision.recalculate(calc)
    assert replay.impact.requires_reissue and replay.result.approval_fingerprint == second.fingerprint
    assert new.verify_calculation(replay.result)
    authority.revoke(second)
    assert revision.recalculate(calc).reason == "target_not_authorized"
    assert not revision.verify_recalculation(replay, calc)
    assert "PRIVATE" not in revision.assess_calculation(calc).model_dump_json()
    authority.revoke(first)
    assert revision.recalculate(calc).reason == "source_receipt_unverifiable"


def test_actual_pdf_revisions_and_replayed_calculation(tmp_path):
    pytest.importorskip("pdfplumber")
    pytest.importorskip("reportlab")
    from agentforge_sdk import load_multimodal
    from evals.document_change_fixtures import create_change_fixtures

    files = create_change_fixtures(tmp_path)
    before = load_multimodal(files["v1"], document_id="atlas", source="atlas.pdf", version="v1")
    after = load_multimodal(files["v2"], document_id="atlas", source="atlas.pdf", version="v2")
    old, new = MultimodalRAG(before), MultimodalRAG(after)
    replay = DocumentRevision(old, new).recalculate(old.calculate("percent_change", row_cells(before, "Requests")))
    assert replay.status == "recalculated" and replay.result.value == "50"
    assert all(c.page == 2 for c in replay.result.citations)
