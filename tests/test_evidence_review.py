from __future__ import annotations

import hashlib
import json

import pytest

from agentforge_sdk import (
    Box,
    PageCorrection,
    ReviewApproval,
    ReviewAuthority,
    ReviewedEvidence,
    ReviewedRAG,
    ReviewedRegionInput,
    ReviewError,
    apply_review,
    prepare_review,
)
from agentforge_sdk.models import fingerprint
from evals.evidence_review_fixtures import create_review_fixture


@pytest.fixture
def review_case(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("pdfplumber")
    path, original, correction = create_review_fixture(tmp_path)
    now = [1000]
    authority = ReviewAuthority({"authored-reviewer": bytes(range(32))}, clock=lambda: now[0])
    draft = prepare_review(original, [correction], rationale="Authored complete-page transcription control")
    approval = authority.decide(draft, reviewer_id="authored-reviewer", decision="approve", lifetime_seconds=60)
    evidence = apply_review(original, draft, approval, authority=authority)
    return path, original, correction, now, authority, draft, approval, evidence


def cell(evidence, row, column):
    return next(r.region_id for r in evidence.document.pages[0].regions
                if r.kind == "table_cell" and r.row == row and r.column == column)


def reseal(model):
    return model.model_copy(update={"fingerprint": fingerprint(model.model_dump(mode="json", exclude={"fingerprint"}))})


def test_held_scan_can_be_explicitly_reviewed_without_rewriting_machine_evidence(review_case):
    path, original, _correction, _now, authority, draft, approval, evidence = review_case
    assert not original.ready and original.pages[0].reason == "ocr_required"
    assert original.pages[0].regions == () and evidence.original == original
    assert evidence.document.ready and evidence.document.pages[0].method == "reviewed"
    assert all(r.confidence is None for r in evidence.document.pages[0].regions)
    assert evidence.document.fingerprint != original.fingerprint
    assert evidence.document.file_sha256 == original.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert authority.permits(approval, draft)
    rag = ReviewedRAG(ReviewedEvidence.model_validate_json(evidence.model_dump_json()), authority=authority, source_path=path)
    answer = rag.answer("Requests After", top_k=1)
    assert [c.spatial.quote for c in answer.citations] == ["Requests", "After", "125"]
    assert all(c.origin == "reviewer_supplied" and c.reviewer_id == "authored-reviewer" for c in answer.citations)
    assert all(rag.verify_citation(c) for c in answer.citations)
    assert rag.verify_answer(answer, question="Requests After")
    assert not rag.verify_answer(answer, question="other")
    assert rag.verify_answer(rag.answer("xylophonium"))


def test_reviewed_scanned_table_arithmetic_has_replayable_review_lineage(review_case):
    path, _original, _correction, _now, authority, _draft, approval, evidence = review_case
    rag = ReviewedRAG(evidence, authority=authority, source_path=path)
    result = rag.calculate("percent_change", [cell(evidence, 1, 1), cell(evidence, 1, 2)])
    assert result.result.status == "calculated" and result.result.value == "25"
    assert result.approval_fingerprint == approval.fingerprint and rag.verify_calculation(result)
    assert all(c.origin == "reviewer_supplied" for c in result.citations)
    assert not rag.verify_calculation(reseal(result.model_copy(update={"original_document_fingerprint": "f" * 64})))
    assert "signature" not in repr(approval)
    assert "authored-reviewer" not in repr(authority)


def test_reviewed_manifest_cannot_accidentally_bypass_the_authorized_wrapper(review_case):
    from agentforge_sdk import MultimodalRAG

    evidence = review_case[-1]
    with pytest.raises(ValueError, match="ReviewedRAG"):
        MultimodalRAG(evidence.document)


@pytest.mark.parametrize("decision", ["reject", "wrong_key", "wrong_reviewer", "rehashed_signature", "changed_draft"])
def test_missing_or_forged_authority_cannot_apply_review(review_case, decision):
    _path, original, _correction, _now, authority, draft, approval, _evidence = review_case
    if decision == "reject":
        approval = authority.decide(draft, reviewer_id="authored-reviewer", decision="reject")
    elif decision == "wrong_key":
        authority = ReviewAuthority({"authored-reviewer": b"z" * 32}, clock=lambda: 1000)
    elif decision == "wrong_reviewer":
        approval = reseal(approval.model_copy(update={"reviewer_id": "someone-else"}))
    elif decision == "rehashed_signature":
        approval = reseal(approval.model_copy(update={"signature": "f" * 64}))
    else:
        draft = reseal(draft.model_copy(update={"rationale": "Changed after signing"}))
    with pytest.raises(ReviewError):
        apply_review(original, draft, approval, authority=authority)


@pytest.mark.parametrize("boundary", [999, 1060, 2000])
def test_future_and_expired_approval_are_not_usable(review_case, boundary):
    path, _original, _correction, now, authority, _draft, _approval, evidence = review_case
    rag = ReviewedRAG(evidence, authority=authority, source_path=path)
    answer = rag.answer("Requests After", top_k=1)
    now[0] = boundary
    assert not rag.verify_answer(answer) and not rag.verify_citation(answer.citations[0])
    with pytest.raises(ReviewError):
        rag.answer("Requests After")
    with pytest.raises(ReviewError):
        rag.calculate("sum", [cell(evidence, 1, 1)])


def test_revocation_affects_existing_instances_and_old_receipt_verification(review_case):
    path, _original, _correction, _now, authority, _draft, approval, evidence = review_case
    rag = ReviewedRAG(evidence, authority=authority, source_path=path)
    result = rag.calculate("sum", [cell(evidence, 1, 1)])
    authority.revoke(approval)
    assert not rag.verify_calculation(result)
    with pytest.raises(ReviewError):
        rag.answer("Requests")
    with pytest.raises(ReviewError):
        ReviewedRAG(evidence, authority=authority, source_path=path)


def test_source_replacement_or_removal_invalidates_an_existing_review(review_case):
    path, _original, _correction, _now, authority, _draft, _approval, evidence = review_case
    rag = ReviewedRAG(evidence, authority=authority, source_path=path)
    result = rag.answer("Requests After", top_k=1)
    path.write_bytes(b"PRIVATE-REPLACEMENT")
    assert not rag.verify_answer(result)
    with pytest.raises(ReviewError) as exc:
        rag.answer("Requests")
    assert "PRIVATE" not in str(exc.value)
    path.unlink()
    with pytest.raises(ReviewError):
        ReviewedRAG(evidence, authority=authority, source_path=path)


def test_rehashed_derived_text_or_provenance_tampering_is_rejected(review_case):
    path, _original, _correction, _now, authority, _draft, _approval, evidence = review_case
    rag = ReviewedRAG(evidence, authority=authority, source_path=path)
    result = rag.answer("Requests After", top_k=1)
    bad = result.citations[0].model_copy(update={"origin": "machine_extracted", "reviewer_id": None})
    assert not rag.verify_citation(bad)
    forged = reseal(result.model_copy(update={"citations": (bad, *result.citations[1:])}))
    assert not rag.verify_answer(forged)
    with pytest.raises(ReviewError):
        ReviewedRAG(reseal(evidence.model_copy(update={"document": evidence.original})), authority=authority, source_path=path)


@pytest.mark.parametrize("change", [{"page": 2}, {"original_page_sha256": "f" * 64}])
def test_stale_or_unknown_page_proposals_are_rejected(review_case, change):
    _path, original, correction, *_rest = review_case
    with pytest.raises(ReviewError):
        prepare_review(original, [correction.model_copy(update=change)], rationale="Authored invalid control")


@pytest.mark.parametrize("fault", ["missing", "duplicate", "overlap", "column_geometry"])
def test_manual_grids_are_complete_and_spatially_consistent(review_case, fault):
    _path, _original, correction, *_rest = review_case
    regions = list(correction.regions)
    if fault == "missing":
        regions.pop()
    elif fault == "duplicate":
        regions.append(regions[-1])
    elif fault == "overlap":
        for index, region in enumerate(regions):
            if region.column == 1:
                regions[index] = region.model_copy(update={"box": region.box.model_copy(update={"x0": 0.1})})
    else:
        regions[-1] = regions[-1].model_copy(update={"box": regions[-1].box.model_copy(update={"x0": 0.67})})
    with pytest.raises(ValueError):
        PageCorrection(**{**correction.model_dump(), "regions": tuple(r.model_dump() for r in regions)})


@pytest.mark.parametrize("keys", [{}, {"reviewer": b"short"}, {"bad actor": b"x" * 32}, {"reviewer": "secret"}])
def test_authority_keys_are_explicit_bounded_and_not_in_errors(keys):
    with pytest.raises(ValueError) as exc:
        ReviewAuthority(keys)
    assert "secret" not in str(exc.value)


def test_drafts_require_attestation_and_reject_unsigned_or_modified_records(review_case):
    _path, original, correction, _now, _authority, draft, approval, _evidence = review_case
    data = correction.model_dump()
    data.pop("attestation")
    with pytest.raises(ValueError):
        PageCorrection.model_validate(data)
    with pytest.raises(ValueError):
        prepare_review(original, [correction, correction], rationale="duplicate pages")
    with pytest.raises(ValueError):
        ReviewApproval.model_validate({**approval.model_dump(), "expires_at": 9999})
    assert "regions" not in repr(correction)
    assert "corrections" not in repr(draft)
    assert json.loads(draft.model_dump_json())["corrections"][0]["attestation"] == "complete_page_review"


def test_clock_rollback_does_not_revive_an_expired_approval(review_case):
    _path, _original, _correction, now, authority, draft, approval, _evidence = review_case
    now[0] = 1060
    assert not authority.permits(approval, draft)
    now[0] = 1000
    assert not authority.permits(approval, draft)


def test_revocation_capacity_failure_disables_authority_instead_of_leaving_a_lease_live(review_case):
    path, _original, _correction, _now, authority, draft, approval, evidence = review_case
    rag = ReviewedRAG(evidence, authority=authority, source_path=path)
    authority._revoked = {format(index, "064x") for index in range(10_000)}
    with pytest.raises(ReviewError):
        authority.revoke(approval)
    assert not authority.permits(approval, draft)
    with pytest.raises(ReviewError):
        rag.answer("Requests")
    with pytest.raises(ReviewError):
        authority.decide(draft, reviewer_id="authored-reviewer", decision="approve")


def test_clock_failure_does_not_leak_callback_errors(review_case):
    _path, _original, _correction, _now, authority, draft, approval, _evidence = review_case

    def broken():
        raise RuntimeError("PRIVATE-CLOCK-STATE")

    authority._clock = broken
    assert not authority.permits(approval, draft)
    with pytest.raises(ReviewError) as exc:
        authority.decide(draft, reviewer_id="authored-reviewer", decision="approve")
    assert "PRIVATE" not in str(exc.value)


@pytest.mark.parametrize("lifetime", [True, 0, 86_401, "60"])
def test_lease_lifetime_is_a_strict_bounded_integer(review_case, lifetime):
    _path, _original, _correction, _now, authority, draft, *_rest = review_case
    with pytest.raises(ReviewError):
        authority.decide(draft, reviewer_id="authored-reviewer", decision="approve", lifetime_seconds=lifetime)


def test_revocation_during_selector_call_prevents_return_of_answer(review_case):
    from agentforge_sdk import Selection

    path, _original, _correction, _now, authority, _draft, approval, evidence = review_case

    class RevokingSelector:
        name, external_processing = "authored-revoking-selector", False

        def select(self, question, selected):
            authority.revoke(approval)
            return [Selection(evidence_id=selected[0].evidence_id, quote=selected[0].text)]

    rag = ReviewedRAG(evidence, authority=authority, source_path=path, selector=RevokingSelector())
    with pytest.raises(ReviewError):
        rag.answer("Requests After", top_k=1)


def test_proposals_cannot_bypass_extraction_limits_or_rereview_a_derived_manifest(review_case):
    _path, original, correction, _now, authority, _draft, _approval, evidence = review_case
    # Small text budget remains in force for review even though the original page is held.
    payload = original.model_dump()
    payload["config"]["max_text_chars"] = 1000
    payload["fingerprint"] = fingerprint({k: v for k, v in payload.items() if k != "fingerprint"})
    from agentforge_sdk import MultimodalDocument

    bounded = MultimodalDocument.model_validate(payload)
    oversized = correction.model_copy(update={"regions": (ReviewedRegionInput(text="x" * 1001, box=correction.regions[0].box),)})
    with pytest.raises(ValueError):
        prepare_review(bounded, [oversized], rationale="Authored budget control")
    with pytest.raises(ReviewError):
        prepare_review(evidence.document, [correction], rationale="Attempted nested review")


def test_mixed_document_retains_machine_provenance_and_partial_review_stays_held(tmp_path):
    pytest.importorskip("reportlab")
    pytest.importorskip("pdfplumber")
    pytest.importorskip("pypdf")
    from agentforge_sdk import load_multimodal
    from evals.multimodal_fixtures import create_fixtures

    files = create_fixtures(tmp_path / "pdf")
    original = load_multimodal(files["mixed"], document_id="mixed")
    assert original.pages[0].status == "ready" and not original.ready
    native_regions = tuple(ReviewedRegionInput(**r.model_dump(exclude={"region_id", "confidence"})) for r in original.pages[0].regions)
    first = PageCorrection(page=1, original_page_sha256=fingerprint(original.pages[0].model_dump(mode="json")),
                           regions=native_regions, attestation="complete_page_review")
    authority = ReviewAuthority({"authored-reviewer": b"x" * 32}, clock=lambda: 1000)
    draft = prepare_review(original, [first], rationale="Authored partial-document control")
    approval = authority.decide(draft, reviewer_id="authored-reviewer", decision="approve")
    partial = apply_review(original, draft, approval, authority=authority)
    assert not partial.document.ready
    result = ReviewedRAG(partial, authority=authority, source_path=files["mixed"]).answer("Atlas latency")
    assert result.result.reason == "incomplete_extraction" and not result.citations
    # Explicit complete transcription of the scan, with authored geometry.
    lines = [("Atlas recovery drill", (0.06, 0.16, 0.7, 0.25)),
             ("Recovery time is 42 seconds.", (0.06, 0.29, 0.85, 0.38)),
             ("Approved by operator.", (0.06, 0.41, 0.75, 0.5))]
    second = PageCorrection(page=2, original_page_sha256=fingerprint(original.pages[1].model_dump(mode="json")),
                            regions=tuple(ReviewedRegionInput(text=t, box=Box(x0=b[0], y0=b[1], x1=b[2], y1=b[3])) for t, b in lines),
                            attestation="complete_page_review")
    draft = prepare_review(original, [second], rationale="Authored full scan transcription")
    approval = authority.decide(draft, reviewer_id="authored-reviewer", decision="approve")
    evidence = apply_review(original, draft, approval, authority=authority)
    rag = ReviewedRAG(evidence, authority=authority, source_path=files["mixed"])
    assert evidence.document.pages[0] == original.pages[0]
    native = rag.answer("Atlas p95 latency", top_k=1)
    reviewed = rag.answer("Recovery time seconds", top_k=1)
    assert all(c.origin == "machine_extracted" and c.reviewer_id is None for c in native.citations)
    assert all(c.origin == "reviewer_supplied" and c.spatial.page == 2 for c in reviewed.citations)
    assert rag.verify_answer(native) and rag.verify_answer(reviewed)
