"""Batch lifecycle controls; no provider requests or approval UI are simulated as real."""

from __future__ import annotations

import pytest
from test_document_changes import document, row_cells

from agentforge_sdk import (
    CheckpointError,
    MultimodalRAG,
    RevalidationBatch,
    RevalidationCheckpoint,
    RevalidationConfig,
    SavedAnswer,
    SavedCalculation,
    SDKError,
    Selection,
)
from agentforge_sdk.models import fingerprint


def inputs(reader):
    return [
        SavedAnswer(item_id="latency", question="Atlas p95 latency", receipt=reader.answer("Atlas p95 latency", top_k=1)),
        SavedCalculation(item_id="requests", receipt=reader.calculate("percent_change", row_cells(reader.document, "Requests"))),
    ]


def readers(**changes):
    return MultimodalRAG(document()), MultimodalRAG(document(version="v2", **changes))


def test_assess_only_is_default_and_does_not_call_target_answer(monkeypatch):
    old, new = readers(after="150")
    monkeypatch.setattr(new, "answer", lambda *a, **k: pytest.fail("implicit target execution"))
    batch = RevalidationBatch(old, new, inputs(old))
    checkpoint = batch.advance()
    assert checkpoint.complete and checkpoint.answer_attempts == 0 and checkpoint.provider_cost_usd == 0
    assert [e.status for e in checkpoint.entries] == ["refresh_required", "refresh_required"]
    assert [q.item_id for q in batch.review_queue(checkpoint)] == ["latency", "requests"]
    assert batch.verify_checkpoint(checkpoint)


def test_bounded_steps_json_resume_and_finished_run_do_not_repeat_answer_calls(monkeypatch):
    old, new = readers(after="150", cover=True, reverse=True)
    actual, calls = new.answer, []

    def answer(*args, **kwargs):
        calls.append(args[0])
        return actual(*args, **kwargs)

    monkeypatch.setattr(new, "answer", answer)
    batch = RevalidationBatch(old, new, inputs(old), config=RevalidationConfig(
        refresh_answers=True, replay_calculations=True, max_items_per_step=1, answer_top_k=1))
    first = batch.advance()
    assert not first.complete and len(first.entries) == 1 and first.answer_attempts == 1
    restored = batch.restore(first.model_dump_json())
    final = batch.advance(restored)
    assert final.complete and final.entries[1].replay.result.value == "50"
    assert all(c.page == 2 for c in final.entries[1].replay.result.citations)
    assert final.entries[0].result_changed is False and final.entries[1].result_changed is True
    assert final.provider_cost_usd == 0 and final.answer_attempts == 1
    assert [q.item_id for q in batch.review_queue(final)] == ["requests"]
    assert batch.advance(final) == final and batch.verify_checkpoint(final) and calls == ["Atlas p95 latency"]


def test_resume_with_a_reconstructed_batch_keeps_identical_input_bindings():
    old, new = readers()
    saved = inputs(old)
    config = RevalidationConfig(max_items_per_step=1)
    first = RevalidationBatch(old, new, saved, config=config).advance()
    fresh = RevalidationBatch(old, new, saved, config=config)
    assert fresh.advance(fresh.restore(first.model_dump_json())).complete


@pytest.mark.parametrize("limit", [0, 1])
def test_answer_budget_survives_resume_and_does_not_count_calculation_tools(limit):
    old, new = readers(after="150")
    saved = [inputs(old)[0], SavedAnswer(item_id="again", question="Atlas p95 latency", receipt=inputs(old)[0].receipt), inputs(old)[1]]
    batch = RevalidationBatch(old, new, saved, config=RevalidationConfig(
        refresh_answers=True, replay_calculations=True, max_answer_attempts=limit, max_items_per_step=1))
    checkpoint = batch.advance()
    while not checkpoint.complete:
        checkpoint = batch.advance(batch.restore(checkpoint.model_dump_json()))
    assert checkpoint.answer_attempts == limit and checkpoint.entries[-1].status == "refreshed"
    assert sum(e.reason == "answer_budget_exhausted" for e in checkpoint.entries) == 2 - limit


def test_same_context_positive_receipts_require_no_refresh_even_when_enabled(monkeypatch):
    old = MultimodalRAG(document())
    saved = inputs(old)
    monkeypatch.setattr(old, "answer", lambda *a, **k: pytest.fail("unnecessary refresh"))
    batch = RevalidationBatch(old, old, saved, config=RevalidationConfig(refresh_answers=True, replay_calculations=True))
    result = batch.advance()
    assert all(e.status == "current" for e in result.entries) and not batch.review_queue(result)


def test_old_abstention_can_be_retrieved_again_but_is_not_marked_current():
    old, new = readers()
    item = SavedAnswer(item_id="unknown", question="xylophonium", receipt=old.answer("xylophonium"))
    batch = RevalidationBatch(old, new, [item], config=RevalidationConfig(refresh_answers=True))
    result = batch.advance()
    assert result.entries[0].status == "abstained" and result.answer_attempts == 1
    assert batch.review_queue(result)[0].category == "abstained"


def test_new_document_can_answer_a_previously_unanswered_saved_question():
    old = MultimodalRAG(document())
    new = MultimodalRAG(document(version="v2", title="Xylophonium report"))
    item = SavedAnswer(item_id="previously-unknown", question="Xylophonium", receipt=old.answer("Xylophonium"))
    batch = RevalidationBatch(old, new, [item], config=RevalidationConfig(refresh_answers=True, answer_top_k=1))
    result = batch.advance()
    assert result.entries[0].status == "refreshed" and result.entries[0].result_changed is True
    assert new.verify_answer(result.entries[0].answer, question=item.question)
    assert batch.review_queue(result)[0].category == "changed_result"


@pytest.mark.parametrize("changes", [{"held": True}, {"extractor": "changed-parser"}])
def test_held_extraction_or_parser_drift_never_calls_answer_backend(changes, monkeypatch):
    old, new = readers(**changes)
    monkeypatch.setattr(new, "answer", lambda *a, **k: pytest.fail("held target executed"))
    batch = RevalidationBatch(old, new, inputs(old), config=RevalidationConfig(refresh_answers=True, replay_calculations=True))
    result = batch.advance()
    assert all(e.status == "blocked" for e in result.entries) and result.answer_attempts == 0


def test_ambiguous_calculation_goes_to_review_not_positional_replay():
    old, new = readers(duplicate=True)
    batch = RevalidationBatch(old, new, [inputs(old)[1]], config=RevalidationConfig(replay_calculations=True))
    result = batch.advance()
    assert result.entries[0].reason == "alignment_unavailable" and result.entries[0].replay is None
    assert batch.review_queue(result)[0].category == "blocked"


def test_replay_that_abstains_preserves_target_tool_receipt():
    old, new = readers(currency="EUR")
    item = SavedCalculation(item_id="money", receipt=old.calculate("sum", row_cells(old.document, "Revenue")))
    batch = RevalidationBatch(old, new, [item], config=RevalidationConfig(replay_calculations=True))
    result = batch.advance()
    assert result.entries[0].status == "abstained"
    assert result.entries[0].replay.result.reason == "ambiguous_number" and batch.verify_checkpoint(result)


def test_wrong_question_and_tampered_source_do_not_refresh(monkeypatch):
    old, new = readers()
    item = inputs(old)[0].model_copy(update={"question": "wrong question"})
    monkeypatch.setattr(new, "answer", lambda *a, **k: pytest.fail("invalid source executed"))
    batch = RevalidationBatch(old, new, [item], config=RevalidationConfig(refresh_answers=True))
    result = batch.advance()
    assert result.entries[0].reason == "source_unverifiable" and not result.answer_attempts


def test_sdk_failure_consumes_attempt_once_without_retry_or_error_text(monkeypatch):
    old, new = readers()
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise SDKError("PRIVATE PROVIDER CONTENT")

    monkeypatch.setattr(new, "answer", fail)
    batch = RevalidationBatch(old, new, [inputs(old)[0]], config=RevalidationConfig(refresh_answers=True))
    result = batch.advance()
    assert result.entries[0].status == "failed" and result.answer_attempts == 1
    assert result.provider_cost_usd is None and "PRIVATE" not in result.model_dump_json()
    assert batch.advance(batch.restore(result.model_dump_json())) == result and calls == [1]


def test_custom_selector_cost_remains_unknown_and_checkpoint_verification_never_calls_it():
    class Selector:
        name, external_processing = "authored-local-control", False
        calls = 0

        def select(self, question, evidence):
            self.calls += 1
            return [Selection(evidence_id=evidence[0].evidence_id, quote=evidence[0].text)]

    selector = Selector()
    old = MultimodalRAG(document())
    new = MultimodalRAG(document(version="v2"), selector=selector)
    batch = RevalidationBatch(old, new, [inputs(old)[0]], config=RevalidationConfig(refresh_answers=True, answer_top_k=1))
    result = batch.advance()
    assert result.provider_cost_usd is None and selector.calls == 1
    assert batch.verify_checkpoint(result) and batch.restore(result.model_dump_json()) == result
    assert selector.calls == 1


@pytest.mark.parametrize("field,value", [("complete", False), ("answer_attempts", 0), ("provider_cost_usd", 99.0)])
def test_rehashed_summary_tampering_is_rejected(field, value):
    old, new = readers()
    batch = RevalidationBatch(old, new, [inputs(old)[0]], config=RevalidationConfig(refresh_answers=True))
    checkpoint = batch.advance().model_copy(update={field: value})
    checkpoint = checkpoint.model_copy(update={"fingerprint": fingerprint(checkpoint.model_dump(mode="json", exclude={"fingerprint"}))})
    assert not batch.verify_checkpoint(checkpoint)
    with pytest.raises(CheckpointError):
        batch.advance(checkpoint)


@pytest.mark.parametrize("change", ["order", "question", "target", "config"])
def test_cross_batch_or_changed_inputs_cannot_resume(change):
    old, new = readers()
    saved, config = inputs(old), RevalidationConfig(max_items_per_step=1)
    checkpoint = RevalidationBatch(old, new, saved, config=config).advance()
    if change == "order":
        saved.reverse()
    elif change == "question":
        saved[0] = saved[0].model_copy(update={"question": "Atlas report"})
    elif change == "target":
        new = MultimodalRAG(document(version="v3"))
    else:
        config = RevalidationConfig(max_items_per_step=2)
    batch = RevalidationBatch(old, new, saved, config=config)
    with pytest.raises(CheckpointError):
        batch.restore(checkpoint.model_dump_json())


def test_content_minimal_queue_contains_no_quotes_questions_or_calculation_values():
    old, new = readers(after="150")
    batch = RevalidationBatch(old, new, inputs(old), config=RevalidationConfig(replay_calculations=True))
    result = batch.advance()
    queue = " ".join(q.model_dump_json() for q in batch.review_queue(result))
    assert all(s not in queue for s in ("180 ms", "Atlas", '"50"', "quote", "question"))
    assert "entries=" not in repr(result) and "180 ms" not in repr(inputs(old)[0])


@pytest.mark.parametrize("config", [{"max_answer_attempts": True}, {"max_items_per_step": 0},
                                   {"answer_top_k": 21}, {"refresh_answers": "yes"}])
def test_strict_finite_budgets(config):
    with pytest.raises(ValueError):
        RevalidationConfig(**config)


def test_empty_duplicate_untyped_or_oversized_batches_rejected():
    old, new = readers()
    saved = inputs(old)
    for items in ([], [saved[0], saved[0]], [saved[0].model_dump()], [saved[0]] * 65):
        with pytest.raises(ValueError):
            RevalidationBatch(old, new, items)


def test_bad_checkpoint_admission_is_content_free():
    old, new = readers()
    batch = RevalidationBatch(old, new, inputs(old))
    for value in ("PRIVATE", b"{", "x" * 16_000_001, 42):
        with pytest.raises(CheckpointError, match="checkpoint is invalid") as error:
            batch.restore(value)
        assert "PRIVATE" not in str(error.value)
    assert not batch.verify_checkpoint(None)


def test_checkpoint_roundtrip_preserves_union_receipts():
    old, new = readers(after="150")
    batch = RevalidationBatch(old, new, inputs(old), config=RevalidationConfig(refresh_answers=True, replay_calculations=True))
    result = batch.advance()
    assert batch.verify_checkpoint(RevalidationCheckpoint.model_validate_json(result.model_dump_json()))


def test_rehashed_entry_that_skips_enabled_execution_is_rejected():
    old, new = readers()
    saved = inputs(old)
    assess = RevalidationBatch(old, new, saved).advance()
    batch = RevalidationBatch(old, new, saved, config=RevalidationConfig(refresh_answers=True, replay_calculations=True))
    forged = assess.model_copy(update={"batch_fingerprint": batch.fingerprint})
    forged = forged.model_copy(update={"fingerprint": fingerprint(forged.model_dump(mode="json", exclude={"fingerprint"}))})
    assert not batch.verify_checkpoint(forged)


def test_rehashed_fresh_answer_tampering_does_not_pass_resume():
    old, new = readers()
    batch = RevalidationBatch(old, new, [inputs(old)[0]], config=RevalidationConfig(refresh_answers=True))
    result = batch.advance()
    entry = result.entries[0]
    fake_answer = entry.answer.model_copy(update={"answer": "invented"})
    fake_answer = fake_answer.model_copy(update={"fingerprint": fingerprint(fake_answer.model_dump(mode="json", exclude={"fingerprint"}))})
    entry = entry.model_copy(update={"answer": fake_answer})
    entry = entry.model_copy(update={"fingerprint": fingerprint(entry.model_dump(mode="json", exclude={"fingerprint"}))})
    forged = result.model_copy(update={"entries": (entry,)})
    forged = forged.model_copy(update={"fingerprint": fingerprint(forged.model_dump(mode="json", exclude={"fingerprint"}))})
    assert not batch.verify_checkpoint(forged)


@pytest.mark.parametrize("invalidation", ["revoke", "expire", "source"])
def test_review_authority_or_source_changes_invalidate_resume_without_target_calls(tmp_path, monkeypatch, invalidation):
    pytest.importorskip("pdfplumber")
    from agentforge_sdk import ReviewAuthority, ReviewedRAG, apply_review, prepare_review
    from evals.evidence_review_fixtures import create_review_fixture

    path, original, correction = create_review_fixture(tmp_path)
    clock = [1000]
    authority = ReviewAuthority({"a": b"a" * 32, "b": b"b" * 32}, clock=lambda: clock[0])
    draft = prepare_review(original, [correction], rationale="Authored batch controls")
    approvals = [authority.decide(draft, reviewer_id=r, decision="approve") for r in ("a", "b")]
    old, new = [ReviewedRAG(apply_review(original, draft, a, authority=authority), authority=authority, source_path=path) for a in approvals]
    saved = [SavedAnswer(item_id="requests", question="Requests After", receipt=old.answer("Requests After", top_k=1))]
    batch = RevalidationBatch(old, new, saved, config=RevalidationConfig(refresh_answers=True, answer_top_k=1))
    checkpoint = batch.advance()
    assert checkpoint.entries[0].answer.approval_fingerprint == approvals[1].fingerprint
    assert batch.verify_checkpoint(checkpoint)
    if invalidation == "revoke":
        authority.revoke(approvals[1])
    elif invalidation == "expire":
        clock[0] += 3601
    else:
        path.write_bytes(b"modified source bytes")
    monkeypatch.setattr(new, "answer", lambda *a, **k: pytest.fail("revoked target executed"))
    assert not batch.verify_checkpoint(checkpoint)
    with pytest.raises(CheckpointError):
        batch.advance(checkpoint)
    assert batch.advance().entries[0].status == "blocked"
