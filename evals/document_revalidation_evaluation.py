"""Batch/resume controls on authored PDF revisions, not a real-world answer-quality benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentforge_sdk import (
    CheckpointError,
    MultimodalRAG,
    RevalidationBatch,
    RevalidationConfig,
    SavedAnswer,
    SavedCalculation,
    load_multimodal,
)
from agentforge_sdk.models import fingerprint
from evals.document_change_fixtures import create_change_fixtures


def evaluate(directory: Path) -> dict:
    directory = directory.absolute()
    root = (Path.cwd() / "data/evaluations").resolve()
    if directory.resolve() == root or not directory.resolve().is_relative_to(root):
        raise ValueError("revalidation evidence must be below data/evaluations/")
    for part in (directory, *directory.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("revalidation output links are unsupported")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / ".gitignore").write_text("*\n", encoding="utf-8")
    files = create_change_fixtures(directory / "fixtures")
    documents = {name: load_multimodal(path, document_id="atlas", source="atlas.pdf", version=name) for name, path in files.items()}
    old, new = MultimodalRAG(documents["v1"]), MultimodalRAG(documents["v2"])
    regions = documents["v1"].pages[0].regions

    def cells(label):
        row = next(r.row for r in regions if r.kind == "table_cell" and r.column == 0 and r.text == label)
        return [next(r.region_id for r in regions if r.row == row and r.column == col) for col in (1, 2)]

    question = "Atlas p95 latency"
    saved = [SavedAnswer(item_id="latency", question=question, receipt=old.answer(question, top_k=1)),
             SavedCalculation(item_id="requests", receipt=old.calculate("percent_change", cells("Requests"))),
             SavedCalculation(item_id="revenue", receipt=old.calculate("sum", cells("Revenue")))]
    config = RevalidationConfig(refresh_answers=True, replay_calculations=True, max_items_per_step=1, answer_top_k=1)
    batch = RevalidationBatch(old, new, saved, config=config)
    first = batch.advance()
    restored_batch = RevalidationBatch(old, new, saved, config=config)
    second = restored_batch.advance(restored_batch.restore(first.model_dump_json()))
    final = restored_batch.advance(second)
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    check("actual_pdf_extraction_ready", all(doc.ready for doc in documents.values()))
    check("bounded_first_step", len(first.entries) == 1 and not first.complete)
    check("fresh_retrieval_not_old_receipt_reuse", new.verify_answer(first.entries[0].answer, question=question)
          and first.entries[0].answer.fingerprint != saved[0].receipt.fingerprint)
    check("new_answer_citations_on_page_two", all(c.page == 2 for c in first.entries[0].answer.citations))
    check("reconstructed_batch_resumes", restored_batch.fingerprint == batch.fingerprint and len(second.entries) == 2)
    check("numeric_replay_25_to_50", saved[1].receipt.value == "25" and second.entries[1].replay.result.value == "50")
    check("tool_replay_target_verified", new.verify_calculation(second.entries[1].replay.result))
    check("completed_all_saved_items", final.complete and len(final.entries) == 3)
    check("one_answer_attempt_not_one_provider_request_claim", final.answer_attempts == 1)
    check("offline_cost_zero", final.provider_cost_usd == 0)
    check("completed_resume_is_noop", batch.advance(final) == final)
    check("checkpoint_verified", batch.verify_checkpoint(final))
    queue = batch.review_queue(final)
    check("only_changed_result_needs_review", len(queue) == 1 and queue[0].item_id == "requests" and queue[0].category == "changed_result")
    queue_json = json.dumps([q.model_dump(mode="json") for q in queue])
    check("queue_excludes_excerpts_and_questions", "180 ms" not in queue_json and question not in queue_json)
    assess = RevalidationBatch(old, new, saved).advance()
    check("assessment_default_never_refreshes", assess.answer_attempts == 0 and all(e.status == "refresh_required" for e in assess.entries))
    zero = RevalidationBatch(old, new, saved, config=RevalidationConfig(refresh_answers=True, replay_calculations=True, max_answer_attempts=0)).advance()
    check("zero_budget_holds_only_answer", zero.entries[0].reason == "answer_budget_exhausted" and zero.entries[1].status == "refreshed")
    ambiguous = RevalidationBatch(old, MultimodalRAG(documents["ambiguous"]), [saved[1]], config=config).advance()
    check("ambiguous_labels_block_calculation", ambiguous.entries[0].status == "blocked" and ambiguous.entries[0].replay is None)
    mixed = RevalidationBatch(old, MultimodalRAG(documents["unit-change"]), [saved[2]], config=config).advance()
    check("mixed_units_keep_abstention", mixed.entries[0].status == "abstained" and mixed.entries[0].replay.result.reason == "ambiguous_number")
    foreign = RevalidationBatch(old, MultimodalRAG(documents["same-content"]), saved, config=config)
    try:
        foreign.restore(first.model_dump_json())
        rejected = False
    except CheckpointError:
        rejected = True
    check("wrong_target_checkpoint_rejected", rejected)
    check("tampered_summary_rejected", not batch.verify_checkpoint(final.model_copy(update={"answer_attempts": 0})))
    report = {"protocol": "document-revalidation-controls-v1",
              "evidence_kind": "authored_pdfs_actual_extraction_fresh_retrieval_and_tool_replay",
              "total": len(checks), "passed": sum(c["passed"] for c in checks), "checks": checks,
              "batch_fingerprint": batch.fingerprint, "checkpoint_fingerprint": final.fingerprint,
              "document_fingerprints": {name: doc.fingerprint for name, doc in documents.items()},
              "live_provider_calls": 0, "human_review_performed": False, "real_world_quality_measured": False,
              "scope": "bounded lifecycle and evidence lineage; no exactly-once, semantic truth, or production-quality claim"}
    report["fingerprint"] = fingerprint(report)
    for name, checkpoint in (("first", first), ("second", second), ("final", final), ("assessment", assess)):
        (directory / f"{name}.json").write_text(checkpoint.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (directory / "queue.json").write_text(json.dumps([q.model_dump(mode="json") for q in queue], indent=2) + "\n", encoding="utf-8")
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
