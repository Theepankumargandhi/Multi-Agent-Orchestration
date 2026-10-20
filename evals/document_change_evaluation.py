"""Extraction/receipt revision controls on authored PDFs; not a visual or semantic accuracy benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentforge_sdk import DocumentRevision, MultimodalRAG, load_multimodal
from agentforge_sdk.models import fingerprint
from evals.document_change_fixtures import create_change_fixtures


def evaluate(directory: Path) -> dict:
    directory = directory.absolute()
    root = (Path.cwd() / "data/evaluations").resolve()
    if directory.resolve() == root or not directory.resolve().is_relative_to(root):
        raise ValueError("document change evidence must be below data/evaluations/")
    for part in (directory, *directory.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("revision output links are unsupported")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / ".gitignore").write_text("*\n", encoding="utf-8")
    files = create_change_fixtures(directory / "fixtures")
    documents = {name: load_multimodal(path, document_id="atlas", source="atlas.pdf", version=name) for name, path in files.items()}
    readers = {name: MultimodalRAG(doc) for name, doc in documents.items()}
    old, new = readers["v1"], readers["v2"]
    revision = DocumentRevision(old, new)

    def cells(doc, label):
        regions = doc.pages[-1].regions
        row = next(r.row for r in regions if r.kind == "table_cell" and r.column == 0 and r.text == label)
        return [next(r.region_id for r in regions if r.row == row and r.column == col) for col in (1, 2)]

    calculation = old.calculate("percent_change", cells(documents["v1"], "Requests"))
    replay = revision.recalculate(calculation)
    answer = old.answer("Atlas p95 latency", top_k=1)
    impact = revision.assess_answer(answer, question="Atlas p95 latency")
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    check("actual_pdfs_extracted", all(doc.ready for doc in documents.values()))
    check("page_insertion_aligned", any(p.before_page == 1 and p.after_page == 2 for p in revision.changes.pages))
    check("changed_number_flagged", any("numeric" in c.flags for c in revision.changes.regions))
    check("row_reorder_flagged", any("address" in c.flags for c in revision.changes.regions))
    check("source_calculation_verified", calculation.value == "25" and old.verify_calculation(calculation))
    check("replayed_calculation_changed", replay.status == "recalculated" and replay.result.value == "50" and replay.value_changed is True)
    check("new_citations_use_actual_page_two", bool(replay.result.citations) and all(c.page == 2 for c in replay.result.citations))
    check("replayed_receipt_verified", revision.verify_recalculation(replay, calculation) and new.verify_calculation(replay.result))
    check("old_receipt_not_accepted_by_new_corpus", not new.verify_calculation(calculation) and not new.verify_answer(answer))
    check("unchanged_quote_still_needs_reissue", impact.status == "dependencies_unchanged" and impact.requires_reissue)
    check("changed_corpus_requires_fresh_retrieval", "rerun_retrieval_for_changed_corpus" in impact.reasons)
    check("impact_verified", revision.verify_impact(impact, answer, question="Atlas p95 latency"))
    currency = DocumentRevision(old, readers["unit-change"])
    money = old.calculate("sum", cells(documents["v1"], "Revenue"))
    mixed = currency.recalculate(money)
    check("currency_change_flagged", any("unit" in c.flags for c in currency.changes.regions))
    check("no_invented_currency_conversion", mixed.status == "abstained" and mixed.result.reason == "ambiguous_number")
    ambiguous = DocumentRevision(old, readers["ambiguous"])
    check("duplicate_labels_hold_replay", ambiguous.recalculate(calculation).reason == "dependency_alignment_unavailable")
    same = DocumentRevision(old, readers["same-content"])
    same_impact = same.assess_calculation(calculation)
    check("same_content_new_version_still_reissues", same_impact.status == "dependencies_unchanged" and same_impact.requires_reissue)
    check("tampered_diff_rejected", not revision.verify_changes(revision.changes.model_copy(update={"regions": ()})))
    check("tampered_replay_rejected", not revision.verify_recalculation(replay.model_copy(update={"value_changed": False}), calculation))
    report = {"protocol": "document-change-controls-v1", "evidence_kind": "authored_pdfs_actual_extraction_and_tool_replay",
              "total": len(checks), "passed": sum(c["passed"] for c in checks), "checks": checks,
              "document_fingerprints": {name: doc.fingerprint for name, doc in documents.items()},
              "changes_fingerprint": revision.changes.fingerprint, "replay_fingerprint": replay.fingerprint,
              "live_provider_calls": 0, "real_world_quality_measured": False,
              "scope": "extraction-level alignment and verified dependencies, not semantic truth or pixel comparison"}
    report["fingerprint"] = fingerprint(report)
    (directory / "changes.json").write_text(revision.changes.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (directory / "impact.json").write_text(impact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (directory / "replay.json").write_text(replay.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (directory / "documents.json").write_text(json.dumps({name: doc.model_dump(mode="json") for name, doc in documents.items()}, indent=2) + "\n", encoding="utf-8")
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
