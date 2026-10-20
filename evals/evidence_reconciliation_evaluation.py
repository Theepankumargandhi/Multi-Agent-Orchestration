"""Numerical cross-document controls on authored PDFs; not a semantic or factuality benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentforge_sdk import (
    EvidenceSet,
    EvidenceSource,
    FactQuery,
    MultimodalRAG,
    ReconciliationConfig,
    load_multimodal,
)
from agentforge_sdk.models import fingerprint
from evals.document_change_fixtures import create_change_fixtures


def evaluate(directory: Path) -> dict:
    directory = directory.absolute()
    root = (Path.cwd() / "data/evaluations").resolve()
    if directory.resolve() == root or not directory.resolve().is_relative_to(root):
        raise ValueError("reconciliation evidence must be below data/evaluations/")
    for part in (directory, *directory.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("reconciliation output links are unsupported")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / ".gitignore").write_text("*\n", encoding="utf-8")
    files = create_change_fixtures(directory / "fixtures")
    readers = {name: MultimodalRAG(load_multimodal(path, document_id="atlas-" + name, source=name + ".pdf", version=name))
               for name, path in files.items()}
    scope = "atlas:2026-Q1:authored-report"

    def source(name, *, alternate_scope=None):
        return EvidenceSource(name, alternate_scope or scope, readers[name])

    before_query = FactQuery(row_label="Requests", column_label="Before", scope=scope)
    after_query = FactQuery(row_label="Requests", column_label="After", scope=scope)
    money_query = FactQuery(row_label="Revenue", column_label="After", scope=scope)
    collection = EvidenceSet([source("v1"), source("same-content")])
    before, after = [collection.lookup(q) for q in (before_query, after_query)]
    calculation = collection.calculate("percent_change", [before, after])
    conflicting = EvidenceSet([source("v1"), source("same-content"), source("v2")])
    conflict = conflicting.lookup(after_query)
    stopped = conflicting.calculate("sum", [conflict])
    units = EvidenceSet([source("v1"), source("unit-change")]).lookup(money_query)
    ambiguous = EvidenceSet([source("v1"), source("ambiguous")]).lookup(after_query)
    alias = EvidenceSet([EvidenceSource("alias-a", scope, readers["v1"]), EvidenceSource("alias-b", scope, readers["v1"])]).lookup(after_query)
    scoped = EvidenceSet([source("v1"), source("same-content"), source("v2", alternate_scope="atlas:2026-Q2:authored-report")]).lookup(after_query)
    missing = collection.lookup(FactQuery(row_label="Unknown row", column_label="After", scope=scope))
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    check("actual_pdf_extraction", all(r.document.ready for r in readers.values()))
    check("two_distinct_artifacts_agree", after.status == "agreed" and after.distinct_artifacts == 2 and after.value == "125")
    check("exact_decimal_tool", calculation.status == "calculated" and calculation.value == "25" and calculation.unit == "%")
    check("tool_receipt_verified", collection.verify_calculation(calculation))
    check("fact_receipts_verified", collection.verify_fact(before) and collection.verify_fact(after))
    check("each_observation_cites_axes_and_value", all(len(o.citations) == 3 for o in after.observations))
    source_by_id = {"v1": readers["v1"], "same-content": readers["same-content"]}
    check("all_literal_citations_verify", all(source_by_id[o.source_id].verify_citation(c) for o in after.observations for c in o.citations))
    check("labels_are_literal", all([c.quote for c in o.citations] == ["Requests", "After", "125"] for o in after.observations))
    check("disagreement_detected", conflict.status == "conflict" and len(conflict.groups) == 2)
    check("majority_not_used_as_truth", conflict.value is None and "numeric_values_disagree" in conflict.reasons)
    check("conflicting_values_are_cited", {g.value for g in conflict.groups} == {"125", "150"})
    moved = next(o for o in conflict.observations if o.source_id == "v2")
    check("moved_table_citations_on_physical_page_two", all(c.page == 2 for c in moved.citations))
    check("unresolved_fact_blocks_calculation", stopped.status == "abstained" and stopped.reason == "unresolved_fact")
    check("blocked_calculation_receipt_verifies", conflicting.verify_calculation(stopped))
    check("incompatible_units_hold", units.status == "abstained" and "incompatible_units" in units.reasons)
    check("duplicate_labels_hold", ambiguous.status == "abstained" and "ambiguous_labels" in ambiguous.reasons)
    check("file_aliases_not_corroboration", alias.status == "abstained" and alias.distinct_artifacts == 1)
    check("other_reporting_scope_not_compared", scoped.status == "agreed" and any(o.status == "out_of_scope" for o in scoped.observations))
    check("missing_fact_not_a_zero", missing.status == "abstained" and missing.value is None)
    check("single_artifact_requires_explicit_policy", EvidenceSet([source("v1")], config=ReconciliationConfig(min_distinct_artifacts=1)).lookup(after_query).status == "agreed")
    check("tampered_fact_rejected", not collection.verify_fact(after.model_copy(update={"value": "999"})))
    check("tampered_calculation_rejected", not collection.verify_calculation(calculation.model_copy(update={"value": "999"})))
    report = {"protocol": "evidence-reconciliation-controls-v1", "evidence_kind": "authored_pdfs_actual_extraction_and_exact_label_reconciliation",
              "total": len(checks), "passed": sum(c["passed"] for c in checks), "checks": checks,
              "collection_fingerprint": collection.fingerprint, "conflict_fingerprint": conflict.fingerprint,
              "calculation_fingerprint": calculation.fingerprint, "live_provider_calls": 0,
              "real_world_accuracy_measured": False, "human_review_performed": False,
              "scope": "caller-scoped numeric table facts; no semantic contradiction inference or source-truth claim"}
    report["fingerprint"] = fingerprint(report)
    for name, result in (("before", before), ("after", after), ("calculation", calculation), ("conflict", conflict), ("units", units)):
        (directory / f"{name}.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
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
