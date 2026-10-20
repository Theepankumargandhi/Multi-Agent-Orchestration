"""Authored planner controls on actual PDF extraction; not a live-model benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentforge_sdk import (
    DocumentPage,
    DocumentRAG,
    DocumentReasoningAgent,
    EvidenceSet,
    EvidenceSource,
    MultimodalRAG,
    PlannerOutput,
    PlannerUsage,
    ReasoningPlan,
    load_multimodal,
)
from agentforge_sdk.models import fingerprint
from evals.document_change_fixtures import create_change_fixtures
from evals.financebench_pilot import private_directory


class _AuthoredPlanner:
    name, external_processing = "authored-reasoning-control", False

    def __init__(self, mode):
        self.mode, self.calls = mode, 0

    def plan(self, request):
        self.calls += 1
        if self.mode == "repair" and self.calls == 1:
            raw = '{"action":"execute_code","code":"not executed"}'
        elif self.mode == "quote" or self.mode == "repair":
            chosen = next(c.candidate_id for c in request.candidates if c.kind == "quote")
            raw = ReasoningPlan(action="quote", candidate_ids=(chosen,)).model_dump_json()
        else:
            cards = {c.fact_query.column_label: c.candidate_id for c in request.candidates
                     if c.kind == "fact" and c.fact_query.row_label == "Requests"}
            raw = ReasoningPlan(action="calculate", operation="percent_change",
                                candidate_ids=(cards["Before"], cards["After"])).model_dump_json()
        return PlannerOutput(plan_json=raw, usage=PlannerUsage(provider_requests=0, provider_cost_usd=0.0))


def evaluate(directory: Path) -> dict:
    directory = private_directory(directory)
    files = create_change_fixtures(directory / "fixtures")
    readers = {name: MultimodalRAG(load_multimodal(path, document_id="atlas-" + name)) for name, path in files.items()}
    scope = "atlas:authored-Q1"
    agreeing = EvidenceSet([EvidenceSource(name, scope, readers[name]) for name in ("v1", "same-content")])
    conflicting = EvidenceSet([EvidenceSource(name, scope, readers[name]) for name in ("v1", "same-content", "v2")])
    # These are actual extracted lines, not the gold answer to a held-out question.
    text = DocumentRAG([DocumentPage(document_id="atlas", source="v1.pdf", page=1,
        text="\n".join(r.text for r in readers["v1"].document.pages[0].regions if r.kind == "text"))])
    quote_agent = DocumentReasoningAgent(text, _AuthoredPlanner("quote"))
    quote = quote_agent.run("Atlas p95 latency")
    numeric_agent = DocumentReasoningAgent(text, _AuthoredPlanner("numeric"), evidence=agreeing)
    question = "Requests Before After percent change"
    numeric = numeric_agent.run(question)
    conflict_planner = _AuthoredPlanner("numeric")
    conflict_agent = DocumentReasoningAgent(text, conflict_planner, evidence=conflicting)
    conflict = conflict_agent.run(question)
    repair_planner = _AuthoredPlanner("repair")
    repair_agent = DocumentReasoningAgent(text, repair_planner)
    repaired = repair_agent.run("Atlas p95 latency")
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    check("pdfs_actually_extracted", all(r.document.ready for r in readers.values()))
    check("literal_quote_answer", quote.status == "answered" and "180 ms" in quote.answer)
    check("quote_lineage_verified", quote_agent.verify_result(quote, question="Atlas p95 latency"))
    check("numeric_plan_bound_to_catalog", numeric.status == "answered" and numeric.answer == "25 %")
    check("numeric_tool_verified", numeric_agent.verify_result(numeric, question=question))
    check("fact_approvals_not_manufactured", all(c.__class__.__name__ == "SpatialCitation" for f in numeric.calculation.facts for o in f.observations for c in o.citations))
    check("conflict_stops_execution", conflict.status == "abstained" and conflict.calculation.reason == "unresolved_fact")
    check("conflict_is_not_retried", conflict_planner.calls == 1)
    check("abstention_receipt_verified", conflict_agent.verify_result(conflict, question=question))
    check("invalid_tool_not_executed", repaired.attempts[0].outcome == "invalid_plan")
    check("bounded_recovery_works", repaired.status == "answered" and repair_planner.calls == 2)
    check("recovery_feedback_typed", repaired.attempts[1].request.previous_failure == "invalid_plan")
    check("recovered_receipt_verifies_without_model", repair_agent.verify_result(repaired, question="Atlas p95 latency") and repair_planner.calls == 2)
    check("tampered_answer_rejected", not quote_agent.verify_result(quote.model_copy(update={"answer": "invented"}), question="Atlas p95 latency"))
    check("wrong_question_rejected", not numeric_agent.verify_result(numeric, question="another question"))
    check("registered_fact_catalog_has_axes", len(agreeing.query_catalog()) == 4)
    report = {"protocol": "document-reasoning-controls-v1", "evidence_kind": "authored_planner_responses_actual_pdf_extraction",
              "total": len(checks), "passed": sum(c["passed"] for c in checks), "checks": checks,
              "live_provider_calls": 0, "real_world_answer_accuracy_measured": False,
              "scope": "planner admission, grounded tools, recovery and provenance; semantic plan correctness is separate"}
    report["fingerprint"] = fingerprint(report)
    for name, receipt in (("quote", quote), ("numeric", numeric), ("conflict", conflict), ("recovery", repaired)):
        (directory / (name + ".json")).write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    report = evaluate(parser.parse_args().output_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
