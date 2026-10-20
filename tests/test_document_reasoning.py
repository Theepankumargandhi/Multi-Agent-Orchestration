"""Reasoning lifecycle controls with authored planner responses, not live AI quality evidence."""

from __future__ import annotations

import json

import pytest
from test_document_changes import document

from agentforge_sdk import (
    DocumentPage,
    DocumentRAG,
    DocumentReasoningAgent,
    EvidenceSet,
    EvidenceSource,
    ExternalProcessingDenied,
    MultimodalRAG,
    PlannerOutput,
    PlannerUsage,
    ReasoningConfig,
    ReasoningPlan,
)
from agentforge_sdk.models import SDKError, fingerprint


class Planner:
    name, external_processing = "authored-planner", False

    def __init__(self, choose):
        self.choose, self.calls, self.requests = choose, 0, []

    def plan(self, request):
        self.calls += 1
        self.requests.append(request)
        result = self.choose(request, self.calls)
        return PlannerOutput(plan_json=result if isinstance(result, str) else result.model_dump_json(),
                             usage=PlannerUsage(input_tokens=10, output_tokens=5, provider_requests=0, provider_cost_usd=0.0))


def corpus():
    return DocumentRAG([DocumentPage(document_id="atlas", source="atlas.pdf", page=2,
        text="Atlas p95 latency is 180 ms. Requests increase from 100 to 125.")])


def evidence(conflict=False):
    return EvidenceSet([EvidenceSource(v, "atlas:Q1:counts", MultimodalRAG(document(version=v, after="150" if conflict and v == "b" else "125"))) for v in ("a", "b")])


def select_quote(request, number):
    return ReasoningPlan(action="quote", candidate_ids=(next(c.candidate_id for c in request.candidates if c.kind == "quote"),))


def select_growth(request, number):
    cards = {c.fact_query.column_label: c.candidate_id for c in request.candidates
             if c.kind == "fact" and c.fact_query.row_label == "Requests"}
    return ReasoningPlan(action="calculate", candidate_ids=(cards["Before"], cards["After"]), operation="percent_change")


def test_natural_language_selects_literal_evidence_without_generated_claims():
    planner = Planner(select_quote)
    agent = DocumentReasoningAgent(corpus(), planner)
    result = agent.run("What is Atlas p95 latency?")
    assert result.status == "answered" and '"Atlas p95 latency is 180 ms."' in result.answer
    assert result.citations[0].page == 2 and result.planner_calls == 1 and result.provider_cost_usd == 0
    assert agent.verify_result(result, question="What is Atlas p95 latency?") and planner.calls == 1
    assert not agent.verify_result(result, question="another question")


def test_registered_numeric_plan_executes_existing_reconciled_decimal_tool():
    planner = Planner(select_growth)
    agent = DocumentReasoningAgent(corpus(), planner, evidence=evidence())
    question = "Requests Before After percent change"
    result = agent.run(question)
    assert result.status == "answered" and result.answer == "25 %"
    assert result.calculation.value == "25" and agent.verify_result(result, question=question)
    assert planner.calls == 1


def test_conflict_is_terminal_not_retried_until_a_convenient_quote_is_found():
    planner = Planner(select_growth)
    result = DocumentReasoningAgent(corpus(), planner, evidence=evidence(True)).run("Requests Before After percent change")
    assert result.status == "abstained" and result.reason == "tool_abstained"
    assert result.calculation.reason == "unresolved_fact" and planner.calls == 1


def test_invalid_plan_gets_one_bounded_recovery_with_structured_feedback():
    planner = Planner(lambda r, n: '{"action":"execute_python","code":"PRIVATE"}' if n == 1 else select_quote(r, n))
    agent = DocumentReasoningAgent(corpus(), planner)
    result = agent.run("Atlas p95 latency")
    assert result.status == "answered" and result.planner_calls == 2
    assert result.attempts[0].outcome == "invalid_plan"
    assert planner.requests[1].previous_failure == "invalid_plan" and planner.requests[1].attempts_remaining == 0
    assert "PRIVATE" not in result.model_dump_json() and agent.verify_result(result, question="Atlas p95 latency")


def test_unknown_candidate_never_becomes_a_fictional_quote():
    planner = Planner(lambda r, n: ReasoningPlan(action="quote", candidate_ids=("f" * 64,)))
    agent = DocumentReasoningAgent(corpus(), planner)
    result = agent.run("Atlas p95 latency")
    assert result.status == "abstained" and result.reason == "candidate_unavailable" and planner.calls == 2
    assert not result.citations and agent.verify_result(result, question="Atlas p95 latency")


def test_provider_exception_is_terminal_and_content_free():
    def fail(request, number):
        raise RuntimeError("PRIVATE PROVIDER KEY")
    planner = Planner(fail)
    agent = DocumentReasoningAgent(corpus(), planner)
    result = agent.run("Atlas p95 latency")
    assert result.reason == "planner_failed" and result.provider_cost_usd is None and planner.calls == 1
    assert "PRIVATE" not in result.model_dump_json()
    assert agent.verify_result(result, question="Atlas p95 latency")


def test_no_candidates_uses_no_planner_requests():
    planner = Planner(select_quote)
    agent = DocumentReasoningAgent(corpus(), planner)
    result = agent.run("xylophonium")
    assert result.reason == "no_candidates" and result.planner_calls == 0 and result.provider_cost_usd == 0
    assert planner.calls == 0 and agent.verify_result(result, question="xylophonium")


def test_model_abstention_is_not_retried():
    planner = Planner(lambda r, n: ReasoningPlan(action="abstain"))
    result = DocumentReasoningAgent(corpus(), planner).run("Atlas p95 latency")
    assert result.reason == "model_abstained" and planner.calls == 1


def test_external_consent_and_backend_identity_are_checked_before_calls():
    planner = Planner(select_quote)
    planner.external_processing = True
    with pytest.raises(ExternalProcessingDenied):
        DocumentReasoningAgent(corpus(), planner)
    agent = DocumentReasoningAgent(corpus(), planner, config=ReasoningConfig(allow_external_processing=True))
    planner.name = "different"
    with pytest.raises(SDKError, match="identity"):
        agent.run("Atlas p95 latency")
    assert planner.calls == 0


def test_card_budget_keeps_complete_bounded_candidates():
    planner = Planner(select_quote)
    config = ReasoningConfig(max_prompt_chars=2000)
    result = DocumentReasoningAgent(corpus(), planner, evidence=evidence(), config=config).run("Atlas Requests latency Before After")
    request = result.attempts[0].request
    assert sum(len(c.model_dump_json()) for c in request.candidates) + len(request.question) <= 2000


@pytest.mark.parametrize("field,value", [("answer", "invented"), ("planner_calls", 0), ("reason", "model_abstained")])
def test_rehashed_result_tampering_is_rejected(field, value):
    agent = DocumentReasoningAgent(corpus(), Planner(select_quote))
    result = agent.run("Atlas p95 latency").model_copy(update={field: value})
    result = result.model_copy(update={"fingerprint": fingerprint(result.model_dump(mode="json", exclude={"fingerprint"}))})
    assert not agent.verify_result(result, question="Atlas p95 latency")


def test_search_receipt_verification_rejects_foreign_and_tampered_evidence():
    rag = corpus()
    result = rag.search("Atlas")
    assert rag.verify_search(result, question="Atlas")
    assert not rag.verify_search(result, question="different")
    assert not rag.verify_search(result.model_copy(update={"corpus_sha256": "f" * 64}))
    assert not DocumentRAG([]).verify_search(result)


@pytest.mark.parametrize("payload", [
    {"action": "calculate", "candidate_ids": ["f" * 64], "operation": "ratio"},
    {"action": "quote", "candidate_ids": ["f" * 64], "operation": "sum"},
    {"action": "abstain", "candidate_ids": ["f" * 64]},
    {"action": "quote", "candidate_ids": ["f" * 64, "f" * 64]},
    {"action": "lookup", "candidate_ids": ["not-an-id"]},
    {"action": "quote", "candidate_ids": [], "shell_command": "PRIVATE"},
])
def test_strict_plan_arity_and_tool_allowlist(payload):
    with pytest.raises(ValueError):
        ReasoningPlan.model_validate_json(json.dumps(payload))


def test_ambiguous_question_interpretation_is_not_claimed_by_lineage_verification():
    # The checker deliberately cannot prove the selected quote answers the question.
    agent = DocumentReasoningAgent(corpus(), Planner(select_quote))
    result = agent.run("Requests Atlas latency")
    assert agent.verify_result(result, question="Requests Atlas latency")
    assert result.status == "answered"  # Semantic correctness needs separate labelled evaluation.
