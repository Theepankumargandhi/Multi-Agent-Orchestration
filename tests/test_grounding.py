import importlib

import pytest

from agent.grounding import (
    GroundingPolicy,
    evidence_from_state,
    repair_answer,
    verify_grounding,
    verify_report,
)


def _web_evidence():
    return evidence_from_state(
        {
            "web_notes": (
                "- Architecture report\n"
                "  Link: https://example.com/architecture\n"
                "  Snippet: AgentForge uses LangGraph for agent orchestration."
            )
        }
    )


def test_supported_claim_and_allowlisted_citation_pass_with_integrity_receipt():
    report = verify_grounding(
        route="web",
        answer=(
            "AgentForge uses LangGraph for agent orchestration "
            "[source](https://example.com/architecture)."
        ),
        evidence=_web_evidence(),
        integrity_key=b"grounding-test-integrity-key",
    )
    assert report.passed and report.action == "pass"
    assert report.claim_coverage == 1
    assert report.citation_precision == 1
    assert verify_report(report, b"grounding-test-integrity-key")


def test_fabricated_citation_and_unsupported_high_risk_claim_fail_closed():
    fake = verify_grounding(
        route="web",
        answer="AgentForge uses LangGraph [source](https://malicious.example/fake).",
        evidence=_web_evidence(),
    )
    risky = verify_grounding(
        route="web",
        answer="AgentForge is guaranteed to reduce costs by 97% in 2026.",
        evidence=_web_evidence(),
    )
    assert not fake.passed and fake.citation_precision == 0
    assert fake.action == "abstain"
    assert risky.action == "abstain"
    assert risky.unsupported_high_risk_claims == 1


def test_partial_support_uses_bounded_repair_and_removes_unsupported_claim():
    report = verify_grounding(
        route="rag",
        answer=(
            "AgentForge uses LangGraph for agent orchestration. "
            "The system was invented on the Moon."
        ),
        evidence=evidence_from_state(
            {"rag_notes": "AgentForge uses LangGraph for agent orchestration."}
        ),
        policy=GroundingPolicy(min_coverage=0.8),
    )
    repaired = repair_answer(
        report,
        evidence_from_state({"rag_notes": "AgentForge uses LangGraph for agent orchestration."}),
    )
    assert report.action == "repair"
    assert "uses LangGraph" in repaired
    assert "Moon" not in repaired
    assert "omitted claims" in repaired


def test_math_result_is_verified_and_general_chat_is_not_forced_to_retrieve():
    math_report = verify_grounding(
        route="math",
        answer="The result of 19 multiplied by 7 is 133.",
        evidence=evidence_from_state({"math_result": "19 * 7 = 133"}),
    )
    general = verify_grounding(
        route="general",
        answer="Groundedness means connecting claims to supporting evidence.",
        evidence=[],
    )
    assert math_report.passed
    assert general.action == "not_required" and general.passed


def test_grounding_report_tampering_is_detected():
    report = verify_grounding(
        route="web",
        answer="AgentForge uses LangGraph for orchestration.",
        evidence=_web_evidence(),
    )
    assert verify_report(report)
    report.confidence = 0
    assert not verify_report(report)


@pytest.mark.asyncio
async def test_graph_nodes_hold_draft_until_verification_then_release_or_abstain(
    tmp_path, monkeypatch
):
    research = importlib.import_module("agent.research_assistant")
    monkeypatch.setattr(research, "GROUNDING_VERIFICATION_ENABLED", True)
    supported_state = {
        "route": "web",
        "final_response": (
            "AgentForge uses LangGraph for agent orchestration "
            "[source](https://example.com/architecture)."
        ),
        "web_notes": (
            "Link: https://example.com/architecture\n"
            "Snippet: AgentForge uses LangGraph for agent orchestration."
        ),
    }
    draft = research._draft_response_update(supported_state["final_response"], "web")
    verified = await research.grounding_verifier_agent(supported_state, {"configurable": {}})
    assert "messages" not in draft
    assert verified["grounding_action"] == "pass"
    assert verified["messages"][0].content == supported_state["final_response"]

    monkeypatch.setattr(research, "UNCERTAINTY_CALIBRATION_ENABLED", True)
    monkeypatch.setattr(research, "UNCERTAINTY_CALIBRATOR_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(research, "_uncertainty_calibrator", None)
    gated = await research.grounding_verifier_agent(supported_state, {"configurable": {}})
    assert gated["grounding_action"] == "abstain"
    assert gated["uncertainty_receipt"]["status"] == "calibrator_unavailable"
    assert "messages" not in gated
    monkeypatch.setattr(research, "UNCERTAINTY_CALIBRATION_ENABLED", False)

    unsupported = {
        "route": "web",
        "final_response": "Revenue is guaranteed to grow by 99% in 2026.",
        "web_notes": supported_state["web_notes"],
    }
    verdict = await research.grounding_verifier_agent(unsupported, {"configurable": {}})
    repaired = await research.grounding_repair_agent(
        {**unsupported, **verdict}, {"configurable": {}}
    )
    assert verdict["grounding_action"] == "abstain"
    assert "enough verified evidence" in repaired["final_response"]
    assert repaired["messages"][0].content == repaired["final_response"]


@pytest.mark.asyncio
async def test_adaptive_deliberation_releases_only_consensus_candidate(monkeypatch):
    research = importlib.import_module("agent.research_assistant")
    monkeypatch.setattr(research, "UNCERTAINTY_CALIBRATION_ENABLED", False)
    monkeypatch.setattr(research, "ADAPTIVE_COMPUTE_INTEGRITY_KEY", None)

    async def candidate_llm(*args, **kwargs):
        return (
            "AgentForge uses LangGraph for agent orchestration "
            "[source](https://example.com/architecture)."
        )

    monkeypatch.setattr(research, "_call_llm", candidate_llm)
    plan = research.plan_compute(
        research.ComputeSignals(
            route="web",
            grounding_action="repair",
            grounding_confidence=0.5,
            uncertainty_decision="not_evaluated",
            evidence_count=1,
        ),
        research._adaptive_compute_policy(),
    )
    state = {
        "route": "web",
        "query": "How is AgentForge orchestrated?",
        "final_response": "The architecture is unclear.",
        "web_notes": (
            "Link: https://example.com/architecture\n"
            "Snippet: AgentForge uses LangGraph for agent orchestration."
        ),
        "adaptive_compute_plan": plan.model_dump(mode="json"),
        "answer_source_meta": "web",
    }
    result = await research.adaptive_deliberation_agent(state, {"configurable": {}})
    assert result["grounding_action"] == "pass"
    assert result["adaptive_compute_receipt"]["status"] == "released"
    assert result["adaptive_compute_receipt"]["selected_consensus"] == 1
    assert result["messages"][0].content == result["final_response"]
