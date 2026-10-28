import importlib

import pytest

from agent.grounding import (
    GroundingPolicy,
    evidence_from_state,
    fragment_is_supported,
    repair_answer,
    take_complete_sentences,
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
async def test_graph_nodes_hold_draft_until_verification_then_release_or_abstain(monkeypatch):
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


def test_streaming_buffer_splits_only_finished_sentences():
    sentences, rest = take_complete_sentences("First fact here. Second fact")
    assert sentences == ["First fact here. "]
    assert rest == "Second fact"
    # The separator at the end may still grow, so it is not split yet.
    assert take_complete_sentences("One. ") == ([], "One. ")
    fenced, rest = take_complete_sentences("Run this:\n```\nx = 1.\ny = 2\n")
    assert fenced == ["Run this:\n"]
    assert rest.startswith("```")


def test_streamed_fragment_needs_every_claim_supported():
    evidence = _web_evidence()
    assert fragment_is_supported("AgentForge uses LangGraph for agent orchestration.", evidence)
    assert not fragment_is_supported("Revenue is guaranteed to grow by 99% in 2026.", evidence)
    assert not fragment_is_supported(
        "AgentForge does not use LangGraph for agent orchestration.", evidence
    )
    assert not fragment_is_supported(
        "AgentForge uses LangGraph [source](https://invented.example/page).", evidence
    )
    assert fragment_is_supported("Would you like more detail on this?", evidence)


@pytest.mark.asyncio
async def test_response_streams_verified_sentences_and_holds_unsupported_ones(monkeypatch):
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableLambda

    research = importlib.import_module("agent.research_assistant")
    monkeypatch.setattr(research, "GROUNDING_VERIFICATION_ENABLED", True)
    draft = (
        "AgentForge uses LangGraph for agent orchestration. "
        "Revenue is guaranteed to grow by 99% in 2026. "
        "LangGraph handles the agent orchestration in AgentForge."
    )
    monkeypatch.setattr(research, "_get_model", lambda _config: FakeListChatModel(responses=[draft]))

    class Client(AsyncCallbackHandler):
        def __init__(self):
            self.tokens: list[str] = []
            self.released: list[str] = []

        async def on_llm_new_token(self, token, **_kwargs):
            self.tokens.append(token)

        async def on_custom_event(self, name, data, **_kwargs):
            if name == research.VERIFIED_TEXT_EVENT:
                self.released.append(data)

    client = Client()
    state = {
        "messages": [HumanMessage(content="What does AgentForge use?")],
        "query": "What does AgentForge use?",
        "route": "web",
        "web_notes": (
            "- Architecture report\n"
            "  Link: https://example.com/architecture\n"
            "  Snippet: AgentForge uses LangGraph for agent orchestration."
        ),
    }
    node = RunnableLambda(research.response_agent)
    update = await node.ainvoke(state, {"configurable": {}, "callbacks": [client]})

    assert client.tokens == []  # raw draft tokens never reach the client
    assert client.released == [
        "AgentForge uses LangGraph for agent orchestration. ",
        "LangGraph handles the agent orchestration in AgentForge. ",
    ]
    assert "messages" not in update  # the verifier still decides the released message
    assert "99%" in update["final_response"]


@pytest.mark.asyncio
async def test_general_chat_streams_tokens_with_grounding_on(monkeypatch):
    from unittest.mock import AsyncMock

    from langchain_core.messages import HumanMessage

    research = importlib.import_module("agent.research_assistant")
    monkeypatch.setattr(research, "GROUNDING_VERIFICATION_ENABLED", True)
    llm = AsyncMock(return_value="Hello there.")
    monkeypatch.setattr(research, "_call_llm", llm)
    await research.response_agent(
        {"messages": [HumanMessage(content="hi")], "query": "hi", "route": "general"},
        {"configurable": {}},
    )
    assert llm.await_args.kwargs["stream_to_client"] is True


@pytest.mark.asyncio
async def test_service_forwards_verified_text_as_tokens():
    import asyncio

    from service.service import VERIFIED_TEXT_EVENT, TokenQueueStreamingHandler

    queue: asyncio.Queue = asyncio.Queue()
    handler = TokenQueueStreamingHandler(queue)
    await handler.on_custom_event(VERIFIED_TEXT_EVENT, "Checked sentence. ", run_id=None)
    await handler.on_custom_event("other_event", "ignored", run_id=None)
    assert queue.get_nowait() == "Checked sentence. "
    assert queue.empty()


def test_unsupported_high_risk_claim_is_removed_not_the_whole_answer():
    evidence = evidence_from_state(
        {
            "rag_notes": (
                "Web approval uses LangGraph interrupt and resumes with Command. "
                "The service stores the checkpoint for each user thread."
            )
        }
    )
    answer = (
        "Web approval uses a LangGraph interrupt and resumes with Command.\n"
        "- The service stores the checkpoint for each user thread.\n"
        "- Approved searches always finish within 2 seconds."
    )
    report = verify_grounding(route="rag", answer=answer, evidence=evidence)
    assert report.action == "repair"
    assert report.reason == "remove_unsupported_high_risk_claims"
    repaired = repair_answer(report, evidence, answer)
    assert "2 seconds" not in repaired
    # The supported part keeps its layout, matching what the stream released.
    assert repaired.startswith(
        "Web approval uses a LangGraph interrupt and resumes with Command.\n"
        "- The service stores the checkpoint for each user thread."
    )
