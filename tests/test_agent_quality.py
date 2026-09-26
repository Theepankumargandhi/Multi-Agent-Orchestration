from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from agent.graph_rag import _chunk_text as graph_chunk_text
from agent.knowledge_graph import _build_graph, _extract_triples, _select_relationships
from agent.llama_guard import LlamaGuardOutput, SafetyAssessment, parse_llama_guard_output
from agent.local_rag import _chunk_id
from agent.local_rag import _chunk_text as rag_chunk_text
from agent.research_assistant import (
    _build_response_context,
    _ensure_web_citations,
    _evaluate_response_quality,
    _next_node_after_safety,
    intent_router_agent,
    safety_agent,
    web_hitl_gate_agent,
)


def test_guard_parser_is_fail_closed_on_malformed_output():
    result = parse_llama_guard_output("maybe safe")
    assert result.safety_assessment is SafetyAssessment.ERROR


@pytest.mark.asyncio
async def test_safety_failure_blocks_and_clears_stale_evidence():
    moderation_error = LlamaGuardOutput(
        safety_assessment=SafetyAssessment.ERROR, error_message="provider timeout"
    )
    state = {
        "messages": [HumanMessage(content="hello")],
        "web_notes": "old web data",
        "rag_notes": "old rag data",
        "kg_notes": "old graph data",
        "math_result": "42",
    }
    with patch("agent.research_assistant.llama_guard", AsyncMock(return_value=moderation_error)):
        update = await safety_agent(state, {"configurable": {}})
    assert update["safety_blocked"] is True
    assert _next_node_after_safety(update) == "response_agent"
    assert update["web_notes"] == update["rag_notes"] == update["kg_notes"] == ""
    assert update["math_result"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("calculate 19 * 7", "math"),
        ("latest AI security news", "web"),
        ("explain this repository architecture", "rag"),
        ("latest news compared with this codebase", "hybrid"),
        ("how does FastAPI connect to LangGraph in this project", "kg"),
    ],
)
async def test_deterministic_router_paths(query, expected):
    result = await intent_router_agent(
        {"messages": [HumanMessage(content=query)], "query": query},
        {"configurable": {"model": "unused"}},
    )
    assert result["route"] == expected


def test_hybrid_context_keeps_web_and_rag_but_not_stale_kg():
    context = _build_response_context(
        {
            "query": "compare",
            "route": "hybrid",
            "web_notes": "WEB_EVIDENCE",
            "rag_notes": "RAG_EVIDENCE",
            "kg_notes": "STALE_KG_EVIDENCE",
            "math_result": "STALE_MATH",
        }
    )
    assert "WEB_EVIDENCE" in context and "RAG_EVIDENCE" in context
    assert "STALE_KG_EVIDENCE" not in context and "STALE_MATH" not in context


def test_web_quality_score_rewards_citations():
    base = {"route": "web", "web_notes": "- source", "final_response": "A supported answer."}
    without, _ = _evaluate_response_quality(base)
    with_links, _ = _evaluate_response_quality(
        {**base, "final_response": "See [one](https://example.com/1) and [two](https://example.com/2)."}
    )
    assert with_links > without


def test_missing_web_citation_is_repaired_from_retrieved_evidence():
    answer = _ensure_web_citations(
        "A grounded summary.",
        "- Example report\n  Link: https://example.com/report\n  Snippet: Evidence.",
    )
    assert "[Example report](https://example.com/report)" in answer


@pytest.mark.parametrize("chunker", [rag_chunk_text, graph_chunk_text])
def test_semantic_chunkers_bound_long_unbroken_text(chunker):
    chunks = chunker("x" * 1000, chunk_size=250, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 250 for chunk in chunks)


def test_chunk_ids_are_deterministic_and_content_sensitive():
    first = _chunk_id("a.pdf", "abc", 1, 0, "hello")
    assert first == _chunk_id("a.pdf", "abc", 1, 0, "hello")
    assert first != _chunk_id("a.pdf", "abc", 1, 0, "changed")


@pytest.mark.asyncio
async def test_live_evaluation_can_bypass_interactive_hitl_without_public_api_flag():
    result = await web_hitl_gate_agent(
        {"route": "web", "query": "latest AI news"},
        {"configurable": {"evaluation_bypass_hitl": True}},
    )
    assert result == {"web_hitl_decision": "evaluation_bypass"}


def test_knowledge_graph_extracts_predicates_and_multihop_path():
    triples = _extract_triples("FastAPI calls LangGraph. LangGraph uses ChromaDB.")
    assert ("fastapi", "calls", "langgraph") in triples
    graph = _build_graph([{"source": "architecture.pdf", "snippet": "FastAPI calls LangGraph. LangGraph uses ChromaDB."}])
    relationships = _select_relationships(graph, "How does FastAPI reach ChromaDB?", limit=5)
    predicates = {row[2] for row in relationships}
    assert {"calls", "uses"}.issubset(predicates)
