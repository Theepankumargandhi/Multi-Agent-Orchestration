"""Word overlap must not count a claim as supported by evidence that states the opposite."""
import pytest

from agent.grounding import evidence_from_state, repair_answer, verify_grounding


def _claim(claim: str, evidence: str):
    report = verify_grounding(route="rag", answer=claim, evidence=evidence_from_state({"rag_notes": evidence}))
    return report.claims[0]


@pytest.mark.parametrize(
    ("claim", "evidence", "kind"),
    [
        ("The model is available on Groq.", "The model is not available on Groq.", "negation"),
        ("This dependency is not safe to use.", "This dependency is safe to use in production.", "negation"),
        ("The API key is never stored on disk.", "The API key is stored on disk in plain text.", "negation"),
        ("The sandbox has network access.", "The sandbox has no network access.", "negation"),
        ("AgentForge does not use LangGraph.", "AgentForge uses LangGraph.", "negation"),
        ("The library was released in 2023.", "The library was released in 2025.", "numeric"),
        ("Python 3.11 is required.", "Python 3.12 is required for this release.", "numeric"),
        ("Requests are limited to 60 per minute.", "Requests are limited to 120 per minute.", "numeric"),
        ("Revenue fell 5% in 2024.", "Revenue rose 5% in 2024.", "opposite"),
        ("Error rates rose after the release.", "After the release, error rates dropped sharply.", "opposite"),
        ("The test suite passed on Windows.", "The test suite failed on Windows.", "opposite"),
        ("Memory is enabled by default.", "Memory is disabled by default.", "opposite"),
        ("The feature is enabled in production, but disabled in tests.", "The feature is disabled in production.",
         "opposite"),
    ],
)
def test_contradicting_evidence_is_not_support(claim, evidence, kind):
    assessment = _claim(claim, evidence)
    assert not assessment.supported
    assert assessment.contradiction == kind


@pytest.mark.parametrize(
    ("claim", "evidence"),
    [
        ("Revenue rose 5% in 2024.", "In 2024, revenue rose by 5%."),
        ("The API key is never stored on disk.", "The API key is never written to disk; it stays in memory."),
        ("Memory is disabled by default.", "By default, memory is disabled and must be enabled explicitly."),
        ("Tests failed before the fix and passed after it.",
         "Before the fix the tests failed; after the fix they passed."),
        ("The router does not send world facts to the knowledge graph.",
         "World facts are not sent to the knowledge graph by the router."),
        ("The ingestion stored 1,513 chunks.", "The ingestion stored 1513 chunks in 66 seconds."),
        ("Approval is required before web results are used.", "Web results are used only after approval."),
        # Found on real answers: a negation about other words is not a contradiction.
        ("The sandbox runs tests.", "The sandbox runs tests, not shell commands."),
        ("Evidence is missing or not independent enough.", "Evidence is absent or insufficiently independent."),
        ("Budget and cache isolation and rollback controls are in place.",
         "| Budget/cache isolation, routing artifacts and rollback controls | No provider load or failover |"),
    ],
)
def test_faithful_paraphrases_stay_supported(claim, evidence):
    assessment = _claim(claim, evidence)
    assert assessment.contradiction == ""
    assert assessment.supported


def test_citation_markers_and_retrieval_scores_are_not_compared_as_facts():
    # gpt-oss cites as "【source=..., page=22】"; RAG lines carry page and score numbers.
    claim = "Architecture decisions are documented in decisions.md【source=README.md, page=22】."
    evidence = "- source=docs/architecture/decisions.md, page=1, fused=0.0159, bm25=8.7: Architecture decisions are documented in decisions.md"
    assert _claim(claim, evidence).contradiction == ""


def test_one_contradiction_blocks_release_even_with_high_coverage():
    answer = (
        "AgentForge uses LangGraph for orchestration. The API uses FastAPI. The store uses SQLite. "
        "The cache uses Redis. AgentForge is not built on Python."
    )
    evidence = evidence_from_state({"rag_notes": (
        "AgentForge uses LangGraph for orchestration. The API uses FastAPI. The store uses SQLite. "
        "The cache uses Redis. AgentForge is built on Python."
    )})
    report = verify_grounding(route="rag", answer=answer, evidence=evidence)
    assert report.claim_coverage >= 0.8
    assert report.contradicted_claims == 1
    assert not report.passed and report.action == "repair"
    assert report.reason == "remove_contradicted_claims"
    repaired = repair_answer(report, evidence)
    assert "uses LangGraph" in repaired and "not built on Python" not in repaired
