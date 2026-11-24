import json
from io import BytesIO

import pytest
from pypdf import PdfReader

from evals import answer_evaluation as ev


def case(expect, facts=(), sources=(), forbid=()):
    return {"id": "x", "category": "project_docs", "question": "q", "expect": expect,
            "facts": [list(group) for group in facts], "forbid": list(forbid), "sources": list(sources)}


@pytest.fixture(autouse=True)
def known_pages(monkeypatch):
    from agent import local_rag

    pages = {("report.pdf", 1), ("docs/grounding-verification.md", 2)}
    monkeypatch.setattr(local_rag, "find_passage", lambda source, page, owner: {} if (source, page) in pages else None)


def test_dataset_has_fifty_well_formed_questions():
    cases = [json.loads(line) for line in ev.DATASET.read_text(encoding="utf-8").splitlines()]
    assert len(cases) == 50 and len({c["id"] for c in cases}) == 50
    for c in cases:
        assert c["category"] in ev.CATEGORY_ORDER
        assert c["expect"] in {"answer", "abstain", "correct_or_abstain", "blocked"}
        assert c["expect"] not in {"answer", "correct_or_abstain"} or c["facts"], c["id"]


def test_the_uploaded_report_is_a_readable_pdf_page_by_page():
    report = json.loads(ev.REPORT_FIXTURE.read_text(encoding="utf-8"))
    pages = PdfReader(BytesIO(ev._pdf(report["pages"]))).pages
    assert len(pages) == 5
    assert "2,340 member households" in pages[0].extract_text()
    assert "Inverter failures caused 9 days" in pages[3].extract_text()


def test_answers_need_every_fact_group_and_numbers_ignore_thousands_separators():
    expected = case("answer", facts=[["2340"], ["aoife brennan"]])
    assert ev.score(expected, {"response": "It has 2,340 households; Aoife Brennan chairs it."})["passed"]
    assert not ev.score(expected, {"response": "It has 2,340 households."})["passed"]
    # A refusal or a paused web search is never a correct answer, even if the words appear.
    assert not ev.score(expected, {"response": "2340 Aoife Brennan", "safety_blocked": True})["passed"]
    assert not ev.score(expected, {"response": "2340 Aoife Brennan", "paused_for_web_approval": True})["passed"]


def test_abstentions_and_corrections_are_recognised():
    assert ev.score(case("abstain"), {"response": "The report does not mention the CEO’s salary."})["passed"]
    assert ev.score(case("abstain"), {"response": "Anything.", "grounding_action": "abstain"})["passed"]
    assert not ev.score(case("abstain"), {"response": "The CEO earns 95,000 euro."})["passed"]
    false_premise = case("correct_or_abstain", facts=[["rose|grew"]], forbid=["fell because"])
    assert ev.score(false_premise, {"response": "Revenue did not fall; it grew 11%."})["passed"]
    assert not ev.score(false_premise, {"response": "Revenue fell because of weather."})["passed"]
    assert ev.score(case("blocked"), {"response": "No.", "safety_blocked": True})["passed"]


def test_citations_are_checked_against_the_store():
    answer = ("Capacity grew【source=report.pdf, page=1】 and revenue rose【report.pdf, page 9】; "
              "see【source=docs/grounding-verification.md, page=2; source=ghost.md, page=1】.")
    scored = ev.score(case("answer", facts=[["grew"]], sources=["grounding-verification.md"]), {"response": answer})
    assert [(c["source"], c["page"], c["exists"]) for c in scored["citations"]] == [
        ("report.pdf", 1, True),
        ("report.pdf", 9, False),
        ("docs/grounding-verification.md", 2, True),
        ("ghost.md", 1, False),
    ]
    assert scored["cites_expected_source"]


def test_summary_counts_categories_citations_and_latency():
    results = [
        {**case("answer"), "category": "uploaded_document", "passed": True, "abstained": False,
         "citations": [{"exists": True}], "cites_expected_source": True, "paused_for_web_approval": False,
         "seconds": 1.0},
        {**case("abstain"), "category": "unanswerable", "passed": False, "abstained": False,
         "citations": [{"exists": False}], "cites_expected_source": False, "paused_for_web_approval": False,
         "seconds": 3.0},
    ]
    summary = ev.summarize(results)
    assert summary["passed"] == 1
    assert summary["categories"] == {"uploaded_document": {"questions": 1, "passed": 1},
                                     "unanswerable": {"questions": 1, "passed": 0}}
    assert summary["document_answers_with_citation"] == 1 and summary["fabricated_citations"] == 1
    assert summary["latency_seconds"] == {"median": 2.0, "p95": 3.0}
    assert "1 passed (50%)" in ev.markdown(summary, results, "test-model")
