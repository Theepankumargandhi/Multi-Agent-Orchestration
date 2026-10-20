from __future__ import annotations

from fastapi.testclient import TestClient

from agentforge_sdk import DocumentRAG
from agentforge_sdk.demo import demo_pages
from examples.document_sdk_fastapi import create_app


def test_http_developer_integration_returns_structured_page_citations():
    rag = DocumentRAG(demo_pages())
    with TestClient(create_app(rag)) as client:
        result = client.post("/answer", json={"question": "What is Atlas p95 latency?", "top_k": 1})
        assert result.status_code == 200
        body = result.json()
        assert body["status"] == "evidence_found" and "180 ms" in body["answer"]
        assert body["citations"][0]["page"] == 2 and body["selector_invocations"] == 0
        from agentforge_sdk import AnswerResult
        assert rag.verify_answer(AnswerResult.model_validate_json(result.content))


def test_http_empty_or_invalid_requests_do_not_echo_private_inputs():
    with TestClient(create_app(DocumentRAG(demo_pages()))) as client:
        for body in ({"question": "PRIVATE-QUESTION", "unexpected": "PRIVATE-SECRET"},
                     {"question": "PRIVATE-QUESTION", "top_k": True}, {"question": "\x00PRIVATE-QUESTION"}):
            result = client.post("/answer", json=body)
            assert result.status_code in {400, 422}
            assert "PRIVATE" not in result.text
        result = client.post("/answer", json={"question": "xylophonium"})
        assert result.status_code == 200 and result.json()["status"] == "abstained"
