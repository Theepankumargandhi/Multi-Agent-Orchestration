"""Offline, authored technical-report example; no model, service, Docker or persistence."""
from agentforge_sdk import DocumentPage, DocumentRAG


def main():
    rag = DocumentRAG([
        DocumentPage(document_id="atlas-report", source="atlas-report.pdf", title="Atlas technical report",
            version="r1", page=2, text="Atlas p95 latency is 180 ms at 100 requests per second. The report uses a warm cache."),
        DocumentPage(document_id="deployment-guide", source="deployment-guide.md", title="Deployment guide",
            version="r3", text="Deployments require owner approval. The health endpoint returns readiness status."),
    ])
    question = "What is Atlas p95 latency?"
    result = rag.answer(question, top_k=1)
    assert result.status == "evidence_found"
    assert rag.verify_answer(result, question=question)
    assert result.citations[0].page == 2 and "180 ms" in result.answer
    print(result.model_dump_json(indent=2))  # Authored demo only; real documents may be confidential.


if __name__ == "__main__":
    main()
