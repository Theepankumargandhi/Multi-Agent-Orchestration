"""Offline authored technical-report demonstration. No files, models or services are loaded."""

from __future__ import annotations

import argparse
import json

from agentforge_sdk.models import DocumentPage
from agentforge_sdk.rag import DocumentRAG


def demo_pages():
    return (
        DocumentPage(
            document_id="atlas-report",
            source="atlas-report.pdf",
            title="Atlas technical report",
            version="r1",
            page=2,
            text="Atlas p95 latency is 180 ms at 100 requests per second. The report uses a warm cache.",
        ),
        DocumentPage(
            document_id="deployment-guide",
            source="deployment-guide.md",
            title="Deployment guide",
            version="r3",
            text="Deployments require owner approval. The health endpoint returns readiness status.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="What is Atlas p95 latency?")
    args = parser.parse_args()
    rag = DocumentRAG(demo_pages())
    result = rag.answer(args.question, top_k=1)
    print(
        json.dumps(
            {
                "evidence_kind": "authored_offline_demo",
                "source_provenance_verified": rag.verify_answer(result, question=args.question),
                "result": result.model_dump(mode="json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
