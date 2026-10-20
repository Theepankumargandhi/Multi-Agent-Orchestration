"""Reproducible authored developer-integration controls, not a real QA benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentforge_sdk import DocumentRAG, ExternalProcessingDenied, RAGConfig, Selection
from agentforge_sdk.demo import demo_pages
from agentforge_sdk.models import fingerprint


def evaluate():
    pages = demo_pages()
    rag = DocumentRAG(pages)
    answer = rag.answer("What is Atlas p95 latency?", top_k=1)
    checks = [
        {"name": "literal_answer", "passed": answer.status == "evidence_found" and "180 ms" in answer.answer},
        {"name": "one_based_page_provenance", "passed": len(answer.citations) == 1 and answer.citations[0].page == 2},
        {"name": "quote_and_receipt_integrity", "passed": rag.verify_answer(answer, question="What is Atlas p95 latency?")},
        {"name": "plain_text_has_no_invented_page", "passed": rag.answer("Deployments owner approval", top_k=1).citations[0].page is None},
        {"name": "unrelated_question_abstains", "passed": rag.answer("xylophonium").status == "abstained"},
        {"name": "isolated_corpus", "passed": DocumentRAG([pages[1]]).answer("Atlas latency").status == "abstained"},
        {"name": "tampered_quote_rejected", "passed": not rag.verify_citation(answer.citations[0].model_copy(update={"quote": "fabricated"}))},
    ]

    class BadSelector:
        name, external_processing = "authored-bad-selector", False

        def select(self, question, evidence):
            return [Selection(evidence_id=evidence[0].evidence_id, quote="Atlas latency is 1 ms.")]

    held = DocumentRAG(pages, selector=BadSelector()).answer("Atlas p95 latency", top_k=1)
    checks.append({"name": "unsupported_selector_quote_held", "passed": held.status == "abstained" and held.reason == "selection_rejected"})

    class ExternalSelector(BadSelector):
        external_processing = True

    try:
        DocumentRAG(pages, selector=ExternalSelector())
    except ExternalProcessingDenied:
        denied = True
    else:
        denied = False
    checks.append({"name": "unconsented_external_backend_blocked", "passed": denied})
    large = pages[0].model_copy(update={"text": "Atlas " * 100})
    held = DocumentRAG([large], config=RAGConfig(max_context_chars=256)).answer("Atlas")
    checks.append({"name": "context_budget_preserves_complete_chunks", "passed": held.status == "abstained" and held.retrieval.context_chars == 0})
    result = {"protocol": "document-sdk-controls-v1", "evidence_kind": "authored_offline_controls",
              "corpus_sha256": fingerprint([page.model_dump() for page in pages]), "total": len(checks),
              "passed": sum(row["passed"] for row in checks), "checks": checks,
              "live_provider_calls": 0, "real_world_quality_measured": False, "multimodal_exercised": False}
    result["fingerprint"] = fingerprint(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
