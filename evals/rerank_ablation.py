"""Measure whether reranking improves local retrieval, and what it costs.

Runs every question in a labelled retrieval dataset through the real document index three
ways (fused order, word-overlap reorder, LLM reranker) and scores the top-k sources.

    python -m evals.rerank_ablation --k 3

Needs an ingested index (scripts/ingestion/ingest_repo_docs.py); the "llm" variant also
needs the provider key for RAG_RERANK_MODEL. Not a CI gate: it calls the live index.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_DATASET = Path("evals/datasets/retrieval_repo_docs.jsonl")


def _word_overlap(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    # The fallback that local_rag used before this measurement, kept for comparison.
    from agent.retrieval import _tokenize

    terms = set(_tokenize(query))
    scored = [
        (
            float(c.get("fused_score") or 0.0)
            + 0.05 * sum(t in (c.get("snippet") or "").lower() for t in terms)
            + 0.3 * sum(t in (c.get("file_name") or "").lower() for t in terms),
            index,
        )
        for index, c in enumerate(candidates)
    ]
    return [candidates[index] for _, index in sorted(scored, key=lambda pair: -pair[0])][:top_k]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--variants", default="fused,word_overlap,llm")
    args = parser.parse_args()

    load_dotenv()
    import agent.local_rag as rag
    from evals.retrieval_metrics import RetrievalCase, evaluate_rankings

    cases = [
        RetrievalCase.model_validate_json(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    llm_reranker = rag._llm_rerank_candidates
    rerankers = {
        "fused": lambda query, candidates, top_k: candidates[:top_k],
        "word_overlap": _word_overlap,
        "llm": llm_reranker,
    }
    captured: list[str] = []
    format_context = rag.format_hybrid_context

    def capture(candidates):
        captured[:] = [candidate["source"] for candidate in candidates]
        return format_context(candidates)

    rag.format_hybrid_context = capture
    rag._search_chroma_hybrid_knowledge("load the embedding model", limit=args.k)
    results = {}
    for name in args.variants.split(","):
        rag._llm_rerank_candidates = rerankers[name]
        rag.RAG_ENABLE_LLM_RERANKER = name == "llm"
        rankings, seconds = {}, []
        for case in cases:
            start = time.perf_counter()
            rag._search_chroma_hybrid_knowledge(case.query, limit=args.k)
            seconds.append(time.perf_counter() - start)
            rankings[case.id] = list(dict.fromkeys(captured))
        report = evaluate_rankings(cases, rankings, k=args.k)
        results[name] = {
            f"hit_rate_at_{args.k}": round(report.hit_rate, 3),
            "mrr": round(report.mrr, 3),
            "median_seconds": round(statistics.median(seconds), 2),
            "misses": [score.case_id for score in report.cases if not score.hit],
        }
    rag._llm_rerank_candidates = llm_reranker
    print(json.dumps({"dataset": str(args.dataset), "cases": len(cases), "variants": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
