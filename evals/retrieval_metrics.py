"""Information-retrieval metrics and rank fusion for RAG ablations."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from pydantic import BaseModel, Field


class RetrievalCase(BaseModel):
    id: str
    query: str
    relevant_ids: set[str] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class RetrievalCaseScore(BaseModel):
    case_id: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    hit: bool


class RetrievalReport(BaseModel):
    k: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float
    slices: dict[str, dict[str, float]]
    cases: list[RetrievalCaseScore]


def reciprocal_rank_fusion(rankings: list[list[str]], rank_constant: int = 60) -> list[str]:
    """Fuse lexical/vector/graph rankings without requiring comparable raw scores."""
    scores: defaultdict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    seen_index = 0
    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] += 1.0 / (rank_constant + rank)
            if document_id not in first_seen:
                first_seen[document_id] = seen_index
                seen_index += 1
    return sorted(scores, key=lambda item: (-scores[item], first_seen[item]))


def _score_case(case: RetrievalCase, ranked_ids: list[str], k: int) -> RetrievalCaseScore:
    top = ranked_ids[:k]
    relevant = case.relevant_ids
    hits = [document_id for document_id in top if document_id in relevant]
    first_rank = next((index for index, item in enumerate(ranked_ids, start=1) if item in relevant), None)
    dcg = sum(1.0 / math.log2(index + 1) for index, item in enumerate(top, start=1) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return RetrievalCaseScore(
        case_id=case.id,
        recall_at_k=len(set(hits)) / len(relevant),
        precision_at_k=len(hits) / k,
        reciprocal_rank=1.0 / first_rank if first_rank else 0.0,
        ndcg_at_k=dcg / idcg if idcg else 0.0,
        hit=bool(hits),
    )


def evaluate_rankings(
    cases: list[RetrievalCase],
    rankings: dict[str, list[str]],
    k: int = 5,
) -> RetrievalReport:
    if not cases:
        raise ValueError("at least one retrieval case is required")
    if k < 1:
        raise ValueError("k must be positive")
    scores = [_score_case(case, rankings.get(case.id, []), k) for case in cases]

    def aggregate(selected: list[RetrievalCaseScore]) -> dict[str, float]:
        return {
            "recall_at_k": statistics.fmean(item.recall_at_k for item in selected),
            "precision_at_k": statistics.fmean(item.precision_at_k for item in selected),
            "mrr": statistics.fmean(item.reciprocal_rank for item in selected),
            "ndcg_at_k": statistics.fmean(item.ndcg_at_k for item in selected),
            "hit_rate": statistics.fmean(float(item.hit) for item in selected),
        }

    by_id = {item.case_id: item for item in scores}
    slice_members: defaultdict[str, list[RetrievalCaseScore]] = defaultdict(list)
    for case in cases:
        for tag in case.tags:
            slice_members[tag].append(by_id[case.id])
    overall = aggregate(scores)
    return RetrievalReport(
        k=k,
        **overall,
        slices={tag: aggregate(items) for tag, items in sorted(slice_members.items())},
        cases=scores,
    )
