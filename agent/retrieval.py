"""Shared text-RAG primitives with reusable, corpus-scoped lexical indexing."""

from __future__ import annotations

import re
from collections import Counter
from math import log
from typing import Any, Dict, List, Sequence, Tuple


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1]


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    content = (text or "").strip()
    if not content:
        return []
    chunk_size = max(200, int(chunk_size))
    chunk_overlap = max(0, int(chunk_overlap))
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 5

    # Prefer paragraph/sentence boundaries; fall back to hard splits only for
    # unusually long unbroken text. This keeps citations semantically coherent.
    units = [u.strip() for u in re.split(r"(?<=\.)\s+|\n{2,}", content) if u.strip()]
    chunks: List[str] = []
    current = ""
    for unit in units:
        if len(unit) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            step = max(1, chunk_size - chunk_overlap)
            chunks.extend(unit[start : start + chunk_size].strip() for start in range(0, len(unit), step))
            continue
        candidate = f"{current} {unit}".strip()
        if current and len(candidate) > chunk_size:
            chunks.append(current.strip())
            overlap = current[-chunk_overlap:].strip() if chunk_overlap else ""
            current = f"{overlap} {unit}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


class _BM25Index:
    """Precompute term contributions once; queries visit only matching postings.

    Each instance belongs to one fixed corpus. Repeated query terms and ties retain
    the original ranker's exact scoring and document-order semantics.
    """

    def __init__(self, docs: Sequence[str], *, terms_to_index: set[str] | None = None) -> None:
        frequencies = [Counter(_tokenize(text)) for text in docs]
        lengths = [sum(terms.values()) for terms in frequencies]
        average_length = (sum(lengths) / len(lengths)) if lengths else 1.0
        average_length = average_length or 1.0
        document_frequency = Counter(
            term
            for terms in frequencies
            for term in terms
            if terms_to_index is None or term in terms_to_index
        )
        count = max(1, len(docs))
        inverse_frequency = {
            term: log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        postings: Dict[str, List[Tuple[int, float]]] = {}
        for index, terms in enumerate(frequencies):
            length = max(1, lengths[index])
            for term, frequency in terms.items():
                if term not in inverse_frequency:
                    continue
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * (length / average_length))
                contribution = inverse_frequency[term] * ((frequency * 2.5) / max(1e-9, denominator))
                postings.setdefault(term, []).append((index, contribution))
        self._postings = {term: tuple(rows) for term, rows in postings.items()}

    def rank(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = {}
        for term in _tokenize(query):
            for index, contribution in self._postings.get(term, ()):
                scores[index] = scores.get(index, 0.0) + contribution
        return sorted(scores.items(), key=lambda row: (-row[1], row[0]))[: max(1, top_k)]


def _bm25_rank_documents(query: str, docs: List[str], top_k: int) -> List[Tuple[int, float]]:
    """One-shot BM25 ranking; fixed corpora can retain a _BM25Index instead."""
    terms = set(_tokenize(query))
    if not terms or not docs:
        return []
    return _BM25Index(docs, terms_to_index=terms).rank(query, top_k)


def _rrf_fuse_rankings(
    vector_ranked: List[Tuple[int, float]],
    bm25_ranked: List[Tuple[int, float]],
    rrf_k: int = 60,
) -> Dict[int, Dict[str, float]]:
    """
    Reciprocal Rank Fusion over vector + lexical ranks.
    Returns fused score map keyed by doc index.
    """
    fused: Dict[int, Dict[str, float]] = {}

    for rank, (doc_idx, score) in enumerate(vector_ranked, start=1):
        rec = fused.setdefault(doc_idx, {"fused": 0.0, "vector": 0.0, "bm25": 0.0})
        rec["vector"] = float(score)
        rec["fused"] += 1.0 / (rrf_k + rank)

    for rank, (doc_idx, score) in enumerate(bm25_ranked, start=1):
        rec = fused.setdefault(doc_idx, {"fused": 0.0, "vector": 0.0, "bm25": 0.0})
        rec["bm25"] = float(score)
        rec["fused"] += 1.0 / (rrf_k + rank)

    return fused


def _heuristic_rerank(query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """
    Cheap fallback reranker: boosts title/file-name and lexical overlap against snippet.
    """
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return candidates[: max(1, top_k)]

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for c in candidates:
        snippet = (c.get("snippet") or "").lower()
        file_name = (c.get("file_name") or "").lower()
        overlap = sum(1 for t in q_tokens if t in snippet)
        file_boost = sum(1 for t in q_tokens if t in file_name) * 0.3
        base = float(c.get("fused_score") or 0.0)
        score = base + overlap * 0.05 + file_boost
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[: max(1, top_k)]]
