"""Ranking parity and input/backend invariants for the reusable SDK index."""

from __future__ import annotations

import io
import math
import random
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pytest

from agentforge_sdk import (
    BackendError,
    DocumentInputError,
    DocumentPage,
    DocumentRAG,
    RAGConfig,
    Selection,
    _retrieval,
    load_text,
)


def _reference_bm25(query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
    """Frozen pre-index scoring equation, independent of the optimized implementation."""

    def tokenize(text):
        return [term for term in re.findall(r"[a-z0-9]+", text.lower()) if len(term) > 1]

    query_terms = tokenize(query)
    if not query_terms or not documents:
        return []
    tokens = [tokenize(text) for text in documents]
    lengths = [len(row) for row in tokens]
    average = sum(lengths) / len(lengths) or 1.0
    frequency = Counter(term for row in tokens for term in set(row))
    scores = []
    for index, row in enumerate(tokens):
        counts = Counter(row)
        score = 0.0
        for term in query_terms:
            if term not in counts:
                continue
            inverse = math.log(1 + (len(tokens) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            denominator = counts[term] + 1.5 * (1 - 0.75 + 0.75 * (max(1, lengths[index]) / average))
            score += inverse * ((counts[term] * (1.5 + 1)) / max(1e-9, denominator))
        if score > 0:
            scores.append((index, score))
    return sorted(scores, key=lambda row: row[1], reverse=True)[: max(1, top_k)]


@pytest.mark.parametrize("seed", range(12))
def test_reusable_and_one_shot_ranking_exactly_match_original_scores(seed):
    randomizer = random.Random(seed)
    vocabulary = ["latency", "Atlas", "deploy", "100", "a", "évidence", "Rust::async", "no-hit"]
    corpus = [" ".join(randomizer.choices(vocabulary, k=randomizer.randrange(60))) for _ in range(seed * 3)]
    corpus += ["Atlas latency", "Atlas latency"]  # Explicit tie ordering.
    index = _retrieval._BM25Index(corpus)
    for query in ("", "!", "a", "missing", "latency latency Atlas", "évidence 100", "Rust::async"):
        for top_k in (0, 1, 3, 100):
            expected = _reference_bm25(query, corpus, top_k)
            assert index.rank(query, top_k) == expected
            assert _retrieval._bm25_rank_documents(query, corpus, top_k) == expected


def test_immutable_corpus_is_not_retokenized_for_each_query(monkeypatch):
    text = "Atlas latency is 180 ms."
    rag = DocumentRAG([DocumentPage(document_id="atlas", source="report", text=text)])
    original = _retrieval._tokenize

    def query_only(value):
        assert value != text, "immutable corpus was tokenized again"
        return original(value)

    monkeypatch.setattr(_retrieval, "_tokenize", query_only)
    assert rag.search("Atlas latency").evidence
    assert rag.answer("Atlas latency").citations


@pytest.mark.parametrize("field", ["text", "source", "title", "version"])
def test_document_rejects_unpaired_surrogates_without_echoing_input(field):
    payload = {"document_id": "atlas", "source": "report", "text": "Atlas latency is 180 ms."}
    payload[field] = "PRIVATE-INPUT\ud800"
    with pytest.raises(ValueError) as error:
        DocumentPage(**payload)
    assert "PRIVATE-INPUT" not in str(error.value)


class _Embedder:
    name = "local-test-v1"
    external_processing = False

    def __init__(self):
        self.calls = 0

    def encode_documents(self, texts):
        self.calls += 1
        return [[1.0] for _ in texts]

    def encode_query(self, text):
        self.calls += 1
        return [1.0]


def test_invalid_unicode_question_is_rejected_before_any_backend_call():
    backend = _Embedder()
    rag = DocumentRAG(
        [DocumentPage(document_id="atlas", source="report", text="Atlas latency is 180 ms.")],
        embedder=backend,
    )
    with pytest.raises(ValueError, match="valid UTF-8"):
        rag.answer("PRIVATE-QUESTION\ud800")
    assert backend.calls == 0
    answer = rag.answer("Atlas")
    assert not rag.verify_answer(answer, question="Atlas\ud800")


def test_embedding_identity_change_cannot_reuse_cached_document_vectors():
    backend = _Embedder()
    rag = DocumentRAG(
        [DocumentPage(document_id="atlas", source="report", text="Atlas latency is 180 ms.")],
        embedder=backend,
    )
    assert rag.search("Atlas").evidence
    assert backend.calls == 2
    backend.name = "different-embedding-space"
    with pytest.raises(BackendError, match="identity changed"):
        rag.search("Atlas")
    assert backend.calls == 2


def test_selector_identity_change_is_rejected_before_invocation():
    class Selector:
        name, external_processing = "selector-v1", False
        calls = 0

        def select(self, question, evidence):
            self.calls += 1
            return [Selection(evidence_id=evidence[0].evidence_id, quote="Atlas latency is 180 ms.")]

    selector = Selector()
    rag = DocumentRAG(
        [DocumentPage(document_id="atlas", source="report", text="Atlas latency is 180 ms.")],
        selector=selector,
    )
    selector.name = "selector-v2"
    with pytest.raises(BackendError, match="identity changed"):
        rag.answer("Atlas")
    assert selector.calls == 0


def test_file_growth_cannot_turn_a_bounded_text_read_into_an_unbounded_one(tmp_path, monkeypatch):
    path = tmp_path / "report.txt"
    path.write_text("small preflight size", encoding="utf-8")

    class GrowingFile(io.BytesIO):
        def read(self, size=-1):
            assert size == 200_001
            return super().read(size)

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: GrowingFile(b"x" * 200_001))
    with pytest.raises(DocumentInputError):
        load_text(path, document_id="report")


def test_oversized_embedding_matrix_is_rejected_before_numeric_conversion():
    class OversizedRow(Sequence):
        accessed = False

        def __len__(self):
            return 4096

        def __getitem__(self, key):
            type(self).accessed = True
            raise AssertionError("over-budget matrix must not be converted")

    class OversizedEmbedder(_Embedder):
        def encode_documents(self, texts):
            self.calls += 1
            return [OversizedRow()] * len(texts)

    backend = OversizedEmbedder()
    pages = [
        DocumentPage(document_id=f"report-{number}", source="report", text="Atlas latency is 180 ms.")
        for number in range(500)
    ]
    rag = DocumentRAG(pages, config=RAGConfig(max_pages=500), embedder=backend)
    result = rag.search("Atlas latency")
    assert result.status == "degraded" and result.reason == "embedding_fallback"
    assert backend.calls == 1
    assert not OversizedRow.accessed
