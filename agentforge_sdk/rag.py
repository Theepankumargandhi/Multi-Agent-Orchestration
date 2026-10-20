"""Instance-isolated text retrieval and literal, provenance-bound evidence answers."""

from __future__ import annotations

import hashlib
import math
import re
from threading import RLock
from typing import Protocol, Sequence

from agentforge_sdk._retrieval import (
    _BM25Index,
    _chunk_text,
    _heuristic_rerank,
    _rrf_fuse_rankings,
    _tokenize,
)
from agentforge_sdk.models import (
    AnswerResult,
    BackendError,
    Citation,
    DocumentPage,
    Evidence,
    ExternalProcessingDenied,
    RAGConfig,
    SearchResult,
    Selection,
    fingerprint,
)


class EmbeddingBackend(Protocol):
    name: str
    external_processing: bool

    def encode_documents(self, texts: list[str]) -> Sequence[Sequence[float]]: ...

    def encode_query(self, text: str) -> Sequence[float]: ...


class EvidenceSelector(Protocol):
    name: str
    external_processing: bool

    def select(self, question: str, evidence: tuple[Evidence, ...]) -> Sequence[Selection]: ...


def _backend_name(backend: EmbeddingBackend | EvidenceSelector, config: RAGConfig) -> str:
    name, external = getattr(backend, "name", None), getattr(backend, "external_processing", None)
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,120}", name)
        or type(external) is not bool
    ):
        raise ValueError("backends must declare a bounded name and external_processing boolean")
    if external and not config.allow_external_processing:
        raise ExternalProcessingDenied("external processing requires explicit instance-level consent")
    return name


def _check_backend(backend: EmbeddingBackend | EvidenceSelector, name: str, config: RAGConfig) -> None:
    if _backend_name(backend, config) != name:
        raise BackendError("backend identity changed; construct a new corpus instance")


def _normalize_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip() or len(question) > 4000 or "\x00" in question:
        raise ValueError("question must be nonempty bounded text")
    try:
        question.encode("utf-8")
    except UnicodeError:
        raise ValueError("question must be valid UTF-8 text") from None
    return question.strip()


def _vector(values: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not 1 <= len(values) <= 4096:
        raise ValueError("embedding shape is unsupported")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("embeddings must contain finite numbers")
    norm = math.sqrt(math.fsum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("zero or overflowing embeddings are unsupported")
    return tuple(float(value / norm) for value in values)


def _quotes(question: str, evidence: tuple[Evidence, ...], maximum: int) -> list[Selection]:
    terms = set(_tokenize(question))
    selected = []
    for item in evidence:
        sentences = [
            match.group().strip() for match in re.finditer(r".+?(?:[.!?](?=\s|$)|\n|$)", item.text, re.DOTALL)
        ]
        sentences = [value for value in sentences if value]
        if not sentences:
            continue
        quote = max(enumerate(sentences), key=lambda pair: (len(terms & set(_tokenize(pair[1]))), -pair[0]))[
            1
        ]
        if len(quote) > maximum:
            # A literal prefix is an excerpt, never presented as a generated claim.
            quote = quote[:maximum].rsplit(" ", 1)[0] or quote[:maximum]
        if len(quote) >= 3:
            selected.append(Selection(evidence_id=item.evidence_id, quote=quote))
    return selected


class DocumentRAG:
    """Immutable in-memory corpus. No environment, provider, tracing or persistent-store defaults."""

    def __init__(
        self,
        pages: Sequence[DocumentPage],
        *,
        config: RAGConfig | None = None,
        embedder: EmbeddingBackend | None = None,
        selector: EvidenceSelector | None = None,
    ) -> None:
        self._config = RAGConfig.model_validate((config or RAGConfig()).model_dump())
        if (
            not isinstance(pages, Sequence)
            or isinstance(pages, (str, bytes))
            or len(pages) > self._config.max_pages
        ):
            raise ValueError("pages must be a bounded sequence")
        clean = [DocumentPage.model_validate(page.model_dump()) for page in pages]
        identities = [(page.document_id, page.version, page.page) for page in clean]
        if len(set(identities)) != len(identities):
            raise ValueError("document/version/page identities must be unique")
        if sum(len(page.text.encode()) for page in clean) > self._config.max_corpus_bytes:
            raise ValueError("corpus byte budget exceeded")
        self._pages = tuple(sorted(clean, key=lambda page: (page.document_id, page.version, page.page or 0)))
        self._corpus_hash = fingerprint([page.model_dump() for page in self._pages])
        chunks = []
        for page in self._pages:
            page_hash = hashlib.sha256(page.text.encode()).hexdigest()
            for number, text in enumerate(
                _chunk_text(page.text, self._config.chunk_size, self._config.chunk_overlap)
            ):
                chunk_hash = hashlib.sha256(text.encode()).hexdigest()
                identity = fingerprint(
                    {
                        "document_id": page.document_id,
                        "source": page.source,
                        "version": page.version,
                        "page": page.page,
                        "input_text_sha256": page_hash,
                        "chunk": number,
                        "chunk_sha256": chunk_hash,
                    }
                )
                chunks.append(
                    {
                        "evidence_id": identity,
                        "document_id": page.document_id,
                        "source": page.source,
                        "version": page.version,
                        "page": page.page,
                        "title": page.title,
                        "input_text_sha256": page_hash,
                        "chunk_sha256": chunk_hash,
                        "text": text,
                    }
                )
                if len(chunks) > self._config.max_chunks:
                    raise ValueError("corpus chunk budget exceeded")
        self._chunks = tuple(chunks)
        self._chunks_by_id = {item["evidence_id"]: item for item in self._chunks}
        self._texts = tuple(item["text"] for item in self._chunks)
        self._lexical_index = _BM25Index(self._texts)
        self._embedder, self._selector = embedder, selector
        self._embedding_name = _backend_name(embedder, self._config) if embedder is not None else "none"
        self._selector_name = _backend_name(selector, self._config) if selector is not None else "none"
        self._vectors = None
        self._embedding_lock = RLock()

    @property
    def config(self) -> RAGConfig:
        return self._config

    @property
    def corpus_fingerprint(self) -> str:
        return self._corpus_hash

    @property
    def page_count(self) -> int:
        return len(self._pages)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(self, question: str, *, top_k: int = 3) -> SearchResult:
        clean = _normalize_question(question)
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer between 1 and 20")
        texts = self._texts
        capacity = max(top_k, self._config.candidate_top_k)
        lexical = self._lexical_index.rank(clean, capacity)
        vector_ranked, invocations, degraded = [], 0, False
        if self._embedder is not None and texts:
            # Backend declarations can change; validate consent again before every invocation.
            _check_backend(self._embedder, self._embedding_name, self._config)
            try:
                with self._embedding_lock:
                    if self._vectors is None:
                        _check_backend(self._embedder, self._embedding_name, self._config)
                        invocations += 1
                        vectors = self._embedder.encode_documents(list(texts))
                        if not isinstance(vectors, Sequence) or len(vectors) != len(texts):
                            raise ValueError("embedding count mismatch")
                        if (
                            any(
                                not isinstance(row, Sequence) or isinstance(row, (str, bytes))
                                for row in vectors
                            )
                            or sum(len(row) for row in vectors) > 2_000_000
                        ):
                            raise ValueError("embedding storage budget exceeded")
                        converted = tuple(_vector(row) for row in vectors)
                        if (
                            len({len(row) for row in converted}) != 1
                            or len(converted) * len(converted[0]) > 2_000_000
                        ):
                            raise ValueError("embedding dimension/storage budget exceeded")
                        self._vectors = converted
                    _check_backend(self._embedder, self._embedding_name, self._config)
                    invocations += 1
                    query = _vector(self._embedder.encode_query(clean))
                    if len(query) != len(self._vectors[0]):
                        raise ValueError("query embedding dimension mismatch")
                    scores = [
                        (index, math.fsum(left * right for left, right in zip(row, query, strict=True)))
                        for index, row in enumerate(self._vectors)
                    ]
                    vector_ranked = sorted(
                        [(index, score) for index, score in scores if score > 0],
                        key=lambda row: (-row[1], row[0]),
                    )[:capacity]
            except ExternalProcessingDenied:
                raise
            except Exception:
                if not self._config.allow_lexical_fallback:
                    raise BackendError("embedding backend failed; no automatic retry") from None
                degraded = True
        fused = _rrf_fuse_rankings(vector_ranked, lexical, self._config.rrf_k)
        vector_ids = {index for index, _ in vector_ranked}
        candidates = []
        for index, score in sorted(fused.items(), key=lambda pair: (-pair[1]["fused"], pair[0]))[:capacity]:
            item = self._chunks[index]
            candidates.append(
                {
                    **item,
                    "doc_idx": index,
                    "snippet": item["text"],
                    "file_name": item["title"],
                    "fused_score": score["fused"],
                    "bm25_score": score["bm25"],
                    "vector_score": score["vector"] if index in vector_ids else None,
                }
            )
        ranked = _heuristic_rerank(clean, candidates, top_k=top_k)
        packed, used = [], 0
        for item in ranked:
            # Keep complete chunks; truncating them would break their identity/hash bindings.
            if used + len(item["text"]) > self._config.max_context_chars:
                continue
            packed.append(
                Evidence(
                    **{
                        name: item[name]
                        for name in (
                            "evidence_id",
                            "document_id",
                            "source",
                            "version",
                            "page",
                            "title",
                            "input_text_sha256",
                            "chunk_sha256",
                            "text",
                            "vector_score",
                            "fused_score",
                        )
                    },
                    lexical_score=float(item["bm25_score"]),
                )
            )
            used += len(item["text"])
        payload = {
            "protocol": "agentforge-document-sdk-v1",
            "query_sha256": hashlib.sha256(clean.encode()).hexdigest(),
            "corpus_sha256": self._corpus_hash,
            "config_sha256": fingerprint(self._config.model_dump()),
            "status": "degraded" if degraded else "ok" if packed else "empty",
            "reason": "embedding_fallback"
            if degraded
            else "ranked"
            if packed
            else "context_budget_empty"
            if ranked
            else "no_lexical_hits",
            "evidence": tuple(packed),
            "backend": "bm25-rrf-heuristic:" + self._embedding_name,
            "candidate_count": len(candidates),
            "context_chars": used,
            "embedding_invocations": invocations,
            "external_processing_authorized": self._config.allow_external_processing,
        }
        payload["fingerprint"] = fingerprint({**payload, "evidence": [row.model_dump() for row in packed]})
        return SearchResult(**payload)

    def answer(self, question: str, *, top_k: int = 3) -> AnswerResult:
        retrieval = self.search(question, top_k=top_k)
        invocations, citations, reason = 0, [], "no_evidence"
        if retrieval.evidence:
            selected = _quotes(question, retrieval.evidence, self._config.max_quote_chars)
            reason = "literal_evidence_quotes"
            if self._selector is not None:
                _check_backend(self._selector, self._selector_name, self._config)
                invocations = 1
                try:
                    selected = self._selector.select(question, retrieval.evidence)
                except Exception:
                    selected, reason = [], "selector_failed"
            by_id = {row.evidence_id: row for row in retrieval.evidence}
            try:
                if (
                    not isinstance(selected, Sequence)
                    or isinstance(selected, (str, bytes))
                    or len(selected) > top_k
                ):
                    raise ValueError("selector returned an invalid selection count")
                seen = set()
                for raw in selected:
                    item = Selection.model_validate(raw.model_dump())
                    evidence = by_id[item.evidence_id]
                    if item.evidence_id in seen or len(item.quote) > self._config.max_quote_chars:
                        raise ValueError("duplicate or oversized selected quote")
                    start = evidence.text.find(item.quote)
                    if start < 0:
                        raise ValueError("selected quote is not in the returned evidence")
                    seen.add(item.evidence_id)
                    citations.append(
                        Citation(
                            evidence_id=item.evidence_id,
                            document_id=evidence.document_id,
                            source=evidence.source,
                            version=evidence.version,
                            page=evidence.page,
                            input_text_sha256=evidence.input_text_sha256,
                            chunk_sha256=evidence.chunk_sha256,
                            quote=item.quote,
                            chunk_start=start,
                            chunk_end=start + len(item.quote),
                        )
                    )
                if not citations and reason != "selector_failed":
                    reason = "selector_abstained"
            except (ValueError, KeyError, TypeError, AttributeError):
                citations, reason = [], "selection_rejected"
        payload = {
            "protocol": "agentforge-document-sdk-v1",
            "status": "evidence_found" if citations else "abstained",
            "reason": reason,
            "answer": "\n".join(f'"{row.quote}" [{index}]' for index, row in enumerate(citations, start=1))
            if citations
            else "No validated evidence answer is available.",
            "citations": tuple(citations),
            "retrieval": retrieval,
            "answer_mode": "extractive_selector" if self._selector is not None else "extractive",
            "selector_backend": self._selector_name,
            "selector_invocations": invocations,
            "provider_cost_usd": 0.0 if invocations == retrieval.embedding_invocations == 0 else None,
        }
        payload["fingerprint"] = fingerprint(
            {
                **payload,
                "citations": [row.model_dump() for row in citations],
                "retrieval": retrieval.model_dump(mode="json"),
            }
        )
        return AnswerResult(**payload)

    async def asearch(self, question: str, *, top_k: int = 3) -> SearchResult:
        import asyncio

        return await asyncio.to_thread(self.search, question, top_k=top_k)

    async def aanswer(self, question: str, *, top_k: int = 3) -> AnswerResult:
        import asyncio

        return await asyncio.to_thread(self.answer, question, top_k=top_k)

    def verify_search(self, result: SearchResult, *, question: str | None = None) -> bool:
        """Validate frozen evidence membership and receipt binding, not ranking quality."""
        try:
            result = SearchResult.model_validate(result.model_dump())
            if (result.fingerprint != fingerprint(result.model_dump(mode="json", exclude={"fingerprint"}))
                    or result.corpus_sha256 != self._corpus_hash
                    or result.config_sha256 != fingerprint(self._config.model_dump())
                    or (question is not None and result.query_sha256 != hashlib.sha256(_normalize_question(question).encode()).hexdigest())):
                return False
            if len({r.evidence_id for r in result.evidence}) != len(result.evidence):
                return False
            return all(all(self._chunks_by_id[row.evidence_id][key] == getattr(row, key)
                           for key in self._chunks_by_id[row.evidence_id]) for row in result.evidence)
        except (ValueError, TypeError, AttributeError, KeyError):
            return False

    def verify_citation(self, citation: Citation) -> bool:
        """Check literal quote provenance against THIS immutable corpus, not factual truth."""
        try:
            citation = Citation.model_validate(citation.model_dump())
            item = self._chunks_by_id[citation.evidence_id]
            return (
                all(
                    item[name] == getattr(citation, name)
                    for name in (
                        "document_id",
                        "source",
                        "version",
                        "page",
                        "input_text_sha256",
                        "chunk_sha256",
                    )
                )
                and 0 <= citation.chunk_start < citation.chunk_end <= len(item["text"])
                and item["text"][citation.chunk_start : citation.chunk_end] == citation.quote
            )
        except (ValueError, TypeError, AttributeError, KeyError):
            return False

    def verify_answer(self, answer: AnswerResult, *, question: str | None = None) -> bool:
        """Validate receipt/quote provenance, NOT the answer's relevance or the document's truth."""
        try:
            answer = AnswerResult.model_validate(answer.model_dump())
            data = answer.model_dump(mode="json")
            if answer.fingerprint != fingerprint(
                {key: value for key, value in data.items() if key != "fingerprint"}
            ):
                return False
            retrieval = answer.retrieval.model_dump(mode="json")
            if answer.retrieval.fingerprint != fingerprint(
                {key: value for key, value in retrieval.items() if key != "fingerprint"}
            ):
                return False
            if (
                answer.retrieval.corpus_sha256 != self._corpus_hash
                or answer.retrieval.config_sha256 != fingerprint(self._config.model_dump())
                or (
                    question is not None
                    and answer.retrieval.query_sha256
                    != hashlib.sha256(_normalize_question(question).encode()).hexdigest()
                )
            ):
                return False
            actual = self._chunks_by_id
            supplied = {row.evidence_id: row for row in answer.retrieval.evidence}
            if len(supplied) != len(answer.retrieval.evidence):
                return False
            for item in supplied.values():
                if any(
                    actual[item.evidence_id][name] != getattr(item, name) for name in actual[item.evidence_id]
                ):
                    return False
            if any(
                item.evidence_id not in supplied or not self.verify_citation(item)
                for item in answer.citations
            ):
                return False
            expected = "\n".join(
                f'"{row.quote}" [{index}]' for index, row in enumerate(answer.citations, start=1)
            )
            return (
                answer.status == "evidence_found" and bool(answer.citations) and answer.answer == expected
            ) or (
                answer.status == "abstained"
                and not answer.citations
                and answer.answer == "No validated evidence answer is available."
            )
        except (ValueError, TypeError, AttributeError, KeyError):
            return False
