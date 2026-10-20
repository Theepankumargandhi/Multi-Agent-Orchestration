from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from agentforge_sdk import (
    BackendError,
    DocumentInputError,
    DocumentPage,
    DocumentRAG,
    ExternalProcessingDenied,
    RAGConfig,
    Selection,
    load_chroma_records,
    load_pdf,
    load_sqlite,
    load_text,
)
from agentforge_sdk.models import fingerprint

ROOT = Path(__file__).parent.parent


def pages():
    return [DocumentPage(document_id="atlas", source="atlas.pdf", title="Atlas report", page=2, version="r1",
                         text="Atlas p95 latency is 180 ms. Atlas serves 100 requests per second."),
            DocumentPage(document_id="guide", source="guide.md", title="Deployment guide", text="Deployments require owner approval.")]


def test_offline_evidence_answer_has_literal_versioned_provenance():
    rag = DocumentRAG(pages())
    result = rag.answer("Atlas p95 latency", top_k=1)
    assert result.status == "evidence_found" and "180 ms" in result.answer
    citation = result.citations[0]
    assert citation.document_id == "atlas" and citation.page == 2 and citation.version == "r1"
    evidence = result.retrieval.evidence[0]
    assert evidence.text[citation.chunk_start:citation.chunk_end] == citation.quote
    assert evidence.chunk_sha256 == hashlib.sha256(evidence.text.encode()).hexdigest()
    assert citation.input_text_sha256 == hashlib.sha256(pages()[0].text.encode()).hexdigest()
    assert rag.verify_answer(result, question="Atlas p95 latency")
    assert not rag.verify_answer(result, question="A different question")
    assert result.provider_cost_usd == 0 and result.selector_invocations == result.retrieval.embedding_invocations == 0
    assert result.answer_mode == "extractive" and result.selector_backend == "none"


def test_corpus_is_immutable_order_independent_and_instance_isolated():
    documents = pages()
    first, second = DocumentRAG(documents), DocumentRAG(list(reversed(documents)))
    documents.clear()
    assert first.corpus_fingerprint == second.corpus_fingerprint
    assert first.answer("Atlas latency").model_dump() == second.answer("Atlas latency").model_dump()
    assert DocumentRAG([]).answer("Atlas latency").status == "abstained"
    with pytest.raises(ValueError):
        first.config.chunk_size = 500


@pytest.mark.parametrize("question", ["", " ", "a" * 4001, "\x00", 10], ids=["empty", "whitespace", "oversized", "nul", "non_text"])
def test_invalid_questions_are_rejected(question):
    with pytest.raises(ValueError):
        DocumentRAG(pages()).answer(question)


@pytest.mark.parametrize("top_k", [0, 21, True, "3"])
def test_top_k_is_strictly_bounded(top_k):
    with pytest.raises(ValueError):
        DocumentRAG(pages()).search("Atlas latency", top_k=top_k)


@pytest.mark.parametrize("change", [
    {"page": 0}, {"page": True}, {"text": " "}, {"text": "x\x00y"}, {"document_id": "../private"},
    {"source": "https://user:password@example.com/report"}, {"source": "https://example.com/report?key=private"},
    {"source": "line\nheader"}, {"version": "bad\nversion"}, {"unknown": True},
])
def test_metadata_and_payload_contracts_reject_unsafe_inputs(change):
    with pytest.raises(ValueError):
        DocumentPage.model_validate({**pages()[0].model_dump(), **change})


def test_page_validation_does_not_echo_document_text():
    with pytest.raises(ValueError) as exc:
        DocumentPage(document_id="valid", source="file.txt", text="PRIVATE-INPUT\x00")
    assert "PRIVATE-INPUT" not in str(exc.value)


def test_corpus_and_context_budgets_are_explicit():
    with pytest.raises(ValueError, match="unique"):
        DocumentRAG([pages()[0], pages()[0]])
    with pytest.raises(ValueError, match="bounded"):
        DocumentRAG(pages(), config=RAGConfig(max_pages=1))
    with pytest.raises(ValueError, match="byte"):
        DocumentRAG([pages()[0].model_copy(update={"text": "Atlas " * 300})], config=RAGConfig(max_corpus_bytes=1000))
    with pytest.raises(ValueError, match="chunk"):
        DocumentRAG([pages()[0].model_copy(update={"text": "Atlas " * 300})], config=RAGConfig(chunk_size=200, max_chunks=1))
    rag = DocumentRAG([pages()[0].model_copy(update={"text": "Atlas " * 100})], config=RAGConfig(max_context_chars=256))
    result = rag.answer("Atlas")
    assert result.status == "abstained" and result.retrieval.reason == "context_budget_empty"


def test_sentence_aware_overlap_at_largest_configuration_remains_retrievable():
    text = "Atlas " * 166 + ". " + "Latency " * 498 + "."
    rag = DocumentRAG([pages()[0].model_copy(update={"text": text})], config=RAGConfig(chunk_size=4000, chunk_overlap=1000))
    result = rag.search("Atlas Latency", top_k=2)
    assert result.evidence and all(len(item.text) <= 5001 for item in result.evidence)


def test_no_hits_do_not_create_seeded_knowledge_or_call_selector():
    class Selector:
        name, external_processing = "test-selector", False

        def select(self, *_):
            raise AssertionError("must not run without evidence")

    result = DocumentRAG(pages(), selector=Selector()).answer("xylophonium")
    assert result.status == "abstained" and result.reason == "no_evidence"
    assert not result.citations and result.selector_invocations == 0


class LocalEmbedder:
    name, external_processing = "authored-local-embedding-control", False

    def __init__(self):
        self.calls = []

    def encode_documents(self, values):
        self.calls.append("documents")
        return [[1.0, 0.0] if "Atlas" in value else [0.0, 1.0] for value in values]

    def encode_query(self, value):
        self.calls.append("query")
        return [1.0, 0.0]


def test_explicit_embedding_ranks_semantic_only_hits_and_caches_only_per_instance():
    embedder = LocalEmbedder()
    rag = DocumentRAG(pages(), embedder=embedder)
    assert embedder.calls == []
    first = rag.answer("response duration", top_k=1)
    second = rag.answer("response duration", top_k=1)
    assert first.status == "evidence_found" and first.citations[0].document_id == "atlas"
    assert first.retrieval.embedding_invocations == 2 and second.retrieval.embedding_invocations == 1
    assert first.provider_cost_usd is None and first.retrieval.evidence[0].vector_score == 1.0
    DocumentRAG(pages(), embedder=embedder).search("Atlas")
    assert embedder.calls.count("documents") == 2


def test_external_processing_is_explicit_and_revalidated():
    backend = LocalEmbedder()
    backend.external_processing = True
    with pytest.raises(ExternalProcessingDenied):
        DocumentRAG(pages(), embedder=backend)
    assert not backend.calls
    consented = DocumentRAG(pages(), embedder=backend, config=RAGConfig(allow_external_processing=True))
    assert consented.search("Atlas").external_processing_authorized
    backend.external_processing = False
    unconsented = DocumentRAG(pages(), embedder=backend)
    backend.external_processing = True
    with pytest.raises(ExternalProcessingDenied):
        unconsented.search("Atlas")


@pytest.mark.parametrize("vector", [[float("nan")], [float("inf")], [0.0, 0.0], [], [True, 1.0], [1.0] * 4097])
def test_malformed_embeddings_are_degraded_or_blocked_without_retry(vector):
    class Broken(LocalEmbedder):
        def encode_documents(self, values):
            self.calls.append("documents")
            return [vector for _ in values]

    backend = Broken()
    result = DocumentRAG(pages(), embedder=backend).answer("Atlas", top_k=1)
    assert result.retrieval.status == "degraded" and result.retrieval.reason == "embedding_fallback"
    assert backend.calls == ["documents"] and result.provider_cost_usd is None
    backend = Broken()
    with pytest.raises(BackendError):
        DocumentRAG(pages(), embedder=backend, config=RAGConfig(allow_lexical_fallback=False)).search("Atlas")
    assert backend.calls == ["documents"]


def test_selector_can_only_return_exact_quotes_from_packed_evidence():
    class Selector:
        name, external_processing = "authored-selector", False

        def select(self, question, evidence):
            return [Selection(evidence_id=evidence[0].evidence_id, quote="Atlas p95 latency is 180 ms.")]

    rag = DocumentRAG(pages(), selector=Selector())
    result = rag.answer("Atlas p95 latency", top_k=1)
    assert result.status == "evidence_found" and result.selector_invocations == 1
    assert result.provider_cost_usd is None and rag.verify_answer(result)
    assert result.selector_backend == "authored-selector"


@pytest.mark.parametrize("fault", ["invented_quote", "invented_id", "duplicate", "exception", "empty", "too_many"])
def test_invalid_or_failed_selector_is_held_with_no_fallback_generation(fault):
    class Selector:
        name, external_processing = "authored-failing-selector", False

        def select(self, question, evidence):
            if fault == "exception":
                raise RuntimeError("PRIVATE-PROVIDER-ERROR")
            if fault == "empty":
                return []
            choice = Selection(evidence_id="0" * 64 if fault == "invented_id" else evidence[0].evidence_id,
                               quote="Invented latency is 1 ms." if fault == "invented_quote" else "Atlas p95 latency is 180 ms.")
            return [choice, choice] if fault in {"duplicate", "too_many"} else [choice]

    result = DocumentRAG(pages(), selector=Selector()).answer("Atlas p95 latency", top_k=1)
    assert result.status == "abstained" and not result.citations
    assert result.selector_invocations == 1 and result.provider_cost_usd is None
    assert "PRIVATE-PROVIDER-ERROR" not in result.model_dump_json()


def test_tampered_or_other_corpus_citations_and_receipts_are_rejected():
    rag = DocumentRAG(pages())
    result = rag.answer("Atlas latency", top_k=1)
    for update in ({"quote": "made up"}, {"page": 1}, {"version": "r2"}, {"chunk_start": 1}, {"input_text_sha256": "0" * 64}):
        assert not rag.verify_citation(result.citations[0].model_copy(update=update))
    assert not rag.verify_answer(result.model_copy(update={"answer": "fabricated answer"}))
    assert not DocumentRAG([pages()[1]]).verify_answer(result)
    assert rag.verify_answer(type(result).model_validate_json(result.model_dump_json()))
    assert result.fingerprint == fingerprint(result.model_dump(mode="json", exclude={"fingerprint"}))


def test_readonly_sqlite_bridge_preserves_original_store_and_has_no_fake_page(tmp_path):
    database = tmp_path / "rag.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT, content TEXT, source TEXT)")
    connection.execute("INSERT INTO docs VALUES (1, 'Atlas', 'Atlas latency is 180 ms.', 'local://atlas')")
    connection.commit()
    connection.close()
    before = database.read_bytes()
    records = load_sqlite(database)
    assert database.read_bytes() == before and len(records) == 1 and records[0].page is None
    assert DocumentRAG(records).answer("Atlas latency").status == "evidence_found"
    with pytest.raises(DocumentInputError):
        load_sqlite(tmp_path / "missing.db")
    assert not (tmp_path / "missing.db").exists()


def test_chroma_data_bridge_retains_one_based_page_and_rejects_ambiguous_metadata():
    payload = {"ids": ["chunk-1"], "documents": ["Atlas latency is 180 ms."],
               "metadatas": [{"source": "atlas.pdf", "page": 2, "file_name": "Atlas", "file_hash": "revision-hash"}]}
    records = load_chroma_records(payload)
    assert records[0].page == 2 and records[0].version == "revision-hash"
    assert DocumentRAG(records).answer("Atlas latency").citations[0].page == 2
    for broken in ({**payload, "ids": []}, {**payload, "ids": ["same", "same"]}, {**payload, "metadatas": [{"page": 0}]},
                   {**payload, "metadatas": [{"page": "n/a"}]}, {**payload, "metadatas": [None]}):
        with pytest.raises(DocumentInputError):
            load_chroma_records(broken)


def test_text_loader_has_no_invented_page_or_absolute_path(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("Owner approval is required.", encoding="utf-8")
    result = load_text(source, document_id="guide")
    assert result[0].source == "guide.md" and result[0].page is None
    assert str(tmp_path) not in result[0].model_dump_json()
    for path in (tmp_path / "missing.txt", tmp_path / ".env"):
        with pytest.raises(DocumentInputError):
            load_text(path, document_id="invalid")


def test_pdf_loader_extracts_actual_embedded_text_and_holds_blank_pages(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    path = tmp_path / "report.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=400)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 300 Td (Atlas latency is 180 ms.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)
    records = load_pdf(path, document_id="report")
    assert records[0].page == 1 and "180 ms" in records[0].text
    assert DocumentRAG(records).answer("Atlas latency").status == "evidence_found"
    writer = PdfWriter()
    writer.add_blank_page(width=400, height=400)
    with path.open("wb") as output:
        writer.write(output)
    with pytest.raises(DocumentInputError, match="no OCR"):
        load_pdf(path, document_id="scanned-or-blank")


@pytest.mark.asyncio
async def test_async_wrappers_match_synchronous_provenance():
    rag = DocumentRAG(pages())
    assert (await rag.asearch("Atlas latency")).model_dump() == rag.search("Atlas latency").model_dump()
    assert (await rag.aanswer("Atlas latency")).model_dump() == rag.answer("Atlas latency").model_dump()


def test_sdk_import_is_independent_of_service_environment_and_network(tmp_path):
    script = '''import os, socket, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
class ForbiddenSocket(socket.socket):
    def __new__(cls, *a, **kw):
        raise AssertionError("network access during import")
socket.socket = ForbiddenSocket
from agentforge_sdk import DocumentRAG, DocumentPage
r = DocumentRAG([DocumentPage(document_id="one", source="test", text="Atlas latency is 180 ms.")])
assert r.answer("Atlas latency").status == "evidence_found"
assert not any(name == "agent" or name.startswith(("langchain", "langgraph", "chromadb", "dotenv", "openai", "groq", "opentelemetry")) for name in sys.modules)
assert os.getenv("SDK_DOTENV_MARKER") is None
assert list(Path.cwd().iterdir()) == [Path(".env")]
'''
    (tmp_path / ".env").write_text("SDK_DOTENV_MARKER=must-not-load\n")
    # Path.iterdir returns paths prefixed with cwd; compare names instead.
    script = script.replace('list(Path.cwd().iterdir()) == [Path(".env")]', '[p.name for p in Path.cwd().iterdir()] == [".env"]')
    result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT)], cwd=tmp_path,
        env={**os.environ, "RAG_CHUNK_SIZE": "invalid-service-setting", "LANGSMITH_TRACING": "true"},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_shared_production_ranking_primitives_match_sdk():
    from agent import local_rag
    from agentforge_sdk import _retrieval
    corpus = [page.text for page in pages()]
    assert local_rag._bm25_rank_documents("Atlas latency", corpus, 3) == _retrieval._bm25_rank_documents("Atlas latency", corpus, 3)
    assert local_rag._chunk_text(corpus[0], 200, 20) == _retrieval._chunk_text(corpus[0], 200, 20)
    assert local_rag._rrf_fuse_rankings([(0, 1.0)], [(1, 2.0)], 40) == _retrieval._rrf_fuse_rankings([(0, 1.0)], [(1, 2.0)], 40)
