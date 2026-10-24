"""Local embeddings must work without an OpenAI key and never share a collection."""
import sys
from types import ModuleType

import pytest

from agent import rag_embeddings


@pytest.fixture
def local_backend(monkeypatch):
    calls = []

    class FakeMiniLM:
        def __call__(self, texts):
            calls.append(list(texts))
            return [[0.5, 0.25] for _ in texts]

    module = ModuleType("chromadb.utils.embedding_functions")
    module.ONNXMiniLM_L6_V2 = FakeMiniLM
    monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", module)
    monkeypatch.setattr(rag_embeddings, "RAG_EMBEDDING_BACKEND", "local")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return calls


def test_local_backend_needs_no_openai_key(local_backend):
    embeddings = rag_embeddings.build_embeddings("text-embedding-3-small")
    assert embeddings.embed_documents(["a", "b"]) == [[0.5, 0.25], [0.5, 0.25]]
    assert embeddings.embed_query("q") == [0.5, 0.25]
    assert local_backend == [["a", "b"], ["q"]]


def test_local_vectors_use_their_own_collection_and_model_label(local_backend):
    assert rag_embeddings.collection_name("local_pdf_docs") == "local_pdf_docs_local"
    assert rag_embeddings.embedding_model_name("text-embedding-3-small") == "all-MiniLM-L6-v2"


def test_openai_backend_keeps_existing_collection(monkeypatch):
    monkeypatch.setattr(rag_embeddings, "RAG_EMBEDDING_BACKEND", "openai")
    assert rag_embeddings.collection_name("local_pdf_docs") == "local_pdf_docs"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        rag_embeddings.build_embeddings("text-embedding-3-small")


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(rag_embeddings, "RAG_EMBEDDING_BACKEND", "nope")
    with pytest.raises(RuntimeError, match="'openai' or 'local'"):
        rag_embeddings.build_embeddings("text-embedding-3-small")
