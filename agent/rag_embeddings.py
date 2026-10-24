"""Embedding backend shared by the local RAG and graph RAG vector stores."""

import os

# "openai" uses OPENAI_API_KEY; "local" runs all-MiniLM-L6-v2 on this machine
# through the ONNX runtime bundled with ChromaDB, with no key or per-call cost.
RAG_EMBEDDING_BACKEND = os.getenv("RAG_EMBEDDING_BACKEND", "openai").strip().lower()
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class LocalMiniLMEmbeddings:
    """LangChain-compatible wrapper around ChromaDB's ONNX MiniLM (384 dimensions)."""

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._function = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(value) for value in vector] for vector in self._function(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings(openai_model: str):
    if RAG_EMBEDDING_BACKEND == "local":
        return LocalMiniLMEmbeddings()
    if RAG_EMBEDDING_BACKEND != "openai":
        raise RuntimeError("RAG_EMBEDDING_BACKEND must be 'openai' or 'local'.")
    from langchain_openai import OpenAIEmbeddings

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to use ChromaDB embedding retrieval.")
    return OpenAIEmbeddings(model=openai_model)


def embedding_model_name(openai_model: str) -> str:
    return LOCAL_EMBEDDING_MODEL if RAG_EMBEDDING_BACKEND == "local" else openai_model


def collection_name(base: str) -> str:
    # Vectors of different sizes cannot share a collection, so local embeddings
    # get their own and switching backends never mixes them.
    return f"{base}_local" if RAG_EMBEDDING_BACKEND == "local" else base
