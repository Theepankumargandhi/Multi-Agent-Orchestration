import os
import shutil
import time
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4


GRAPH_RAG_ENABLED = os.getenv("GRAPH_RAG_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
GRAPH_RAG_PDF_DIR = os.getenv("GRAPH_RAG_PDF_DIR", "graph_rag_docs")
GRAPH_CHROMA_PERSIST_DIR = os.getenv("GRAPH_CHROMA_PERSIST_DIR", "graph_chroma_db")
GRAPH_CHROMA_COLLECTION_NAME = os.getenv("GRAPH_CHROMA_COLLECTION_NAME", "graph_pdf_docs")
GRAPH_OPENAI_EMBEDDING_MODEL = os.getenv("GRAPH_OPENAI_EMBEDDING_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
GRAPH_RAG_CHUNK_SIZE = int(os.getenv("GRAPH_RAG_CHUNK_SIZE", "1000"))
GRAPH_RAG_CHUNK_OVERLAP = int(os.getenv("GRAPH_RAG_CHUNK_OVERLAP", "150"))
GRAPH_RAG_CACHE_TTL_SECONDS = int(os.getenv("GRAPH_RAG_CACHE_TTL_SECONDS", "600"))

_graph_chroma_store_cache = None
_graph_rag_cache: Dict[Tuple[str, int], Tuple[float, str]] = {}
_redis_client = None
_redis_disabled = False


def _build_embeddings():
    from langchain_openai import OpenAIEmbeddings

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for graph RAG Chroma retrieval.")
    return OpenAIEmbeddings(model=GRAPH_OPENAI_EMBEDDING_MODEL)


def _get_chroma_store():
    global _graph_chroma_store_cache
    if _graph_chroma_store_cache is not None:
        return _graph_chroma_store_cache

    from langchain_community.vectorstores import Chroma

    os.makedirs(GRAPH_CHROMA_PERSIST_DIR, exist_ok=True)
    _graph_chroma_store_cache = Chroma(
        collection_name=GRAPH_CHROMA_COLLECTION_NAME,
        persist_directory=GRAPH_CHROMA_PERSIST_DIR,
        embedding_function=_build_embeddings(),
    )
    return _graph_chroma_store_cache


def _reset_chroma_store_cache() -> None:
    global _graph_chroma_store_cache
    _graph_chroma_store_cache = None


def _get_redis_client():
    global _redis_client, _redis_disabled
    if _redis_disabled:
        return None
    cache_use_redis = os.getenv("CACHE_USE_REDIS", "true").strip().lower() in {"1", "true", "yes", "on"}
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not cache_use_redis or not redis_url:
        return None
    if _redis_client is not None:
        return _redis_client
    try:
        import redis  # type: ignore

        _redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
        return _redis_client
    except Exception:
        _redis_disabled = True
        _redis_client = None
        return None


def _redis_key(prefix: str, cache_key: Tuple[str, int]) -> str:
    payload = json.dumps(cache_key, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _cache_get_graph(cache_key: Tuple[str, int]) -> tuple[str | None, str | None]:
    ttl = max(0, GRAPH_RAG_CACHE_TTL_SECONDS)
    if ttl <= 0:
        return None, None

    client = _get_redis_client()
    if client is not None:
        try:
            value = client.get(_redis_key("graph_rag", cache_key))
            if isinstance(value, str) and value:
                return value, "redis"
        except Exception:
            pass

    cached = _graph_rag_cache.get(cache_key)
    if cached and (time.time() - cached[0]) <= ttl:
        return cached[1], "memory"
    return None, None


def _cache_set_graph(cache_key: Tuple[str, int], value: str) -> None:
    ttl = max(0, GRAPH_RAG_CACHE_TTL_SECONDS)
    if ttl <= 0:
        return

    _graph_rag_cache[cache_key] = (time.time(), value)
    client = _get_redis_client()
    if client is not None:
        try:
            client.setex(_redis_key("graph_rag", cache_key), ttl, value)
        except Exception:
            pass


def _reset_graph_rag_cache() -> None:
    _graph_rag_cache.clear()
    client = _get_redis_client()
    if client is not None:
        try:
            for key in client.scan_iter(match="graph_rag:*"):
                client.delete(key)
        except Exception:
            pass


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    content = (text or "").strip()
    if not content:
        return []
    chunk_size = max(200, int(chunk_size))
    chunk_overlap = max(0, int(chunk_overlap))
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 5

    chunks: List[str] = []
    step = max(1, chunk_size - chunk_overlap)
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def _chroma_has_documents(store) -> bool:
    try:
        payload = store.get(limit=1)
    except TypeError:
        payload = store.get()
    except Exception:
        return False

    if not isinstance(payload, dict):
        return False
    ids = payload.get("ids") or []
    if isinstance(ids, list):
        return len(ids) > 0
    return bool(ids)


def ingest_pdfs_to_graph_chroma(
    pdf_dir: str = GRAPH_RAG_PDF_DIR,
    reset: bool = False,
    chunk_size: int = GRAPH_RAG_CHUNK_SIZE,
    chunk_overlap: int = GRAPH_RAG_CHUNK_OVERLAP,
) -> Dict[str, Any]:
    if not GRAPH_RAG_ENABLED:
        return {
            "ok": False,
            "message": "GRAPH_RAG_ENABLED is disabled. Set GRAPH_RAG_ENABLED=true in .env.",
        }

    root = Path(pdf_dir)
    if not root.exists():
        return {"ok": False, "message": f"PDF directory not found: {root}"}

    pdf_files = sorted(root.rglob("*.pdf"))
    if not pdf_files:
        return {"ok": False, "message": f"No PDF files found under: {root}"}

    if reset and Path(GRAPH_CHROMA_PERSIST_DIR).exists():
        shutil.rmtree(GRAPH_CHROMA_PERSIST_DIR, ignore_errors=True)
        _reset_chroma_store_cache()
        _reset_graph_rag_cache()

    try:
        from langchain_community.document_loaders import PyPDFLoader
    except Exception as e:
        return {"ok": False, "message": f"PDF loader import failed: {e}"}

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []
    page_count = 0

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
        except Exception:
            continue

        for page in pages:
            content = (getattr(page, "page_content", "") or "").strip()
            if not content:
                continue

            raw_page = getattr(page, "metadata", {}).get("page", 0)
            try:
                page_no = int(raw_page) + 1
            except Exception:
                page_no = 1

            page_count += 1
            chunks = _chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for chunk_index, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append(
                    {
                        "source": str(pdf_file),
                        "file_name": pdf_file.name,
                        "page": page_no,
                        "chunk_index": chunk_index,
                    }
                )
                ids.append(str(uuid4()))

    if not texts:
        return {
            "ok": False,
            "message": "No extractable text found in provided PDFs.",
            "pdf_count": len(pdf_files),
            "page_count": page_count,
            "chunk_count": 0,
        }

    try:
        store = _get_chroma_store()
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        persist = getattr(store, "persist", None)
        if callable(persist):
            persist()
        _reset_graph_rag_cache()
    except Exception as e:
        return {"ok": False, "message": f"Graph ChromaDB ingestion failed: {e}"}

    return {
        "ok": True,
        "message": "Graph PDF ingestion completed.",
        "pdf_count": len(pdf_files),
        "page_count": page_count,
        "chunk_count": len(texts),
        "persist_dir": GRAPH_CHROMA_PERSIST_DIR,
        "collection": GRAPH_CHROMA_COLLECTION_NAME,
    }


def _search_graph_chroma(query: str, limit: int = 5) -> str:
    if not GRAPH_RAG_ENABLED:
        return "Graph RAG is disabled."

    clean_query = (query or "").strip()
    if not clean_query:
        return "No graph knowledge hits."

    try:
        store = _get_chroma_store()
    except Exception as e:
        return f"Graph RAG retrieval failed: {e}"

    if not _chroma_has_documents(store):
        return (
            "No graph ChromaDB knowledge found yet. Ingest PDFs first with "
            "`python scripts/ingestion/ingest_graph_rag_pdfs.py --pdf-dir graph_rag_docs --reset`."
        )

    pairs: List[Tuple[Any, float]] = []
    score_mode = "relevance"
    try:
        pairs = store.similarity_search_with_relevance_scores(clean_query, k=max(1, limit))
    except Exception:
        try:
            pairs = store.similarity_search_with_score(clean_query, k=max(1, limit))
            score_mode = "distance"
        except Exception as e:
            return f"Graph ChromaDB retrieval failed: {e}"

    if not pairs:
        return "No graph knowledge hits in ChromaDB."

    lines: List[str] = []
    for doc, score in pairs[: max(1, limit)]:
        metadata = getattr(doc, "metadata", {}) or {}
        source = str(metadata.get("source", "unknown"))
        page = metadata.get("page", "n/a")
        snippet = (getattr(doc, "page_content", "") or "").strip().replace("\n", " ")
        if len(snippet) > 260:
            snippet = snippet[:257] + "..."
        label = f"{score_mode}={score:.4f}" if isinstance(score, (int, float)) else f"{score_mode}=n/a"
        lines.append(f"- source={source}, page={page}, {label}: {snippet}")
    return "\n".join(lines)


def search_graph_knowledge(
    query: str,
    limit: int = 5,
    return_meta: bool = False,
) -> str | tuple[str, Dict[str, Any]]:
    clean_query = (query or "").strip()
    cache_key = (clean_query.lower(), int(limit))
    cached_value, cache_backend = _cache_get_graph(cache_key)
    if cached_value is not None:
        meta = {"cache_hit": True, "cache_backend": cache_backend or "unknown", "source": "graph_rag_cache"}
        return (cached_value, meta) if return_meta else cached_value

    result = _search_graph_chroma(clean_query, limit=limit)
    _cache_set_graph(cache_key, result)
    meta = {"cache_hit": False, "cache_backend": "none", "source": "graph_rag_live"}
    return (result, meta) if return_meta else result
