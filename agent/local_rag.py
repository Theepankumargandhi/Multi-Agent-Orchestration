import os
import re
import shutil
import sqlite3
import time
import json
import hashlib
from collections import Counter
from math import log
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4


DEFAULT_RAG_DB_PATH = os.getenv("LOCAL_RAG_DB_PATH", "local_rag.db")
DEFAULT_RAG_PDF_DIR = os.getenv("RAG_PDF_DIR", "rag_docs")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "local_pdf_docs")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
USE_CHROMA_RAG = os.getenv("USE_CHROMA_RAG", "true").strip().lower() not in {"0", "false", "no", "off"}
RAG_VECTOR_TOP_K = int(os.getenv("RAG_VECTOR_TOP_K", "8"))
RAG_BM25_TOP_K = int(os.getenv("RAG_BM25_TOP_K", "8"))
RAG_RERANK_TOP_K = int(os.getenv("RAG_RERANK_TOP_K", "6"))
RAG_RRF_K = int(os.getenv("RAG_RRF_K", "60"))
RAG_ENABLE_LLM_RERANKER = os.getenv("RAG_ENABLE_LLM_RERANKER", "true").strip().lower() in {"1", "true", "yes", "on"}
RAG_CACHE_TTL_SECONDS = int(os.getenv("RAG_CACHE_TTL_SECONDS", "600"))

_chroma_store_cache = None
_local_rag_cache: Dict[Tuple[str, int, str, str], Tuple[float, str]] = {}
_redis_client = None
_redis_disabled = False

DEFAULT_LOCAL_DOCS: List[Dict[str, str]] = [
    {
        "title": "Project Architecture",
        "source": "local://architecture",
        "content": (
            "This project runs a LangGraph agent behind a FastAPI service and a Streamlit client. "
            "The service exposes /invoke, /stream, and /feedback endpoints."
        ),
    },
    {
        "title": "Agent Routing",
        "source": "local://agent-routing",
        "content": (
            "The orchestrator routes tasks to specialized nodes such as safety checks, web search, "
            "local RAG retrieval, math evaluation, and final response generation."
        ),
    },
    {
        "title": "Streaming Behavior",
        "source": "local://streaming",
        "content": (
            "The FastAPI stream endpoint emits Server-Sent Events with token and message payloads. "
            "The client supports both sync and async streaming consumption."
        ),
    },
]


def _connect(db_path: str = DEFAULT_RAG_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_local_knowledge_store(db_path: str = DEFAULT_RAG_DB_PATH) -> None:
    """Create local doc table if missing and seed basic docs once."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        row = conn.execute("SELECT COUNT(*) AS total FROM docs").fetchone()
        if row and row["total"] == 0:
            for doc in DEFAULT_LOCAL_DOCS:
                conn.execute(
                    "INSERT INTO docs (title, content, source) VALUES (?, ?, ?)",
                    (doc["title"], doc["content"], doc["source"]),
                )
        conn.commit()


def add_local_document(
    title: str,
    content: str,
    source: str = "local://manual",
    db_path: str = DEFAULT_RAG_DB_PATH,
) -> None:
    """Add one document into the local SQLite knowledge store."""
    if not title.strip() or not content.strip():
        raise ValueError("Both title and content are required.")
    init_local_knowledge_store(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO docs (title, content, source) VALUES (?, ?, ?)",
            (title.strip(), content.strip(), source.strip() or "local://manual"),
        )
        conn.commit()


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1]


def _score_doc(query_tokens: List[str], title: str, content: str, source: str) -> int:
    haystack_title = title.lower()
    haystack_content = content.lower()
    haystack_source = source.lower()
    score = 0
    for token in query_tokens:
        score += haystack_title.count(token) * 3
        score += haystack_content.count(token)
        score += haystack_source.count(token) * 2
    return score


def _search_sqlite_knowledge(query: str, limit: int = 3, db_path: str = DEFAULT_RAG_DB_PATH) -> str:
    init_local_knowledge_store(db_path)
    query_tokens = _tokenize(query)
    if not query_tokens:
        return "No local knowledge hits."

    with _connect(db_path) as conn:
        rows = conn.execute("SELECT title, content, source FROM docs").fetchall()

    scored: List[Tuple[int, sqlite3.Row]] = []
    for row in rows:
        score = _score_doc(query_tokens, row["title"], row["content"], row["source"])
        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, limit)]
    if not top:
        return "No local knowledge hits."

    lines = []
    for score, row in top:
        snippet = row["content"].strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        lines.append(
            f"- {row['title']} (score={score}, source={row['source']}): {snippet}"
        )
    return "\n".join(lines)


def _build_embeddings():
    from langchain_openai import OpenAIEmbeddings

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required to use ChromaDB embedding retrieval."
        )
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


def _get_chroma_store():
    global _chroma_store_cache
    if _chroma_store_cache is not None:
        return _chroma_store_cache

    from langchain_community.vectorstores import Chroma

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    _chroma_store_cache = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=_build_embeddings(),
    )
    return _chroma_store_cache


def _reset_chroma_store_cache() -> None:
    global _chroma_store_cache
    _chroma_store_cache = None


def _reset_local_rag_cache() -> None:
    _local_rag_cache.clear()
    client = _get_redis_client()
    if client is not None:
        try:
            # Best-effort clear for this app namespace only.
            for key in client.scan_iter(match="rag_local:*"):
                client.delete(key)
        except Exception:
            pass


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


def _redis_key(prefix: str, cache_key: Tuple[str, int, str, str]) -> str:
    payload = json.dumps(cache_key, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _cache_get_rag(cache_key: Tuple[str, int, str, str]) -> tuple[str | None, str | None]:
    ttl = max(0, RAG_CACHE_TTL_SECONDS)
    if ttl <= 0:
        return None, None

    client = _get_redis_client()
    if client is not None:
        try:
            value = client.get(_redis_key("rag_local", cache_key))
            if isinstance(value, str) and value:
                return value, "redis"
        except Exception:
            pass

    cached = _local_rag_cache.get(cache_key)
    if cached and (time.time() - cached[0]) <= ttl:
        return cached[1], "memory"
    return None, None


def _cache_set_rag(cache_key: Tuple[str, int, str, str], value: str) -> None:
    ttl = max(0, RAG_CACHE_TTL_SECONDS)
    if ttl <= 0:
        return

    _local_rag_cache[cache_key] = (time.time(), value)
    client = _get_redis_client()
    if client is not None:
        try:
            client.setex(_redis_key("rag_local", cache_key), ttl, value)
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


def ingest_pdfs_to_chroma(
    pdf_dir: str = DEFAULT_RAG_PDF_DIR,
    reset: bool = False,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Dict[str, Any]:
    """
    Ingest PDF files from a directory into persistent ChromaDB.

    Returns a dict summary:
    - ok: bool
    - message: str
    - pdf_count / page_count / chunk_count
    """
    if not USE_CHROMA_RAG:
        return {
            "ok": False,
            "message": "USE_CHROMA_RAG is disabled. Set USE_CHROMA_RAG=true in .env.",
        }

    root = Path(pdf_dir)
    if not root.exists():
        return {"ok": False, "message": f"PDF directory not found: {root}"}

    pdf_files = sorted(root.rglob("*.pdf"))
    if not pdf_files:
        return {"ok": False, "message": f"No PDF files found under: {root}"}

    if reset and Path(CHROMA_PERSIST_DIR).exists():
        shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
        _reset_chroma_store_cache()
        _reset_local_rag_cache()

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
            # Skip problematic files but continue ingestion.
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
        _reset_local_rag_cache()
    except Exception as e:
        return {"ok": False, "message": f"ChromaDB ingestion failed: {e}"}

    return {
        "ok": True,
        "message": "PDF ingestion completed.",
        "pdf_count": len(pdf_files),
        "page_count": page_count,
        "chunk_count": len(texts),
        "persist_dir": CHROMA_PERSIST_DIR,
        "collection": CHROMA_COLLECTION_NAME,
    }


def _search_chroma_knowledge(query: str, limit: int = 3) -> str | None:
    if not USE_CHROMA_RAG:
        return None

    clean_query = (query or "").strip()
    if not clean_query:
        return "No local knowledge hits."

    try:
        store = _get_chroma_store()
    except Exception:
        # Chroma is not ready (e.g., missing dependencies/key), fallback to SQLite.
        return None

    if not _chroma_has_documents(store):
        return (
            "No ChromaDB knowledge found yet. Ingest PDFs first with "
            "`python ingest_pdfs.py --pdf-dir rag_docs --reset`."
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
            return f"ChromaDB retrieval failed: {e}"

    if not pairs:
        return "No local knowledge hits in ChromaDB."

    lines: List[str] = []
    for doc, score in pairs[: max(1, limit)]:
        metadata = getattr(doc, "metadata", {}) or {}
        source = str(metadata.get("source", "unknown"))
        page = metadata.get("page", "n/a")
        snippet = (getattr(doc, "page_content", "") or "").strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        label = f"{score_mode}={score:.4f}" if isinstance(score, (int, float)) else f"{score_mode}=n/a"
        lines.append(f"- source={source}, page={page}, {label}: {snippet}")
    return "\n".join(lines)


def _bm25_rank_documents(query: str, docs: List[str], top_k: int) -> List[Tuple[int, float]]:
    """
    Lightweight BM25-style lexical ranking over a local document list.
    Returns (doc_index, score) sorted descending.
    """
    q_tokens = _tokenize(query)
    if not q_tokens or not docs:
        return []

    tokenized_docs = [_tokenize(d) for d in docs]
    doc_lens = [len(toks) for toks in tokenized_docs]
    avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 1.0
    if avgdl <= 0:
        avgdl = 1.0

    # Document frequency per token
    df: Dict[str, int] = {}
    for toks in tokenized_docs:
        for tok in set(toks):
            df[tok] = df.get(tok, 0) + 1

    N = max(1, len(tokenized_docs))
    k1 = 1.5
    b = 0.75
    scores: List[Tuple[int, float]] = []

    for idx, toks in enumerate(tokenized_docs):
        tf = Counter(toks)
        dl = max(1, doc_lens[idx])
        score = 0.0
        for tok in q_tokens:
            if tok not in tf:
                continue
            dft = df.get(tok, 0)
            # BM25 IDF variant (positive-smoothed)
            idf = log(1 + (N - dft + 0.5) / (dft + 0.5))
            freq = tf[tok]
            denom = freq + k1 * (1 - b + b * (dl / avgdl))
            score += idf * ((freq * (k1 + 1)) / max(1e-9, denom))
        if score > 0:
            scores.append((idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[: max(1, top_k)]


def _rrf_fuse_rankings(
    vector_ranked: List[Tuple[int, float]],
    bm25_ranked: List[Tuple[int, float]],
    rrf_k: int = RAG_RRF_K,
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


def _llm_rerank_candidates(query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """
    Optional LLM reranker over fused candidates.
    Falls back to heuristic reranking on any error or missing key/dependencies.
    """
    if not RAG_ENABLE_LLM_RERANKER or not os.getenv("OPENAI_API_KEY"):
        return _heuristic_rerank(query, candidates, top_k)

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        return _heuristic_rerank(query, candidates, top_k)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    candidate_lines: List[str] = []
    for i, c in enumerate(candidates, start=1):
        candidate_lines.append(
            (
                f"{i}. file={c.get('file_name','unknown')} page={c.get('page','n/a')}\n"
                f"snippet: {c.get('snippet','')}"
            )
        )

    system = (
        "You rerank local RAG chunks for relevance to the user query.\n"
        "Return only one line with comma-separated candidate numbers in best-first order.\n"
        "Example: 2,1,3\n"
    )
    user = f"Query: {query}\n\nCandidates:\n" + "\n\n".join(candidate_lines)

    try:
        response = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        text = (getattr(response, "content", "") or "").strip()
        ranked_indices: List[int] = []
        for token in re.findall(r"\d+", text):
            idx = int(token)
            if 1 <= idx <= len(candidates) and idx not in ranked_indices:
                ranked_indices.append(idx)
        if not ranked_indices:
            return _heuristic_rerank(query, candidates, top_k)
        ordered = [candidates[i - 1] for i in ranked_indices]
        # append any missing candidates in original fused order
        used_ids = {id(c) for c in ordered}
        ordered.extend([c for c in candidates if id(c) not in used_ids])
        return ordered[: max(1, top_k)]
    except Exception:
        return _heuristic_rerank(query, candidates, top_k)


def _search_chroma_hybrid_knowledge(query: str, limit: int = 3) -> str | None:
    """
    Hybrid local retrieval:
    1) Chroma vector search candidates
    2) BM25 lexical ranking over Chroma corpus documents
    3) Reciprocal Rank Fusion (RRF)
    4) Optional LLM reranker (fallback heuristic reranker)
    """
    if not USE_CHROMA_RAG:
        return None

    clean_query = (query or "").strip()
    if not clean_query:
        return "No local knowledge hits."

    try:
        store = _get_chroma_store()
    except Exception:
        return None

    if not _chroma_has_documents(store):
        return (
            "No ChromaDB knowledge found yet. Ingest PDFs first with "
            "`python ingest_pdfs.py --pdf-dir rag_docs --reset`."
        )

    # Vector candidates
    vector_pairs: List[Tuple[Any, float]] = []
    vector_mode = "relevance"
    try:
        vector_pairs = store.similarity_search_with_relevance_scores(clean_query, k=max(limit, RAG_VECTOR_TOP_K))
    except Exception:
        try:
            vector_pairs = store.similarity_search_with_score(clean_query, k=max(limit, RAG_VECTOR_TOP_K))
            vector_mode = "distance"
        except Exception as e:
            return f"ChromaDB retrieval failed: {e}"

    if not vector_pairs:
        return "No local knowledge hits in ChromaDB."

    # Fetch corpus to run lexical BM25 over the same Chroma collection
    try:
        payload = store.get()
    except Exception as e:
        return f"ChromaDB corpus fetch failed: {e}"

    ids = list(payload.get("ids") or [])
    docs = list(payload.get("documents") or [])
    metas = list(payload.get("metadatas") or [])
    if not ids or not docs:
        return "No local knowledge hits in ChromaDB."

    # Normalize vector ranking by mapping docs to corpus indices
    def _doc_key(page_content: str, metadata: Dict[str, Any]) -> tuple[str, str, str]:
        source = str((metadata or {}).get("source", ""))
        page = str((metadata or {}).get("page", ""))
        snippet = (page_content or "").strip()
        return source, page, snippet[:200]

    corpus_keys = [_doc_key(d, metas[i] if i < len(metas) else {}) for i, d in enumerate(docs)]
    key_to_indices: Dict[tuple[str, str, str], List[int]] = {}
    for idx, key in enumerate(corpus_keys):
        key_to_indices.setdefault(key, []).append(idx)

    vector_ranked: List[Tuple[int, float]] = []
    for doc, score in vector_pairs:
        content = getattr(doc, "page_content", "") or ""
        metadata = getattr(doc, "metadata", {}) or {}
        key = _doc_key(content, metadata)
        idx_list = key_to_indices.get(key, [])
        if not idx_list:
            continue
        # use first match and pop to reduce duplicate mapping collisions
        corpus_idx = idx_list.pop(0)
        # For distance mode, lower is better. Convert roughly to higher-is-better.
        norm_score = float(score)
        if vector_mode == "distance":
            norm_score = 1.0 / (1.0 + max(0.0, norm_score))
        vector_ranked.append((corpus_idx, norm_score))

    bm25_ranked = _bm25_rank_documents(clean_query, docs, top_k=max(limit, RAG_BM25_TOP_K))
    fused_map = _rrf_fuse_rankings(vector_ranked, bm25_ranked, rrf_k=RAG_RRF_K)
    if not fused_map:
        return "No local knowledge hits in ChromaDB."

    fused_sorted = sorted(fused_map.items(), key=lambda x: x[1]["fused"], reverse=True)

    # Build candidate objects for reranking/formatting
    candidates: List[Dict[str, Any]] = []
    for doc_idx, score_blob in fused_sorted[: max(limit, RAG_RERANK_TOP_K)]:
        metadata = metas[doc_idx] if doc_idx < len(metas) and isinstance(metas[doc_idx], dict) else {}
        snippet = (docs[doc_idx] or "").strip().replace("\n", " ")
        if len(snippet) > 260:
            snippet = snippet[:257] + "..."
        candidates.append(
            {
                "doc_idx": doc_idx,
                "source": str(metadata.get("source", "unknown")),
                "file_name": str(metadata.get("file_name", metadata.get("source", "unknown"))),
                "page": metadata.get("page", "n/a"),
                "snippet": snippet,
                "vector_score": score_blob.get("vector", 0.0),
                "bm25_score": score_blob.get("bm25", 0.0),
                "fused_score": score_blob.get("fused", 0.0),
            }
        )

    reranked = _llm_rerank_candidates(clean_query, candidates, top_k=max(1, limit))

    lines: List[str] = []
    for c in reranked[: max(1, limit)]:
        lines.append(
            (
                f"- source={c['source']}, page={c['page']}, "
                f"fused={c['fused_score']:.4f}, bm25={c['bm25_score']:.4f}, vector={c['vector_score']:.4f}: "
                f"{c['snippet']}"
            )
        )
    return "\n".join(lines)


def search_local_knowledge(
    query: str,
    limit: int = 3,
    db_path: str = DEFAULT_RAG_DB_PATH,
    return_meta: bool = False,
) -> str | tuple[str, Dict[str, Any]]:
    """
    Retrieve local context.

    Priority:
    1) ChromaDB semantic retrieval (if USE_CHROMA_RAG=true and available)
    2) SQLite token-overlap fallback
    """
    clean_query = (query or "").strip()
    cache_key = (
        clean_query.lower(),
        int(limit),
        str(bool(USE_CHROMA_RAG)),
        str(Path(db_path).name),
    )
    cached_value, cache_backend = _cache_get_rag(cache_key)
    if cached_value is not None:
        meta = {"cache_hit": True, "cache_backend": cache_backend or "unknown", "source": "rag_cache"}
        return (cached_value, meta) if return_meta else cached_value

    chroma_result = _search_chroma_hybrid_knowledge(query=clean_query, limit=limit)
    if chroma_result is None:
        chroma_result = _search_chroma_knowledge(query=clean_query, limit=limit)
    if chroma_result is not None:
        _cache_set_rag(cache_key, chroma_result)
        meta = {"cache_hit": False, "cache_backend": "none", "source": "rag_live"}
        return (chroma_result, meta) if return_meta else chroma_result
    sqlite_result = _search_sqlite_knowledge(query=clean_query, limit=limit, db_path=db_path)
    _cache_set_rag(cache_key, sqlite_result)
    meta = {"cache_hit": False, "cache_backend": "none", "source": "rag_sqlite_fallback"}
    return (sqlite_result, meta) if return_meta else sqlite_result
