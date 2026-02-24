import os
import re
import shutil
import sqlite3
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

_chroma_store_cache = None

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


def search_local_knowledge(query: str, limit: int = 3, db_path: str = DEFAULT_RAG_DB_PATH) -> str:
    """
    Retrieve local context.

    Priority:
    1) ChromaDB semantic retrieval (if USE_CHROMA_RAG=true and available)
    2) SQLite token-overlap fallback
    """
    chroma_result = _search_chroma_knowledge(query=query, limit=limit)
    if chroma_result is not None:
        return chroma_result
    return _search_sqlite_knowledge(query=query, limit=limit, db_path=db_path)
