# Agent Orchestration (LangGraph + FastAPI + Streamlit)

A production-style multi-agent GenAI assistant built with LangGraph, FastAPI, Streamlit, PostgreSQL persistence, and local RAG (ChromaDB + PDF ingestion).

## Highlights

- 10-agent LangGraph orchestration graph
- FastAPI service with `/invoke` and `/stream`
- Streamlit chat UI
- Hybrid supervisor routing (`intent_router_agent`: rules first, LLM fallback for low-signal cases)
- LlamaGuard moderation (`safety_agent`)
- Web retrieval with recency preferences + relevance filtering
- Local RAG with ChromaDB + hybrid retrieval (vector + BM25 + reranking), SQLite fallback
- TTL caching for web search and local RAG retrieval
- Optional Redis-backed caching (with in-memory fallback)
- Dual-layer persistence:
  - LangGraph PostgreSQL checkpointer (graph state continuity)
  - Conversation Store (PostgreSQL with SQLite fallback)
- Evaluation agent (heuristic quality audit)

## Agent Architecture (10 Agents)

1. `safety_agent`
2. `intent_router_agent`
3. `clarification_agent`
4. `query_rewriter_agent`
5. `recency_guard_agent`
6. `web_search_agent`
7. `rag_agent`
8. `math_agent`
9. `response_agent`
10. `evaluation_agent`

## Full Architecture Flow (Mermaid)

```mermaid
flowchart TD
    U[User input in Streamlit] --> ST[streamlit_app.py]
    ST -->|message, model, thread_id| API[FastAPI /invoke or /stream]

    API --> STOREH[Store human message]
    API --> CFG[Build RunnableConfig - thread_id checkpoint model]
    CFG --> LG[LangGraph research_assistant]

    LG --> SA[safety_agent]
    SA --> IR[intent_router_agent]

    IR -->|clarify| CA[clarification_agent]
    IR -->|rewrite| QR[query_rewriter_agent]
    QR --> IR

    IR -->|math| MA[math_agent]
    IR -->|web| RG[recency_guard_agent]
    IR -->|rag| RA[rag_agent]
    IR -->|hybrid| RG
    IR -->|general| RESP[response_agent]

    RG --> WS[web_search_agent - web cache check]
    WS -->|web| RESP
    WS -->|hybrid| RA
    RA[RAG agent - local RAG cache check] --> RESP
    MA --> RESP

    CA --> EVA[evaluation_agent]
    RESP --> EVA
    EVA --> END[Graph END]

    END --> APIOUT[FastAPI serializes final AI message]
    APIOUT --> STOREA[Store AI message + metadata]
    STOREA --> RET[Return JSON or SSE]
    RET --> UI[Render in Streamlit]
```

### Flow Notes

- `clarification_agent` asks a follow-up question and ends the current run.
- The next user message starts a new run and is routed again.
- `local:` prefix forces routing to `rag_agent`.
- `recency_guard_agent` applies recency as a preference (fallback to most recent relevant results).
- Mermaid graph edges remain the same after hybrid-router / RAG reranking upgrades because those improvements happen inside `intent_router_agent` and `rag_agent` internals.
- Cache checks happen inside `web_search_agent` and `rag_agent` retrieval functions (Redis/in-memory fallback), so graph topology still does not change.

## Project Structure (Key Files)

- `agent/research_assistant.py` - LangGraph orchestration, agents, routing logic
- `agent/tools.py` - web search + filtering logic
- `agent/local_rag.py` - local RAG (ChromaDB + SQLite fallback)
- `agent/llama_guard.py` - moderation logic
- `service/service.py` - FastAPI service + endpoints + checkpointer/store wiring
- `service/persistence_store.py` - conversation Store layer (Postgres/SQLite)
- `streamlit_app.py` - Streamlit UI
- `ingest_pdfs.py` - PDF ingestion to ChromaDB
- `FLOW_CHART.md` / `flowchart.mmd` - flow diagram references

## Endpoints

- `POST /invoke` - non-streaming chat response
- `POST /stream` - streaming chat response
- `GET /store/{thread_id}` - inspect persisted conversation records
- `POST /feedback` - user feedback/rating

## Data & Persistence

### Dual-layer persistence

1. **Checkpointer (LangGraph, PostgreSQL)**
- Stores graph execution/checkpoint state by `thread_id`
- Used for workflow state continuity/resume

2. **Conversation Store (PostgreSQL, SQLite fallback)**
- Stores durable human/AI messages + metadata
- Used for history/debugging/audit (`/store/{thread_id}`)

### PostgreSQL tables you will see

- Checkpointer tables:
  - `checkpoints`
  - `checkpoint_writes`
  - `checkpoint_blobs`
- Conversation store table:
  - `conversation_store`

## Routing Summary (Supervisor Logic)

`intent_router_agent` uses a hybrid strategy:

1. Deterministic rules first (high precision, low latency)
2. Optional LLM classifier fallback for low-signal queries (confidence-gated)
3. Safe fallback to `general` if classifier confidence is low/unavailable

Primary deterministic rules:

- `local:` prefix -> `rag`
- ambiguous (`help me`, `this`, `that`) -> `clarify`
- vague but rewritable (`news`, `latest news`) -> `rewrite`
- math-like query -> `math`
- web keywords -> `web`
- local/project keywords -> `rag`
- both web + local keywords -> `hybrid`
- fallback -> `general`

Router debug metadata written into state:
- `route_confidence`
- `route_reason`

## Setup

### 1. Clone

```bash
git clone https://github.com/Theepankumargandhi/Agent-Orchestration.git
cd Agent-Orchestration
```

### 2. Environment variables (`.env`)

Set at least the following (adjust values to your machine):

```env
OPENAI_API_KEY=...
GROQ_API_KEY=...

# FastAPI service port (your current setup uses 8080)
PORT=8080
API_BASE_URL=http://localhost:8080

# LangGraph checkpointer (PostgreSQL)
POSTGRES_CHECKPOINT_URI=postgresql://postgres:password@localhost:5432/agentdb
CHECKPOINT_FALLBACK_SQLITE=true

# Conversation Store (defaults to checkpoint URI if omitted)
POSTGRES_STORE_URI=postgresql://postgres:password@localhost:5432/agentdb
STORE_FALLBACK_SQLITE=true
STORE_DB_PATH=store.db
STORE_NAMESPACE=default

# RAG / ChromaDB
USE_CHROMA_RAG=true
CHROMA_PERSIST_DIR=chroma_db
CHROMA_COLLECTION_NAME=local_pdf_docs
RAG_PDF_DIR=rag_docs
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=150

# Optional hybrid router (rules + LLM fallback)
HYBRID_ROUTER_ENABLE=true
HYBRID_ROUTER_MIN_CONFIDENCE=0.75

# Optional hybrid RAG retrieval + reranking (Chroma path)
RAG_VECTOR_TOP_K=8
RAG_BM25_TOP_K=8
RAG_RERANK_TOP_K=6
RAG_RRF_K=60
RAG_ENABLE_LLM_RERANKER=true

# Optional TTL caches (latency/cost optimization)
WEB_CACHE_TTL_SECONDS=300
RAG_CACHE_TTL_SECONDS=600

# Optional Redis cache backend (shared cache across processes/containers)
CACHE_USE_REDIS=true
REDIS_URL=redis://localhost:6379/0

# Optional API auth
AUTH_SECRET=
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run backend

```bash
python run_service.py
```

### 5. Run Streamlit UI (separate terminal)

```bash
streamlit run streamlit_app.py
```

## Optional: Ingest PDFs into ChromaDB

1. Put PDFs in `rag_docs/`
2. Run:

```bash
python ingest_pdfs.py --pdf-dir rag_docs --reset
```

3. Ask a forced local query:

```text
local: summarize the uploaded pdfs
```

### Local RAG retrieval pipeline (current)

When ChromaDB is enabled, local retrieval now uses a hybrid pipeline inside `agent/local_rag.py`:

1. Chroma vector retrieval (semantic)
2. BM25-style lexical ranking over local Chroma corpus
3. Reciprocal Rank Fusion (RRF)
4. Optional LLM reranker (falls back to heuristic reranker)

This improves local retrieval precision for paraphrases and keyword-heavy queries.

### Caching (current)

- **Web search TTL cache** (`agent/tools.py`)
  - caches `perform_web_search(...)` results by query/recency/relevance key
  - default TTL: `300s`
- **Local RAG TTL cache** (`agent/local_rag.py`)
  - caches `search_local_knowledge(...)` results by query/limit/backend key
  - default TTL: `600s`
  - cache is cleared automatically after PDF ingestion/reset
- **Redis support (optional)**
  - if `REDIS_URL` is configured and reachable, web/RAG caches use Redis (`setex`)
  - if Redis is unavailable, system falls back to local in-memory TTL cache automatically
- **UI cache visibility**
  - the response footer shows cache usage only on cache hits (for example: `Cache: web via memory cache`)
  - live retrievals do not add extra source text

## Verify Persistence

### Conversation Store (human-readable)

Open in browser (replace with your sidebar thread ID and correct backend port):

```text
http://localhost:8080/store/<thread_id>?limit=50
```

### PostgreSQL (pgAdmin)

- `conversation_store` -> human/AI messages + metadata
- `checkpoints`, `checkpoint_writes`, `checkpoint_blobs` -> LangGraph checkpoint internals

## Known Limitations (Honest / Interview-ready)

- `evaluation_agent` is heuristic (not factual verification)
- `clarification_agent` ends current run; final answer comes on next user turn
- hybrid router still depends on classifier confidence thresholds and can misroute low-signal queries
- RAG quality still depends on chunking, embeddings, ingestion quality, and reranker behavior

