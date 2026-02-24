# Agent Orchestration (LangGraph + FastAPI + Streamlit)

A production-style multi-agent GenAI assistant built with LangGraph, FastAPI, Streamlit, PostgreSQL persistence, and local RAG (ChromaDB + PDF ingestion).

## Highlights

- 10-agent LangGraph orchestration graph
- FastAPI service with `/invoke` and `/stream`
- Streamlit chat UI
- Rule-based supervisor routing (`intent_router_agent`)
- LlamaGuard moderation (`safety_agent`)
- Web retrieval with recency preferences + relevance filtering
- Local RAG with ChromaDB (SQLite fallback)
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
    API --> CFG[Build RunnableConfig<br/>thread_id/checkpoint/model]
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

    RG --> WS[web_search_agent]
    WS -->|web| RESP
    WS -->|hybrid| RA
    RA --> RESP
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

`intent_router_agent` is rule-based (not LLM-based) and routes using keyword/regex heuristics:

- `local:` prefix -> `rag`
- ambiguous (`help me`, `this`, `that`) -> `clarify`
- vague but rewritable (`news`, `latest news`) -> `rewrite`
- math-like query -> `math`
- web keywords -> `web`
- local/project keywords -> `rag`
- both web + local keywords -> `hybrid`
- fallback -> `general`

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
- router is rule-based and can miss nuanced/paraphrased intent
- RAG quality depends on chunking, embeddings, and clean PDF ingestion

## License

MIT (see `LICENSE`)
