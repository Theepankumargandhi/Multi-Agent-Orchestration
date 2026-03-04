# Agent Orchestration: LangGraph Multi-Agent Supervisor (FastAPI + Streamlit + RAG)

Production-ready **LangGraph multi-agent orchestration** project with a **supervisor routing agent**, FastAPI backend, Streamlit frontend, PostgreSQL persistence, local RAG (ChromaDB), knowledge graph retrieval, and MCP tool integration.

If you are searching for a **LangGraph multi-agent orchestration example**, **supervisor agent architecture**, or a **production multi-agent AI template**, this repository is built for that exact use case.

[![CI](https://github.com/Theepankumargandhi/Multi-Agent-Orchestration/actions/workflows/ci.yml/badge.svg)](https://github.com/Theepankumargandhi/Multi-Agent-Orchestration/actions/workflows/ci.yml)
[![CD](https://github.com/Theepankumargandhi/Multi-Agent-Orchestration/actions/workflows/cd-release.yml/badge.svg)](https://github.com/Theepankumargandhi/Multi-Agent-Orchestration/actions/workflows/cd-release.yml)

Keywords: `langgraph`, `multi-agent orchestration`, `supervisor agent`, `agent routing`, `fastapi`, `streamlit`, `rag`, `knowledge graph`, `mcp`, `genai`.

## What You Can Build With This Repo

- Multi-agent assistants with a supervisor/router pattern
- Agentic RAG systems with hybrid retrieval (vector + BM25 + reranking)
- Real-time chat services using FastAPI + Streamlit + SSE streaming
- Tool-using agents via MCP (`web_search`, `calculator`)

## Highlights

- 11-agent LangGraph orchestration graph
- FastAPI service with `/invoke` and `/stream`
- Operational endpoints: `/metrics`, `/healthz`, `/readyz`
- Streamlit chat UI
- Hybrid supervisor routing (`intent_router_agent`: rules first, LLM fallback for low-signal cases)
- LlamaGuard moderation (`safety_agent`)
- Web retrieval with recency preferences + relevance filtering
- Local RAG with ChromaDB + hybrid retrieval (vector + BM25 + reranking), SQLite fallback
- Knowledge graph reasoning agent (NetworkX) for relationship-style local queries
- TTL caching for web search and local RAG retrieval
- Optional Redis-backed caching (with in-memory fallback)
- MCP tool bridge for `web_search` and `calculator` (with local fallback)
- Monitoring stack with Prometheus + Grafana
- Dual-layer persistence:
  - LangGraph PostgreSQL checkpointer (graph state continuity)
  - Conversation Store (PostgreSQL with SQLite fallback)
- Evaluation agent (heuristic quality audit)
- Per-user authentication (user ID + password) with bearer tokens

## Agent Architecture (11 Agents)

1. `safety_agent`
2. `intent_router_agent`
3. `clarification_agent`
4. `query_rewriter_agent`
5. `recency_guard_agent`
6. `web_search_agent`
7. `knowledge_graph_agent`
8. `rag_agent`
9. `math_agent`
10. `response_agent`
11. `evaluation_agent`

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
    IR -->|kg| KG[knowledge_graph_agent]
    IR -->|rag| RA[rag_agent]
    IR -->|hybrid| RG
    IR -->|general| RESP[response_agent]

    RG --> WS[web_search_agent - web cache check]
    WS -->|web| RESP
    WS -->|hybrid| RA
    KG --> RA
    RA[RAG agent - local RAG cache check] --> RESP
    MA --> RESP

    CA --> EVA[evaluation_agent]
    RESP --> EVA
    EVA --> END[Graph END]

    END --> APIOUT[FastAPI serializes final AI message]
    APIOUT --> STOREA[Store AI message + metadata]
    STOREA --> RET[Return JSON or SSE]
    RET --> UI[Render in Streamlit]

    API --> OPS["/healthz | /readyz | /metrics"]
    OPS --> PROM[Prometheus scrape]
    PROM --> GRAF[Grafana dashboards]

    subgraph K8S[Optional Local Kubernetes Deployment]
      KAGENT[agent-service Deployment + Service]
      KUI[streamlit-app Deployment + Service]
      KPROM[prometheus Deployment + Service]
      KGRAF[grafana Deployment + Service]
    end
    KAGENT --> KPROM
    KPROM --> KGRAF
    KUI --> KAGENT
```

### Flow Notes

- `clarification_agent` asks a follow-up question and ends the current run.
- The next user message starts a new run and is routed again.
- `local:` prefix routes to `rag_agent` or `knowledge_graph_agent` for relationship questions.
- `recency_guard_agent` applies recency as a preference (fallback to most recent relevant results).
- Mermaid graph edges remain stable for retrieval internals; new relationship reasoning runs in `knowledge_graph_agent`.
- Cache checks happen inside `web_search_agent`, `rag_agent`, and graph-rag retrieval functions (Redis/in-memory fallback).

## Project Structure (Key Files)

- `agent/research_assistant.py` - LangGraph orchestration, agents, routing logic
- `agent/tools.py` - web search + filtering logic
- `agent/local_rag.py` - local RAG (ChromaDB + SQLite fallback)
- `agent/graph_rag.py` - dedicated graph RAG Chroma retrieval store
- `agent/knowledge_graph.py` - NetworkX relationship extraction over graph RAG evidence
- `agent/llama_guard.py` - moderation logic
- `service/service.py` - FastAPI service + endpoints + checkpointer/store wiring
- `service/persistence_store.py` - conversation Store layer (Postgres/SQLite)
- `streamlit_app.py` - Streamlit UI
- `monitoring/prometheus.yml` - Prometheus scrape config for `/metrics`
- `scripts/ingestion/ingest_local_rag_pdfs.py` - PDF ingestion to local RAG ChromaDB
- `scripts/ingestion/ingest_graph_rag_pdfs.py` - separate PDF ingestion for knowledge graph store
- `scripts/ingestion/generate_synthetic_graph_pdfs.py` - synthetic graph PDFs for KG testing
- `k8s/` - local Kubernetes manifests (app + monitoring stack)
- `docs/architecture/agent_runtime_flow.md` - runtime flow explainer (human-readable)
- `docs/architecture/agent_runtime_flow.mmd` - raw Mermaid source for the same flow

## GitHub Actions CI/CD

- `CI` workflow (`.github/workflows/ci.yml`)
  - triggers on `push` and `pull_request` to `main`
  - runs Python tests: `service/test_service.py` and `schema/test_schema.py`
  - verifies both Docker images build successfully
- `CD` workflow (`.github/workflows/cd-release.yml`)
  - triggers on version tags (`v*`) or manual dispatch
  - builds and publishes Docker images to Docker Hub:
    - `<namespace>/multi-agent-orchestration-service:<version>`
    - `<namespace>/multi-agent-orchestration-app:<version>`
  - required repo secrets:
    - `DOCKERHUB_USERNAME`
    - `DOCKERHUB_TOKEN` (Docker Hub access token)
  - optional repo variable:
    - `DOCKERHUB_NAMESPACE` (if omitted, username is used as namespace)

### Release Tag for CD

```bash
git tag v1.0.0
git push origin v1.0.0
```

## Endpoints

- `POST /auth/register` - register user and receive access token
- `POST /auth/login` - login user and receive access token
- `POST /invoke` - non-streaming chat response
- `POST /stream` - streaming chat response
- `GET /store/{thread_id}` - inspect persisted conversation records
- `POST /feedback` - user feedback/rating
- `GET /healthz` - liveness probe endpoint
- `GET /readyz` - readiness probe endpoint
- `GET /metrics` - Prometheus scrape endpoint

When `ENABLE_USER_AUTH=true`, all non-auth endpoints require `Authorization: Bearer <access_token>`.

## Monitoring (Prometheus + Grafana)

### Docker Compose (Local)

Bring up full stack:

```bash
docker compose up -d --build
```

Access:

- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3001` (default `admin` / `admin`)

Prometheus target should show `agent_service` as `UP` at:

```text
http://localhost:9090/targets
```

### Recommended Grafana Panels

- Request Rate: `sum(rate(http_requests_total[5m]))`
- Error Rate (5xx):
  - `sum(rate(http_requests_total{status_code=~"5.."}[15m])) / clamp_min(sum(rate(http_requests_total[15m])), 1e-9)`
- P95 Latency:
  - `histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket[15m])))`
- Traffic by endpoint:
  - `sum by (path) (rate(http_requests_total[5m]))`
- Status code distribution:
  - `sum by (status_code) (rate(http_requests_total[5m]))`

## Kubernetes (Local)

Beginner-friendly Kubernetes manifests are available in `k8s/` with step-by-step instructions:

- See [k8s/README.md](k8s/README.md)
- Includes app + streamlit + Prometheus + Grafana manifests
- Includes health/readiness probes
- Supports one-command apply via `kubectl apply -k k8s`

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
  - `users`

## Routing Summary (Supervisor Logic)

`intent_router_agent` uses a hybrid strategy:

1. Deterministic rules first (high precision, low latency)
2. Optional LLM classifier fallback for low-signal queries (confidence-gated)
3. Safe fallback to `general` if classifier confidence is low/unavailable

Primary deterministic rules:

- `local:` prefix -> `rag` (or `kg` when relation intent is detected)
- relationship reasoning + local context -> `kg`
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
CHECKPOINT_DB_PATH=data/checkpoints/checkpoints.db

# Conversation Store (defaults to checkpoint URI if omitted)
POSTGRES_STORE_URI=postgresql://postgres:password@localhost:5432/agentdb
STORE_FALLBACK_SQLITE=true
STORE_DB_PATH=data/store/store.db
STORE_NAMESPACE=default

# RAG / ChromaDB
USE_CHROMA_RAG=true
CHROMA_PERSIST_DIR=data/chroma_db
CHROMA_COLLECTION_NAME=local_pdf_docs
RAG_PDF_DIR=rag_docs
LOCAL_RAG_DB_PATH=data/rag/local_rag.db
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

# Graph RAG (separate store for knowledge_graph_agent)
GRAPH_RAG_ENABLED=true
GRAPH_RAG_PDF_DIR=graph_rag_docs
GRAPH_CHROMA_PERSIST_DIR=data/graph_chroma_db
GRAPH_CHROMA_COLLECTION_NAME=graph_pdf_docs
GRAPH_RAG_CHUNK_SIZE=1000
GRAPH_RAG_CHUNK_OVERLAP=150
GRAPH_RAG_CACHE_TTL_SECONDS=600

# Optional TTL caches (latency/cost optimization)
WEB_CACHE_TTL_SECONDS=300
RAG_CACHE_TTL_SECONDS=600

# Optional Redis cache backend (shared cache across processes/containers)
CACHE_USE_REDIS=true
REDIS_URL=redis://localhost:6379/0

# Optional API auth
AUTH_SECRET=

# Per-user auth (enabled by default)
ENABLE_USER_AUTH=true
USER_AUTH_SECRET=change_me_to_a_long_random_secret
USER_AUTH_TOKEN_TTL_SECONDS=86400
PASSWORD_HASH_ITERATIONS=210000

# Optional MCP tools (web_search + calculator)
MCP_TOOLS_ENABLED=true
MCP_TOOL_SERVER_COMMAND=.venv\Scripts\python.exe
MCP_TOOL_SERVER_SCRIPT=agent\mcp_tool_server.py
MCP_TOOL_SERVER_ARGS=
MCP_BRIDGE_COMMAND=.venv\Scripts\python.exe
MCP_BRIDGE_SCRIPT=agent\mcp_bridge_client.py
MCP_CALL_TIMEOUT_SECONDS=20
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run backend

```bash
python run_service.py
```

### 4b. MCP sidecar dependencies (only in `.venv`, not in `llm_env`)

If your main runtime is `llm_env`, install MCP only in the separate `.venv` used by `MCP_BRIDGE_COMMAND` / `MCP_TOOL_SERVER_COMMAND`:

```bash
.venv\Scripts\python.exe -m pip install mcp==1.12.4
```

If `.venv\Scripts\python.exe` shows `No Python at ...`, recreate `.venv` first:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install mcp==1.12.4
```

### 5. Run Streamlit UI (separate terminal)

```bash
streamlit run streamlit_app.py
```

On first launch, use the sidebar `Register` flow (user ID + password), then sign in.

## Optional: Ingest PDFs into ChromaDB

1. Put PDFs in `rag_docs/`
2. Run:

```bash
python scripts/ingestion/ingest_local_rag_pdfs.py --pdf-dir rag_docs --reset
```

3. Ask a forced local query:

```text
local: summarize the uploaded pdfs
```

## Optional: Ingest Graph PDFs (Separate KG Store)

1. Put relationship-focused PDFs in `graph_rag_docs/`
2. Run:

```bash
python scripts/ingestion/ingest_graph_rag_pdfs.py --pdf-dir graph_rag_docs --reset
```

3. Ask a relationship query:

```text
local: how is streamlit connected to fastapi in this project
```

### Ingestion Separation 

- `scripts/ingestion/ingest_local_rag_pdfs.py` updates **Local RAG** only (`rag_docs` -> `chroma_db`).
- `scripts/ingestion/ingest_graph_rag_pdfs.py` updates **Graph RAG / KG** only (`graph_rag_docs` -> `graph_chroma_db`).
- `knowledge_graph_agent` uses Graph RAG retrieval, then forwards to `rag_agent` for extra grounding before `response_agent`.

### Local RAG retrieval pipeline 

When ChromaDB is enabled, local retrieval now uses a hybrid pipeline inside `agent/local_rag.py`:

1. Chroma vector retrieval (semantic)
2. BM25-style lexical ranking over local Chroma corpus
3. Reciprocal Rank Fusion (RRF)
4. Optional LLM reranker (falls back to heuristic reranker)

This improves local retrieval precision for paraphrases and keyword-heavy queries.

### Caching 

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
  - the response footer shows source path when available (for example: `Sources: web via mcp` or `Sources: web via memory cache`)
  - live retrievals may omit source footer when no source metadata is present

### MCP tools 

- `web_search_agent` calls an MCP stdio tool server first for web search.
- `math_agent` calls an MCP stdio calculator tool first.
- Main app env (`llm_env`) does not need the `mcp` package anymore.
- MCP client calls a sidecar bridge script (`agent/mcp_bridge_client.py`) using `MCP_BRIDGE_COMMAND`.
- In `llm_env`, point both `MCP_BRIDGE_COMMAND` and `MCP_TOOL_SERVER_COMMAND` to a separate `.venv` Python where `mcp` is installed.
- If MCP is disabled/unavailable, both agents automatically fall back to existing local logic.
- Defaults:
  - `MCP_TOOL_SERVER_SCRIPT` -> `agent/mcp_tool_server.py`
  - `MCP_BRIDGE_SCRIPT` -> `agent/mcp_bridge_client.py`

## Verify Persistence

### Conversation Store 

Open in browser (replace with your sidebar thread ID and correct backend port):

```text
http://localhost:8080/store/<thread_id>?limit=50
```

### PostgreSQL (pgAdmin)

- `conversation_store` -> human/AI messages + metadata
- `checkpoints`, `checkpoint_writes`, `checkpoint_blobs` -> LangGraph checkpoint internals

## Known Limitations 

- `evaluation_agent` is heuristic (not factual verification)
- `clarification_agent` ends current run; final answer comes on next user turn
- hybrid router still depends on classifier confidence thresholds and can misroute low-signal queries
- RAG quality still depends on chunking, embeddings, ingestion quality, and reranker behavior

