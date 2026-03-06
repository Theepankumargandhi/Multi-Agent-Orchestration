# Agent Runtime Flow

## File Purpose

- Human-readable architecture explanation for the runtime execution flow.
- Best for onboarding and quick understanding of how requests move across agents.

This flowchart matches the current codebase behavior (`agent/research_assistant.py`, `service/service.py`, `streamlit_app.py`).

```mermaid
flowchart TD
    U[User in Streamlit] --> AUTH[Login or Register]
    AUTH --> HIST[Load conversation history]
    HIST --> Q[User sends message]
    Q --> API[FastAPI invoke or stream]

    API --> STOREH[Store human message]
    API --> HMAP[HITL mapper for plain approve or reject]
    HMAP --> CFG[Build graph config model + thread_id]
    CFG --> LG[LangGraph research_assistant]
    CFG --> CHK[Postgres checkpoints]

    LG --> SA[safety_agent]
    SA --> IR[intent_router_agent]

    IR -->|clarify| CA[clarification_agent]
    CA --> EVA[evaluation_agent]

    IR -->|rewrite| QR[query_rewriter_agent]
    QR --> IR

    IR -->|math| MA[math_agent]
    MA --> RESP[response_agent]

    IR -->|rag| RA[rag_agent]
    RA --> RESP

    IR -->|kg| KG[knowledge_graph_agent]
    KG --> RA

    IR -->|general| RESP

    IR -->|web or hybrid| RG[recency_guard_agent]
    RG --> WH[web_hitl_gate_agent]

    WH -->|awaiting| WAIT[Wait for decision]
    WAIT --> BTN[UI buttons Approve or Reject]
    BTN --> API

    WH -->|approved web| WS[web_search_agent]
    WS -->|web route| RESP

    WH -->|approved hybrid| WS
    WS -->|hybrid route| RA

    WH -->|rejected| REJ[Reject follow-up message]
    REJ --> EVA

    RESP --> EVA
    EVA --> END[Graph end]
    END --> APIRESP[Return answer to Streamlit]
    APIRESP --> STOREA[Store AI message]
    STOREA --> CST[conversation_store]
    END --> HITLDB[Store HITL decision]
    HITLDB --> HITL[hitl_events]

    API --> METRICS["/healthz | /readyz | /metrics"]
    METRICS --> PROM[Prometheus]
    PROM --> GRAF[Grafana]
```

## Notes

- `clarification_agent` does not continue to `response_agent` in the same run.
- Clarified user reply comes as a new turn and is routed again by `intent_router_agent`.
- On login, UI calls `GET /store/threads`, auto-loads latest thread, and can switch older thread history.
- For recency/news prompts, graph-level HITL runs in `web_hitl_gate_agent`.
- Streamlit shows `Approve`/`Reject` buttons for waiting HITL decisions (typing `approve` or `reject: <reason>` also works).
- Service rewrites plain approve/reject input into internal HITL control payload using pending context for the same `user_id` + `thread_id`.
- HITL decisions are audited automatically and persisted in `hitl_events`.
- `local:` prefix routes to `rag` or `kg` depending on relationship intent.
- `knowledge_graph_agent` reads from dedicated Graph RAG ingestion (`graph_rag_docs` -> `graph_chroma_db`).
- `rag_agent` reads from local RAG ingestion (`rag_docs` -> `chroma_db`).
- Web/RAG/Graph-RAG cache checks happen inside retrieval internals (Redis or in-memory fallback).
- UI can show source-path footer metadata (for example: `Sources: web via mcp` or cache backend labels).
- Prometheus scrapes `/metrics`; Grafana visualizes Prometheus data.
- Kubernetes manifests for this flow are available in `k8s/`.
