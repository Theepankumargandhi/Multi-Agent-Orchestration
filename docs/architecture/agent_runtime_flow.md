# Agent Runtime Flow

## File Purpose

- Human-readable architecture explanation for the runtime execution flow.
- Best for onboarding and quick understanding of how requests move across agents.

This flowchart matches the current codebase behavior (`agent/research_assistant.py`, `service/service.py`, `streamlit_app.py`).

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

    API --> OPS[/healthz /readyz /metrics]
    OPS --> PROM[Prometheus scrape]
    PROM --> GRAF[Grafana dashboards]

    subgraph K8S[Optional Local Kubernetes Deployment]
      KCM[ConfigMap + Secret]
      KAGENT[agent-service Deployment + Service]
      KUI[streamlit-app Deployment + Service]
      KPROM[prometheus Deployment + Service]
      KGRAF[grafana Deployment + Service]
    end

    KCM --> KAGENT
    KCM --> KUI
    KAGENT --> KPROM
    KPROM --> KGRAF
    KUI --> KAGENT
```

## Notes

- `clarification_agent` does not continue to `response_agent` in the same run.
- Clarified user reply comes as a new turn and is routed again by `intent_router_agent`.
- `local:` prefix routes to `rag` or `kg` depending on relationship intent.
- `knowledge_graph_agent` reads from dedicated Graph RAG ingestion (`graph_rag_docs` -> `graph_chroma_db`).
- `rag_agent` reads from local RAG ingestion (`rag_docs` -> `chroma_db`).
- Web/RAG/Graph-RAG cache checks happen inside retrieval internals (Redis or in-memory fallback).
- UI can show source-path footer metadata (for example: `Sources: web via mcp` or cache backend labels).
- Prometheus scrapes `/metrics`; Grafana visualizes Prometheus data.
- Kubernetes manifests for this flow are available in `k8s/`.
