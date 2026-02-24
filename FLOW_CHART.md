# Agent Input-to-Output Flow

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

## Notes

- `clarification_agent` does not continue to `response_agent` in the same run.
- Clarified user reply comes as a new turn and is routed again by `intent_router_agent`.
- `local:` prefix forces the RAG route.
- Web/RAG cache checks happen inside `web_search_agent` / `rag_agent` internals (Redis or in-memory fallback).
- UI shows a cache footer only on cache hits (e.g., `Cache: web via memory cache`).
