# CiteRight

A research assistant that answers from your documents and the web, cites where each claim came
from, and refuses to make things up.

[![CI](https://github.com/Theepankumargandhi/Multi-Agent-Orchestration/actions/workflows/ci.yml/badge.svg)](https://github.com/Theepankumargandhi/Multi-Agent-Orchestration/actions/workflows/ci.yml)

Most chat assistants write a fluent answer first and leave you to work out which parts are true.
CiteRight checks the answer before you see it. Every factual claim is matched against the evidence
it retrieved; claims with no support, or that contradict a source, are removed. If nothing
survives, it says it doesn't have enough evidence instead of guessing.

## What it does

- **Answers from your documents.** Ingest PDFs or Markdown; questions about them use hybrid
  retrieval (vector + BM25, fused and reranked) with source and section citations.
- **Searches the web, with your permission.** Questions about recent events trigger a web search
  that pauses for your approval first. Answers include links to the pages used.
- **Checks every answer.** A grounding verifier maps each claim to evidence, rejects citations to
  pages it never retrieved, and catches claims that contradict their source (numbers, negation,
  opposite words). See [docs/grounding-verification.md](docs/grounding-verification.md).
- **Refuses unsafe requests.** A moderation model checks each message first and fails closed.
- **Does exact maths.** Arithmetic goes to a calculator, not the language model.
- **Remembers conversations.** Threads persist per user; approval decisions survive an API restart.

## How a question flows

```
question → safety check → router ─┬─ your documents (RAG / knowledge graph)
                                  ├─ web search (after you approve)
                                  ├─ calculator
                                  └─ general chat
                       → draft answer → grounding check → release / repair / abstain
```

The flow is a LangGraph state machine with one node per step (17 in total), served by FastAPI
with a React UI. Details: [docs/architecture/agent_runtime_flow.md](docs/architecture/agent_runtime_flow.md).

## Quick start

You need Python 3.11+, Node 22.12+ and a free [Groq API key](https://console.groq.com/keys)
(or an OpenAI key).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --editable .

cp .env.example .env                 # then set GROQ_API_KEY
python scripts/ingestion/ingest_repo_docs.py   # index this repo's docs so you can ask about them
python run_service.py                # API on http://127.0.0.1:8000
```

In a second terminal:

```bash
cd frontend
npm ci --ignore-scripts
npm run dev                          # UI on http://127.0.0.1:5173
```

Create an account in the UI and ask, for example:

- "How does web search approval work in this project?" (your documents)
- "What is 17 percent of 850?" (calculator)
- "Latest news about open-weight LLMs" (web, asks for approval)

### Your own documents

```bash
python scripts/ingestion/ingest_local_rag_pdfs.py --pdf-dir path/to/pdfs   # PDFs
python scripts/ingestion/ingest_repo_docs.py notes/ more-notes.md          # Markdown
```

With `RAG_EMBEDDING_BACKEND=local` (the default in `.env.example`), embeddings run on your machine
for free.

### Docker

```bash
cp .env.example .env                 # set GROQ_API_KEY
docker compose up --build            # UI on http://127.0.0.1:8080
```

Data (accounts, conversations, document index) persists in the `citeright_data` volume.

## Configuration that matters

| Variable | Default | Effect |
|---|---|---|
| `GROQ_API_KEY` / `OPENAI_API_KEY` | – | At least one is required. |
| `GROUNDING_VERIFICATION_ENABLED` | `true` in `.env.example` | Check claims before release. While on, answers from evidence stream sentence by sentence, each shown once its sources support it. |
| `EVIDENCE_QUALITY_ENABLED` | `false` | Drop duplicate, stale, injected or conflicting web evidence; web answers then need 2+ independent sources. |
| `GRAPH_WEB_HITL_ENABLED` | `true` | Ask before each web search. |
| `RAG_EMBEDDING_BACKEND` | `local` | `local` (free) or `openai`. |
| `AGENT_MEMORY_ENABLED` | `false` | Opt-in long-term memory per user. |

Everything else is documented in [.env.example](.env.example).

## Testing and evaluation

```bash
python -m pytest -q                  # ~180 unit and integration tests, no API key needed
python -m ruff check .
cd frontend && npm test              # UI tests with recorded responses
```

CI also runs offline quality gates that must pass on every pull request:

| Gate | What it checks |
|---|---|
| `evals.run_offline_evals` | Router picks the right path on the routing benchmark (≥ 95%). |
| `evals.run_experiments` | Same benchmark through the experiment runner, against a keyword baseline. |
| `evals.grounding_evaluation` | 14 cases: supported, unsupported, fabricated citations, contradictions, repair. |
| `evals.evidence_quality_evaluation` | Prompt injection, duplicates, stale and conflicting evidence. |
| `evals.memory_evaluation` | Tenant isolation, memory poisoning, deletion. |

A live smoke test against Groq is opt-in:
`CITERIGHT_LIVE_SMOKE=1 python -m pytest tests/test_live_smoke.py -m integration`.

## Limits

- The grounding check is a deterministic heuristic, not an entailment model. It misses paraphrased
  opposites outside its word lists, unit conversions and implicit negation. Measured results are in
  [docs/grounding-verification.md](docs/grounding-verification.md).
- The offline gates test the routing, verification and memory logic with authored cases. They are
  not a measure of answer accuracy on real questions; that evaluation is still to be built.
- Groq's free tier rate-limits the safety model after a few quick requests.
- Web search uses DuckDuckGo results and can return nothing for narrow queries.

## Repository map

| Path | Contents |
|---|---|
| `agent/` | LangGraph agent, retrieval, grounding, safety, memory, tools |
| `service/` | FastAPI app: auth, streaming, approvals, conversation store |
| `frontend/` | React UI |
| `evals/` | Offline evaluation gates and datasets |
| `scripts/ingestion/` | Document ingestion |
| `client/`, `schema/` | Python client and shared request/response models |
| `docs/` | Architecture, grounding, evidence quality, memory, security |

The repository started as a broader agent platform (coding agent, document workbench, training
experiments). Those parts are preserved on the `archive/full-platform` branch.

## License

MIT
