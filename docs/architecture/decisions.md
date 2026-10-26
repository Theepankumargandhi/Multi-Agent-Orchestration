# Architecture decisions

## ADR-001: Explicit graph nodes over one tool-calling agent

Status: accepted.

Safety, routing, retrieval, approval, synthesis, and evaluation remain separate nodes. This makes failure policy, latency, state transitions, and tests visible. The cost is more orchestration code and schema evolution work.

## ADR-002: Checkpoint-native human approval

Status: accepted.

Web approval uses LangGraph `interrupt()` and resumes with `Command`. The service only interprets `approve`/`reject` as control input when the user/thread checkpoint contains an active interrupt. The checkpoint is authoritative; the in-process cache is only a latency optimization.

## ADR-003: Fail closed before tools

Status: accepted.

Unsafe requests and moderation-provider failures branch directly to the response node by default. No web, RAG, graph, MCP, or calculator node runs. Deployments may opt out with `SAFETY_FAIL_CLOSED=false`, but that is not recommended for public systems.

## ADR-004: Hybrid retrieval with bounded lexical work

Status: accepted for the single-host deployment.

Vector retrieval is fused with BM25 using reciprocal-rank fusion and optionally reranked. The lexical corpus is bounded to protect memory. For a larger corpus, move BM25 to OpenSearch/Postgres FTS and keep only retrieved candidate IDs in application memory.

## ADR-005: PostgreSQL first, SQLite fallback

Status: accepted.

Compose uses PostgreSQL for durable graph checkpoints and application data. SQLite keeps local development friction low. Graph instances are built with their saver rather than mutating a global compiled graph, preventing cross-request backend changes.

## ADR-006: Provider-neutral thin client

Status: accepted.

The UI calls `/capabilities` and never requires provider secrets. Model names come from backend environment configuration, which prevents UI/backend configuration drift.

## ADR-007: Adjudicate evidence before synthesis

Status: accepted as an opt-in policy.

Retrieved text is not treated as trustworthy merely because retrieval ranked it highly. Evidence-bound
routes may pass through a distinct pre-synthesis node that quarantines injection-shaped content,
collapses copied material across domains, applies source-independence and temporal-validity rules, and
constructs explicit numeric or negation conflict edges. Downstream stages consume a typed filtered list,
not the original retrieval strings, preventing rejected text from returning through citation repair,
grounding, or adaptive candidate generation.

The first implementation is deterministic, auditable, and credential-free. It intentionally does not
claim general semantic contradiction detection or universal publisher trust. Domain deployments should
version their source policy, calibrate an NLI ensemble on human-labelled pairs, track multilingual and
temporal errors, preserve publisher lineage, and route consequential unresolved conflicts to people.
