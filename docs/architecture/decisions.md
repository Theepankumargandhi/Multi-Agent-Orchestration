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

Status: accepted for the portfolio scale.

Vector retrieval is fused with BM25 using reciprocal-rank fusion and optionally reranked. The lexical corpus is bounded to protect memory. For a larger corpus, move BM25 to OpenSearch/Postgres FTS and keep only retrieved candidate IDs in application memory.

## ADR-005: PostgreSQL first, SQLite fallback

Status: accepted.

Compose uses PostgreSQL for durable graph checkpoints and application data. SQLite keeps local development friction low. Graph instances are built with their saver rather than mutating a global compiled graph, preventing cross-request backend changes.

## ADR-006: Provider-neutral thin client

Status: accepted.

The UI calls `/capabilities` and never requires provider secrets. Model names come from backend environment configuration, which prevents UI/backend configuration drift.

## ADR-007: Lease-based durable coding jobs

Status: accepted for the single-host reference deployment.

Coding requests are persisted before execution and workers claim them through an atomic compare-and-swap transition. Expiring leases plus heartbeats make abandoned work recoverable without allowing two workers to commit the same result. Idempotency keys deduplicate client retries, transient failures receive bounded exponential retries, exhausted jobs enter a visible dead-letter state, and every transition is append-only audited.

Generated diffs are not stored inline with API-visible metadata. They are written atomically to a separate artifact directory, addressed by task ID, and verified against a recorded SHA-256 digest before release. Rejection deletes the artifact and approval never applies it automatically.

SQLite WAL keeps this implementation portable and supports multiple processes on one host. It is not claimed as a multi-node queue: that deployment requires a PostgreSQL or managed-queue adapter plus encrypted object storage. The API remains separated from Docker in that topology; only isolated workers receive sandbox-runtime access.

## ADR-008: Independent review plus deterministic gates

Status: accepted.

The coding workflow separates analyst, implementation, test-author, and reviewer roles. The reviewer receives the issue, patch, plan, and test/gate evidence but does not share an implementation role. This reduces confirmation bias but does not make model judgment authoritative: deterministic failures and critical/high findings always block, even when the reviewer emits `approved=true`.

The test author is technically restricted to recognized test paths. Repair is bounded by both round count and a workflow-wide write budget. Reviewer or structured-output failure fails closed. Ruff is compared to its pre-edit baseline so existing lint debt is not presented as a new regression.

Every run produces typed verification metadata and patch-free JSON/Markdown evidence. Artifact hashes establish integrity, not correctness; human approval remains mandatory and never applies or merges code.

## ADR-009: Lexical plus dependency-graph code context

Status: superseded by ADR-010; retained as the credential-free V1 baseline.

The verified PR workflow selects model context using BM25-style lexical relevance, path/symbol matches, and graph propagation across statically extracted imports and calls. It supplies bounded line windows rather than whole files and persists a receipt with ranking components and content hashes. Repository code remains explicitly untrusted prompt data.

The index never imports repository modules. Python uses the standard AST; JavaScript/TypeScript support is deliberately conservative and pattern-based. Unchanged in-memory entries are reused by content digest, but raw source is not persisted in a shared cache without tenant isolation, encryption, and retention policy.

Retrieval is evaluated separately from patch correctness with Recall@K, MRR, and NDCG. The evaluation corpus excludes its labels, tests, documentation, and other answer-repeating files by default to avoid contamination.

## ADR-010: Multi-language hybrid code intelligence with explicit ablations

Status: accepted.

Code Intelligence V2 uses Tree-sitter for Python, TypeScript, JavaScript, Java, Go, and Rust, retaining bounded AST/pattern fallbacks. Unchanged content reuses indexed documents and vectors; edited content passes an edited previous tree into Tree-sitter for incremental parsing. Query decomposition exposes symbol, path, test, dependency, and concept facets instead of hiding retrieval intent inside one opaque embedding.

Candidate generation combines BM25, semantic similarity, and dependency propagation. A small candidate pool can be reranked by an auditable offline feature model or an explicitly enabled local cross-encoder. Syntax spans drive context compression, cross-file duplicate lines are removed, and a hard approximate token budget is enforced. Persisted receipts bind the query plan, backend identities, scores, compressed snippets, and file content by digest without persisting raw source or embeddings.

The additional complexity is admitted only with ablation evidence. CI compares lexical, lexical-plus-graph, hybrid, and hybrid-plus-reranking strategies across token budgets and publishes the JSON artifact. On the current small repository regression set, the simpler graph configuration has better NDCG than the offline semantic/reranking baseline; this result is documented instead of tuning on the five-case evaluation set. Learned-model promotion requires a larger human-labeled development set and untouched holdout.

## ADR-011: Adaptive inference depth with evidence consensus

Status: accepted as an opt-in policy.

The research graph does not run a fixed reflection loop. It uses grounding confidence, conformal
correctness sets, out-of-distribution status, evidence availability, and claim risk to choose an early
exit, bounded deliberation, or immediate abstention. Deliberation candidates receive provider-side
completion limits and a cumulative deadline; each candidate is independently grounded and optionally
conformal-filtered. Agreement is measured over supporting evidence IDs instead of surface wording.

Candidate text and hidden reasoning are not persisted in the decision receipt or streamed to clients.
The receipt records only hashes, control outcomes, budgets, confidence, and consensus. Missing required
calibration, unsupported high-risk claims, insufficient evidence, invalid signatures, disagreement, and
budget exhaustion fail closed.

This adds latency and cost on selected requests and correlated candidates can still agree on the same
error. Production use therefore requires provider/model diversity where practical, human-labelled
calibration of the stopping policy, real token/cost telemetry, distributed tenant budgets, and repeated
live-model evaluation. The deterministic checked-in ablation validates orchestration mechanics only.

## ADR-012: Adjudicate evidence before synthesis

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
