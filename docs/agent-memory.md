# Trustworthy long-term agent memory

AgentForge includes an opt-in memory subsystem that turns explicit user requests into typed,
tenant-isolated records. It is disabled by default, and the graph behaves as before when it is off.

## Architecture

The LangGraph runtime reads memory immediately after the safety gate and writes it only after the
evaluation node. The read node receives the authenticated `user_id`, performs hybrid temporal
retrieval, and injects a token-bounded block marked as untrusted informational data. The write node
uses a conservative structured extractor and persists only explicit phrases such as `remember that`
or `I prefer`; it does not store whole conversations.

Four record types are supported:

- `episodic`: a specific interaction or event;
- `semantic`: a durable fact;
- `preference`: a user preference; and
- `procedural`: a reusable workflow.

The SQLite reference backend stores confidence, importance, trust, provenance, timestamps, TTL,
usefulness, version lineage, and an HMAC/SHA-256 record fingerprint. Every query includes the tenant
identifier in the storage predicate. Conflicting active memories are superseded instead of silently
overwritten, exact repeats are idempotent, and expired or tombstoned records cannot be retrieved.

## Retrieval and learning

Retrieval combines a dependency-free hashed semantic vector, lexical overlap, exponential recency,
confidence, trust, importance, and outcome-derived usefulness. A retrieval receipt records component
scores, considered/rejected counts, token usage, policy version, and integrity fingerprint without
storing the raw query. The built-in vector is an offline baseline, not a claim of learned-embedding
quality.

Downstream code can report whether a selected memory was helpful. The rolling usefulness score then
participates in later ranking. Episodic memories with at least three helpful uses can be consolidated
into semantic memories; the original is retained as a signed superseded record for auditability.

## Safety and privacy controls

- Prompt-injection and secret-like patterns are quarantined and excluded from retrieval.
- Email addresses and phone numbers are redacted before persistence.
- Retrieved text is explicitly labelled untrusted and must never be interpreted as instructions.
- Low-confidence writes are rejected; TTLs bound temporary records.
- Authenticated users can inspect, correct, export, delete, or forget all of their records.
- Deletion replaces content with a tombstone, reseals the record, and emits an audit event.
- Integrity fingerprints detect database or report tampering.

SQLite files are not application-layer encrypted. Production deployments should use an encrypted
managed database, row-level tenant policies, key rotation, retention jobs, backups with deletion
propagation, and a privacy review appropriate to the data classification.

## Enable and use

Set a dedicated integrity key of at least 16 bytes:

```dotenv
AGENT_MEMORY_ENABLED=true
AGENT_MEMORY_DB_PATH=data/memory/agent_memory.db
AGENT_MEMORY_INTEGRITY_KEY=replace-with-at-least-16-random-characters
AGENT_MEMORY_TOKEN_BUDGET=384
```

Authenticated lifecycle endpoints are:

- `POST /memories`
- `GET /memories` and `GET /memories/search?q=...`
- `PATCH /memories/{memory_id}`
- `POST /memories/{memory_id}/outcome`
- `POST /memories/consolidate`
- `GET /memories/audit`
- `GET /memories/export`
- `DELETE /memories/{memory_id}` and `DELETE /memories`

The API returns `404` while the feature is disabled. OpenAPI at `/docs` contains the request schemas.

## Evaluation gate

Run the deterministic 12-scenario governance suite:

```bash
python -m evals.memory_evaluation \
  --output data/evaluations/memory/latest.json \
  --min-pass-rate 1 \
  --max-cross-tenant-leakage 0 \
  --max-poisoning-asr 0 \
  --max-deletion-violations 0
```

It covers relevant recall, irrelevant rejection, tenant isolation, expiry, preference updates,
conflict lineage, poisoning quarantine, PII redaction, deletion, token budgets, explicit consent, and
artifact tamper detection. CI publishes the integrity-bound report, and the AgentOps command center
shows the same evidence under **Memory governance**.

These deterministic cases validate control flow, not broad conversational quality. Before publishing
production metrics, build a consented, human-reviewed holdout; measure Recall@K, stale-memory rate,
false-write rate, poisoning attack success, cross-tenant leakage, deletion propagation, latency, and
cost; and calibrate any LLM extractor or learned embedding model against that holdout.
