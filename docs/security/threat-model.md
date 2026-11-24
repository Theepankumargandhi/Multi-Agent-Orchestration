# Security threat model

## Assets and trust boundaries

Assets are the model-provider keys, user accounts and sign-in tokens, conversations and checkpoints,
uploaded and ingested documents, and long-term memory when it is enabled. Trust boundaries sit at the
public HTTP API, retrieved web pages and documents, the model provider, the optional MCP subprocess,
and the SQLite or PostgreSQL stores.

## Principal threats and controls

| Threat | Control in this repository | Residual risk / next step |
|---|---|---|
| Prompt injection from retrieved content | Retrieved text is labelled untrusted; route-scoped state prevents stale reuse; the optional evidence gate quarantines injection-shaped content and passes only adjudicated evidence to synthesis and grounding | Pattern-based; calibrate a classifier on adaptive and multilingual attacks |
| Hallucinated claims or fabricated citations | Each sentence is released only after its claims are checked against retrieved evidence; cited URLs must come from the retrieved set; repair or abstention when claims are unsupported or contradicted | Deterministic heuristic, not entailment; paraphrased contradictions can pass (see [grounding-verification.md](../grounding-verification.md)) |
| Citation laundering, stale or conflicting sources | Optional evidence gate: near-duplicate collapse, independent-source minimum, recency checks, numeric and negation conflict detection | Add domain-calibrated NLI and publisher lineage |
| Unsafe requests | The safety node runs before routing and fails closed: an unsafe verdict or a moderation error goes straight to the response node, so no retrieval or tool runs | Moderation model false positives and negatives; measured in the answer evaluation |
| Forged web-search approval | Approval is a native LangGraph interrupt; the service treats approve/reject as control input only when that user's thread has an active interrupt | Add role-based approval for teams |
| Cross-user data access | Conversations, checkpoints, memory and uploads are keyed by the authenticated user; uploaded chunks carry an owner tag and searches read only shared docs plus the asker's files; tests attempt cross-user reads and deletes | SQLite has no row-level security; use database policies for multi-tenant production |
| Malicious uploads | Only PDF, Markdown and text are accepted; only extracted text is stored; per-route body limits (10 MB uploads, 64 KB elsewhere); chunk cap per file; file names are sanitised | PDF parsing runs in-process; isolate the parser for hostile files |
| Resource exhaustion and quota abuse | Pydantic length bounds, body limits, bounded caches and BM25 corpus, timeouts, process-local rate limit; `DEMO_MODE` caps questions per account per hour and per server per day | Limits are per process and reset on restart; use a gateway or Redis for distributed limits |
| Long-term memory poisoning | Opt-in; explicit-consent extraction, confidence/TTL gates, PII redaction, instruction and secret quarantine, untrusted prompt boundaries, deletion | Add human-authored adaptive attacks and encrypted storage |
| Arbitrary calculator execution | `numexpr` receives empty globals and a small constant set | Add an expression allowlist if the calculator grows |
| Secret leakage | Keys stay on the backend; `.env` is excluded from Git and the Docker context; clients get generic errors; uploaded text is kept out of logs | Use a managed secret store and automated secret scanning |
| Weak sign-in secrets | Startup refuses a published placeholder secret, and with `APP_ENV=production` requires a configured secret of 32+ characters; passwords are hashed with Argon2 | Add account lockout and email verification for a real deployment |
| Container compromise | Images run as a non-root user; Compose sets `no-new-privileges`, and the web container is read-only with all capabilities dropped | Read-only API root filesystem with dedicated writable volumes; signed images |
| Supply-chain vulnerability | Bounded dependency ranges, clean installs in CI, `pip-audit` on every pull request | Add lockfiles, SBOMs and image scanning |

Time-bounded vulnerability waivers and their deployment constraints are tracked in [audit-exceptions.md](audit-exceptions.md).

## Data retention

Conversations, approval records and uploads persist until the user or operator removes them; the
public demo image keeps nothing across restarts. Long-term memory has per-record and forget-all
endpoints. A production deployment must also propagate deletion to backups, enforce retention and
rotate signing keys.

## Incident behaviour

Readiness fails when the conversation store is unavailable. Agent exceptions return generic client
messages, while structured server logs keep a request ID. Moderation failures block the run.
