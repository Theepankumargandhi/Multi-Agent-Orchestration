# Security threat model

## Assets and trust boundaries

Assets include model-provider keys, user credentials/tokens, conversation data, checkpoints, ingested documents, source repositories, generated patches, tool outputs, and trace data. Trust boundaries exist at the public HTTP API, retrieved web/PDF content, MCP subprocess transport, persistence backends, model providers, temporary coding workspaces, and the Docker sandbox runtime.

## Principal threats and controls

| Threat | Control in this repository | Residual risk / production next step |
|---|---|---|
| Prompt injection from retrieved content | Retrieval is labeled untrusted; route-scoped state prevents stale reuse; the optional evidence gate quarantines injection patterns and passes only adjudicated objects to synthesis, citation completion, grounding, and deliberation | Calibrate multilingual content classifiers and adaptive injection attacks on representative retrieved corpora |
| Citation laundering, stale sources, or contradictory retrieved claims | Cross-domain near-duplicate collapse, independent-source minimums, requested-window freshness checks, numeric/negation conflict graphs, filtered context, and integrity-bound evidence reports | Add domain-calibrated NLI, publisher lineage, signed provenance, multilingual temporal extraction, and human escalation for material conflicts |
| Prompt injection from repository content | Code is never executed during indexing; Tree-sitter visits are bounded; compressed snippets are token-bounded and labeled as untrusted evidence; specialist roles cannot approve their own output | Add language-aware content classifiers and isolate the reviewer model behind a separate policy service |
| Agent goal hijacking across repository, retrieval, tool, memory, or peer-agent content | Provenance artifacts carry hashes and taints; deterministic complete mediation blocks high-confidence unsafe actions and routes ambiguous injected-context mutations through owner approval; versioned red-team cases gate attack success and benign utility in CI | Add a separately deployed policy decision point and continuous human-authored adversarial holdouts |
| Code-context cache disclosure | Incremental trees, source, and embeddings remain process-local; persisted receipts contain paths, scores, backend IDs, and hashes but not raw snippets or vectors | Encrypt any future shared index and enforce tenant-scoped cache keys and retention |
| Retrieval-model supply-chain compromise | Learned embeddings/cross-encoders are optional, disabled by default, and identified in receipts; deterministic local fallbacks keep safety gates operational | Pin model revisions, verify model artifacts, use offline model stores, and scan serialization formats before production enablement |
| Unsafe tool execution | Safety node precedes routing and fails closed; unsafe/error state has a direct response edge | Use a dedicated moderation service with SLOs and policy versioning |
| Forged approval | Native checkpoint interrupt plus user-specific namespace/thread validation | Put API behind TLS; add role-based approval for teams |
| Cross-user data access | Authenticated user ID is required in conversation, checkpoint, and memory queries; API tests attempt cross-tenant list, search, export, and deletion | Replace the SQLite reference backend with database row-level security and defense-in-depth authorization policy |
| Long-term memory poisoning or stale facts | Explicit-consent extraction, confidence/TTL gates, PII redaction, instruction/secret quarantine, untrusted prompt boundaries, conflict lineage, correction/deletion, and a deterministic governance suite | Add human-authored adaptive attacks, calibrated classifiers, encrypted storage, retention enforcement, and independent policy review |
| Hallucinated claims or fabricated citations | Optional pre-release claim/evidence alignment, retrieved-URL allowlisting, stricter high-risk thresholds, draft withholding, bounded repair/abstention, integrity receipts, and a CI regression corpus | Calibrate NLI/judge ensembles on independently labelled domain claims and continuously evaluate adversarial paraphrases and fresh sources |
| Overconfident verifier release or calibration tampering | Route-aware split-conformal correctness sets, singleton-only release, fail-closed artifact loading, HMAC/SHA integrity, selective-risk evaluation, and confidence-distribution drift alerts | Use representative time/user/document-separated labels, protected calibration storage, minimum slice sizes, scheduled recalibration, and operator-reviewed risk targets |
| Runaway deliberation, hidden-draft leakage, or confident candidate collusion | Adaptive inference has hard candidate/provider-token/latency ceilings; drafts never stream or enter receipts; every candidate is independently grounded and, when enabled, conformal-filtered; evidence disagreement and receipt tampering fail closed | Enforce distributed tenant cost quotas, diversify providers for failure independence, and calibrate consensus on adversarial human-labelled live traces |
| Resource exhaustion | Pydantic length bounds, HTTP body limit, bounded caches/corpora, timeouts, process-local rate limit | Use gateway/Redis distributed limits and per-user cost budgets |
| Arbitrary calculator execution | `numexpr` receives empty globals and a small local constant set | Add expression AST allowlisting and complexity limits if expanded |
| Secret leakage | Backend-only credentials, `.env` excluded from Docker context/git, generic client errors | Use a managed secret store and automated secret scanning |
| Container compromise | Non-root UID, dropped Kubernetes capabilities, no privilege escalation, network policy | Use read-only root filesystems with dedicated writable volumes and signed images |
| Supply-chain vulnerability | Bounded dependency ranges, clean-install verification, CI `pip-audit` | Add lockfiles, SBOM generation, image scanning, and Dependabot/Renovate |
| Malicious repository or tests | Secret-filtered ephemeral copy, no symlinks, network disabled, non-root UID, read-only root, dropped capabilities, no-new-privileges, resource/time limits | Run public workloads in dedicated microVM workers; containers share the host kernel |
| Coding-agent path traversal | Repository roots and file actions are resolved and checked below fixed roots; absolute and parent paths are rejected | Add OS-level per-tenant storage isolation for multi-tenant operation |
| Model-generated shell injection | Model emits a validated typed action; tests use a fixed owner-supplied argv and an executable allowlist; no shell is invoked | Maintain per-language argument policies and fuzz the command validator |
| Model-generated exfiltration or unsafe code | Pre-tool rules reject credential literals, network-plus-sensitive-source flows, dynamic execution, unsafe deserialization, TLS bypass, test tampering, and common resource attacks; post-patch gates independently rescan | Replace pattern rules with AST/data-flow policies and Semgrep/CodeQL in isolated workers |
| Reviewer prompt injection or self-approval | Reviewer is a separate structured role; patch and evidence are explicitly untrusted; deterministic gates and critical/high findings override approval; reviewer failure blocks | Use a separately deployed reviewer model and policy service for stronger failure independence |
| Test-author scope expansion | Test-author writes are technically restricted to recognized test paths; workflow-wide write and repair budgets cap mutations | Add language-aware ownership policies and CODEOWNERS enforcement |
| Patch contains credentials or unsafe primitives | Added lines are scanned for credential-like values, shell/dynamic execution, unsafe deserialization, disabled TLS verification, and unsafe YAML loading | Replace regex screening with Semgrep/CodeQL plus a reviewed exception workflow |
| Unauthorized patch disclosure | Job lookup is user-scoped; public job records redact diffs; explicit owner approval gates patch retrieval | Persist encrypted artifacts with RBAC, expiry, and immutable approval audit logs |
| Evidence or patch artifact tampering | Patch and JSON/Markdown dossiers are stored separately, byte-counted, and SHA-256 verified on every read; blocked jobs never persist an accessible patch | Sign attestations with KMS-backed keys and store them in immutable object storage |
| Docker control-plane compromise | Compose/Kubernetes keep the feature disabled and never mount the Docker socket | Use a separate queue worker on an isolated host; never expose Docker access to the public API container |
| Evaluation artifact disclosure | Reports omit patch bodies; patches and trajectories are separated, content-hashed, gitignored, and mounted read-only in the optional dashboard | Encrypt artifacts, repeat secret scanning/redaction, add tenant RBAC and retention/deletion jobs |
| Post-training data or model poisoning | Reviewer provenance, protected-set contamination checks, `trust_remote_code=false`, dataset/run fingerprints, artifact re-verification before canary | Pin immutable base-model revisions, scan weights, isolate GPU workers, sign attestations, and require two-person review for high-risk labels |

Time-bounded vulnerability waivers and their deployment constraints are tracked in [audit-exceptions.md](audit-exceptions.md).

## Data retention

Conversation and audit data persist until removed by the operator. Long-term memory has authenticated
per-record and forget-all APIs, but production deployments must also propagate erasure to backups and
replicas, enforce retention jobs, rotate signing keys, and complete a privacy review before handling
regulated or customer data.

## Incident behavior

Readiness fails when the conversation store is unavailable. Agent exceptions return generic client messages while structured server logs retain a request/run identifier. Moderation failures block the run by default.
