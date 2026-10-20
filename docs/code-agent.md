# Sandboxed coding agent

The local [portable verifier](portable-patch-verification.md) now accepts external
Python patches or selected mini-SWE-agent batch predictions without invoking a
repair team. Fresh repeated Docker runs distinguish owner-preservation violations
from missing evidence. The [regression benchmark](patch-regression-benchmark.md)
compares matched verification strategies, while [consented migration pilots](maintainer-migration-pilots.md)
record human-reported outcomes separately. These entry points do not broaden
service API permissions or claim independently reviewed real-task adoption.

The optional local [stateful migration verifier](stateful-migration-verification.md)
adds ordered session evidence to UpgradePilot repair tasks. Only discovery
witnesses reach the team; withheld scenarios are evaluated after repair. It does
not widen the service API, authorize inference, add generic tool permissions or
write to the original repository.

[Generated workflow laws](metamorphic-workflow-verification.md) add calibrated
retry/read/rollback/order checks to that local migration path. The source must
satisfy declared laws before repair; final target variants remain terminal.
This is deterministic verification for the existing team, not additional agent
autonomy, model training or a new service API permission.

The separate local [migration-study runner](crash-safe-migration-studies.md) now
reuses the benchmark checkpoint for arm-boundary recovery and uncertain-request
reservations. It does not resume a live agent mid-tool call or change the coding
service's durable worker/lease protocol.

The coding agent turns a repository-scoped issue into a tested unified diff. It never edits the source repository. It copies an allowlisted, size-bounded view into a temporary workspace, plans with structured actions, executes only a fixed owner-supplied test command inside a hardened Docker container, and withholds the patch until the submitting user explicitly approves it.

The [October 4 implementation audit](live-upgrade-audit-2026-10-04.md) records
real Docker execution, a bounded live Groq smoke candidate, and 17 authenticated
HTTP checks. It also records the failures: corrupt missing-newline patch framing
(now repaired), unreliable provider schema responses, the unexecuted test-author
role, and the held real-issue pilot. The stock Groq provider selector was not
repaired by the audit's injected client. Do not treat offline coverage as proof
that every specialist or the normal credentialed API-to-worker path passed live.

## October 14 reliability integration

The gaps above now have implemented controls and provider-free regression checks;
this does not turn the October 4 live failures into successful live reruns.
The normal API/worker policy enables targeted edits, required specialist completion,
and fresh delivery verification. Programmatic `SandboxPolicy` defaults remain
compatible with older authored controls; enable the flags explicitly in experiments.

`read_lines` supplies a bounded source window and the file's byte hash. An `edit`
requires that observed hash and exactly one matching block, then publishes the
change atomically. Stale, ambiguous, overlapping, no-op and protected-path edits
fail closed. The agent cannot rewrite an existing file in targeted-only mode.
Explicit `groq:<model>` and `openai:<model>` routing validates original function-call
arguments without SDK schema retries or guessed required fields.

Final delivery copies the original repository again, applies the exact emitted
patch, compares candidate source/configuration hashes, and runs owner tests in a
fresh Docker container. Unsupported diff formats or missing execution evidence
hold completion. The report records completed specialist stages and binds the
delivery receipt to the patch. Generated repository code never executes on the host.

The scripted four-role Docker check passed with all specialist stages completed,
fresh patch application and owner tests passing, and original/candidate integrity
preserved. It made seven scripted model invocations and zero provider requests.
It is a workflow regression control, not a real coding-task benchmark.

```bash
python -m evals.targeted_delivery_evaluation --output data/evaluations/targeted-delivery/my-check.json
python -m code_agent.benchmark_release validate evals/experiments/targeted_delivery_matrix.json
```

The three-arm matrix is a frozen authored-smoke comparison definition, not an
executed live comparison. See [AI failure investigation](ai-failure-investigation.md)
for the new private replay workflow and the remaining evidence needed.

## Execution lifecycle

[UpgradePilot](upgradepilot.md) reuses this team for a pinned Pydantic migration
profile, adding a real codemod baseline and repeated old/new JSON contracts.
Controller-managed dependency paths are denied to model edit tools but may be
applied by the trusted fresh-diff replay controller. Ordinary protected files
remain immutable even when listed as controller-managed. The default managed-path
list is empty, so existing coding policies retain their behavior.

The opt-in [patch-targeted mutation gate](patch-targeted-mutation-testing.md) runs
after existing quality gates and before independent review. It holds changed
assertion tests constant across repeated clean/mutant Docker copies and never edits
the candidate. Report mode is advisory; enforce mode holds insufficient sensitivity
or unavailable evidence. Source drift blocks both modes. Defaults remain off.

1. The authenticated API validates the repository as a relative path below `CODE_AGENT_REPOSITORY_ROOT` and transactionally enqueues the request. A per-user `Idempotency-Key` safely deduplicates client retries.
2. A worker atomically claims one available job with an expiring lease. Its heartbeat extends ownership; a later worker recovers an expired lease after a crash or restart.
3. Files are copied to a unique temporary directory. Git history, virtual environments, data directories, symlinks, `.env` files, keys, and certificates are excluded.
4. A non-executing Tree-sitter pass builds a hybrid code index across six languages. Decomposed query facets drive lexical, semantic, graph, and reranking signals; a typed analyst receives only the compressed context pack.
5. The implementer runs a bounded structured-action loop. It can list, read bounded lines, search, make hash-bound exact edits, create files, delete, test, or finish, but it cannot construct shell commands. Existing-file rewrites are denied when targeted-only mode is enabled.
6. A separate test-author role inspects the changed workspace and may write only recognized test paths. Attempts to change production files are rejected by policy.
7. Deterministic gates rerun the owner-approved tests and check patch scope, credential-like additions, unsafe execution/deserialization/TLS patterns, source-repository integrity, and baseline-aware Ruff results.
8. An independent structured reviewer sees the issue, analyst plan, patch, changed files, and gate evidence. Critical/high findings override an inconsistent model approval.
9. Failed gates and blocking findings feed a bounded repair loop. Tests and independent review run again after every repair; the total mutation budget applies across specialist stages.
10. Completion requires a non-empty patch, passing blocking gates, and reviewer approval. A reviewer outage fails closed.
11. The container and temporary workspace are destroyed. Metadata is committed to SQLite while verified diffs and JSON/Markdown evidence dossiers are written atomically to separate artifact storage with byte counts and SHA-256 digests.
12. Job metadata never includes the diff. The owner reviews the dossier before approval. Rejection erases all task artifacts; approval releases but never applies the verified patch.

Every transition is appended to `code_job_events`. Transient worker failures use bounded exponential backoff and move to `dead_letter` after `CODE_AGENT_MAX_ATTEMPTS`. Terminal agent outcomes are not retried. Queue state survives API and worker restarts.

## Hybrid code-context selection

An optional [multi-agent repair tournament](multi-agent-repair-tournament.md) runs several
verified specialist teams from one frozen source snapshot, challenges eligible patches, and
reruns execution gates before selecting a winner. It integrates with the same queue,
worker, approval API, and evidence dossiers. `verified_pr` remains the default; the tournament
requires explicit operator selection and can multiply sandbox and inference resource use.

Before analyst and implementation calls, the worker creates a non-executing code index from the filtered workspace. Tree-sitter extracts symbols, imports, calls, and syntax spans for Python, TypeScript, JavaScript, Java, Go, and Rust. Query decomposition separates symbols, paths, test intent, dependency intent, and concepts. BM25, embedding similarity, and dependency propagation generate candidates, which an auditable feature model or optional cross-encoder reranks. Syntax-aware windows are compressed, deduplicated, and limited by an approximate token budget.

The verification result and PR dossier retain a context receipt—not raw source snippets or embeddings—with the query plan, backend identities, selected paths, rank and score components, reasons, file/snippet hashes, compression statistics, candidate counts, and a fingerprint. See [code-intelligence.md](code-intelligence.md) for incremental parse-tree updates, learned adapters, and the strategy/token-budget ablation dashboard.

## Container controls

[Cycle-aware stage control](agent-cycle-control.md) supports `off`, observational
`shadow`, and opt-in `enforce` across all coding workflows. It hashes filtered
source and tool outcomes to detect short repeated cycles; a stage stop is not
verification or approval of a patch. Existing gates and budgets remain unchanged.

[Execution-guided patch minimization](execution-guided-patch-minimization.md) is an
optional pre-review experiment on extra Python production-file changes. It keeps
tests intact, binds repeated outcomes to candidate snapshots, charges deterministic
restorations/rollback against workflow writes, and reruns normal verification before
independent review. It never applies a patch to the original repository.

Optional [execution-spectrum localization](spectrum-fault-localization.md) compares
passing/failing Python test coverage in repeated fresh copies and supplies advisory
production-line context to coding stages. It is off by default, invalidates after
edits, and does not change the owner command or approval gates.

The worker creates the sandbox with networking disabled, a read-only root filesystem, a bounded writable `/tmp`, CPU/memory/PID limits, all Linux capabilities dropped, `no-new-privileges`, and numeric UID/GID `65532`. Only the temporary repository copy is bind-mounted. Provider credentials are used by the host-side planner and are never injected into the container.

Build the minimal default image:

```bash
docker build -f docker/Dockerfile.code-sandbox -t agentforge-code-sandbox:local .
```

The included image supports Python, pytest, Ruff, and Coverage.py for optional spectrum instrumentation. For another stack, create a prebuilt image containing all dependencies, set `CODE_AGENT_IMAGE` to its exact trusted tag or digest, and keep runtime networking disabled. Do not allow request payloads to choose images.

## Local API workflow

Place a repository below `repositories/`, build the sandbox image, and set:

```dotenv
CODE_AGENT_ENABLED=true
CODE_AGENT_REPOSITORY_ROOT=repositories
CODE_AGENT_IMAGE=agentforge-code-sandbox:local
CODE_AGENT_NETWORK_ENABLED=false
CODE_AGENT_EXECUTION_MODE=embedded
CODE_AGENT_WORKFLOW=verified_pr
CODE_AGENT_DB_PATH=data/code-agent/jobs.db
CODE_AGENT_ARTIFACT_ROOT=data/code-agent/artifacts
CODE_AGENT_MAX_ARTIFACT_BYTES=4000000
CODE_AGENT_MAX_ATTEMPTS=3
CODE_AGENT_LEASE_SECONDS=90
CODE_AGENT_MAX_CHANGED_FILES=20
CODE_AGENT_MAX_REPAIR_ROUNDS=2
CODE_AGENT_MAX_WORKFLOW_WRITES=16
CODE_AGENT_MIN_CHANGED_LINE_COVERAGE=0.8
# Optional operator-maintained pricing used only for cost estimates:
CODE_AGENT_INPUT_COST_PER_MILLION=0
CODE_AGENT_OUTPUT_COST_PER_MILLION=0
```

Start the API directly on a machine with access to Docker. After obtaining a bearer token from `/auth/login`, submit a task:

```bash
curl -X POST http://localhost:8000/code/tasks \
  -H "Authorization: Bearer $TOKEN" \
  -H "Idempotency-Key: parser-fix-2026-001" \
  -H "Content-Type: application/json" \
  -d '{"repository":"sample","issue":"Fix the failing parser boundary test.","test_command":["python","-m","pytest","-q"]}'
```

Poll `GET /code/tasks/{task_id}`. Its public result includes the typed analysis, every review round, quality gates, blocking reasons, specialist-tagged observations, token counts, operator-priced cost estimate, and changed files—but never the patch body. Fetch the review artifact as JSON or Markdown:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/code/tasks/$TASK_ID/dossier?format=markdown"
```

If its status is `awaiting_approval`, review the dossier and `GET /code/tasks/{task_id}/events`, then call `POST /code/tasks/{task_id}/decision` with `{"approve":true}`. Only then will `GET /code/tasks/{task_id}/patch` return the integrity-verified diff. When the approved test command enables Python coverage (`coverage` or `--cov`), the gate extracts new-line locations from the unified diff and enforces `CODE_AGENT_MIN_CHANGED_LINE_COVERAGE`; otherwise it records an explicit skip rather than inventing a coverage result. Blocked jobs retain a patch-free dossier for diagnosis. Rejection erases both patch and dossier. `GET /code/queue` returns only the authenticated user's status counts; Prometheus exports global job-state, verification-outcome, artifact-byte, and lease-recovery metrics.

For control-plane separation, set `CODE_AGENT_EXECUTION_MODE=external` on the API and run this command on an isolated host that can access the same database/artifact volume and the sandbox runtime:

```bash
agentforge-code-worker
# Operational probes and one-shot schedulers can use:
agentforge-code-worker --once
```

SQLite WAL supports multiple worker processes sharing a local filesystem and is intentionally easy to demonstrate. A multi-node deployment should replace the store with a PostgreSQL/managed-queue adapter and object storage while retaining the same claim, lease, approval, and artifact contracts. Do not place the SQLite database on an unreliable network filesystem.

The supplied Compose and Kubernetes configurations leave this capability disabled and do not mount the Docker socket. For a public deployment, run the code agent as a dedicated worker on an isolated sandbox host or microVM runtime. Giving the internet-facing API access to the host Docker daemon would turn a container escape or control-plane flaw into a host-level incident.

## Evaluation

The deterministic outcome grader requires a non-empty patch, changed files, a passing non-timeout final test, completed status, and confirmation that the source repository was untouched. Validate the five-case credential-free smoke dataset with:

```bash
python -m code_agent.evaluation evals/datasets/code_agent_smoke.jsonl --validate-only
```

Live runs record pass@1 with confidence intervals, final-test and regression rates, sandbox violations, latency, iterations, tool/model calls, changed files, provider token usage, estimated cost, failure categories, and slices. They produce separate integrity-bound patches, full replay trajectories, a manifest, and SWE-bench-compatible predictions. See [coding-agent-evaluation.md](coding-agent-evaluation.md) for dataset import, multi-model comparison, repository-specific images, artifact contracts, and honest reporting rules.

## Independent behavioral regression probes

The optional tournament supports a [pre-patch regression-design agent](regression-challenge-agent.md). Its output is JSON calls to operator-allowlisted Python functions, not executable model-generated tests. A frozen suite runs against the baseline and candidates in Docker; repeated outcomes, workspace integrity, fresh owner tests, and human approval remain required. Exact expectations are retained for review because incorrect model oracles can cause false holds. The default workflow does not change.

The optional [reference and mutation gate](oracle-calibration.md) additionally checks generated expectations against a pinned, separate operator implementation and measures which controlled code faults the suite detects. It produces a kill matrix and redundancy diagnostics without exposing reference code to repair models. Source revocation, invalid executions, and insufficient sensitivity hold the tournament.

## Deliberate limitations

The optional local [schema-guided migration path](schema-guided-migration.md)
uses a separately attested image probe and owner-allowlisted model/contract
mappings. It does not enable schema introspection for ordinary coding jobs or
widen their tool permissions. Actual behavioral gates and owner approval remain
required, even when normalized schemas match.

- The reference queue is durable and multi-process on one host, not a multi-region broker. Use PostgreSQL or a managed queue plus encrypted object storage for a multi-node production deployment.
- Tree-sitter extraction covers six languages; other file types remain lexical-only and each language still needs repository-specific build/test images and command policy.
- Approval releases a diff; it does not prove semantic correctness and it does not merge or deploy code.
- A repository's tests are untrusted code. Container isolation reduces risk but is not equivalent to a strong VM or microVM boundary.
- The agent does not install dependencies at runtime and has no network. Required tools must be built into the allowlisted image.
- The deterministic unsafe-code scanner is intentionally conservative. A legitimate blocked pattern must be redesigned or handled through an explicit future policy-exception mechanism; the model cannot override it.
