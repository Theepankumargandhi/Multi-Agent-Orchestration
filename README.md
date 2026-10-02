# AgentForge

**A reliability-first platform for building, evaluating, and operating AI agents.**

[![CI](https://github.com/Theepankumargandhi/Multi-Agent-Orchestration/actions/workflows/ci.yml/badge.svg)](https://github.com/Theepankumargandhi/Multi-Agent-Orchestration/actions/workflows/ci.yml)
[![CD](https://github.com/Theepankumargandhi/Multi-Agent-Orchestration/actions/workflows/cd-release.yml/badge.svg)](https://github.com/Theepankumargandhi/Multi-Agent-Orchestration/actions/workflows/cd-release.yml)

AgentForge began as a LangGraph research assistant. It has grown into a practical AI-engineering platform that answers a harder question: *how do you let an agent use retrieval, memory, tools, and additional inference without silently releasing an unsafe or unsupported answer?*

The repository contains two working agent systems:

- an evidence-aware research agent with hybrid RAG, durable human approval, memory, grounding, calibrated uncertainty, and adaptive test-time compute;
- a sandboxed coding agent that retrieves repository context, proposes a patch, tests it in an isolated container, and produces a reviewable evidence dossier without modifying the source repository.

Around those agents is the part that matters in production: an inference gateway, evaluation suites, adversarial testing, trace-based observability, release gates, persistence, and deployable FastAPI, Docker, and Kubernetes infrastructure.

> This is not presented as a production factuality guarantee or an official benchmark result. The checked-in evaluations are reproducible engineering evidence; the [limitations](#what-the-results-do-not-claim) explain where human-labelled and live-model validation is still required.

## Why this project exists

Most agent demos stop after the model produces a plausible answer. AgentForge treats generation as the middle of the workflow, not the end.

A request may need to be rejected by a safety policy, clarified, routed to a tool, approved by a person, grounded in several sources, repaired, or withheld when confidence is too low. Each of those decisions must be observable and testable. AgentForge makes those decisions explicit in the graph and attaches evaluation evidence to them.

That makes the project useful for demonstrating the work expected from an AI engineer:

- designing agent workflows rather than a single prompt;
- building and measuring retrieval systems;
- controlling hallucination, uncertainty, cost, and tool risk;
- evaluating complete trajectories, not only final text;
- operating model calls behind stable service boundaries;
- shipping with tests, telemetry, persistence, and deployment controls.

## System at a glance

```mermaid
flowchart TD
    User[User or API client] --> API[FastAPI service]
    API --> Auth[Authentication, limits, and tenant context]
    Auth --> Safety{Safety gate}

    Safety -->|blocked or unavailable| Stop[Safe response]
    Safety -->|allowed| Memory[Retrieve consented memory]
    Memory --> Router{Structured intent router}

    Router -->|needs context| Clarify[Ask for clarification]
    Router -->|math| Math[Math tool]
    Router -->|local knowledge| RAG[Hybrid RAG]
    Router -->|relationships| Graph[Knowledge graph retrieval]
    Router -->|current information| Approval{{Human approval}}
    Router -->|general| Draft[Draft answer]

    Approval -->|approved| Web[Web retrieval]
    Approval -->|rejected| Stop
    Graph --> RAG
    Math --> Evidence
    RAG --> Evidence[Evidence adjudication]
    Web --> Evidence
    Evidence -->|injection, stale, duplicated, or conflicting| Abstain[Abstain or degrade]
    Evidence -->|acceptable| Draft

    Draft --> Ground[Claim and citation verification]
    Ground -->|unsupported| Repair[Bounded repair]
    Ground -->|supported| Uncertainty{Conformal uncertainty gate}
    Repair --> Final
    Uncertainty -->|confident| Final[Release answer]
    Uncertainty -->|recoverable uncertainty| Compute[Adaptive compute controller]
    Compute --> Search[Verifier-guided bounded MCTS]
    OfflineRL[Conservative offline-RL action prior] --> Search
    Student[Distilled search policy alternative] --> Search
    Search -->|simulate typed action| World[Learned transition world model]
    World -->|next-state LCB, success, or OOD| Search
    Search --> PRM[Calibrated process-reward ensemble]
    PRM -->|verified plan| Candidates[Generate and verify real answer candidates]
    Candidates --> Consensus[Bounded consensus selection]
    Consensus --> Preferences[Optional reviewed preference reranking]
    Preferences -->|supported margin or baseline fallback| Final
    Preferences -->|consented candidate metadata| Replay[(Private execution replay)]
    Replay --> Outcome[Delayed correctness and safety review]
    Outcome --> Calibrate[Request-group held-out calibration gate]
    Outcome --> PreferenceTrain[Family-grouped Bradley-Terry training]
    PreferenceTrain --> PreferenceGate[Future holdout risk and utility gate]
    PreferenceGate --> PreferenceCandidate[Owner-reviewed tenant-bound ranker]
    PreferenceCandidate --> Preferences
    Consensus -->|non-serving same-pool comparison| Shadow[Preference shadow selector]
    Shadow --> ShadowPairs[Pre-label paired decision snapshots]
    ShadowPairs --> ShadowReview[Disagreement and audit review queue]
    ShadowReview --> ShadowGate[Fresh-family paired risk and utility gate]
    ShadowGate --> ApprovalLease[Owner-reviewed model and policy approval lease]
    ApprovalLease --> Preferences
    ApprovalLease --> Registry[Optional owner-activated tenant deployment registry]
    Registry --> LiveGuard[Live lineage and delayed-outcome sentinel]
    LiveGuard -->|healthy and atomic serving capture| Preferences
    Outcome --> LiveGuard
    LiveGuard -->|unsafe, degraded, expired, or source revoked| Baseline[Revoke learned selector and preserve baseline]
    Outcome -->|optional stronger evaluation| Cohort[Frozen forward-time task-family cohort]
    Cohort --> Prospective[Drift and paired incumbent checks]
    HoldoutLedger[(One-use holdout ledger)] --> Prospective
    HoldoutLedger --> PreferenceGate
    HoldoutLedger --> ShadowGate
    Candidates -->|consented pre-review workflow features| StepSnapshots[Frozen workflow snapshots]
    StepSnapshots --> StepReviews[Independent per-step correctness reviews]
    StepReviews --> StepTrain[Explicit-label training and validation-only calibration]
    StepTrain --> StepGate[Future-family step-quality and missingness gate]
    HoldoutLedger --> StepGate
    StepGate --> StepCandidate[Signed verifier candidate for review, not activation]
    Prospective --> CandidateArtifact
    Calibrate --> CandidateArtifact[Candidate calibrator for owner review]
    PRM -->|bad reasoning step or no consensus| Abstain
    PRM -->|ensemble disagreement or OOD| ReviewQueue[(Private verifier review queue)]
    ReviewQueue --> ReviewLabel[Human safe, unsafe, or ambiguous label]
    ReviewLabel --> Offline

    Final --> Evaluate[Record quality and feedback signals]
    Abstain --> Evaluate
    Stop --> Evaluate
    Clarify --> Evaluate
    Evaluate --> WriteMemory[Write memory only with explicit consent]

    API <--> State[(PostgreSQL or SQLite checkpoints)]
    API <--> Cache[(Redis or local cache)]
    API --> Telemetry[Prometheus, OpenTelemetry, and LangSmith]
    API --> Gateway[Inference gateway]
    Gateway --> Bandit{Constrained contextual bandit}
    Bandit --> Models[Economy, balanced, and quality models]
    Bandit --> Feedback[Propensity-scored feedback]
    Feedback --> Offline[IPS, SNIPS, and doubly robust gate]
    Gateway --> Online[Online SLO and rollback controller]
```

The research workflow is an **18-node LangGraph state machine**. The trust controls are independently configurable, so they can be evaluated in isolation or enabled together. Native LangGraph interrupts persist web-approval state in the checkpointer, which means an approval can survive an API restart.

The line-by-line graph description lives in [the runtime flow](docs/architecture/agent_runtime_flow.md), and important trade-offs are recorded as [architecture decisions](docs/architecture/decisions.md).

## What is implemented

| Area | What the implementation demonstrates |
|---|---|
| Agent orchestration | Typed LangGraph state, structured routing, clarification, bounded repair, native interrupts, and route-scoped evidence |
| Retrieval | Semantic chunking, deterministic document IDs, vector + BM25 fusion, reranking, graph predicates, multi-hop retrieval, hard-negative learning-to-rank, caching, and retrieval ablations |
| Evidence intelligence | Prompt-injection quarantine, source-independence checks, cross-domain duplicate detection, freshness policy, and numeric/negation conflict graphs |
| Trustworthy generation | Claim-to-evidence alignment, citation allowlisting, high-risk thresholds, conformal selective answering, and fail-closed abstention |
| Test-time compute | Confidence-aware early exit, offline-RL or search-distilled PUCT priors, learned action-conditioned world-model rollouts, calibrated PRM ensembles, uncertainty penalties, OOD fallback/abstention, active learning, and measured quality/compute curves |
| Execution-to-learning loop | Consented content-free candidate observations, keyed tenant isolation, immutable delayed labels, request-grouped calibration/test splits, coverage/risk gates, and candidate-only recalibration |
| Prospective AI validation | Pre-execution task-family IDs, frozen chronological cohorts, embargo and label-availability cutoffs, one-use family exposure tracking, live-lineage checks, distribution-shift holds, and paired bootstrap comparisons |
| Preference learning | Human-reviewed candidate pairs, family-bootstrap Bradley–Terry metadata reward models, training-only feature scaling, conservative margin/OOD fallback, tenant-bound artifacts, safety-gated runtime reranking, and future-family evaluation |
| Controlled AI rollout | Preregistered non-serving selector comparisons, actual PRM/release eligibility snapshots, disagreement plus audit review, missing-label sensitivity bounds, fresh-family holdout gates, and optional expiring model/policy/scope-bound approvals |
| Post-activation AI control | Tenant-scoped signed deployment revisions, live approval-source checks, atomic serving provenance, fixed review deadlines, family-deduplicated sequential error monitoring, immediate reviewed-unsafe holds, and irreversible-per-approval baseline rollback |
| Reviewed workflow supervision | Frozen runtime workflow features, independent immutable step annotations, explicit-label-only PRM training, chronological family splits, validation-only temperature calibration, constant/weak-credit ablations, and conservative missing-review gates |
| Long-term memory | Episodic, semantic, preference, and procedural memory with consent, tenant isolation, provenance, TTLs, corrections, deletion, and poisoning controls |
| Model operations | Tenant budgets, provider deadlines, circuit breakers, fallback, isolated semantic caching, constrained contextual-bandit routing, canaries, shadow evaluation, and online rollback decisions |
| Evaluation | Versioned datasets, fingerprints, trace replay, confidence intervals, failure slices, Pareto analysis, human-review provenance, adversarial arenas, and CI gates |
| Observability | Content-free OpenTelemetry GenAI spans, Prometheus metrics, LangSmith hooks, request/node latency, privacy checks, and reliability fault injection |
| Deployment | FastAPI + SSE, PostgreSQL/SQLite, Redis, Docker Compose, non-root containers, health checks, Kubernetes manifests, and network policy |

The controls are designed to compose. For example, retrieved documents do not go straight into a prompt: the evidence layer first removes suspicious or redundant material; the grounding layer then checks claims against the surviving evidence; the uncertainty layer decides whether the answer can be released; and adaptive compute is reserved for cases that are uncertain but still recoverable.

## Sandboxed coding agent

The coding workflow is deliberately separated from the research graph because code execution has a different risk model.

```mermaid
flowchart LR
    Issue[Repository-scoped issue] --> Queue[Durable job queue]
    Queue --> Lease[Leased worker]
    Lease --> Copy[Filtered ephemeral copy]
    Copy --> Context[Tree-sitter code intelligence]
    Context --> Roles[Analyst, implementer, and test author]
    Roles --> Policy{Tool policy and taint checks}
    Policy -->|denied| Dossier[Evidence dossier]
    Policy -->|allowed| Sandbox[Network-disabled Docker sandbox]
    Sandbox --> Verify{Tests, lint, scope, secrets, integrity}
    Verify -->|repairable| Repair[Bounded repair loop]
    Repair --> Verify
    Verify -->|failed| Dossier
    Verify -->|verified| Review[Independent reviewer]
    Review --> Dossier
    Dossier --> Owner{{Owner approval}}
    Owner -->|approved| Patch[Release unified diff]
```

Code intelligence uses Tree-sitter for Python, TypeScript, JavaScript, Java, Go, and Rust. A decomposed query searches symbols, files, tests, and dependencies using BM25, semantic embeddings, graph propagation, and reranking. Hard-negative mining trains an integrity-sealed pairwise fusion model, with optional bi-encoder and cross-encoder fine-tuning. The resulting context is deduplicated, compressed to a token budget, and accompanied by provenance receipts.

The worker never edits the original repository. It operates on a filtered temporary copy and executes tests inside a non-root container with no network, a read-only root filesystem, dropped capabilities, and CPU, memory, PID, and time limits. Jobs are transactional and recoverable: submissions are deduplicated, workers hold expiring leases, abandoned jobs can be reclaimed, and completed artifacts are checked by digest.

See [the coding-agent design](docs/code-agent.md), [code intelligence](docs/code-intelligence.md), and [coding-agent evaluation](docs/coding-agent-evaluation.md).

## Evaluation evidence

The repository currently contains **401 automated tests**. The CI floor is intentionally lower than the measured total so platform-specific integration paths can remain optional; focused coverage and the current whole-project percentage are published by every CI run.

| Module | Focused coverage |
|---|---:|
| Adaptive test-time compute | 94% |
| Verifier-guided search | 93% |
| Calibrated verifier ensemble | 90% |
| Evidence quality | 97% |
| Conformal uncertainty | 95% |
| Grounding verification | 94% |
| Inference gateway | 94% |
| Agent memory | 92% |
| Online quality controller | 89% |

The deterministic evaluation suites cover more than happy-path output. They exercise evidence injection, copied-source laundering, stale and contradictory evidence, fabricated citations, unsafe high-risk claims, calibration drift, compute-budget violations, process-level reward failures, unsafe trajectory selection, cross-tenant memory access, poisoning, provider failures, cache isolation, canary rollback, contextual-policy support and propensity errors, unsafe tool calls, worker-lease recovery, artifact tampering, and retrieval regressions.

Run the same core checks used in CI:

```bash
python -m pytest -q --cov --cov-report=term-missing --cov-report=xml
python -m evals.run_offline_evals --min-score 0.95
python -m evals.contextual_bandit evals/datasets/contextual_bandit_feedback.jsonl --require-promotion
python -m evals.process_reward_evaluation --require-promotion
python -m evals.search_planning_evaluation --check evals/experiments/search_planning.report.json --require-promotion
python -m evals.verifier_uncertainty_evaluation --check-artifact evals/experiments/process_reward_ensemble.json --check-report evals/experiments/verifier_uncertainty.report.json --require-promotion
python -m evals.world_model_evaluation --require-promotion
python -m evals.offline_rl_evaluation --require-promotion
python -m evals.distillation_evaluation --require-promotion
python -m evals.replay_calibration --drill --require-gate
python -m evals.prospective_evaluation drill --require-gate
python -m evals.process_supervision_evaluation drill --require-gate
ruff check agent client code_agent evals post_training schema service tests
python -m pip check
```

Run the reliability command center locally:

```bash
python -m evals.reliability --output data/evaluations/reliability/latest.json
streamlit run evals/reliability_dashboard.py --server.port 8506
```

It combines experiment results, retrieval ablations, security attacks, failure recovery, model-gateway behavior, memory governance, grounding, uncertainty, adaptive compute, and evidence-quality results in one place. Every report retains configuration and dataset fingerprints so a result can be tied back to the exact experiment that produced it.

For the evaluation methodology and measured case studies, read [evaluation strategy](docs/evaluation.md) and the [recruiter-facing case study](docs/agentforge-case-study.md).

## Quick start

### 1. Create the environment

Python 3.11–3.13 is supported.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-service.txt -r requirements-app.txt
```

### 2. Configure a model provider

```bash
cp .env.example .env
```

On Windows PowerShell, use `Copy-Item .env.example .env`. Add at least one of `OPENAI_API_KEY` or `GROQ_API_KEY`. Model credentials stay in the service; clients discover only the configured model IDs through `/capabilities`.

The advanced reliability controls are opt-in so the basic application can start without calibration artifacts or signing keys. Their defaults and explanations are in [.env.example](.env.example).

### 3. Start the API and UI

Use two terminals:

```bash
python run_service.py
```

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`. API documentation is available at `http://localhost:8000/docs`.

### Docker Compose

```bash
docker compose up --build
```

The default stack starts the API, UI, PostgreSQL, Redis, Prometheus, and Grafana. Evaluation dashboards are behind the `evaluation` profile:

```bash
docker compose --profile evaluation up --build
```

| Service | URL |
|---|---|
| API documentation | `http://localhost:8000/docs` |
| Agent UI | `http://localhost:8501` |
| Coding-agent evaluation | `http://localhost:8502` |
| Post-training lab | `http://localhost:8503` |
| Code-context lab | `http://localhost:8504` |
| Security lab | `http://localhost:8505` |
| Reliability command center | `http://localhost:8506` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3001` |

Change the example PostgreSQL, Grafana, authentication, and integrity-signing secrets before exposing the stack outside a local environment.

## Try the main workflows

### Research agent

After registering through `POST /auth/register`, call `POST /invoke` or `POST /stream` with a stable `thread_id`. Current-information questions pause at a native graph interrupt. Resume with `approve` or `reject: reason` on the same user and thread; the service verifies that an active checkpoint interrupt exists before accepting the decision.

Useful demo prompts:

- `Calculate (125 * 8) / 4` — deterministic routing and tool execution.
- `local: explain the checkpoint implementation` — source-grounded local RAG.
- `How does FastAPI connect to LangGraph in this project?` — relationship and graph retrieval.
- `What changed in AI news this week?` — web approval, evidence adjudication, and durable resume.

### Ingest local documents

Place PDFs in `rag_docs/` or `graph_rag_docs/`, then run:

```bash
python scripts/ingestion/ingest_local_rag_pdfs.py --pdf-dir rag_docs
python scripts/ingestion/ingest_graph_rag_pdfs.py --pdf-dir graph_rag_docs
```

Ingestion is idempotent for unchanged content. Chunks receive deterministic IDs, and stale chunks for a changed source are removed only after the replacement write succeeds.

### Run a coding task

Build the purpose-specific sandbox image, then enable `CODE_AGENT_ENABLED`:

```bash
docker build -f docker/Dockerfile.code-sandbox -t agentforge-code-sandbox:local .
```

Submit an authenticated issue to `POST /code/tasks` with an `Idempotency-Key`. Poll the task, inspect its event history and evidence dossier, then explicitly approve or reject the verified patch. The original repository remains unchanged throughout the workflow.

## Repository map

```text
agent/          LangGraph runtime, retrieval, memory, grounding, and model controls
code_agent/     Sandboxed coding workflow, code intelligence, security, and benchmarks
service/        FastAPI endpoints, authentication, persistence, and streaming
evals/          Datasets, experiment runners, graders, dashboards, and release gates
post_training/  Reviewed SFT/DPO preparation, LoRA/QLoRA training, and model lineage
monitoring/     Prometheus and Grafana configuration
docker/         Runtime and hardened coding-sandbox images
k8s/            Kubernetes deployment, storage, secrets, and network policy
docs/           Architecture decisions, threat model, subsystem guides, and case studies
tests/          Unit, integration, security, and evaluation tests
```

## Design choices worth discussing

**Why several release gates?** Evidence quality, grounding, uncertainty, and adaptive compute answer different questions. A source can be suspicious before synthesis; a generated claim can be unsupported afterward; a supported answer can still fall outside the calibrated release region; and extra inference is useful only for uncertain cases that have enough evidence to recover.

**Why are most advanced controls opt-in?** Some require a calibration artifact, a dedicated integrity key, or an operational policy that should be owned by the deploying team. Silent defaults would make a demo easier but would hide those dependencies.

**Why deterministic evaluations?** Credential-free tests are stable enough for every pull request. They validate control flow, isolation, accounting, and failure behavior. Live-model and human-labelled evaluations remain a separate layer because they are slower, more expensive, and statistically variable.

**Why not let the coding agent apply changes?** The project keeps generation and authorization separate. A verified patch is evidence for a human decision, not permission to mutate the source repository.

## What the results do not claim

The checked-in datasets are intentionally useful for regression testing, but several are authored or synthetic. They prove that the mechanisms behave as designed; they do not prove broad real-world model quality.

- Evidence, grounding, conformal, and adaptive-compute results need larger human-labelled, multilingual, temporal, and live-model datasets before they support production quality claims.
- The behavioral arena uses synthetic seed scenarios and deterministic policy simulators; it is not a substitute for repeated adversarial testing with real models and calibrated judges.
- Retrieval uses an embedded Chroma deployment and hashed embeddings by default in some offline paths. Larger deployments should use managed/shared indexes and evaluated production embeddings.
- The inference gateway, online event stream, memory store, coding queue, and artifact store are durable single-host reference implementations, not distributed control planes.
- The Docker sandbox is strong process isolation for a portfolio system, but hostile multi-tenant execution should use isolated microVM workers and an external immutable artifact store.
- Kubernetes manifests are a secure starting point, not a complete managed-cloud architecture.
- Post-training plumbing is implemented, but no fine-tuned-model quality claim should be made without reviewed data, accelerator training, and a frozen holdout evaluation.
- Process-reward results use a small synthetic seed to test learning, pruning, integrity, and promotion behavior; they are not evidence that the verifier generalizes to unseen live-model reasoning traces.
- Independently reviewed process learning predicts correctness of typed workflow proxies, not hidden reasoning or chain-of-thought. Its synthetic controls test leakage prevention and evaluation holds; final-answer lift still needs diverse reviewed runtime data and a separate selective-release evaluation. No model is activated automatically.
- Verifier-guided search results use deterministic synthetic transitions. They validate planning, budgets, fail-closed behavior, and replay integrity—not real-world reasoning quality or provider cost savings.
- Verifier-ensemble shift results use one controlled behaviorally inverted member and only three calibration traces. They validate disagreement detection, conservative scoring, and active-learning plumbing—not production OOD coverage.
- Learned world-model results use a small, mostly authored transition dataset. They validate action-conditioned prediction, conservative rollouts, artifact integrity, and OOD abstention—not general real-world environment modeling.
- Conservative offline-RL results use eight authored test episodes with synthetic propensities. They validate CQL, sequential off-policy estimators, PUCT priors, safety masking, and promotion plumbing—not real-traffic policy lift.
- Reviewed preference reranking learns metadata correlations, not answer semantics. Its synthetic 40-family test demonstrates selection and OOD fallback; offline pools do not reconstruct every live eligibility/PRM decision, so the result is not an end-to-end factuality or live-selector improvement claim.
- Preference shadow mode compares the actual selectors on already-generated candidates without changing served answers. Its synthetic controls are not randomized traffic lift; reviewed runtime evidence and an owner-approved rollout are still required. Approval checks are opt-in offline leases, not continuous revocation.

These boundaries are intentional. Good AI engineering includes knowing what the evidence supports—and what it does not.

## Resume summary

> Built AgentForge, an 18-node LangGraph agent platform combining hybrid and graph RAG, durable human approval, evidence-conflict detection, claim-level grounding, conformal selective answering, and PUCT planning with conservative offline-RL action priors, a learned transition world model, calibrated process-reward ensembles, epistemic OOD controls, replayable plans, and a privacy-safe active-learning loop. Added sequential doubly robust policy evaluation, contextual-bandit routing, tenant-isolated memory, a policy-gated coding agent with six-language Tree-sitter retrieval, FastAPI/SSE serving, privacy-safe telemetry, adversarial evaluation, and Docker/Kubernetes deployment assets.

When using this project in a resume or interview, lead with one measurable workflow rather than listing every subsystem. A strong walkthrough is: retrieve evidence, detect a conflict, withhold an unsupported answer, show the trace and evaluation result, then explain the trade-off between answer coverage, accuracy, latency, and cost.

## Documentation

- [Architecture and runtime flow](docs/architecture/agent_runtime_flow.md)
- [Architecture decision records](docs/architecture/decisions.md)
- [Evaluation strategy](docs/evaluation.md)
- [AgentForge case study](docs/agentforge-case-study.md)
- [Evidence intelligence](docs/evidence-intelligence.md)
- [Grounding verification](docs/grounding-verification.md)
- [Uncertainty calibration](docs/uncertainty-calibration.md)
- [Adaptive test-time compute](docs/adaptive-test-time-compute.md)
- [Process reward modeling](docs/process-reward-modeling.md)
- [Verifier-guided reasoning search](docs/verifier-guided-search.md)
- [Uncertainty-aware verifier ensemble](docs/verifier-uncertainty.md)
- [Learned world-model planning](docs/learned-world-model-planning.md)
- [Conservative offline-RL planning](docs/conservative-offline-rl-planning.md)
- [Search policy distillation and compute curves](docs/search-policy-distillation.md)
- [Execution feedback and gated recalibration](docs/execution-feedback-calibration.md)
- [Forward-time AI validation and holdout governance](docs/prospective-ai-validation.md)
- [Reviewed preference learning and conservative reranking](docs/reviewed-preference-reranking.md)
- [Non-serving preference shadows and prospective approval gates](docs/preference-shadow-validation.md)
- [Revocable preference deployments and delayed-outcome sentinel](docs/preference-deployment-sentinel.md)
- [Independent workflow supervision and verifier retraining](docs/reviewed-process-supervision.md)
- [Agent memory](docs/agent-memory.md)
- [Inference gateway](docs/inference-gateway.md)
- [Online AI governance](docs/online-ai-governance.md)
- [Code agent](docs/code-agent.md)
- [Code intelligence](docs/code-intelligence.md)
- [Security threat model](docs/security/threat-model.md)
- [GenAI observability](docs/genai-observability.md)
- [Self-improvement flywheel](docs/self-improvement-flywheel.md)
- [Post-training](docs/post-training.md)

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). AgentForge is released under the [MIT License](LICENSE).
