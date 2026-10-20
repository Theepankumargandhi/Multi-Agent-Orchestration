# Evaluation strategy

The repository separates deterministic CI gates from credentialed model evaluations.

The [October 4 audit](live-upgrade-audit-2026-10-04.md) maps the implemented
subsystems to their actual checks: 84 test modules, 62 credential-free CI
commands, real-container controls, live coding attempts, and local HTTP journeys.
It lists the still-unverified provider, specialist, research and infrastructure
paths explicitly. Passing a synthetic governance drill is not a production
quality result or authorization to activate a candidate.

The coding tournament's [regression-challenge stage](regression-challenge-agent.md) adds pre-patch JSON probe generation and a separate Docker-backed weak-patch ablation. CI runs authored probe/patch controls without a provider; the live generation smoke run is opt-in. These distinguish execution controls from model-oracle quality, and neither automatically activates the workflow.

## Reproducible experiment platform

[Reference-calibrated probe evaluation](oracle-calibration.md) measures the proposed checker itself: agreement with a pinned operator source, controlled AST-fault sensitivity, and a diagnostic probe-to-mutant matrix. CI's strong/weak/wrong-oracle comparison is an authored Docker control, not a provider or serving-promotion experiment.

`python -m evals.run_experiments` loads a versioned JSON experiment matrix and JSONL dataset, executes every variant against the same cases, and produces a portable report plus a SQLite registry entry. The checked-in experiment compares:

- `keyword-baseline-v1`: a deliberately simple control;
- `production-hybrid-router`: the deterministic portion of the production router.

Each report records its dataset SHA-256 and includes per-case traces, quality/pass rates, p50/p95 latency, tokens, estimated cost, failure categories, and quality/cost/latency Pareto status. Open results with `streamlit run evals/dashboard.py`.

Experiment definitions live in `evals/experiments/`; datasets live in `evals/datasets/`. Available adapters are `keyword-baseline`, `research-router`, and `live-graph`. Use the live adapter only for credentialed end-to-end experiments.

The live adapter defaults to `bypass_hitl=true` so batch evaluation can exercise web retrieval and synthesis without pausing 120 times. This flag exists only in direct graph evaluation configuration and is not accepted by the public API. Set `"bypass_hitl": false` in a variant when the approval workflow itself is under test.

## Offline gates

[UpgradePilot's frozen controls](upgradepilot.md) run actual Pydantic 1.10.24 and
2.12.4 environments plus bump-pydantic 0.8.0. They distinguish a codemod-sufficient
constraint case, a union-type regression hidden by passing unit tests, and a
validator-signature failure. Fixture-specific scripted specialists exercise the
repair path; the results must not be presented as live model migration accuracy.
The corpus lock, source preservation, original tests and exact contract inputs
are retained across baseline and repaired candidates.

The [AI incident study](ai-failure-investigation.md) tests actual snapshot retrieval,
ranking and packing transformations against five authored failures/negative controls:

```bash
python -m evals.incident_investigation_evaluation --output data/evaluations/incident-investigation/my-controls.json
```

It uses an extractive generator and simulated exact-fact labels, makes no provider
requests, and cannot promote a repair. Independently reviewed live replay is a
separate opt-in path. The October 14 affected integration suite passed 323 tests
without skips; this is not a rerun of every test in the repository. A separate
scripted four-role coding run passed exact-patch fresh-container verification.

`python -m evals.runtime_rag_evaluation --output data/evaluations/runtime-rag/controls.json`
adds authored controls through actual SQLite RAG, encrypted runtime capture and
a reopened durable worker. Its three cases must pass while still withholding
live repair recommendations. The [frozen comparison protocol](runtime-rag-investigation.md)
separates diagnostic families from held-out snapshot cohorts, preregisters all
members before comparative replays, retains missing/deleted outcomes in bounds,
and reports family-bootstrap correctness differences rather than inventing a lift.
It does not claim prospective random traffic sampling or independent human labels
for the local control study.

`python -m evals.run_offline_evals` keeps the original seven-case compatibility gate. `python -m evals.run_experiments --no-store --min-score 0.95` runs the 25-case ablation and blocks CI when the winning quality score falls below 95%. Pytest separately covers the grading system, experiment registry, safety behavior, state isolation, citation scoring, chunking, graph traversal, durable approval lookup, API auth, storage isolation, and monitoring.

The checked-in set is a regression seed, not evidence of general model quality. Add paraphrases, adversarial prompts, multilingual inputs, and ambiguous cases before publishing a benchmark.

## Evidence retained in coding-agent context

The [evidence-span retrieval study](evidence-span-retrieval.md) complements filename recall
with source-bound line and complete-span coverage measured on the actual packed context.
Baseline/candidate comparisons share queries and source snapshots; task families, not
paraphrases, are the bootstrap unit. A fixed primary budget controls the review decision,
while other budgets and query slices explain compression failures. CI publishes the authored
study's JSON and Markdown card without activating a model or treating synthetic labels as
real-quality evidence. Stale source labels and inconsistent receipts remain code failures.

## Recommended RAG evaluation set

For each document set, maintain question, expected source/chunk, reference facts, and unanswerable status. Track:

- retrieval recall@k and mean reciprocal rank;
- groundedness/faithfulness;
- answer relevance and citation precision;
- refusal accuracy for unanswerable questions;
- p50/p95 end-to-end and per-node latency;
- tokens and estimated cost per successful answer;
- cache hit rate and tool/provider error rate.

Use deterministic checks for citations and source IDs, then an LLM judge with a fixed rubric. Manually review a rotating sample because judge models can share the same blind spots as the system under test.

## Iteration-two evaluation and optimization

The repository now includes:

- a 120-case synthetic candidate set split 72/24/24 across train/validation/test;
- explicit `synthetic_seed` versus `human_reviewed` provenance;
- CSV label-review export/import with reviewer and timestamp audit metadata;
- bootstrap confidence intervals and per-tag slice metrics;
- pairwise structured judging with forward/reversed candidate ordering;
- judge accuracy and high-confidence error measurement only against reviewed preferences;
- Recall@K, precision@K, MRR, NDCG, hit rate, slice metrics, and reciprocal-rank fusion;
- an auditable logistic model router trained from observed small/strong model outcomes;
- validation-set threshold tuning under a minimum-quality constraint and separate held-out reporting.
- an opt-in production `adaptive` model that loads the trained artifact, routes high-risk requests to the strong tier, records probability/threshold/fingerprint metadata, and preserves explicit model choices.

## Contextual policy learning

The next-generation adaptive router goes beyond the original binary classifier. It learns a
separate linear reward model for economy, balanced, and quality profiles from logged feedback.
The reward combines quality, cost, latency, and a large unsafe-outcome penalty. At inference
time, LinUCB uncertainty supports bounded exploration, while hard feasibility rules remove
actions that exceed the request's cost or latency ceiling. Economy models are never eligible
for high-risk prompts.

Every exploratory decision reports the probability with which its action was selected. That
propensity makes the feedback useful for counterfactual evaluation instead of turning it into
biased click data. Before a policy can be promoted, the offline gate reports:

- inverse propensity scoring (IPS) and self-normalized IPS (SNIPS);
- a doubly robust estimate with a context-clustered confidence interval;
- effective sample size and unsupported-context counts;
- target-policy safety violations and an explicit promotion decision.

The checked-in seed demonstrates the machinery rather than claiming general model quality. On
its 12 held-out action observations, the learned policy has a doubly robust utility of 0.8361
versus 0.5375 for the logging policy, with zero matched unsafe outcomes. Replace this synthetic
seed with randomized production or shadow traffic before making a real routing claim.

Reproduce the integrity-sealed artifact and gate locally:

```bash
python -m evals.contextual_bandit evals/datasets/contextual_bandit_feedback.jsonl \
  --artifact data/evaluations/bandit/policy.json \
  --report data/evaluations/bandit/report.json \
  --check evals/experiments/contextual_bandit_policy.json \
  --require-promotion
```

The routing reproducibility check separates exact integrity from numerical agreement. Each
artifact must verify its own unchanged SHA-256 digest. Dataset identity, action configuration,
training settings, observation counts, and matrix dimensions must match exactly. Only learned
coefficients may differ, by at most `1e-12` absolute (no relative tolerance), to accommodate
cross-Python floating-point roundoff. Non-finite values and larger drift fail closed. The CLI
reports both fingerprints, the maximum difference, and whether the match was exact; it never
assigns the reference fingerprint to regenerated weights. The checked-in reference is not
rewritten. Promotion remains a separate required check, and passing it does not activate a
serving policy.

The routing suite passed 22 tests on Python 3.11 and 3.13, including tampering, structural and
lineage changes, non-finite weights, real coefficient drift, and failed-promotion controls.
The CI routing command passed on both runtimes. This is a local reproduction of the previously
failing step, not confirmation that the full GitHub workflow has completed successfully.

Process-reward and verifier-ensemble reproduction use the same absolute weight tolerance,
while dataset identity, stopping thresholds, calibration settings, and member order/count
remain exact. Every member and parent digest must verify independently. Uncertainty report
comparison also verifies each report's exact digest and binds it to its own verified ensemble;
only the per-outcome uncertainty estimate receives numerical tolerance. Selected traces,
safety outcomes, aggregate metrics, and promotion decisions must match exactly. Fresh weights
and reports retain their own fingerprints, and invalid report comparisons write no outputs.
The related checks passed on Python 3.13 and in isolated offline Python 3.11 checks; those
isolated checks omit optional service startup and are not a full CI environment reproduction.

The tooling is implemented, but the included 120 cases remain clearly marked as synthetic candidates. A project owner must review them and run credentialed model experiments before publishing end-to-end quality or cost claims.

## Failure-to-improvement flywheel

`python -m evals.flywheel` extends offline evaluation into an auditable improvement loop. It
redacts and fingerprints traces, clusters recurring failures, proposes quarantined validation
cases, exports them for human review, builds root-cause-specific prompt/policy ablations, and
applies regression, safety, quality, cost, latency, and provenance gates before canary entry.
The canary state machine automatically promotes after sufficient healthy evidence or rolls
back on the first unhealthy aggregate window. See
[self-improvement-flywheel.md](self-improvement-flywheel.md) for the workflow and trust boundaries.

### Calibration protocol

1. Build at least 100 human-reviewed cases split by route, risk, answerability, and difficulty.
2. Freeze a hidden test split before changing prompts or retrieval.
3. Add pairwise LLM judging only for qualities that deterministic graders cannot measure.
4. Measure judge agreement against human labels and publish disagreements.
5. Compare vector, hybrid, reranked, and graph retrieval with retrieval recall@k and answer-level groundedness.
6. Fit a cost-aware router on the training split, then report quality, latency, and cost only on the hidden split.
7. Never promote a configuration solely because it performs better on the cases used to tune it.

## Release policy

Migration comparisons have an optional [counterexample and final-set protocol](counterexample-migration.md).
Discovery feedback is available only to its enabled arm; the frozen final gate is
terminal, not a source of repair hints. The local pilot runner retains held attempts
and missing usage explicitly. Its scripted controls are not real-repository results.

The [schema-guided ablation](schema-guided-migration.md) holds final inputs and
discovery budgets constant while comparing input-shape reach. It reports actual
source-value versus source-error counts, not an invented branch-coverage metric.
Normalized JSON Schema equality never counts as behavioral equivalence.

The optional [stateful migration controls](stateful-migration-verification.md)
compare ordered observations within isolated sessions, including state after a
failed call. Their reduced discovery traces may inform repair; final sessions
are terminal and patch-bound. The October 16 local run passed 194 selected tests
without skips and all four authored Docker expectations, including the overfit
negative control. Those results do not measure live-model or real-repository
migration quality.

[Metamorphic workflow controls](metamorphic-workflow-verification.md) add a distinct
correctness oracle: owner-declared laws with generated variants and explicit
effect checks. Source-law failures stop before repair, while withheld target
variants are terminal. Property inventories describe only the selected laws and
cases; they are not application coverage or evidence of live-model improvement.

[Crash-safe study recovery](crash-safe-migration-studies.md) keeps completed
successes and holds immutable, preserves unknown request reservations, and marks
quality metrics invalid after an unresolved earlier attempt. Final readiness
rates after an explicit retry are descriptive, not first-attempt model accuracy.

Block a release on unit/integration failures, routing accuracy below 95%, coverage below the configured floor, lint failures, or known vulnerable runtime dependencies. Model/RAG metric thresholds should be established from a representative dataset and stored alongside each experiment.
## Developer text-evidence SDK

The [SDK controls](document-sdk.md) exercise the shared production text-RAG
primitives through a typed, instance-isolated public API. Authored cases check
literal quotes, page/version provenance, abstention, consent, held selector output
and corpus/context limits. A separate packaging control installs the lightweight
wheel and runs an application outside the checkout with platform/provider imports
and network sockets blocked. The HTTP example is tested separately from the
existing service; these checks do not exercise live LLM generation or establish
real-document QA quality. See [the local audit](document-sdk-audit-2026-10-16.md).

## Portable patch-regression comparisons

The [portable verifier benchmark](patch-regression-benchmark.md) freezes external
patches, owner contracts/domains, source/license bytes, implementation and image
IDs. It separately executes tests-only, contract and generated-domain arms,
retains inconclusives and reports detection, false alarms, coverage and latency.
No models are called: provider cost is zero while infrastructure cost is unknown.

The starter corpus has four authored controls, not independently reviewed real
patches. Real-case intake prepares exact review bindings but never invents a
reviewer; independent attestations and negative controls are required at admission.
Source drift or mixed task-family splits invalidate a comparison. Public controls
are not a secret holdout. [Maintainer pilot outcomes](maintainer-migration-pilots.md)
are recorded separately and remain human-reported, not automatically verified
GitHub acceptance or evidence of general coding-agent quality.

## Structured failures behind green tests

The [structured regression controls](structured-regression-discovery.md) compare
tests-only, explicit-contract and generated JSON arms, then exercise equivalent
patches, intentionally changed out-of-scope behavior and invalid original domains.
Failures are reduced with fresh single-input confirmation and can be replayed
against source/patch/image-pinned inputs. These six expectations execute in real
offline Docker; they are authored controls, not real upstream benchmark outcomes.
Optional AI domain-proposal tests use injected clients, not paid provider calls.

The [real-patch pilot](real-patch-validation.md) additionally freezes source,
license, test and candidate bytes for four actual forward upstream changes.
It compares all three strategies and confirms exported reduced inputs in fresh
containers. Its public, curator-labelled development results remain provisional;
the independent-review/holdout gate is not bypassed. The manual workflow uploads
only an input-free summary, not raw counterexamples or private snapshots.

## Retrieval query stress controls

The [query robustness study](retrieval-robustness.md) compares clean and perturbed queries
within the same source-bound task families, reports worst-case coverage and paired intervals,
and exercises numerical gates with an empty-context negative control. Generated variants
remain held; CI reports them without model activation. The latest affected robustness,
packing, span, retrieval-learning, and retrieval-CI test run passed all 70 tests. This was not
a full-suite rerun or a downstream coding-quality experiment.
## Multi-agent repair controls

The [repair tournament](multi-agent-repair-tournament.md) adds seven one-team/three-team
orchestration scenarios with controlled model and command outputs. The focused runtime
and control suite passed 33 tests with 96.3% combined statement/branch coverage. These checks
include misleading approvals, exact command binding, stale patches, strict fresh gates,
provider outages, cancellation cleanup, deadline holds, atomic shared reservations, snapshot
integrity, and approval-gated persistence. They are not a full-suite or live-model benchmark.
