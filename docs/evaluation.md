# Evaluation strategy

The repository separates deterministic CI gates from credentialed model evaluations.

## Reproducible experiment platform

`python -m evals.run_experiments` loads a versioned JSON experiment matrix and JSONL dataset, executes every variant against the same cases, and produces a portable report plus a SQLite registry entry. The checked-in experiment compares:

- `keyword-baseline-v1`: a deliberately simple control;
- `production-hybrid-router`: the deterministic portion of the production router.

Each report records its dataset SHA-256 and includes per-case traces, quality/pass rates, p50/p95 latency, tokens, estimated cost, failure categories, and quality/cost/latency Pareto status. Open results with `streamlit run evals/dashboard.py`.

Experiment definitions live in `evals/experiments/`; datasets live in `evals/datasets/`. Available adapters are `keyword-baseline`, `research-router`, and `live-graph`. Use the live adapter only for credentialed end-to-end experiments.

The live adapter defaults to `bypass_hitl=true` so batch evaluation can exercise web retrieval and synthesis without pausing 120 times. This flag exists only in direct graph evaluation configuration and is not accepted by the public API. Set `"bypass_hitl": false` in a variant when the approval workflow itself is under test.

## Offline gates

`python -m evals.run_offline_evals` keeps the original seven-case compatibility gate. `python -m evals.run_experiments --no-store --min-score 0.95` runs the 25-case ablation and blocks CI when the winning quality score falls below 95%. Pytest separately covers the grading system, experiment registry, safety behavior, state isolation, citation scoring, chunking, graph traversal, durable approval lookup, API auth, storage isolation, and monitoring.

The checked-in set is a regression seed, not evidence of general model quality. Add paraphrases, adversarial prompts, multilingual inputs, and ambiguous cases before publishing a benchmark.

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

Block a release on unit/integration failures, routing accuracy below 95%, coverage below the configured floor, lint failures, or known vulnerable runtime dependencies. Model/RAG metric thresholds should be established from a representative dataset and stored alongside each experiment.
