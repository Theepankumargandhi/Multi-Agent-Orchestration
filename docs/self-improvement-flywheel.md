# Agent self-improvement flywheel

AgentForge turns measured failures into reviewed evaluation cases and evidence-gated
configuration changes. The workflow is intentionally closed-loop, but it is not
self-authorizing: a person owns labels and an offline gate must pass before a candidate
can enter canary state.

## Trust boundaries

- Raw reports are treated as sensitive. Ingestion recursively redacts credential-like
  keys, bearer values, common API tokens, email addresses, and secret assignments.
- Redaction is defense in depth, not a universal DLP system. Add organization-specific
  detectors, encryption, access control, and retention limits in production.
- Candidate prompts are accepted only from the in-process evaluation adapter. The public
  API cannot set evaluation prompt suffixes.
- Failed traces can propose validation cases, never protected test cases.
- Proposed labels remain `synthetic_seed` until a named reviewer approves or corrects them.
- A candidate cannot enter canary state unless every configured quality, regression,
  safety, cost, latency, and provenance gate passes.

## 1. Ingest and redact failures

Run an EvalOps experiment, then ingest its portable report:

```bash
python -m evals.flywheel ingest \
  --report data/evaluations/<experiment-report>.json \
  --output data/evaluations/flywheel/traces.jsonl
```

Ingestion is idempotent for the same experiment, variant, and case. Each redacted record
has a content fingerprint so exported evidence can be audited.

## 2. Discover recurring failure modes

```bash
python -m evals.flywheel cluster \
  --traces data/evaluations/flywheel/traces.jsonl \
  --output data/evaluations/flywheel/clusters.json
```

Clustering uses deterministic normalized TF-IDF vectors and cosine similarity, constrained
by the failure taxonomy. This keeps the offline workflow reproducible and credential-free.
It is a lightweight semantic baseline, not a claim of deep learned embeddings. A production
deployment can replace the vectorizer while preserving the portable cluster contract.

## 3. Review regression candidates

```bash
python -m evals.flywheel propose \
  --traces data/evaluations/flywheel/traces.jsonl \
  --output data/evaluations/flywheel/proposals.jsonl

python -m evals.flywheel review-export \
  --proposals data/evaluations/flywheel/proposals.jsonl \
  --output data/evaluations/flywheel/proposal-review.csv
```

Complete `decision`, `reviewer`, and `review_notes`. The expected JSON can be corrected
during review. Promote approved cases into a separate regression dataset:

```bash
python -m evals.flywheel promote-regressions \
  --proposals data/evaluations/flywheel/proposals.jsonl \
  --review-file data/evaluations/flywheel/proposal-review.csv \
  --output data/evaluations/flywheel/reviewed-regressions.jsonl
```

Do not merge these cases into a hidden test split. Use them for training or validation,
then report final results on a separately frozen holdout.

Optionally create conservative formatting and untrusted-instruction challenges. These are
also quarantined and require the same human review:

```bash
python -m evals.flywheel augment-proposals \
  --proposals data/evaluations/flywheel/proposals.jsonl \
  --output data/evaluations/flywheel/adversarial-proposals.jsonl
```

## 4. Build and run failure-driven candidates

```bash
python -m evals.flywheel build-candidates \
  --clusters data/evaluations/flywheel/clusters.json \
  --dataset data/evaluations/flywheel/reviewed-regressions.jsonl \
  --model <configured-model> \
  --output data/evaluations/flywheel/candidates.json

python -m evals.run_experiments \
  --config data/evaluations/flywheel/candidates.json
```

The first variant is the current control. Candidate instructions are derived from the
observed root-cause category, versioned in the experiment report, and evaluated against
identical cases. Generation does not imply promotion.

## 5. Gate, canary, and rollback

```bash
python -m evals.flywheel gate \
  --report data/evaluations/<candidate-report>.json \
  --baseline current-production \
  --candidate candidate-grounding-v1 \
  --output data/evaluations/flywheel/promotion.json

python -m evals.flywheel deployment-init \
  --state data/evaluations/flywheel/deployment.json \
  --version production-v1

python -m evals.flywheel canary-start \
  --state data/evaluations/flywheel/deployment.json \
  --decision data/evaluations/flywheel/promotion.json \
  --version candidate-grounding-v1

python -m evals.flywheel canary-observe \
  --state data/evaluations/flywheel/deployment.json \
  --samples 50 --quality 0.90 --error-rate 0.02 \
  --p95-latency-ms 1800 --cost-per-request-usd 0.01
```

The offline decision is content-fingerprinted. Canary observations promote after the
minimum sample count or roll back immediately when a configured threshold is violated.
The local state file uses atomic replacement and retains an audit log. In a multi-replica
deployment, replace it with a transactional database and make the deployment controller
consume the same state contract.

Open `streamlit run evals/dashboard.py` to inspect failure categories, recurring clusters,
gate checks, regressions, decision fingerprints, and the canary audit timeline.

Human corrections or explicit accepted/rejected pairs can be exported for later SFT/DPO
experiments. The exporter refuses to infer preference labels from model scores:

```bash
python -m evals.flywheel export-preferences \
  --traces data/evaluations/flywheel/traces.jsonl \
  --output data/evaluations/flywheel/preferences.jsonl
```

## Honest interpretation

This subsystem demonstrates an auditable learning loop around an agent. It does not prove
that a model updates its weights autonomously, that TF-IDF clustering is universally
semantic, or that local state is a production control plane. Publish only results from
human-reviewed data and credentialed runs, along with sample size and gate thresholds.
