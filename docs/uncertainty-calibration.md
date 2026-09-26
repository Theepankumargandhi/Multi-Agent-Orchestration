# Conformal uncertainty and selective generation

AgentForge can place a route-aware split-conformal gate after claim grounding. Grounding produces an
explainable confidence score; the conformal layer converts that score into a prediction set over
`correct` and `incorrect`. The answer is released only when the set is exactly `{correct}`. Ambiguous
sets, empty sets, missing artifacts, and failed integrity checks abstain.

## Why this is separate from grounding

A grounding score is not automatically a calibrated probability. A fixed score threshold can behave
differently across web, RAG, knowledge-graph, hybrid, and calculator routes. The calibration step uses
a held-out labelled split and fits Mondrian thresholds per route, falling back to a global threshold
when a route lacks enough examples.

For a labelled calibration example with predicted correctness confidence `p`, nonconformity is
`1 - p` when the decision is correct and `p` when it is incorrect. The finite-sample quantile uses
`ceil((n + 1) * (1 - alpha))`. At inference time:

- `correct` enters the set when `1 - p <= q`;
- `incorrect` enters the set when `p <= q`; and
- release requires the singleton set `{correct}`.

The artifact records its target error rate, global and route quantiles, sample counts, confidence
histogram, dataset fingerprint, and integrity fingerprint. Runtime receipts record the selected
quantile, prediction set, release decision, route fallback, and out-of-distribution flag.

## Training and evaluation

```bash
python -m evals.uncertainty_evaluation \
  --output data/evaluations/uncertainty/latest.json \
  --artifact data/evaluations/uncertainty/calibrator.json \
  --target-error-rate 0.2 \
  --min-selective-accuracy 1 \
  --max-empirical-error 0 \
  --min-incorrect-abstention 1 \
  --require-drift-detection
```

The checked-in dataset contains 25 calibration and 15 disjoint test examples across web, hybrid,
RAG, knowledge-graph, and math routes. It exists to verify the implementation and CI gate—not to
claim statistical validity for production traffic.

## Runtime configuration

Generate the artifact with the same integrity key used by the service, then enable both grounding
and uncertainty control:

```dotenv
GROUNDING_VERIFICATION_ENABLED=true
UNCERTAINTY_CALIBRATION_ENABLED=true
UNCERTAINTY_CALIBRATOR_PATH=data/evaluations/uncertainty/calibrator.json
UNCERTAINTY_INTEGRITY_KEY=replace-with-a-separate-16-character-minimum-secret
```

The artifact is hot-reloaded on file modification. Loading fails closed when the file is absent,
malformed, or has an invalid fingerprint. Jensen–Shannon divergence compares recent confidence
histograms with the calibration reference and supports a distribution-shift alert.

## Statistical limitations

Conformal coverage depends on exchangeability between calibration and deployment examples. Route
shift, label noise, adaptive prompts, temporal changes, or a modified verifier can invalidate the
observed guarantee. Before production use:

- replace synthetic labels with independently reviewed claim-correctness labels;
- keep calibration and evaluation splits isolated by user, document, and time;
- report answer coverage alongside selective accuracy and error;
- recalibrate after changing retrieval, prompts, models, or grounding scores;
- monitor route/slice drift and minimum sample sizes; and
- use risk-specific policies for regulated or safety-critical domains.
