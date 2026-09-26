# Adaptive test-time compute

AgentForge can allocate additional inference only when the first answer is uncertain. The controller consumes claim-grounding, conformal prediction, out-of-distribution, evidence-availability, and high-risk signals and chooses one of three actions:

- `early_exit`: release an already grounded, calibrated answer without extra model calls;
- `deliberate`: generate a bounded set of independent candidates, verify each candidate, and release only when candidates reach evidence-level consensus;
- `abstain`: stop immediately when more inference cannot repair the evidence problem or a required control is unavailable.

This is test-time scaling with an explicit stopping policy, not an unrestricted reflection loop.

## Runtime control flow

The initial private draft passes through claim grounding and, when enabled, route-aware conformal calibration. An uncertain but potentially recoverable result enters `adaptive_deliberation_agent`.

Each candidate is generated without streaming, checked against the same retrieved evidence, filtered by conformal correctness when the calibrator is active, and represented by a content-free assessment. Consensus is computed over independently selected evidence IDs rather than answer wording. The agent releases the highest-confidence candidate only when two or more eligible candidates agree above the policy threshold.

High-risk unsupported claims, missing evidence, unavailable required calibration, candidate disagreement, exhausted budgets, and invalid plan integrity all fail closed.

The runtime never exposes hidden candidate drafts or chain-of-thought. Receipts contain outcome metadata, timings, token estimates, evidence-consensus scores, and SHA-256/HMAC fingerprints—not response content.

## Budgets and stopping rules

The default policy allows at most three candidates, 1,800 approximate extra tokens, and 15 seconds of cumulative candidate latency. Low-risk requests use two candidates by default. A candidate must:

1. pass claim-level grounding;
2. pass conformal filtering when calibration is enabled;
3. improve confidence by at least the configured margin;
4. agree with another eligible candidate on supporting evidence;
5. remain inside the token and latency budgets.

These controls prevent recursive agent loops and make the quality/cost/latency trade-off auditable.

## Evaluate the controller

```bash
python -m evals.adaptive_compute_evaluation \
  --output data/evaluations/adaptive-compute/latest.json \
  --min-pass-rate 1 \
  --min-recovery-rate 1 \
  --max-unsafe-release-rate 0 \
  --max-budget-violation-rate 0
```

Open the **Adaptive compute** tab in the reliability dashboard to compare the fixed one-shot baseline with the adaptive policy and inspect compute allocation per scenario.

The checked-in dataset is a deterministic control-plane drill. Its accuracy, recovery, and compute-reduction results validate orchestration logic only. A production study must use representative human-labelled queries, repeated live-model samples, actual provider token accounting, route and risk slices, and confidence intervals.

## Enable it

Adaptive compute requires grounding verification. Conformal calibration is strongly recommended.

```dotenv
GROUNDING_VERIFICATION_ENABLED=true
UNCERTAINTY_CALIBRATION_ENABLED=true
ADAPTIVE_COMPUTE_ENABLED=true
ADAPTIVE_COMPUTE_MAX_CANDIDATES=3
ADAPTIVE_COMPUTE_MAX_EXTRA_TOKENS=1800
ADAPTIVE_COMPUTE_MAX_LATENCY_MS=15000
ADAPTIVE_COMPUTE_INTEGRITY_KEY=replace-with-an-independent-secret
```

Use independent integrity keys for grounding, uncertainty, and adaptive-compute artifacts. Enabling adaptive compute increases latency and provider cost only for requests selected for deliberation.
