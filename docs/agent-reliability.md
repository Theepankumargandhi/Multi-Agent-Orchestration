# Agent reliability and failure-injection lab

## Purpose

The reliability lab verifies that AgentForge recovery controls behave predictably before those controls are tested against live providers or production infrastructure. It deliberately injects dependency outcomes into a reusable state machine and compares a one-attempt baseline with the configured resilient policy on the same scenarios.

This is deterministic control-flow evidence. Its latency, availability, and cost numbers are simulated from the checked-in plans and must not be presented as production measurements.

## Scenario coverage

The versioned JSONL corpus contains three healthy controls and 12 fault scenarios across:

- model rate limiting, timeout, malformed structured output, and permanent authentication failure;
- unavailable and empty-result retrieval;
- idempotent tool timeout;
- checkpoint unavailability and stream disconnection;
- worker crash and lease recovery;
- context-window overflow; and
- security-policy denial.

Recovery actions include bounded retry, fallback-model selection, structured-output repair, alternate retrieval, checkpoint/stream resume, worker lease recovery, context compression, graceful degradation, and fail-closed containment.

## Run and gate

```bash
python -m evals.reliability \
  evals/datasets/agent_reliability_scenarios.jsonl \
  --output data/evaluations/reliability/latest.json \
  --min-pass-rate 1 \
  --min-recovery-rate 1 \
  --max-p95-ms 250
```

The command exits non-zero if the report or event fingerprints fail verification, a scenario violates its expected outcome or SLO, recovery drops below the configured threshold, or simulated P95 exceeds its gate.

Each scenario records the exact primary, control, and fallback event sequence. Reports include:

- baseline and resilient availability;
- recovery success and safe-containment rates;
- per-scenario SLO compliance;
- attempts, P50/P95 simulated duration, tokens, and cost;
- incremental recovery latency, tokens, and cost; and
- dataset, policy, event-trace, and whole-report fingerprints.

## Reliability command center

Generate the reliability, telemetry, security, and retrieval reports, then launch:

```bash
docker compose --profile evaluation up --build reliability_dashboard
```

Open `http://localhost:8506`. The read-only dashboard provides:

- a release-readiness overview over every available evidence artifact;
- baseline-versus-resilient recovery charts;
- component filtering and scenario-level incident summaries;
- baseline and recovery event replay with integrity receipts;
- an OpenTelemetry parent-child graph, span waterfall, and safe-attribute inspector; and
- consolidated security, retrieval, and benchmark evidence.

The container runs as an unprivileged user with a read-only filesystem, dropped capabilities, `no-new-privileges`, a bounded temporary filesystem, and read-only evidence mounts.

## Extending to real chaos testing

The scripted dependency implements the same invoke/failure boundary used by the evaluator, so additional adapters can translate staging-only failures into its typed fault vocabulary. The next evidence level should run these cases against disposable infrastructure using provider test accounts, network proxies, killed workers, and temporary database outages. Keep those measurements in a separately labelled `staging_chaos` dataset; do not merge them with deterministic CI results.

Useful staging targets are recovery rate under repeated faults, tail latency, retry amplification, duplicate side effects, checkpoint correctness, token/cost amplification, and safety behavior during partial failure.
