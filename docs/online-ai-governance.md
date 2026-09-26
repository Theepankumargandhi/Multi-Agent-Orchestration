# Online AI quality governance

AgentForge now connects inference decisions to a privacy-safe online evaluation controller. The gateway can emit content-free operational events, delayed human or asynchronous grader feedback can be joined later, and rolling release policy can promote, hold, or roll back a canary using quality, safety, latency, and cost evidence.

## Why this exists

Offline benchmarks are necessary but cannot detect a provider regression, traffic shift, safety spike, or cost/latency degradation after deployment. This layer provides the control mechanics needed to turn production signals into an auditable release decision without putting prompts or model responses into monitoring artifacts.

## Implemented controls

- Strict content-free event and delayed-feedback schemas with unknown fields rejected.
- Idempotent ingestion. An identical redelivery is counted once; a conflicting event with the same ID fails closed.
- Optional keyed event and feedback integrity fingerprints.
- Tenant-safe delayed feedback joins that reseal the enriched event.
- Bounded retention and bounded JSONL export with single-generation rotation.
- Short and long rolling windows for quality pass rate, safety rate, P95 latency, average cost, success, cache-hit, and fallback rates.
- Multi-window quality and safety error-budget burn alerts.
- Provider-mix drift using Jensen-Shannon divergence.
- A 95% quality-delta confidence interval and explicit non-inferiority margin.
- Canary promote, hold, or rollback decisions constrained by minimum sample count, quality, safety, latency ratio, cost ratio, and critical SLO alerts.
- Integrity-bound decision and report receipts.

## Runtime feed

Enable the inference gateway and set an event path:

```env
MODEL_GATEWAY_ENABLED=true
MODEL_GATEWAY_FINGERPRINT_KEY=replace-with-at-least-16-random-characters
MODEL_GATEWAY_ONLINE_EVENT_PATH=data/evaluations/online-monitor/events.jsonl
ONLINE_EVAL_INTEGRITY_KEY=replace-with-a-separate-16-character-minimum-secret
```

After each successful controlled inference, the runtime writes provider, cohort, outcome, usage, cost, latency, cache/fallback state, high-risk state, and keyed fingerprints. It never writes the system prompt, user prompt, model response, evidence, API key, or selected model identifier. A live export requires a dedicated integrity key of at least 16 bytes. Export failure does not fail user inference; the gateway receipt remains attached to request state so a durable exporter can retry.

Operational gateway events initially have no quality or safety label. Those fields are intentionally nullable because trustworthy online evaluation often arrives later from human feedback, a calibrated grader, an outcome event, or a sampled audit. `DelayedFeedback` joins the label by event ID and tenant fingerprint.

## Credential-free incident drill

Run the checked-in stream through the same release logic:

```bash
python -m evals.online_monitor_evaluation \
  --output data/evaluations/online-monitor/latest.json \
  --require-decision rollback \
  --min-integrity-rate 1 \
  --max-privacy-violations 0 \
  --require-alert
```

The drill currently accepts 20 unique events, idempotently suppresses one duplicate delivery, joins two delayed labels, verifies 100% of event receipts, finds zero prohibited content fields, raises six multi-window alerts, and rolls back the degraded canary. Control mean quality is `0.901`; canary mean quality is `0.595`, with a 95% delta interval of approximately `[-0.328, -0.284]`. Canary P95 latency is `780 ms` versus `149 ms` for control, and its average cost ratio is about `3.06x`.

These are synthetic incident-response results. They prove the ingestion, statistical, SLO, integrity, and release-control paths; they are not evidence of real traffic volume, production quality, uptime, or financial savings.

## Dashboard and CI

The **Online AI monitor** tab in the AgentOps command center shows rolling windows, error-budget alerts, provider drift, evidence fingerprints, and the statistical canary decision. CI requires the incident drill to detect the degradation and uploads `online-ai-canary-governance` as reviewable evidence.

## Production follow-up

For a multi-replica deployment, replace the JSONL reference feed and in-process store with Kafka or another durable event bus, a schema registry, stream processing with event-time watermarks, an atomic analytical store, and an alert/rollout integration. Calibrate thresholds from real traffic, segment by route and risk tier, correct for repeated testing, and require minimum human-label coverage before autonomous promotion. Automatic rollback should be connected only after staged failure drills and an operator-approved policy.
