# GenAI observability

AgentForge emits one privacy-safe trace for every research-agent invocation and every completed coding-agent workflow. The schema follows the OpenTelemetry GenAI semantic-convention vocabulary for agent and model operations while keeping the convention's potentially sensitive content fields disabled.

## What is captured

- Parent/child agent spans, model/provider identity, workflow outcome, route, latency, token use, and estimated model cost.
- Coding tool action names, iteration and stage; repository and file paths are represented only by SHA-256 digests.
- Security-policy decisions, rule counts, severity, approvals, and policy version.
- Research graph node names and timings already produced by the LangGraph workflow.

Prompts, responses, retrieved evidence, source code, tool output, system instructions, tool definitions, and credentials are never added to span attributes. Attribute keys associated with content are rejected, and secret-looking string values are dropped as a second line of defense. `content_capture_enabled` remains `false` in every local trace artifact.

## Local evaluation

Enable the bounded JSONL sink for a development or evaluation run:

```bash
GENAI_TRACE_JSONL_PATH=data/evaluations/telemetry/traces.jsonl
GENAI_TRACE_MAX_BYTES=10000000
```

Then evaluate the resulting dataset:

```bash
python -m code_agent.observability data/evaluations/telemetry/traces.jsonl \
  --output data/evaluations/telemetry/report.json \
  --min-valid-rate 1.0 \
  --min-semantic-coverage 1.0
```

The command exits non-zero for malformed or fingerprint-mismatched traces, broken parent-child hierarchies, missing semantic attributes, content-bearing attributes, or secret-looking values. Its report also exposes error-span rate, p50/p95 agent latency, tokens, cost, and tool/security span counts. JSONL rotation keeps the active dataset within `GENAI_TRACE_MAX_BYTES`; one previous segment is retained as `.1`.

## OTLP export

Point the service at any OTLP/HTTP-compatible collector:

```bash
GENAI_OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
OTEL_SERVICE_NAME=agentforge-agent-service
```

The exporter accepts either a collector base URL or a URL ending in `/v1/traces`. Export is disabled by default. An exporter/configuration failure never fails an agent request, and exception messages are not logged because they can contain endpoint credentials.

The implementation targets the current [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai). These conventions are still evolving, so the internal versioned trace model isolates dashboards and evaluation artifacts from upstream naming changes.

## Recruiter-facing demo

Run the API, execute one research request and one sandboxed coding task, then load the local JSONL file into the evaluator. Show the parent/child trace hierarchy, per-workflow latency and cost, the security-policy spans, and the zero-leakage gates. This demonstrates the production concern that is usually missing from portfolio agents: proving how an agent behaved without storing the user's private context.
