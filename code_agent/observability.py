"""Privacy-safe OpenTelemetry GenAI spans and trace-driven reliability evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from code_agent.models import CodeAgentResult, CodeTask, TelemetrySpanRecord, TelemetryTrace

INSTRUMENTATION_NAME = "io.agentforge.code-agent"
INSTRUMENTATION_VERSION = "1.0.0"
_WRITE_LOCK = threading.Lock()
_TRACER_LOCK = threading.Lock()
_TRACER: Any = None
_FORBIDDEN_ATTRIBUTE_MARKERS = (
    "prompt",
    "message",
    "content",
    "system_instructions",
    "tool.definitions",
)
_SECRET_VALUE = re.compile(
    r"(?i)(?:sk-|ghp-|github_pat-|AKIA[0-9A-Z]|BEGIN PRIVATE KEY|bearer\s+[A-Za-z0-9])"
)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _provider(model: str) -> str:
    return "groq" if model.startswith(("llama-", "mixtral-", "gemma-")) else "openai"


def _span_id() -> str:
    return uuid4().hex[:16]


def _trace_fingerprint(spans: list[TelemetrySpanRecord]) -> str:
    canonical = [span.model_dump(mode="json") for span in spans]
    for span in canonical:
        for field in (
            "span_id",
            "parent_span_id",
            "start_time_unix_nano",
            "end_time_unix_nano",
        ):
            span.pop(field, None)
    return _canonical_hash(canonical)


def _safe_attributes(attributes: dict[str, Any]) -> dict[str, str | int | float | bool]:
    safe: dict[str, str | int | float | bool] = {}
    for raw_key, value in attributes.items():
        key = str(raw_key).strip()[:200]
        lowered = key.lower()
        if not key or any(marker in lowered for marker in _FORBIDDEN_ATTRIBUTE_MARKERS):
            continue
        if isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, int):
            safe[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            safe[key] = value
        elif isinstance(value, str) and len(value) <= 500 and not _SECRET_VALUE.search(value):
            safe[key] = value
    return safe


def _make_span(
    *,
    name: str,
    kind: str,
    parent_span_id: str,
    start_ns: int,
    duration_ms: float,
    status: str,
    attributes: dict[str, Any],
) -> TelemetrySpanRecord:
    duration_ms = max(0.0, duration_ms)
    return TelemetrySpanRecord(
        span_id=_span_id(),
        parent_span_id=parent_span_id,
        name=name,
        kind=kind,
        start_time_unix_nano=start_ns,
        end_time_unix_nano=start_ns + int(duration_ms * 1_000_000),
        duration_ms=duration_ms,
        status=status,
        attributes=_safe_attributes(attributes),
    )


def build_code_agent_trace(
    task: CodeTask,
    result: CodeAgentResult,
    *,
    workflow: str = "single_agent",
) -> TelemetryTrace:
    """Create semantic, content-free spans from a completed coding-agent result."""
    trace_id = uuid4().hex
    now_ns = time.time_ns()
    total_ms = max(0.0, result.total_duration_ms)
    root_start = max(0, now_ns - int(total_ms * 1_000_000))
    root_id = _span_id()
    root_attributes = _safe_attributes(
        {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.provider.name": _provider(task.model),
            "gen_ai.agent.name": "AgentForge Coding Agent",
            "gen_ai.agent.version": INSTRUMENTATION_VERSION,
            "gen_ai.request.model": task.model,
            "gen_ai.usage.input_tokens": result.prompt_tokens,
            "gen_ai.usage.output_tokens": result.completion_tokens,
            "agentforge.workflow": workflow,
            "agentforge.status": result.status,
            "agentforge.iterations": result.iterations,
            "agentforge.changed_files": len(result.changed_files),
            "agentforge.estimated_cost_usd": result.estimated_cost_usd,
            "agentforge.repository_sha256": hashlib.sha256(
                task.repository.encode("utf-8")
            ).hexdigest(),
        }
    )
    spans = [
        TelemetrySpanRecord(
            span_id=root_id,
            name="invoke_agent AgentForge Coding Agent",
            kind="agent",
            start_time_unix_nano=root_start,
            end_time_unix_nano=now_ns,
            duration_ms=total_ms,
            status="ok" if result.status == "completed" else "error",
            attributes=root_attributes,
        )
    ]
    cursor = root_start
    model_duration = max(0.0, total_ms - sum(item.duration_ms for item in result.observations))
    if result.model_calls:
        model_span = _make_span(
            name=f"chat {task.model}",
            kind="model",
            parent_span_id=root_id,
            start_ns=cursor,
            duration_ms=model_duration,
            status="ok" if result.status != "failed" else "error",
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": _provider(task.model),
                "gen_ai.request.model": task.model,
                "gen_ai.usage.input_tokens": result.prompt_tokens,
                "gen_ai.usage.output_tokens": result.completion_tokens,
                "agentforge.model_calls": result.model_calls,
            },
        )
        spans.append(model_span)
        cursor = model_span.end_time_unix_nano
    for observation in result.observations:
        remaining_ms = max(0.0, (now_ns - min(cursor, now_ns)) / 1_000_000)
        span = _make_span(
            name=f"execute_tool {observation.action}",
            kind="tool",
            parent_span_id=root_id,
            start_ns=min(cursor, now_ns),
            duration_ms=min(observation.duration_ms, remaining_ms),
            status="ok" if observation.ok else "error",
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": observation.action,
                "agentforge.stage": observation.stage,
                "agentforge.iteration": observation.iteration,
                "agentforge.path_sha256": (
                    hashlib.sha256(observation.path.encode("utf-8")).hexdigest()
                    if observation.path
                    else ""
                ),
            },
        )
        spans.append(span)
        cursor = span.end_time_unix_nano
    if result.security:
        for event in result.security.events:
            spans.append(
                _make_span(
                    name="evaluate_policy agentforge-security",
                    kind="security",
                    parent_span_id=root_id,
                    start_ns=root_start,
                    duration_ms=min(event.latency_ms, total_ms),
                    status="ok" if event.allowed else "error",
                    attributes={
                        "gen_ai.operation.name": "evaluate_policy",
                        "agentforge.policy.version": event.policy_version,
                        "agentforge.policy.allowed": event.allowed,
                        "agentforge.policy.requires_approval": event.requires_human_approval,
                        "agentforge.policy.severity": event.severity,
                        "agentforge.policy.rule_count": len(event.rule_ids),
                        "agentforge.policy.decision_id": event.decision_id,
                    },
                )
            )
    return TelemetryTrace(
        trace_id=trace_id,
        generated_at=datetime.now(UTC).isoformat(),
        spans=spans,
        fingerprint=_trace_fingerprint(spans),
    )


def _otel_tracer():
    """Return an OTLP-backed tracer when configured, otherwise the API no-op tracer."""
    global _TRACER
    if _TRACER is not None:
        return _TRACER
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    enabled = os.getenv("GENAI_OTEL_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not enabled or not endpoint:
        _TRACER = trace.get_tracer(INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION)
        return _TRACER
    with _TRACER_LOCK:
        if _TRACER is not None:
            return _TRACER
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider(
                resource=Resource.create(
                    {
                        "service.name": os.getenv(
                            "OTEL_SERVICE_NAME", "agentforge-agent-service"
                        ),
                        "service.version": INSTRUMENTATION_VERSION,
                        "deployment.environment.name": os.getenv("APP_ENV", "development"),
                    }
                )
            )
            export_url = endpoint.rstrip("/")
            if not export_url.endswith("/v1/traces"):
                export_url += "/v1/traces"
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=export_url))
            )
            _TRACER = provider.get_tracer(INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION)
        except (ImportError, ValueError):
            _TRACER = trace.get_tracer(INSTRUMENTATION_NAME, INSTRUMENTATION_VERSION)
        return _TRACER


def emit_otel_trace(trace_record: TelemetryTrace) -> None:
    tracer = _otel_tracer()
    if tracer is None or not trace_record.spans:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode
    except ImportError:
        return
    root_record = trace_record.spans[0]
    root = tracer.start_span(
        root_record.name,
        start_time=root_record.start_time_unix_nano,
        attributes=root_record.attributes,
    )
    root_context = trace.set_span_in_context(root)
    try:
        for item in trace_record.spans[1:]:
            span = tracer.start_span(
                item.name,
                context=root_context,
                start_time=item.start_time_unix_nano,
                attributes=item.attributes,
            )
            if item.status == "error":
                span.set_status(Status(StatusCode.ERROR))
            span.end(end_time=item.end_time_unix_nano)
        if root_record.status == "error":
            root.set_status(Status(StatusCode.ERROR))
    finally:
        root.end(end_time=root_record.end_time_unix_nano)


def append_local_trace(trace_record: TelemetryTrace, path: Path | None = None) -> None:
    configured = os.getenv("GENAI_TRACE_JSONL_PATH", "").strip()
    target = path or (Path(configured) if configured else None)
    if target is None:
        return
    payload = trace_record.model_dump_json() + "\n"
    max_bytes = max(100_000, int(os.getenv("GENAI_TRACE_MAX_BYTES", "10000000")))
    target.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        if target.exists() and target.stat().st_size + len(payload.encode("utf-8")) > max_bytes:
            rotated = target.with_suffix(target.suffix + ".1")
            if rotated.exists():
                rotated.unlink()
            target.replace(rotated)
        with target.open("a", encoding="utf-8", newline="") as stream:
            stream.write(payload)


def capture_code_agent_trace(
    task: CodeTask,
    result: CodeAgentResult,
    *,
    workflow: str = "single_agent",
) -> TelemetryTrace:
    trace_record = build_code_agent_trace(task, result, workflow=workflow)
    emit_otel_trace(trace_record)
    append_local_trace(trace_record)
    return trace_record


def capture_research_agent_trace(
    *,
    run_id: str,
    model: str,
    state: dict[str, Any],
    outcome: str,
    duration_ms: float,
    stream: bool = False,
) -> TelemetryTrace:
    """Export an existing LangGraph node trace without prompts, answers, or evidence."""
    normalized_run_id = re.sub(r"[^0-9a-f]", "", run_id.lower())
    trace_id = (
        normalized_run_id
        if len(normalized_run_id) == 32
        else hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:32]
    )
    now_ns = time.time_ns()
    duration_ms = max(0.0, duration_ms)
    root_start = max(0, now_ns - int(duration_ms * 1_000_000))
    root_id = _span_id()
    steps = state.get("agent_trace_steps") or []
    root = TelemetrySpanRecord(
        span_id=root_id,
        name="invoke_agent AgentForge Research Agent",
        kind="agent",
        start_time_unix_nano=root_start,
        end_time_unix_nano=now_ns,
        duration_ms=duration_ms,
        status="error" if outcome == "error" else "ok",
        attributes=_safe_attributes(
            {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.provider.name": _provider(model),
                "gen_ai.agent.name": "AgentForge Research Agent",
                "gen_ai.agent.version": INSTRUMENTATION_VERSION,
                "gen_ai.request.model": model,
                "agentforge.run.id": run_id,
                "agentforge.route": str(state.get("route") or "unknown"),
                "agentforge.outcome": outcome,
                "agentforge.stream": stream,
                "agentforge.safety_blocked": bool(state.get("safety_blocked")),
                "agentforge.evaluation_score": int(state.get("evaluation_score") or 0),
            }
        ),
    )
    spans = [root]
    cursor = root_start
    for raw_step in steps[:100]:
        if not isinstance(raw_step, dict):
            continue
        agent_name = str(raw_step.get("agent") or "unknown")[:100]
        step_duration = max(0.0, float(raw_step.get("latency_ms") or 0.0))
        remaining_ms = max(0.0, (now_ns - min(cursor, now_ns)) / 1_000_000)
        child = _make_span(
            name=f"invoke_agent {agent_name}",
            kind="agent",
            parent_span_id=root_id,
            start_ns=min(cursor, now_ns),
            duration_ms=min(step_duration, remaining_ms),
            status="ok",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.provider.name": _provider(model),
                "gen_ai.agent.name": agent_name,
                "gen_ai.agent.version": INSTRUMENTATION_VERSION,
                "gen_ai.request.model": model,
                "agentforge.step": int(raw_step.get("step") or len(spans)),
            },
        )
        spans.append(child)
        cursor = child.end_time_unix_nano
    trace_record = TelemetryTrace(
        trace_id=trace_id,
        generated_at=datetime.now(UTC).isoformat(),
        service_name="agentforge-research-agent",
        spans=spans,
        fingerprint=_trace_fingerprint(spans),
    )
    emit_otel_trace(trace_record)
    append_local_trace(trace_record)
    return trace_record


class TelemetryEvalReport(BaseModel):
    schema_version: str = "1.0"
    dataset_path: str
    dataset_sha256: str
    traces: int
    spans: int
    valid_trace_rate: float
    fingerprint_violations: int
    hierarchy_violations: int
    error_span_rate: float
    p50_agent_duration_ms: float
    p95_agent_duration_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: float
    tool_spans: int
    security_spans: int
    content_attribute_violations: int
    secret_value_violations: int
    semantic_attribute_coverage: float


def load_traces(path: Path) -> list[TelemetryTrace]:
    traces = [
        TelemetryTrace.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not traces:
        raise ValueError("telemetry trace dataset is empty")
    return traces


def _valid_hierarchy(trace_record: TelemetryTrace) -> bool:
    by_id = {span.span_id: span for span in trace_record.spans}
    if len(by_id) != len(trace_record.spans):
        return False
    roots = [span for span in trace_record.spans if not span.parent_span_id]
    if len(roots) != 1 or roots[0].kind != "agent":
        return False
    root = roots[0]
    for span in trace_record.spans:
        if span.start_time_unix_nano < root.start_time_unix_nano:
            return False
        if span.end_time_unix_nano > root.end_time_unix_nano:
            return False
        seen: set[str] = set()
        current = span
        while current.parent_span_id:
            if current.span_id in seen or current.parent_span_id not in by_id:
                return False
            seen.add(current.span_id)
            current = by_id[current.parent_span_id]
        if current.span_id != root.span_id:
            return False
    return True


def evaluate_traces(traces: list[TelemetryTrace], dataset_path: str = "") -> TelemetryEvalReport:
    spans = [span for item in traces for span in item.spans]
    roots = [span for span in spans if span.kind == "agent"]
    fingerprint_violations = sum(
        item.fingerprint != _trace_fingerprint(item.spans) for item in traces
    )
    hierarchy_violations = sum(not _valid_hierarchy(item) for item in traces)
    valid = [
        item
        for item in traces
        if len(item.trace_id) == 32
        and all(span.end_time_unix_nano >= span.start_time_unix_nano for span in item.spans)
        and item.fingerprint == _trace_fingerprint(item.spans)
        and _valid_hierarchy(item)
    ]
    content_violations = sum(
        any(marker in key.lower() for marker in _FORBIDDEN_ATTRIBUTE_MARKERS)
        for span in spans
        for key in span.attributes
    )
    secret_violations = sum(
        bool(_SECRET_VALUE.search(str(value)))
        for span in spans
        for value in span.attributes.values()
    )
    required = ("gen_ai.operation.name",)
    semantic_covered = sum(
        all(key in span.attributes for key in required)
        for span in spans
    )
    input_tokens = sum(
        int(span.attributes.get("gen_ai.usage.input_tokens", 0))
        for span in roots
    )
    output_tokens = sum(
        int(span.attributes.get("gen_ai.usage.output_tokens", 0))
        for span in roots
    )
    costs = sum(
        float(span.attributes.get("agentforge.estimated_cost_usd", 0.0))
        for span in roots
    )
    durations = [span.duration_ms for span in roots]
    return TelemetryEvalReport(
        dataset_path=dataset_path,
        dataset_sha256=_canonical_hash(
            [item.model_dump(mode="json", exclude={"generated_at"}) for item in traces]
        ),
        traces=len(traces),
        spans=len(spans),
        valid_trace_rate=len(valid) / len(traces),
        fingerprint_violations=fingerprint_violations,
        hierarchy_violations=hierarchy_violations,
        error_span_rate=sum(span.status == "error" for span in spans) / len(spans)
        if spans
        else 0.0,
        p50_agent_duration_ms=statistics.median(durations) if durations else 0.0,
        p95_agent_duration_ms=_percentile(durations, 0.95),
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_estimated_cost_usd=costs,
        tool_spans=sum(span.kind == "tool" for span in spans),
        security_spans=sum(span.kind == "security" for span in spans),
        content_attribute_violations=content_violations,
        secret_value_violations=secret_violations,
        semantic_attribute_coverage=semantic_covered / len(spans) if spans else 0.0,
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate privacy-safe GenAI telemetry traces")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-valid-rate", type=float, default=1.0)
    parser.add_argument("--min-semantic-coverage", type=float, default=1.0)
    parser.add_argument("--allow-content-attributes", action="store_true")
    args = parser.parse_args()
    report = evaluate_traces(load_traces(args.dataset), args.dataset.as_posix())
    payload = report.model_dump_json(indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    failed = (
        report.valid_trace_rate < max(0.0, min(args.min_valid_rate, 1.0))
        or report.semantic_attribute_coverage
        < max(0.0, min(args.min_semantic_coverage, 1.0))
        or report.secret_value_violations > 0
        or (report.content_attribute_violations > 0 and not args.allow_content_attributes)
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
