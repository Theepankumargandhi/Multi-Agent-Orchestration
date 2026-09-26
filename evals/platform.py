"""Reproducible experiment runner and deterministic graders for agent systems."""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator

ROUTES = {"clarify", "rewrite", "math", "web", "rag", "kg", "hybrid", "general"}
CITATION_RE = re.compile(r"\[[^\]]+\]\((https?://[^\s)]+)\)", re.IGNORECASE)


class ExpectedBehavior(BaseModel):
    route: str | None = None
    answer_contains: list[str] = Field(default_factory=list)
    answer_excludes: list[str] = Field(default_factory=list)
    min_citations: int | None = Field(default=None, ge=0)
    safety_blocked: bool | None = None
    tool_calls: list[str] | None = None
    max_latency_ms: float | None = Field(default=None, gt=0)

    @field_validator("route")
    @classmethod
    def valid_route(cls, value: str | None) -> str | None:
        if value is not None and value not in ROUTES:
            raise ValueError(f"unsupported route: {value}")
        return value


class EvalCase(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    input: str = Field(min_length=1, max_length=20000)
    expected: ExpectedBehavior
    tags: list[str] = Field(default_factory=list)
    split: Literal["train", "validation", "test"] = "test"
    review_status: Literal["synthetic_seed", "human_reviewed"] = "synthetic_seed"
    metadata: dict[str, Any] = Field(default_factory=dict)


class VariantConfig(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    adapter: Literal["keyword-baseline", "research-router", "live-graph", "openai-compatible"]
    model: str = "offline-eval"
    parameters: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    dataset: str
    variants: list[VariantConfig] = Field(min_length=1)
    metric_weights: dict[str, float] = Field(default_factory=dict)
    pass_threshold: float = Field(default=0.8, ge=0, le=1)
    concurrency: int = Field(default=4, ge=1, le=32)
    splits: list[Literal["train", "validation", "test"]] = Field(default_factory=lambda: ["test"])
    bootstrap_samples: int = Field(default=1000, ge=0, le=10000)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)


class AgentRun(BaseModel):
    answer: str = ""
    route: str = "unknown"
    safety_blocked: bool = False
    tool_calls: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    trace: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class MetricResult(BaseModel):
    score: float = Field(ge=0, le=1)
    passed: bool
    detail: str


class CaseResult(BaseModel):
    case_id: str
    input: str
    expected: ExpectedBehavior
    actual: AgentRun
    metrics: dict[str, MetricResult]
    quality_score: float = Field(ge=0, le=1)
    passed: bool
    tags: list[str] = Field(default_factory=list)
    split: str = "test"
    review_status: str = "synthetic_seed"


class VariantReport(BaseModel):
    variant: VariantConfig
    quality_score: float
    pass_rate: float
    metric_scores: dict[str, float]
    p50_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    total_tokens: int
    failure_categories: dict[str, int]
    quality_confidence_interval: list[float] = Field(
        default_factory=lambda: [0.0, 0.0], min_length=2, max_length=2
    )
    slice_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    quality_delta_vs_baseline: float = 0.0
    cost_delta_vs_baseline: float = 0.0
    p95_latency_delta_vs_baseline: float = 0.0
    pareto_optimal: bool = False
    cases: list[CaseResult]


class ExperimentReport(BaseModel):
    schema_version: str = "1.0"
    experiment_id: str
    experiment_name: str
    created_at: str
    dataset_path: str
    dataset_fingerprint: str
    winner: str
    pass_threshold: float
    evaluated_splits: list[str] = Field(default_factory=lambda: ["test"])
    review_status_counts: dict[str, int] = Field(default_factory=dict)
    reports: list[VariantReport]


class AgentAdapter(Protocol):
    async def run(self, case: EvalCase, variant: VariantConfig) -> AgentRun: ...


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    normalized_inputs: dict[str, tuple[str, str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = EvalCase.model_validate_json(line)
        if case.id in seen:
            raise ValueError(f"duplicate case id {case.id!r} at line {line_number}")
        normalized = " ".join(case.input.casefold().split())
        prior = normalized_inputs.get(normalized)
        if prior and prior[1] != case.split:
            raise ValueError(
                f"data leakage: equivalent inputs {prior[0]!r} and {case.id!r} occur across splits"
            )
        seen.add(case.id)
        normalized_inputs[normalized] = (case.id, case.split)
        cases.append(case)
    if not cases:
        raise ValueError(f"dataset contains no cases: {path}")
    return cases


def _keyword_route(query: str) -> str:
    """Deliberately simple baseline used to make routing ablations meaningful."""
    text = query.lower()
    clean = text.strip(" .!?\t\r\n")
    if clean in {"hello", "hi", "hey", "thanks", "thank you"} or clean.startswith(
        ("hello ", "hi ", "hey ")
    ):
        return "general"
    if clean in {"what about this", "what about that", "do it", "help"}:
        return "clarify"
    if re.search(r"\b(calculate|compute|solve|equation|percent)\b|\d\s*[-+*/]\s*\d", text):
        return "math"
    web = bool(re.search(r"\b(latest|today|current|news|recent|weather|price)\b", text))
    local = bool(re.search(r"\b(repository|codebase|project|document|pdf|local)\b", text))
    relationship = bool(re.search(r"\b(connect|depend|relationship|flow|call|architecture)\w*\b", text))
    if web and local:
        return "hybrid"
    if relationship and local:
        return "kg"
    if web:
        return "web"
    if local:
        return "rag"
    if len(text.split()) <= 2:
        return "clarify"
    return "general"


class KeywordBaselineAdapter:
    async def run(self, case: EvalCase, variant: VariantConfig) -> AgentRun:
        started = time.perf_counter()
        route = _keyword_route(case.input)
        latency = (time.perf_counter() - started) * 1000
        return AgentRun(
            route=route,
            latency_ms=latency,
            trace=[{"step": 1, "agent": "keyword_baseline", "latency_ms": latency}],
        )


class ResearchRouterAdapter:
    async def run(self, case: EvalCase, variant: VariantConfig) -> AgentRun:
        from agent.research_assistant import intent_router_agent

        started = time.perf_counter()
        decision = await intent_router_agent(
            {"messages": [HumanMessage(content=case.input)], "query": case.input},
            {
                "configurable": {
                    "model": variant.model,
                    "evaluation_mode": True,
                    "router_instruction_suffix": str(
                        variant.parameters.get("router_instruction_suffix", "")
                    ),
                }
            },
        )
        latency = (time.perf_counter() - started) * 1000
        return AgentRun(
            route=str(decision.get("route") or "unknown"),
            latency_ms=latency,
            trace=[
                {
                    "step": 1,
                    "agent": "intent_router_agent",
                    "latency_ms": latency,
                    "confidence": decision.get("route_confidence"),
                    "reason": decision.get("route_reason"),
                }
            ],
        )


class LiveGraphAdapter:
    """Runs the complete graph. Provider credentials may be required."""

    def __init__(self):
        self._graph = None

    async def run(self, case: EvalCase, variant: VariantConfig) -> AgentRun:
        from agent.research_assistant import build_research_assistant

        started = time.perf_counter()
        if self._graph is None:
            self._graph = build_research_assistant()
        graph = self._graph
        usage_callback = UsageMetadataCallbackHandler()
        config = {
            "configurable": {
                "model": variant.model,
                "thread_id": f"eval-{uuid4()}",
                "evaluation_bypass_hitl": bool(variant.parameters.get("bypass_hitl", True)),
                "evaluation_mode": True,
                "router_instruction_suffix": str(
                    variant.parameters.get("router_instruction_suffix", "")
                ),
                "response_instruction_suffix": str(
                    variant.parameters.get("response_instruction_suffix", "")
                ),
            },
            "callbacks": [usage_callback],
        }
        try:
            state = await graph.ainvoke({"messages": [HumanMessage(content=case.input)]}, config=config)
        except Exception as exc:
            return AgentRun(latency_ms=(time.perf_counter() - started) * 1000, error=type(exc).__name__)
        latency = (time.perf_counter() - started) * 1000
        messages = state.get("messages") or []
        answer = str(getattr(messages[-1], "content", "")) if messages else ""
        usage_by_model = usage_callback.usage_metadata
        prompt_tokens = sum(int(item.get("input_tokens") or 0) for item in usage_by_model.values())
        completion_tokens = sum(int(item.get("output_tokens") or 0) for item in usage_by_model.values())
        trace = list(state.get("agent_trace_steps") or [])
        tool_agents = {
            "web_search_agent",
            "knowledge_graph_agent",
            "rag_agent",
            "math_agent",
        }
        tool_calls = [
            str(step.get("agent")) for step in trace if str(step.get("agent")) in tool_agents
        ]
        input_price = float(variant.parameters.get("input_cost_per_million", 0.0))
        output_price = float(variant.parameters.get("output_cost_per_million", 0.0))
        estimated_cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
        return AgentRun(
            answer=answer,
            route=str(state.get("route") or "unknown"),
            safety_blocked=bool(state.get("safety_blocked")),
            tool_calls=tool_calls,
            citations=CITATION_RE.findall(answer),
            latency_ms=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost,
            trace=trace,
        )


class OpenAICompatibleAdapter:
    """Evaluate a PEFT adapter served by vLLM, llama.cpp, or another compatible runtime."""

    async def run(self, case: EvalCase, variant: VariantConfig) -> AgentRun:
        from post_training.inference import generate

        started = time.perf_counter()
        parameters = variant.parameters
        try:
            result = await generate(
                base_url=str(parameters.get("base_url", "http://127.0.0.1:8001/v1")),
                model=variant.model,
                prompt=case.input,
                api_key_env=str(parameters.get("api_key_env", "LOCAL_MODEL_API_KEY")),
                timeout_seconds=float(parameters.get("timeout_seconds", 120)),
                max_tokens=int(parameters.get("max_tokens", 512)),
                temperature=float(parameters.get("temperature", 0.0)),
                allow_remote=bool(parameters.get("allow_remote", False)),
            )
        except Exception as exc:
            return AgentRun(
                latency_ms=(time.perf_counter() - started) * 1000,
                error=type(exc).__name__,
            )
        latency = (time.perf_counter() - started) * 1000
        input_price = float(parameters.get("input_cost_per_million", 0.0))
        output_price = float(parameters.get("output_cost_per_million", 0.0))
        estimated_cost = (
            result.prompt_tokens * input_price + result.completion_tokens * output_price
        ) / 1_000_000
        reported_route = str(parameters.get("reported_route", "unknown"))
        if reported_route == "unknown":
            try:
                structured_answer = json.loads(result.answer)
                candidate_route = str(structured_answer.get("route") or "unknown")
                if candidate_route in ROUTES:
                    reported_route = candidate_route
            except (json.JSONDecodeError, AttributeError):
                pass
        return AgentRun(
            answer=result.answer,
            route=reported_route,
            tool_calls=list(result.tool_calls),
            citations=CITATION_RE.findall(result.answer),
            latency_ms=latency,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            estimated_cost_usd=estimated_cost,
            trace=[
                {
                    "step": 1,
                    "agent": "openai_compatible_local_model",
                    "latency_ms": latency,
                    "model": result.model,
                }
            ],
        )


ADAPTERS: dict[str, AgentAdapter] = {
    "keyword-baseline": KeywordBaselineAdapter(),
    "research-router": ResearchRouterAdapter(),
    "live-graph": LiveGraphAdapter(),
    "openai-compatible": OpenAICompatibleAdapter(),
}


def _tool_f1(expected: list[str], actual: list[str]) -> float:
    expected_counts, actual_counts = Counter(expected), Counter(actual)
    overlap = sum((expected_counts & actual_counts).values())
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    precision = overlap / sum(actual_counts.values())
    recall = overlap / sum(expected_counts.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def grade_case(case: EvalCase, run: AgentRun, weights: dict[str, float]) -> CaseResult:
    expected = case.expected
    metrics: dict[str, MetricResult] = {}
    if expected.route is not None:
        score = float(run.route == expected.route)
        metrics["route_accuracy"] = MetricResult(
            score=score, passed=bool(score), detail=f"expected={expected.route}, actual={run.route}"
        )
    if expected.answer_contains:
        answer = run.answer.casefold()
        hits = sum(term.casefold() in answer for term in expected.answer_contains)
        score = hits / len(expected.answer_contains)
        metrics["required_term_recall"] = MetricResult(
            score=score,
            passed=math.isclose(score, 1.0),
            detail=f"matched={hits}/{len(expected.answer_contains)}",
        )
    if expected.answer_excludes:
        violations = [term for term in expected.answer_excludes if term.casefold() in run.answer.casefold()]
        score = float(not violations)
        metrics["forbidden_term_safety"] = MetricResult(
            score=score, passed=bool(score), detail=f"violations={violations or 'none'}"
        )
    if expected.min_citations is not None:
        citations = run.citations or CITATION_RE.findall(run.answer)
        score = min(1.0, len(citations) / max(1, expected.min_citations)) if expected.min_citations else 1.0
        metrics["citation_coverage"] = MetricResult(
            score=score,
            passed=len(citations) >= expected.min_citations,
            detail=f"citations={len(citations)}, minimum={expected.min_citations}",
        )
    if expected.safety_blocked is not None:
        score = float(run.safety_blocked == expected.safety_blocked)
        metrics["safety_accuracy"] = MetricResult(
            score=score,
            passed=bool(score),
            detail=f"expected={expected.safety_blocked}, actual={run.safety_blocked}",
        )
    if expected.tool_calls is not None:
        score = _tool_f1(expected.tool_calls, run.tool_calls)
        metrics["tool_call_f1"] = MetricResult(
            score=score,
            passed=math.isclose(score, 1.0),
            detail=f"expected={expected.tool_calls}, actual={run.tool_calls}",
        )
    if expected.max_latency_ms is not None:
        score = min(1.0, expected.max_latency_ms / max(run.latency_ms, expected.max_latency_ms))
        metrics["latency_slo"] = MetricResult(
            score=score,
            passed=run.latency_ms <= expected.max_latency_ms,
            detail=f"actual={run.latency_ms:.2f}ms, maximum={expected.max_latency_ms:.2f}ms",
        )
    if run.error:
        metrics["execution_success"] = MetricResult(score=0, passed=False, detail=f"error={run.error}")

    weighted = [(metric.score, max(0.0, weights.get(name, 1.0))) for name, metric in metrics.items()]
    denominator = sum(weight for _, weight in weighted)
    quality_score = sum(score * weight for score, weight in weighted) / denominator if denominator else 0.0
    if run.error:
        quality_score = 0.0
    return CaseResult(
        case_id=case.id,
        input=case.input,
        expected=expected,
        actual=run,
        metrics=metrics,
        quality_score=quality_score,
        passed=all(metric.passed for metric in metrics.values()),
        tags=case.tags,
        split=case.split,
        review_status=case.review_status,
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def _bootstrap_mean_interval(
    values: list[float], samples: int, confidence_level: float, seed: int = 20260906
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if samples <= 0 or len(values) == 1:
        mean = statistics.fmean(values)
        return [mean, mean]
    rng = random.Random(seed)
    bootstrapped = [
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(samples)
    ]
    tail = (1.0 - confidence_level) / 2.0
    return [_percentile(bootstrapped, tail), _percentile(bootstrapped, 1.0 - tail)]


def _failure_category(case: CaseResult) -> list[str]:
    categories = []
    mapping = {
        "route_accuracy": "routing_failure",
        "required_term_recall": "answer_grounding_failure",
        "forbidden_term_safety": "safety_failure",
        "citation_coverage": "citation_failure",
        "safety_accuracy": "safety_failure",
        "tool_call_f1": "tool_selection_failure",
        "latency_slo": "latency_failure",
        "execution_success": "execution_failure",
    }
    for name, metric in case.metrics.items():
        if not metric.passed:
            categories.append(mapping.get(name, "quality_failure"))
    return categories


async def _run_variant(
    variant: VariantConfig,
    cases: list[EvalCase],
    weights: dict[str, float],
    concurrency: int,
    bootstrap_samples: int,
    confidence_level: float,
) -> VariantReport:
    adapter = ADAPTERS[variant.adapter]
    semaphore = asyncio.Semaphore(concurrency)

    async def execute(case: EvalCase) -> CaseResult:
        async with semaphore:
            try:
                run = await adapter.run(case, variant)
            except Exception as exc:
                run = AgentRun(error=type(exc).__name__)
            return grade_case(case, run, weights)

    results = await asyncio.gather(*(execute(case) for case in cases))
    latencies = [case.actual.latency_ms for case in results]
    metric_names = sorted({name for case in results for name in case.metrics})
    metric_scores = {
        name: statistics.fmean(case.metrics[name].score for case in results if name in case.metrics)
        for name in metric_names
    }
    quality_score = statistics.fmean(case.quality_score for case in results)
    failures = Counter(category for case in results for category in _failure_category(case))
    tagged: defaultdict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        for tag in result.tags:
            tagged[tag].append(result)
    slices = {
        tag: {
            "quality_score": statistics.fmean(item.quality_score for item in items),
            "pass_rate": statistics.fmean(float(item.passed) for item in items),
            "count": float(len(items)),
        }
        for tag, items in sorted(tagged.items())
    }
    return VariantReport(
        variant=variant,
        quality_score=quality_score,
        pass_rate=sum(case.passed for case in results) / len(results),
        metric_scores=metric_scores,
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=_percentile(latencies, 0.95),
        total_cost_usd=sum(case.actual.estimated_cost_usd for case in results),
        total_tokens=sum(case.actual.prompt_tokens + case.actual.completion_tokens for case in results),
        failure_categories=dict(failures),
        quality_confidence_interval=_bootstrap_mean_interval(
            [case.quality_score for case in results], bootstrap_samples, confidence_level
        ),
        slice_scores=slices,
        cases=results,
    )


def _mark_pareto(reports: list[VariantReport]) -> None:
    for candidate in reports:
        candidate.pareto_optimal = not any(
            other.variant.name != candidate.variant.name
            and other.quality_score >= candidate.quality_score
            and other.total_cost_usd <= candidate.total_cost_usd
            and other.p95_latency_ms <= candidate.p95_latency_ms
            and (
                other.quality_score > candidate.quality_score
                or other.total_cost_usd < candidate.total_cost_usd
                or other.p95_latency_ms < candidate.p95_latency_ms
            )
            for other in reports
        )


async def run_experiment(config: ExperimentConfig, base_dir: Path | None = None) -> ExperimentReport:
    root = base_dir or Path.cwd()
    dataset_path = Path(config.dataset)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    all_cases = load_cases(dataset_path)
    cases = [case for case in all_cases if case.split in config.splits]
    if not cases:
        raise ValueError(f"dataset has no cases in configured splits: {config.splits}")
    reports = []
    for variant in config.variants:
        reports.append(
            await _run_variant(
                variant,
                cases,
                config.metric_weights,
                config.concurrency,
                config.bootstrap_samples,
                config.confidence_level,
            )
        )
    baseline = reports[0]
    for report in reports:
        report.quality_delta_vs_baseline = report.quality_score - baseline.quality_score
        report.cost_delta_vs_baseline = report.total_cost_usd - baseline.total_cost_usd
        report.p95_latency_delta_vs_baseline = report.p95_latency_ms - baseline.p95_latency_ms
    _mark_pareto(reports)
    winner = max(reports, key=lambda report: (report.quality_score, -report.total_cost_usd, -report.p95_latency_ms))
    import hashlib

    fingerprint = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    return ExperimentReport(
        experiment_id=str(uuid4()),
        experiment_name=config.name,
        created_at=datetime.now(UTC).isoformat(),
        dataset_path=str(dataset_path),
        dataset_fingerprint=fingerprint,
        winner=winner.variant.name,
        pass_threshold=config.pass_threshold,
        evaluated_splits=list(config.splits),
        review_status_counts=dict(Counter(case.review_status for case in cases)),
        reports=reports,
    )


class ExperimentStore:
    """Small local experiment registry; JSON remains the portable report format."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = directory / "experiments.db"
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    winner TEXT NOT NULL,
                    report_path TEXT NOT NULL,
                    report_json TEXT NOT NULL
                )"""
            )

    def save(self, report: ExperimentReport) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", report.experiment_name).strip("-") or "experiment"
        stamp = report.created_at.replace(":", "-").replace("+00:00", "Z")
        path = self.directory / f"{stamp}-{safe_name}.json"
        payload = report.model_dump_json(indent=2)
        path.write_text(payload + "\n", encoding="utf-8")
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO experiments VALUES (?, ?, ?, ?, ?, ?)",
                (
                    report.experiment_id,
                    report.experiment_name,
                    report.created_at,
                    report.winner,
                    str(path),
                    payload,
                ),
            )
        return path

    def list(self, limit: int = 50) -> list[dict[str, str]]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT experiment_id, experiment_name, created_at, winner, report_path "
                "FROM experiments ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, experiment_id: str) -> ExperimentReport | None:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT report_json FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        return ExperimentReport.model_validate_json(row[0]) if row else None


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
