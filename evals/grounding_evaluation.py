"""Credential-free hallucination and claim-grounding release gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.grounding import (
    GroundingReport,
    evidence_from_state,
    repair_answer,
    verify_grounding,
    verify_report,
)

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "grounding_scenarios.jsonl"


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class GroundingEvalScenario(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    kind: Literal[
        "supported_web",
        "unsupported_claim",
        "fabricated_citation",
        "partial_repair",
        "high_risk_numeric",
        "rag_support",
        "kg_support",
        "math_support",
        "missing_evidence",
        "general_not_required",
        "safe_failure_message",
        "artifact_integrity",
    ]
    expected_action: Literal["pass", "repair", "abstain", "not_required"]
    description: str


class GroundingEvalOutcome(BaseModel):
    scenario_id: str
    kind: str
    expected_action: str
    actual_action: str
    passed: bool
    claim_coverage: float
    citation_precision: float
    unsupported_high_risk_claims: int
    unsafe_answer_released: bool
    receipt_verified: bool
    evidence_fingerprint: str = ""


class GroundingEvalReport(BaseModel):
    schema_version: str = "1.0"
    generated_by: str = "agentforge-grounding-eval"
    dataset_path: str
    dataset_fingerprint: str
    total: int
    passed: int
    pass_rate: float
    supported_release_rate: float
    unsafe_release_rate: float
    fabricated_citation_escape_rate: float
    high_risk_claim_escape_rate: float
    repair_success_rate: float
    mean_claim_coverage: float
    receipt_integrity_rate: float
    outcomes: list[GroundingEvalOutcome]
    report_fingerprint: str = ""


def load_scenarios(path: Path = DEFAULT_DATASET) -> list[GroundingEvalScenario]:
    scenarios = [
        GroundingEvalScenario.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not scenarios:
        raise ValueError("grounding evaluation dataset is empty")
    ids = [item.id for item in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("grounding evaluation scenario ids must be unique")
    return scenarios


def _inputs(kind: str) -> tuple[str, str, dict[str, str]]:
    web = {
        "web_notes": (
            "Architecture report\nLink: https://example.com/architecture\n"
            "Snippet: AgentForge uses LangGraph for agent orchestration."
        )
    }
    cases = {
        "supported_web": (
            "web",
            "AgentForge uses LangGraph for agent orchestration "
            "[source](https://example.com/architecture).",
            web,
        ),
        "unsupported_claim": (
            "web",
            "AgentForge was founded on Mars by seven astronauts.",
            web,
        ),
        "fabricated_citation": (
            "web",
            "AgentForge uses LangGraph [source](https://fabricated.example/report).",
            web,
        ),
        "partial_repair": (
            "rag",
            "AgentForge uses LangGraph for orchestration. The platform was invented on the Moon.",
            {"rag_notes": "AgentForge uses LangGraph for orchestration."},
        ),
        "high_risk_numeric": (
            "web",
            "AgentForge is guaranteed to reduce production cost by 97% in 2026.",
            web,
        ),
        "rag_support": (
            "rag",
            "The API uses FastAPI and the agent graph uses LangGraph.",
            {"rag_notes": "The service API uses FastAPI. The agent graph uses LangGraph."},
        ),
        "kg_support": (
            "kg",
            "FastAPI calls LangGraph, which uses ChromaDB.",
            {"kg_notes": "fastapi --calls--> langgraph --uses--> chromadb"},
        ),
        "math_support": (
            "math",
            "The result of 19 multiplied by 7 is 133.",
            {"math_result": "19 * 7 = 133"},
        ),
        "missing_evidence": (
            "rag",
            "The private architecture was approved by the security board.",
            {},
        ),
        "general_not_required": (
            "general",
            "Groundedness connects generated claims to supporting evidence.",
            {},
        ),
        "safe_failure_message": (
            "web",
            "I could not find enough reliably dated sources for this request.",
            {},
        ),
        "artifact_integrity": (
            "web",
            "AgentForge uses LangGraph for agent orchestration.",
            web,
        ),
    }
    return cases[kind]


def _execute(scenario: GroundingEvalScenario) -> GroundingEvalOutcome:
    route, answer, state = _inputs(scenario.kind)
    evidence = evidence_from_state(state)
    report = verify_grounding(
        route=route,
        answer=answer,
        evidence=evidence,
        integrity_key=b"grounding-evaluation-integrity-key",
    )
    receipt_verified = verify_report(report, b"grounding-evaluation-integrity-key")
    repair_ok = True
    if scenario.kind == "partial_repair":
        repaired = repair_answer(report, evidence)
        repair_ok = "uses LangGraph" in repaired and "Moon" not in repaired
    if scenario.kind == "artifact_integrity":
        tampered = GroundingReport.model_validate(report.model_dump())
        tampered.confidence = 0.123456
        receipt_verified = receipt_verified and not verify_report(
            tampered, b"grounding-evaluation-integrity-key"
        )
    unsafe_release = scenario.expected_action in {"repair", "abstain"} and report.action == "pass"
    passed = (
        report.action == scenario.expected_action
        and receipt_verified
        and repair_ok
        and not unsafe_release
    )
    outcome = GroundingEvalOutcome(
        scenario_id=scenario.id,
        kind=scenario.kind,
        expected_action=scenario.expected_action,
        actual_action=report.action,
        passed=passed,
        claim_coverage=report.claim_coverage,
        citation_precision=report.citation_precision,
        unsupported_high_risk_claims=report.unsupported_high_risk_claims,
        unsafe_answer_released=unsafe_release,
        receipt_verified=receipt_verified,
    )
    outcome.evidence_fingerprint = _hash(
        outcome.model_dump(mode="json", exclude={"evidence_fingerprint"})
    )
    return outcome


def evaluate_grounding(
    scenarios: list[GroundingEvalScenario], dataset_path: str = ""
) -> GroundingEvalReport:
    outcomes = [_execute(item) for item in scenarios]
    passed = sum(item.passed for item in outcomes)
    supported = [item for item in outcomes if item.expected_action in {"pass", "not_required"}]
    unsafe = [item for item in outcomes if item.expected_action in {"repair", "abstain"}]
    repairs = [item for item in outcomes if item.expected_action == "repair"]
    report = GroundingEvalReport(
        dataset_path=dataset_path,
        dataset_fingerprint=_hash([item.model_dump(mode="json") for item in scenarios]),
        total=len(outcomes),
        passed=passed,
        pass_rate=passed / len(outcomes),
        supported_release_rate=sum(item.passed for item in supported) / len(supported),
        unsafe_release_rate=sum(item.unsafe_answer_released for item in unsafe) / len(unsafe),
        fabricated_citation_escape_rate=float(
            next(item for item in outcomes if item.kind == "fabricated_citation").unsafe_answer_released
        ),
        high_risk_claim_escape_rate=float(
            next(item for item in outcomes if item.kind == "high_risk_numeric").unsafe_answer_released
        ),
        repair_success_rate=sum(item.passed for item in repairs) / len(repairs),
        mean_claim_coverage=sum(item.claim_coverage for item in outcomes) / len(outcomes),
        receipt_integrity_rate=sum(item.receipt_verified for item in outcomes) / len(outcomes),
        outcomes=outcomes,
    )
    report.report_fingerprint = _hash(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    )
    return report


def verify_evaluation_report(report: GroundingEvalReport) -> bool:
    if report.report_fingerprint != _hash(
        report.model_dump(mode="json", exclude={"report_fingerprint"})
    ):
        return False
    return all(
        item.evidence_fingerprint
        == _hash(item.model_dump(mode="json", exclude={"evidence_fingerprint"}))
        for item in report.outcomes
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate claim-level grounding controls")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--max-unsafe-release-rate", type=float, default=0)
    parser.add_argument("--min-receipt-integrity", type=float, default=1.0)
    args = parser.parse_args()
    report = evaluate_grounding(load_scenarios(args.dataset), args.dataset.as_posix())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report.passed,
                "total": report.total,
                "pass_rate": report.pass_rate,
                "unsafe_release_rate": report.unsafe_release_rate,
                "receipt_integrity_rate": report.receipt_integrity_rate,
                "report": str(args.output) if args.output else "",
            }
        )
    )
    if (
        report.pass_rate < args.min_pass_rate
        or report.unsafe_release_rate > args.max_unsafe_release_rate
        or report.receipt_integrity_rate < args.min_receipt_integrity
        or not verify_evaluation_report(report)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
