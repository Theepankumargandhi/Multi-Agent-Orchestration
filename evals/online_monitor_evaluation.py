"""Credential-free release gate for online AI monitoring and canary governance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from agent.online_evaluation import (
    DelayedFeedback,
    OnlineEvalPolicy,
    OnlineEvaluationController,
    OnlineEvaluationReport,
    OnlineInferenceEvent,
)

DEFAULT_DATASET = Path(__file__).parent / "datasets" / "online_monitor_events.jsonl"
FORBIDDEN_CONTENT_FIELDS = {
    "prompt",
    "response",
    "content",
    "messages",
    "system",
    "secret",
    "api_key",
    "authorization",
}


class DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    record_type: Literal["event", "feedback"] = "event"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _privacy_violations(value: Any) -> int:
    if isinstance(value, dict):
        return sum(str(key).casefold() in FORBIDDEN_CONTENT_FIELDS for key in value) + sum(
            _privacy_violations(item) for item in value.values()
        )
    if isinstance(value, list):
        return sum(_privacy_violations(item) for item in value)
    return 0


def load_dataset(
    path: Path,
) -> tuple[list[OnlineInferenceEvent], list[DelayedFeedback], str, int]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("online monitoring dataset is empty")
    events: list[OnlineInferenceEvent] = []
    feedback: list[DelayedFeedback] = []
    for raw in rows:
        envelope = DatasetRecord.model_validate(raw)
        payload = dict(raw)
        payload.pop("record_type", None)
        if envelope.record_type == "event":
            events.append(OnlineInferenceEvent.model_validate(payload))
        else:
            feedback.append(DelayedFeedback.model_validate(payload))
    fingerprint = hashlib.sha256(_canonical(rows).encode()).hexdigest()
    return events, feedback, fingerprint, _privacy_violations(rows)


def evaluate_dataset(
    path: Path = DEFAULT_DATASET,
    policy: OnlineEvalPolicy | None = None,
    *,
    integrity_key: bytes | None = None,
) -> OnlineEvaluationReport:
    events, feedback, fingerprint, privacy_violations = load_dataset(path)
    controller = OnlineEvaluationController(policy, integrity_key=integrity_key)
    for event in events:
        controller.ingest(event)
    for item in feedback:
        controller.join_feedback(item)
    return controller.evaluate(
        dataset_fingerprint=fingerprint,
        privacy_violations=privacy_violations,
    )


def verify_report(report: OnlineEvaluationReport, *, integrity_key: bytes | None = None) -> bool:
    return OnlineEvaluationController(integrity_key=integrity_key).verify_report(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate online AI SLO and canary controls")
    parser.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-decision", choices=["promote", "hold", "rollback"])
    parser.add_argument("--min-integrity-rate", type=float, default=1.0)
    parser.add_argument("--max-privacy-violations", type=int, default=0)
    parser.add_argument("--require-alert", action="store_true")
    args = parser.parse_args()

    integrity_key = os.getenv("ONLINE_EVAL_INTEGRITY_KEY", "").encode() or None
    report = evaluate_dataset(args.dataset, integrity_key=integrity_key)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "events": report.accepted_events,
                "duplicates": report.duplicate_events,
                "feedback_joined": report.joined_feedback,
                "alerts": len(report.alerts),
                "decision": report.canary_decision.action,
                "integrity_rate": report.event_integrity_rate,
                "privacy_violations": report.privacy_violations,
                "report": str(args.output) if args.output else "",
            }
        )
    )
    failed = (
        report.event_integrity_rate < max(0, min(1, args.min_integrity_rate))
        or report.privacy_violations > args.max_privacy_violations
        or (args.require_decision and report.canary_decision.action != args.require_decision)
        or (args.require_alert and not report.alerts)
        or not verify_report(report, integrity_key=integrity_key)
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
