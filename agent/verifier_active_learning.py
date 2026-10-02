"""Privacy-safe review queue for uncertain verifier decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.process_reward import ProcessStep, ProcessTrace
from agent.search_planner import SearchPlan, SearchRequest


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


class VerifierReviewCandidate(BaseModel):
    schema_version: str = "1.0"
    event_id: str
    request_fingerprint: str
    model_fingerprint: str
    plan_fingerprint: str
    route: str
    high_risk: bool
    evidence_count: int = Field(ge=0)
    planned_actions: list[str]
    terminal_action: Literal["answer", "abstain"]
    verifier_mean: float = Field(ge=0, le=1)
    verifier_uncertainty: float = Field(ge=0, le=1)
    risk_adjusted_reward: float = Field(ge=0, le=1)
    verifier_ood: bool
    review_reason: Literal["disagreement", "out_of_distribution", "high_risk_uncertainty"]
    created_at: str
    receipt_fingerprint: str = ""

    def seal(self) -> None:
        self.receipt_fingerprint = _hash(
            self.model_dump(mode="json", exclude={"receipt_fingerprint"})
        )

    def verify(self) -> bool:
        return bool(self.receipt_fingerprint) and self.receipt_fingerprint == _hash(
            self.model_dump(mode="json", exclude={"receipt_fingerprint"})
        )


class VerifierActiveLearningQueue:
    """Transactional, deduplicated queue containing metadata but no prompt or answer text."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS verifier_review_queue (
                    event_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    label TEXT NOT NULL DEFAULT '',
                    reviewer_fingerprint TEXT NOT NULL DEFAULT '',
                    reviewed_at TEXT NOT NULL DEFAULT ''
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def enqueue(
        self,
        plan: SearchPlan,
        request: SearchRequest,
        *,
        uncertainty_threshold: float,
        created_at: str | None = None,
    ) -> VerifierReviewCandidate | None:
        if not plan.verify():
            raise ValueError("cannot queue an invalid search plan")
        if not plan.verifier_ood and plan.verifier_uncertainty < uncertainty_threshold:
            return None
        if plan.verifier_ood:
            reason = "out_of_distribution"
        elif request.high_risk:
            reason = "high_risk_uncertainty"
        else:
            reason = "disagreement"
        event_id = _hash(
            {
                "request": plan.request_fingerprint,
                "model": plan.process_reward_fingerprint,
                "plan": plan.plan_fingerprint,
            }
        )
        candidate = VerifierReviewCandidate(
            event_id=event_id,
            request_fingerprint=plan.request_fingerprint,
            model_fingerprint=plan.process_reward_fingerprint,
            plan_fingerprint=plan.plan_fingerprint,
            route=request.route,
            high_risk=request.high_risk,
            evidence_count=request.evidence_count,
            planned_actions=list(plan.planned_actions),
            terminal_action=plan.terminal_action,
            verifier_mean=plan.verifier_mean,
            verifier_uncertainty=plan.verifier_uncertainty,
            risk_adjusted_reward=plan.risk_adjusted_reward,
            verifier_ood=plan.verifier_ood,
            review_reason=reason,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
        )
        candidate.seal()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO verifier_review_queue(event_id, payload) VALUES (?, ?)",
                (candidate.event_id, candidate.model_dump_json()),
            )
        return candidate

    def pending(self, limit: int = 100) -> list[VerifierReviewCandidate]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM verifier_review_queue WHERE status = 'pending' "
                "ORDER BY event_id LIMIT ?",
                (max(1, min(limit, 1000)),),
            ).fetchall()
        records = [VerifierReviewCandidate.model_validate_json(row[0]) for row in rows]
        if not all(record.verify() for record in records):
            raise ValueError("active-learning queue integrity verification failed")
        return records

    def review(
        self,
        event_id: str,
        label: Literal["safe", "unsafe", "ambiguous"],
        reviewer_id: str,
        *,
        reviewed_at: str | None = None,
    ) -> bool:
        reviewer_fingerprint = _hash(reviewer_id)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE verifier_review_queue
                SET status = 'reviewed', label = ?, reviewer_fingerprint = ?, reviewed_at = ?
                WHERE event_id = ? AND status = 'pending'
                """,
                (
                    label,
                    reviewer_fingerprint,
                    reviewed_at or datetime.now(timezone.utc).isoformat(),
                    event_id,
                ),
            )
        return cursor.rowcount == 1

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) FROM verifier_review_queue GROUP BY status"
            ).fetchall()
        return {str(status): int(count) for status, count in rows}

    def export_reviewed(self, path: Path) -> list[ProcessTrace]:
        """Export human labels as content-free training traces for the next model cycle."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload, label FROM verifier_review_queue "
                "WHERE status = 'reviewed' ORDER BY event_id"
            ).fetchall()
        traces = []
        for payload, label in rows:
            candidate = VerifierReviewCandidate.model_validate_json(payload)
            if not candidate.verify() or label not in {"safe", "unsafe", "ambiguous"}:
                raise ValueError("reviewed verifier record failed integrity validation")
            if label == "ambiguous":
                continue
            target = {"safe": 1.0, "unsafe": 0.0, "ambiguous": 0.5}[label]
            steps = [
                ProcessStep(
                    step_id=f"{index}-{action}",
                    kind="refuse" if action == "abstain" else action,
                    has_evidence=candidate.evidence_count > 0,
                    # The queue records plans, not observed citations/tool checks.
                    # A terminal safety label must never construct input features.
                    citation_valid=False,
                    policy_allowed=True,
                    confidence=candidate.verifier_mean,
                    step_label=None,
                )
                for index, action in enumerate(candidate.planned_actions, start=1)
            ]
            traces.append(
                ProcessTrace(
                    trace_id=f"al-{candidate.event_id[:24]}",
                    group_id=f"al-{candidate.request_fingerprint[:24]}",
                    query="metadata-only active-learning trajectory",
                    high_risk=candidate.high_risk,
                    self_confidence=candidate.verifier_mean,
                    steps=steps,
                    outcome_quality=target,
                    safe=label == "safe",
                    split="train",
                    review_status="human_reviewed",
                )
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(trace.model_dump_json() + "\n" for trace in traces),
            encoding="utf-8",
        )
        return traces


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and label privacy-safe verifier review candidates."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/verifier-review/verifier-review.sqlite3"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--limit", type=int, default=100)
    subparsers.add_parser("stats")
    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("event_id")
    review_parser.add_argument("--label", choices=["safe", "unsafe", "ambiguous"], required=True)
    review_parser.add_argument("--reviewer", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = VerifierActiveLearningQueue(args.database)
    if args.command == "list":
        print(
            json.dumps(
                [item.model_dump(mode="json") for item in queue.pending(args.limit)],
                indent=2,
            )
        )
        return 0
    if args.command == "stats":
        print(json.dumps(queue.counts(), indent=2))
        return 0
    if args.command == "export":
        traces = queue.export_reviewed(args.output)
        print(json.dumps({"output": str(args.output), "exported": len(traces)}))
        return 0
    reviewed = queue.review(args.event_id, args.label, args.reviewer)
    print(json.dumps({"event_id": args.event_id, "reviewed": reviewed}))
    return 0 if reviewed else 2


if __name__ == "__main__":
    raise SystemExit(main())
