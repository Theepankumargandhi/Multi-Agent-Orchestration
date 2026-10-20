"""Study-arm recovery on the existing single-host checkpoint; no exactly-once API claim."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from code_agent.benchmark_checkpoint import BenchmarkCheckpoint
from code_agent.upgrade_contracts import _unique_pairs
from code_agent.upgrade_models import StrictModel, fingerprint

RECOVERY_PROTOCOL = "migration-arm-journal-v1"


class StudyRunBudget(StrictModel):
    max_attempts: int = Field(ge=1, le=100)
    max_provider_calls: int = Field(ge=0, le=10000)


class MigrationRecord(StrictModel):
    task_id: str
    arm: str
    status: Literal["held", "ready_for_owner_review"]
    reason: str
    duration_ms: float = Field(ge=0)
    model_calls: int = Field(ge=0)
    live_provider_calls: int = Field(ge=0)
    usage_complete: bool
    provider_input_tokens: int | None = Field(default=None, ge=0)
    provider_output_tokens: int | None = Field(default=None, ge=0)
    billed_cost_usd: float | None = Field(default=None, ge=0)
    final_evaluation: dict[str, Any] | None = None
    final_session_evaluation: dict[str, Any] | None = None
    final_relation_evaluation: dict[str, Any] | None = None
    source_unchanged: bool
    human_approval: Literal["not_measured"] = "not_measured"
    report_fingerprint: str
    patch_path: str | None = None
    trajectory_path: str


def attempt_key(task_id, arm):
    return fingerprint({"task_id": task_id, "arm": arm})


def validate_report(report, task, reservation):
    if (not isinstance(report, dict) or report.get("protocol") != task.spec.protocol
            or report.get("task_id") != task.task_id or report.get("family") != task.spec.family
            or report.get("spec_fingerprint") != fingerprint(task.spec.model_dump())
            or report.get("image_id") != task.spec.image
            or type(report.get("source_unchanged")) is not bool
            or report.get("status") not in {"held", "ready_for_owner_review"}
            or type(report.get("live_provider_calls")) is not int
            or type(report.get("model_calls")) is not int
            or not 0 <= report["live_provider_calls"] <= min(reservation, report["model_calls"])):
        raise ValueError("migration receipt identity or call accounting is invalid")
    if report.get("fingerprint") != fingerprint({key: value for key, value in report.items() if key != "fingerprint"}):
        raise ValueError("migration receipt fingerprint mismatch")
    patch = report.get("patch")
    if not isinstance(patch, str) or len(patch.encode()) > 5_000_000:
        raise ValueError("migration patch missing or oversized")
    if report["status"] == "held" and patch:
        raise ValueError("held migration cannot release a patch")
    if report["status"] == "ready_for_owner_review":
        if not report["source_unchanged"] or not patch or report.get("patch_sha256") != hashlib.sha256(patch.encode()).hexdigest():
            raise ValueError("review-ready migration patch is unbound")
        for name in ("final_evaluation", "final_session_evaluation", "final_relation_evaluation"):
            evidence = report.get(name)
            if evidence is not None and (evidence.get("passed") is not True or evidence.get("patch_sha256") != report["patch_sha256"]):
                raise ValueError("terminal evaluation does not match the released patch")


class MigrationJournal(BenchmarkCheckpoint):
    """Arm-boundary adapter: locks, fsync and artifact hashing stay in the shared checkpoint."""
    def configure(self, entries, budget):
        self.entries = entries
        if self.resume:
            if self.state.get("migration_protocol") != RECOVERY_PROTOCOL:
                raise ValueError("checkpoint is not a recoverable migration study")
            selected = StudyRunBudget.model_validate(self.state.get("run_budget"))
            if budget is not None and budget != selected:
                raise ValueError("run budget changed; resume must retain its reviewed limits")
        else:
            selected = budget
            if (self.state.get("migration_protocol") != RECOVERY_PROTOCOL
                    or self.state.get("run_budget") != budget.model_dump()):
                raise ValueError("migration metadata must be committed with the initial checkpoint")
        self.budget = selected
        for key, attempts in self.state["attempts"].items():
            if key not in entries or not attempts:
                raise ValueError("checkpoint contains an unknown study arm")
            for attempt in attempts:
                if attempt.get("status") not in {"running", "complete"} or type(attempt.get("number")) is not int:
                    raise ValueError("invalid migration attempt marker")
                if attempt["status"] == "complete":
                    record = MigrationRecord.model_validate(attempt["outcome"])
                    task, arm, reservation = entries[key]
                    if record.task_id != task.task_id or record.arm != arm or record.live_provider_calls > reservation:
                        raise ValueError("retained attempt does not match its study arm")
                    if not record.source_unchanged:
                        self.state["revoked"] = True
        if self.state.get("revoked"):
            raise ValueError("study admissions revoked after a source-integrity failure")
        totals = self.accounting()
        if totals["admitted_attempts"] > selected.max_attempts or totals["provider_call_upper_bound_used"] > selected.max_provider_calls:
            raise ValueError("checkpoint exceeds its approved run budget")

    def accounting(self, arm=None):
        count = known = uncertain = interrupted = 0
        for key, attempts in self.state["attempts"].items():
            _task, selected_arm, reservation = self.entries[key]
            if arm is not None and selected_arm != arm:
                continue
            for attempt in attempts:
                count += 1
                if attempt["status"] == "complete":
                    known += attempt["outcome"]["live_provider_calls"]
                else:
                    interrupted += 1
                    uncertain += reservation
        return {"admitted_attempts": count, "known_provider_calls": known,
            "interrupted_attempts": interrupted, "unresolved_provider_call_reservations": uncertain,
            "provider_call_upper_bound_used": known + uncertain}

    def admit(self, key):
        _task, _arm, reservation = self.entries[key]
        totals = self.accounting()
        if (totals["admitted_attempts"] + 1 > self.budget.max_attempts
                or totals["provider_call_upper_bound_used"] + reservation > self.budget.max_provider_calls):
            raise ValueError("study admission budget exhausted before any new provider request")
        return self.start(key)  # Shared checkpoint persists RUNNING before returning.

    def finish_report(self, key, report):
        task, arm, reservation = self.entries[key]
        validate_report(report, task, reservation)
        attempt = self.state["attempts"][key][-1]
        prefix = f"attempts/{attempt['number']}"
        fields = {name: report.get(name) for name in MigrationRecord.model_fields
            if name not in {"task_id", "arm", "human_approval", "report_fingerprint", "patch_path", "trajectory_path"}}
        record = MigrationRecord(task_id=task.task_id, arm=arm, **fields, report_fingerprint=report["fingerprint"],
            patch_path=f"{prefix}/migration.diff" if report["patch"] else None, trajectory_path=f"{prefix}/report.json")
        if not record.source_unchanged:
            self.state["revoked"] = True
        # All returned terminal holds are COMPLETE. They are never retried for a better score.
        self.finish(key, record, None, held=False)
        return record.model_dump(mode="json")

    def _saved_report(self, key):
        task, _arm, reservation = self.entries[key]
        number = self.state["attempts"][key][-1]["number"]
        report_path = self.safe_path(f"attempts/{number}/report.json")
        if report_path.stat().st_size > 10_000_000:
            raise ValueError("saved receipt exceeds its byte budget")
        report = json.loads(report_path.read_bytes(), object_pairs_hook=_unique_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite receipt JSON")))
        if not isinstance(report, dict) or "patch" in report:
            raise ValueError("saved receipt must be an object without embedded patch data")
        patch_path = self.safe_path(f"attempts/{number}/migration.diff")
        if report.get("status") == "ready_for_owner_review":
            if patch_path.stat().st_size > 5_000_000:
                raise ValueError("saved patch exceeds its byte budget")
            report["patch"] = patch_path.read_bytes().decode("utf-8")
        else:
            if patch_path.exists():
                raise ValueError("held receipt unexpectedly has a patch")
            report["patch"] = ""
        validate_report(report, task, reservation)
        return report

    def recover(self, key, *, retry_interrupted):
        attempts = self.state["attempts"].get(key, [])
        if not attempts:
            return None, "new"
        latest = attempts[-1]
        if latest["status"] == "complete":
            # The shared bind verified bytes for ALL retained attempts, not only winners.
            return MigrationRecord.model_validate(latest["outcome"]).model_dump(mode="json"), "reused"
        try:
            report = self._saved_report(key)
        except (OSError, ValueError, TypeError, KeyError):
            if not retry_interrupted:
                raise ValueError("interrupted arm requires --retry-interrupted; prior provider usage remains unknown") from None
            return None, "explicit_retry"
        latest["recovered_saved_receipt"] = True
        return self.finish_report(key, report), "adopted_saved_receipt"

    def write_summary(self, payload):
        destination = self.safe_path("study.json")
        descriptor, temporary = tempfile.mkstemp(prefix=".study-summary-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({**payload, "fingerprint": fingerprint(payload)}, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
