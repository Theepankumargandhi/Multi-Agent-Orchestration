"""Bounded, deterministic execution-evidence routing; never executes repository code."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath

from code_agent.models import (
    CodeAgentResult,
    FailureDiagnosis,
    RepairCandidateEvidence,
    RepairTournamentReport,
)

ROLE_ORDER = {
    "wrong_localization": ("compatibility", "boundary", "defensive"),
    "dependency_or_contract": ("compatibility", "boundary", "defensive"),
    "incomplete_fix": ("boundary", "defensive", "compatibility"),
    "regression": ("defensive", "compatibility", "boundary"),
    "unknown": ("boundary", "compatibility", "defensive"),
}
TERMINAL_CATEGORIES = {"verified", "safety_hold", "provider_error", "budget", "timeout"}


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def diagnose_failure(row: RepairCandidateEvidence, result: CodeAgentResult | None) -> FailureDiagnosis:
    """Heuristic labels, not claims of root-cause certainty. Persist hashes/codes, never raw logs."""
    signals: list[str] = []
    reasons = set(row.reasons)
    final = result.final_test if result else None
    baseline = result.baseline_test if result else None
    failed = bool(final and (final.exit_code != 0 or final.timed_out))
    output = ((final.stdout + "\n" + final.stderr)[:40_000] if failed else "")
    gates = result.verification.rounds[-1].quality_gates if result and result.verification and result.verification.rounds else []
    failed_gates = {gate.name for gate in gates if gate.status == "failed"}
    failed_gates |= {name for name, status in row.final_gate_statuses.items() if status == "failed"}
    observations = result.observations if result else []
    if row.eligible:
        category = "verified"
    elif (failed_gates & {"secret_scan", "unsafe_code_scan", "source_repository_integrity"}
          or ("patch_scope" in failed_gates and row.patch_chars > 0)
          or reasons & {"result_does_not_match_workspace", "candidate_write_budget_exceeded", "patch_changed_after_review",
                        "mandatory_gate_evidence_incomplete", "missing_security_evidence"}
          or (result and result.security and result.security.blocked_actions)):
        category, signals = "safety_hold", ["integrity_or_policy_block"]
    elif (any("ModelBudgetExceeded" in reason for reason in reasons) or (result and result.status == "budget_exhausted")
          or any("ModelBudgetExceeded" in observation.summary for observation in observations)):
        category, signals = "budget", ["model_budget_exhausted"]
    elif row.status == "timed_out" or (final and final.timed_out):
        category, signals = "timeout", ["execution_or_review_deadline"]
    elif result and any(failure.terminal for failure in result.provider_failures):
        category, signals = "provider_error", ["terminal_provider_hold"]
    elif any(observation.summary.startswith("Action model failed") for observation in observations):
        category, signals = "provider_error", ["action_provider_unavailable"]
    elif failed and baseline and baseline.exit_code == 0 and not baseline.timed_out:
        category, signals = "regression", ["passing_baseline_now_fails"]
    elif failed and any(token in output for token in ("ModuleNotFoundError", "ImportError", "TypeError", "AttributeError")):
        category, signals = "dependency_or_contract", ["import_or_interface_exception"]
    elif failed:
        frames = re.findall(r'(?:File "([^"\n]+)", line|(?:^|\n)([\w./\\-]+\.py):\d+:)', output)
        paths = {PurePosixPath((left or right).replace("\\", "/")).as_posix() for left, right in frames}
        production = {path for path in paths if not any(part.startswith("test") for part in PurePosixPath(path).parts)}
        changed = result.changed_files if result else []
        localized = any(path == item or path.endswith("/" + item) for path in production for item in changed)
        if production and changed and not localized:
            category, signals = "wrong_localization", ["failing_production_frame_outside_patch"]
        else:
            category, signals = "incomplete_fix", ["owner_assertion_or_execution_failed"]
    elif row.regression_probes and not all(status == "matched" for status in row.regression_probes.statuses):
        category, signals = "incomplete_fix", ["behavioral_probe_failed"]
    elif "challenge_review_blocked" in reasons:
        category, signals = "incomplete_fix", ["independent_challenge_block"]
    else:
        category, signals = "unknown", ["insufficient_execution_evidence"]
    command_hash = _hash(final.command) if final else ""
    return FailureDiagnosis(category=category, signals=signals, command_sha256=command_hash,
        failure_signature=_hash([category, signals, command_hash, row.patch_sha256]),
        evidence_sha256=_hash({"reasons": row.reasons, "gates": sorted(failed_gates), "command": command_hash,
                              "output": output, "patch": row.patch_sha256}))


def next_perspective(category: str, used: set[str]) -> str | None:
    if category in TERMINAL_CATEGORIES:
        return None
    return next((role for role in ROLE_ORDER.get(category, ROLE_ORDER["unknown"]) if role not in used), None)


def stopping_reason(rows: list[RepairCandidateEvidence], report: RepairTournamentReport | None = None,
                    *, repeated_failure_limit: int = 2) -> str:
    """Evidence-only terminal conditions; resource admission is checked by the runtime."""
    if not rows:
        return ""
    diagnosis = rows[-1].diagnosis
    if diagnosis is None:
        return "invalid_diagnosis"
    if diagnosis.category in TERMINAL_CATEGORIES:
        return diagnosis.category
    limit = report.policy.adaptive_dispatch.repeated_failure_limit if report and report.policy.adaptive_dispatch else repeated_failure_limit
    if sum(row.diagnosis is not None and row.diagnosis.failure_signature == diagnosis.failure_signature for row in rows) >= limit:
        return "repeated_failure"
    return ""


def verify_dispatch_report(report: RepairTournamentReport) -> bool:
    if not report.policy.adaptive_dispatch:
        return not report.dispatch_decisions and not report.dispatch_shadows and not report.stop_reason and all(row.status != "skipped" for row in report.candidates)
    rows = [row for row in report.candidates if row.status != "skipped"]
    if len(rows) != len(report.dispatch_decisions) or not report.stop_reason:
        return False
    used: set[str] = set()
    for index, (row, decision) in enumerate(zip(rows, report.dispatch_decisions, strict=True)):
        expected = "minimal" if index == 0 else next_perspective(rows[index - 1].diagnosis.category, used)
        trigger = "initial" if index == 0 else rows[index - 1].diagnosis.category
        if (row.candidate_id != f"candidate-{index + 1}" or decision.candidate_id != row.candidate_id
                or decision.perspective != row.perspective or row.perspective != expected or decision.trigger != trigger
                or not row.diagnosis or not row.diagnosis.heuristic or row.eligible != (row.diagnosis.category == "verified")):
            return False
        if index and stopping_reason(rows[:index], report):
            return False
        used.add(row.perspective)
    if report.candidates[:len(rows)] != rows:
        return False
    for index, row in enumerate(report.candidates[len(rows):], start=len(rows) + 1):
        if (row.status != "skipped" or row.model_calls or row.reported_writes or row.eligible or row.patch_sha256
                or row.challenge_approved or row.diagnosis or row.reasons != ["not_dispatched:" + report.stop_reason]
                or row.candidate_id != f"candidate-{index}"):
            return False
    if report.workflow != "adaptive_supervisor_v1" or sum(row.eligible for row in rows) > 1:
        return False
    terminal = stopping_reason(rows, report)
    if terminal and report.stop_reason != terminal:
        return False
    if not terminal and report.stop_reason not in {"teams_exhausted", "verification_budget_reserved", "deadline_reserved",
                                                  "regression_preparation_blocked", "global_deadline", "cleanup_failed"}:
        return False
    if report.stop_reason == "teams_exhausted" and len(rows) != report.policy.candidates:
        return False
    for row in rows:
        if row.eligible and (not row.challenge_approved or row.reasons or row.challenge_blocking_findings
                            or not {"mandatory_tests", "patch_scope", "secret_scan", "unsafe_code_scan", "source_repository_integrity"}
                            <= {name for name, status in row.final_gate_statuses.items() if status == "passed"}):
            return False
    shadow_policy = report.policy.adaptive_dispatch.advisor_shadow
    if not shadow_policy and report.dispatch_shadows:
        return False
    if shadow_policy:
        from code_agent.dispatch_learning import context_for, fingerprint
        expected_ids, used, prefix, calls = [], set(), [], report.shared_model_calls
        for index, row in enumerate(rows):
            used.add(row.perspective)
            prefix.append(row)
            calls += row.model_calls
            if index + 1 >= len(rows) or stopping_reason(prefix, report):
                continue
            baseline = next_perspective(row.diagnosis.category, used)
            if baseline is None:
                continue
            expected_ids.append(row.candidate_id)
            shadow = next((item for item in report.dispatch_shadows if item.candidate_id == row.candidate_id), None)
            if (shadow is None or shadow.serving_changed or shadow.baseline_role != baseline
                    or shadow.artifact_fingerprint != shadow_policy.artifact_fingerprint
                    or shadow.context_sha256 != fingerprint(context_for(row, used, calls, report.policy).model_dump(mode="json"))
                    or (shadow.proposed_role is not None and shadow.proposed_role in used)
                    or (shadow.disposition == "supported") != (shadow.proposed_role is not None)):
                return False
        if [item.candidate_id for item in report.dispatch_shadows] != expected_ids:
            return False
    return (report.calls_with_usage <= report.model_calls
            and report.usage_accounting_complete == (report.calls_with_usage == report.model_calls))
