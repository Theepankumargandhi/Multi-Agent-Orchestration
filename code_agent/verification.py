"""Deterministic quality gates and integrity-bound PR evidence generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from code_agent.job_models import ArtifactReference, JobRecord
from code_agent.models import (
    CodeAgentResult,
    CodeTask,
    QualityGateResult,
    SandboxCommandResult,
    VerificationReport,
)
from code_agent.sandbox import DockerSandbox

_TEST_PATHS = (
    re.compile(r"(^|/)tests?/", re.IGNORECASE),
    re.compile(r"(^|/)test_[^/]+\.py$", re.IGNORECASE),
    re.compile(r"(^|/)[^/]+_test\.py$", re.IGNORECASE),
    re.compile(r"\.(spec|test)\.[jt]sx?$", re.IGNORECASE),
)

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
)

_UNSAFE_PATTERNS = (
    (re.compile(r"\bshell\s*=\s*True\b"), "subprocess shell execution"),
    (re.compile(r"\bos\.system\s*\("), "os.system execution"),
    (re.compile(r"\b(?:eval|exec)\s*\("), "dynamic code execution"),
    (re.compile(r"\bpickle\.loads?\s*\("), "unsafe pickle deserialization"),
    (re.compile(r"\bverify\s*=\s*False\b"), "disabled TLS verification"),
    (re.compile(r"\byaml\.load\s*\((?![^\n]*SafeLoader)"), "unsafe YAML loading"),
)


def is_test_path(path: str) -> bool:
    normalized = path.strip().replace("\\", "/")
    return any(pattern.search(normalized) for pattern in _TEST_PATHS)


def repository_fingerprint(root: Path, max_files: int, max_bytes: int) -> str:
    """Hash repository bytes without following symlinks or persisting file contents."""
    digest = hashlib.sha256()
    seen_files = 0
    seen_bytes = 0
    excluded = {".git", ".venv", "venv", "env", "node_modules", "data", "__pycache__"}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts):
            continue
        size = path.stat().st_size
        seen_files += 1
        seen_bytes += size
        if seen_files > max_files or seen_bytes > max_bytes:
            raise ValueError("repository changed beyond configured integrity limits")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _added_patch_lines(patch: str) -> list[str]:
    return [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def changed_lines_by_file(patch: str) -> dict[str, set[int]]:
    """Extract new-file line numbers from a unified diff without applying it."""
    changed: dict[str, set[int]] = {}
    current_file = ""
    new_line = 0
    for line in patch.splitlines():
        if line.startswith("+++ "):
            value = line[4:].strip()
            current_file = "" if value == "/dev/null" else re.sub(r"^b/", "", value)
            if current_file:
                changed.setdefault(current_file, set())
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,\d+)?", line)
            if match:
                new_line = int(match.group(1))
            continue
        if not current_file or line.startswith("---"):
            continue
        if line.startswith("+"):
            changed[current_file].add(new_line)
            new_line += 1
        elif line.startswith(" "):
            new_line += 1
        elif line.startswith("-"):
            continue
    return {path: lines for path, lines in changed.items() if lines}


async def run_quality_gates(
    task: CodeTask,
    sandbox: DockerSandbox,
    *,
    patch: str,
    changed_files: list[str],
    source_fingerprint_before: str,
    baseline_lint: SandboxCommandResult | None,
) -> tuple[list[QualityGateResult], SandboxCommandResult]:
    """Run mandatory tests and deterministic policy checks for the current patch."""
    gates: list[QualityGateResult] = []
    final_test = await asyncio.to_thread(sandbox.run, task.test_command)
    gates.append(
        QualityGateResult(
            name="mandatory_tests",
            status="passed" if final_test.exit_code == 0 and not final_test.timed_out else "failed",
            summary=(
                "Owner-approved test command passed."
                if final_test.exit_code == 0 and not final_test.timed_out
                else "Owner-approved test command failed or timed out."
            ),
            command=task.test_command,
            duration_ms=final_test.duration_ms,
        )
    )

    scope_ok = bool(changed_files) and len(changed_files) <= task.policy.max_changed_files
    gates.append(
        QualityGateResult(
            name="patch_scope",
            status="passed" if scope_ok else "failed",
            summary=(
                f"Patch changes {len(changed_files)} file(s), within the {task.policy.max_changed_files}-file limit."
                if scope_ok
                else f"Patch changes {len(changed_files)} file(s); a non-empty patch within the configured limit is required."
            ),
        )
    )

    added = "\n".join(_added_patch_lines(patch))
    secret_hits = sum(bool(pattern.search(added)) for pattern in _SECRET_PATTERNS)
    gates.append(
        QualityGateResult(
            name="secret_scan",
            status="failed" if secret_hits else "passed",
            summary=(
                f"Detected {secret_hits} credential-like pattern(s) in added lines."
                if secret_hits
                else "No credential-like patterns detected in added lines."
            ),
        )
    )

    unsafe_hits = [description for pattern, description in _UNSAFE_PATTERNS if pattern.search(added)]
    gates.append(
        QualityGateResult(
            name="unsafe_code_scan",
            status="failed" if unsafe_hits else "passed",
            summary=(
                "Detected blocked pattern(s): " + ", ".join(unsafe_hits)
                if unsafe_hits
                else "No blocked dynamic-execution, deserialization, shell, or TLS patterns detected."
            ),
        )
    )

    try:
        source_after = await asyncio.to_thread(
            repository_fingerprint,
            sandbox.repository,
            task.policy.max_files,
            task.policy.max_repository_bytes,
        )
        source_unchanged = source_after == source_fingerprint_before
    except (OSError, ValueError):
        source_unchanged = False
    gates.append(
        QualityGateResult(
            name="source_repository_integrity",
            status="passed" if source_unchanged else "failed",
            summary=(
                "Source repository fingerprint is unchanged."
                if source_unchanged
                else "Source repository integrity could not be verified or changed during execution."
            ),
        )
    )

    python_files = any(path.endswith((".py", ".pyi")) for path in sandbox.workspace.list_files())
    if not python_files or "ruff" not in task.policy.allowed_test_executables:
        gates.append(
            QualityGateResult(
                name="ruff",
                status="skipped",
                summary="Ruff is not applicable or not operator-allowlisted.",
            )
        )
    elif baseline_lint is not None and baseline_lint.exit_code != 0:
        gates.append(
            QualityGateResult(
                name="ruff",
                status="skipped",
                summary="Repository had pre-existing Ruff failures; they are recorded without mislabeling them as patch regressions.",
                command=["ruff", "check", "."],
                duration_ms=baseline_lint.duration_ms,
            )
        )
    else:
        started = time.perf_counter()
        lint = await asyncio.to_thread(sandbox.run, ["ruff", "check", "."])
        gates.append(
            QualityGateResult(
                name="ruff",
                status="passed" if lint.exit_code == 0 and not lint.timed_out else "failed",
                summary="Ruff passed." if lint.exit_code == 0 else "Patch introduced or contains Ruff failures.",
                command=["ruff", "check", "."],
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        )

    coverage_requested = any(
        argument == "coverage" or argument.startswith("--cov")
        for argument in task.test_command
    )
    changed_lines = {
        path: lines
        for path, lines in changed_lines_by_file(patch).items()
        if path.endswith((".py", ".pyi"))
    }
    if not coverage_requested or not changed_lines:
        gates.append(
            QualityGateResult(
                name="changed_line_coverage",
                status="skipped",
                summary="Changed-line coverage requires a coverage-enabled Python test command and changed Python lines.",
            )
        )
    elif ".agentforge-coverage.json" in sandbox.workspace.list_files():
        gates.append(
            QualityGateResult(
                name="changed_line_coverage",
                status="failed",
                summary="Reserved coverage evidence path already exists in the repository.",
            )
        )
    else:
        coverage_command = ["python", "-m", "coverage", "json", "-o", ".agentforge-coverage.json"]
        coverage_result = await asyncio.to_thread(sandbox.run, coverage_command)
        try:
            coverage_payload = json.loads(
                await asyncio.to_thread(
                    sandbox.workspace.read_file, ".agentforge-coverage.json"
                )
            )
            files = coverage_payload.get("files") or {}
            total = covered = 0
            for path, lines in changed_lines.items():
                file_data = files.get(path) or files.get(path.replace("/", "\\")) or {}
                executed = set(file_data.get("executed_lines") or [])
                total += len(lines)
                covered += len(lines & executed)
            ratio = covered / total if total else 0.0
            passed = coverage_result.exit_code == 0 and ratio >= task.policy.min_changed_line_coverage
            gates.append(
                QualityGateResult(
                    name="changed_line_coverage",
                    status="passed" if passed else "failed",
                    summary=(
                        f"Changed-line coverage is {ratio:.1%} ({covered}/{total}); "
                        f"required minimum is {task.policy.min_changed_line_coverage:.1%}."
                    ),
                    command=coverage_command,
                    duration_ms=coverage_result.duration_ms,
                )
            )
        except (OSError, ValueError, json.JSONDecodeError):
            gates.append(
                QualityGateResult(
                    name="changed_line_coverage",
                    status="failed",
                    summary="Coverage was requested but valid changed-line evidence could not be produced.",
                    command=coverage_command,
                    duration_ms=coverage_result.duration_ms,
                )
            )
        finally:
            if ".agentforge-coverage.json" in sandbox.workspace.list_files():
                await asyncio.to_thread(
                    sandbox.workspace.delete_file, ".agentforge-coverage.json"
                )
    return gates, final_test


def blocking_reasons(report: VerificationReport) -> list[str]:
    if not report.rounds:
        return ["No independent review result was produced."]
    latest = report.rounds[-1]
    reasons = [gate.summary for gate in latest.quality_gates if gate.status == "failed"]
    reasons.extend(
        finding.message
        for finding in latest.verdict.findings
        if finding.severity in {"critical", "high"}
    )
    if not latest.verdict.approved and not reasons:
        reasons.append(latest.verdict.summary)
    return reasons


def build_pr_dossier(
    job: JobRecord,
    result: CodeAgentResult,
    patch_reference: ArtifactReference,
) -> tuple[str, str]:
    """Return canonical JSON and human-readable Markdown without embedding the patch."""
    verification = result.verification
    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "task": {
            "task_id": job.task_id,
            "repository": job.request.repository,
            "issue": job.request.issue,
            "model": job.request.model,
        },
        "outcome": {
            "status": result.status,
            "summary": result.summary,
            "changed_files": result.changed_files,
        },
        "verification": verification.model_dump(mode="json") if verification else None,
        "repair_tournament": result.tournament.model_dump(mode="json") if result.tournament else None,
        "execution": {
            "attempt": job.attempt,
            "iterations": result.iterations,
            "writes": result.writes,
            "tool_calls": result.tool_calls,
            "model_calls": result.model_calls,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "estimated_cost_usd": result.estimated_cost_usd,
            "duration_ms": result.total_duration_ms,
            "sandbox": result.sandbox,
        },
        "patch_artifact": patch_reference.model_dump(),
    }
    canonical_evidence = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    evidence["evidence_sha256"] = hashlib.sha256(canonical_evidence.encode("utf-8")).hexdigest()
    json_payload = json.dumps(evidence, indent=2, sort_keys=True)

    analysis = verification.analysis.summary if verification else "Unavailable"
    decision = verification.final_decision if verification else "blocked"
    reasons = verification.blocking_reasons if verification else ["Verification report unavailable."]
    gates = verification.rounds[-1].quality_gates if verification and verification.rounds else []
    findings = verification.rounds[-1].verdict.findings if verification and verification.rounds else []
    markdown = "\n".join(
        [
            f"# Verified PR evidence: {job.task_id}",
            "",
            f"- Repository: `{job.request.repository}`",
            f"- Model: `{job.request.model}`",
            f"- Decision: **{decision}**",
            f"- Patch SHA-256: `{patch_reference.sha256}`",
            f"- Evidence SHA-256: `{evidence['evidence_sha256']}`",
            "",
            "## Analysis",
            "",
            analysis,
            "",
            "## Changed files",
            "",
            *(f"- `{path}`" for path in result.changed_files),
            "",
            "## Quality gates",
            "",
            *(f"- **{gate.name} — {gate.status}:** {gate.summary}" for gate in gates),
            "",
            "## Independent review findings",
            "",
            *(f"- **{finding.severity}/{finding.category}:** {finding.message}" for finding in findings),
            *( ["- No findings."] if not findings else [] ),
            "",
            "## Blocking reasons",
            "",
            *(f"- {reason}" for reason in reasons),
            *( ["- None."] if not reasons else [] ),
            "",
            "This dossier contains evidence and artifact digests, not the patch body. Approval does not apply or merge code.",
            "",
        ]
    )
    if result.tournament:
        tournament = result.tournament
        markdown += "\n## Repair tournament\n\n"
        markdown += f"Selected: `{tournament.winner_id or 'none'}`; evidence SHA-256: `{tournament.fingerprint}`.\n\n"
        markdown += "| Candidate | Perspective | Eligible | Changed files | Changed lines | Reasons |\n"
        markdown += "|---|---|---|---:|---:|---|\n"
        for candidate in tournament.candidates:
            markdown += (f"| {candidate.candidate_id} | {candidate.perspective} | {candidate.eligible} | "
                         f"{candidate.changed_files} | {candidate.changed_lines} | {', '.join(candidate.reasons) or 'none'} |\n")
        if tournament.policy.regression_challenges:
            markdown += "\n### Pre-patch regression probes\n\n"
            markdown += f"Suite SHA-256: `{tournament.regression_suite_sha256 or 'unavailable'}`.\n\n"
            markdown += ("The JSON dossier retains the exact generated calls, expectations, and rationales for owner review. "
                         "Baseline discrimination does not establish oracle correctness. Raw function outputs are not retained.\n\n")
            markdown += "| Candidate | Probe outcomes | Stable repeated run | Workspace unchanged |\n|---|---|---|---|\n"
            for candidate in tournament.candidates:
                probes = candidate.regression_probes
                markdown += (f"| {candidate.candidate_id} | {', '.join(probes.statuses) if probes else 'not run'} | "
                             f"{probes.stable if probes else False} | {probes.workspace_unchanged if probes else False} |\n")
        if tournament.oracle_calibration:
            oracle = tournament.oracle_calibration
            markdown += "\n### Reference agreement and mutation sensitivity\n\n"
            markdown += (f"Eligible: `{oracle.eligible}`; killed `{oracle.killed_mutants}/{len(oracle.mutants)}`; "
                         f"score `{oracle.mutation_score:.3f}`; reference unchanged: `{oracle.reference_unchanged}`.\n\n")
            markdown += "| Mutant | Operator | Outcome | Killing probe indices |\n|---|---|---|---|\n"
            for mutant in oracle.mutants:
                markdown += f"| {mutant.mutant_id} | {mutant.operator} | {mutant.status} | {mutant.killing_probe_indices} |\n"
            markdown += ("\nReference and mutant source are not in this dossier. Agreement is conditional on the operator reference; "
                         "the greedy probe cover is diagnostic only and does not shorten the served suite.\n")
    return json_payload, markdown
