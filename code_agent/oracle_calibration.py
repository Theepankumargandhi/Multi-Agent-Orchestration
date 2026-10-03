"""Private reference agreement and bounded AST mutation sensitivity; execution stays in Docker."""

from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path

from code_agent.models import (
    CodeTask,
    MutationProbeEvidence,
    OracleCalibrationPolicy,
    OracleCalibrationReport,
    RegressionProbeEvidence,
)
from code_agent.regression_challenges import (
    BehavioralProbeSuite,
    blocking_safely,
    candidate_matches,
    execute_suite,
    fingerprint,
)
from code_agent.sandbox import DockerSandbox
from code_agent.workspace import EXCLUDED_PARTS, SECRET_SUFFIXES, EphemeralWorkspace


def reference_fingerprint(root: Path, max_files: int = 5000, max_bytes: int = 50_000_000) -> str:
    """Secret-filtered source identity, with portable CRLF normalization for text only."""
    if not root.is_dir():
        raise ValueError("reference repository does not exist")
    digest = hashlib.sha256()
    count = total = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        name = path.name.lower()
        if (not path.is_file() or path.is_symlink() or any(part in EXCLUDED_PARTS for part in relative.parts)
                or name == ".env" or name.startswith(".env.") or path.suffix.lower() in SECRET_SUFFIXES):
            continue
        count += 1
        total += path.stat().st_size
        if count > max_files or total > max_bytes:
            raise ValueError("reference repository exceeds identity limits")
        data = path.read_bytes()
        if b"\0" not in data[:4096]:
            data = data.replace(b"\r\n", b"\n")
        digest.update(relative.as_posix().encode() + b"\0" + data + b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceMutant:
    path: str
    line: int
    operator: str
    source: str  # Private controller-only code, never serialized into a result or prompt.


_COMPARE = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_ARITHMETIC = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}


def generate_mutants(overlays: dict[str, str], target_names: dict[str, set[str]], limit: int) -> list[SourceMutant]:
    """Parse only: deterministic single AST edits inside explicit public function bodies."""
    if not 1 <= limit <= 8:
        raise ValueError("mutant cap must be between one and eight")
    mutants, seen = [], set()
    for path, source in sorted(overlays.items()):
        if len(source) > 100_000:
            raise ValueError("reference module exceeds bounded mutation input")
        tree = ast.parse(source)
        nodes = list(ast.walk(tree))
        if len(nodes) > 20_000:
            raise ValueError("reference AST exceeds node limit")
        selected = {id(node) for function in tree.body if isinstance(function, ast.FunctionDef)
                    and function.name in target_names[path] for statement in function.body for node in ast.walk(statement)}
        for position, node in enumerate(nodes):
            if id(node) not in selected:
                continue
            operator = None
            if isinstance(node, ast.Compare) and node.ops and type(node.ops[0]) in _COMPARE:
                operator = "comparison_flip"
            elif isinstance(node, ast.If):
                operator = "guard_removal"
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"min", "max"}:
                operator = "min_max_swap"
            elif isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
                operator = "arithmetic_swap"
            if operator is None:
                continue
            changed = copy.deepcopy(tree)
            replacement = list(ast.walk(changed))[position]
            if operator == "comparison_flip":
                replacement.ops[0] = _COMPARE[type(node.ops[0])]()
            elif operator == "guard_removal":
                replacement.test = ast.Constant(value=False)
            elif operator == "min_max_swap":
                replacement.func.id = "max" if node.func.id == "min" else "min"
            else:
                replacement.op = _ARITHMETIC[type(node.op)]()
            ast.fix_missing_locations(changed)
            text = ast.unparse(changed) + "\n"
            key = (path, ast.dump(changed, include_attributes=False))
            if key not in seen:
                seen.add(key)
                mutants.append(SourceMutant(path, node.lineno, operator, text))
            if len(mutants) == limit:
                return mutants
    return mutants


def execution_status(execution: RegressionProbeEvidence | None, suite_hash: str, count: int) -> tuple[str, list[int]]:
    if (execution is None or execution.suite_sha256 != suite_hash or len(execution.statuses) != count
            or not execution.repeated or not execution.stable or not execution.workspace_unchanged
            or "error" in execution.statuses):
        return "invalid", []
    killed = [index for index, status in enumerate(execution.statuses) if status == "mismatched"]
    return ("killed" if killed else "survived"), killed


def minimal_probe_cover(rows: list[MutationProbeEvidence]) -> list[int]:
    """Greedy diagnostic cover of observed kills; never reduces the runtime suite."""
    uncovered = {row.mutant_id for row in rows if row.status == "killed"}
    coverage: dict[int, set[str]] = {}
    for row in rows:
        if row.status == "killed":
            for index in row.killing_probe_indices:
                coverage.setdefault(index, set()).add(row.mutant_id)
    chosen = []
    while uncovered:
        index = min(coverage, key=lambda item: (-len(coverage[item] & uncovered), item))
        chosen.append(index)
        uncovered -= coverage[index]
    return chosen


def assessment(report: OracleCalibrationReport) -> tuple[int, float, list[int], list[str]]:
    killed = sum(row.status == "killed" for row in report.mutants)
    score = killed / len(report.mutants) if report.mutants else 0.0
    blockers = []
    if report.reference_sha256 != report.policy.reference_sha256 or not report.reference_unchanged:
        blockers.append("reference_source_changed_or_unbound")
    if not report.reference_owner_tests_passed:
        blockers.append("reference_owner_tests_not_passed")
    reference = report.reference_execution
    if (not reference or reference.suite_sha256 != report.suite_sha256 or len(reference.statuses) != report.probe_count
            or not candidate_matches(reference)):
        blockers.append("probe_expectations_disagree_with_reference")
    if len(report.mutants) < report.policy.min_mutants:
        blockers.append("insufficient_reference_mutants")
    if any(row.status == "invalid" for row in report.mutants):
        blockers.append("invalid_mutation_execution")
    if score < report.policy.min_mutation_score:
        blockers.append("mutation_sensitivity_below_threshold")
    return killed, score, minimal_probe_cover(report.mutants), blockers


def report_digest(report: OracleCalibrationReport) -> str:
    return fingerprint(report.model_dump(mode="json", exclude={"fingerprint"}))


async def refresh_reference_binding(report: OracleCalibrationReport, reference: Path | None, task: CodeTask) -> None:
    """Recheck the owner source at final selection; revocation cannot be cleared by a later read."""
    try:
        unchanged = bool(reference and report.policy.reference_sha256 == await blocking_safely(
            lambda: reference_fingerprint(reference, task.policy.max_files, task.policy.max_repository_bytes)))
    except (OSError, ValueError):
        unchanged = False
    report.reference_unchanged = report.reference_unchanged and unchanged
    killed, score, cover, blockers = assessment(report)
    report.killed_mutants, report.mutation_score, report.minimal_cover_indices = killed, score, cover
    report.blocking_reasons = list(dict.fromkeys([*report.blocking_reasons, *blockers]))
    report.eligible = not report.blocking_reasons
    report.fingerprint = report_digest(report)


def verify_calibration_report(report: OracleCalibrationReport) -> bool:
    if (report.fingerprint != report_digest(report) or len(report.mutants) > report.policy.max_mutants
            or len({row.mutant_id for row in report.mutants}) != len(report.mutants)
            or not 0 <= report.command_runs <= 1 + 2 * (1 + report.policy.max_mutants)):
        return False
    for row in report.mutants:
        status, indices = execution_status(row.execution, report.suite_sha256, report.probe_count)
        if row.status != status or row.killing_probe_indices != indices:
            return False
    killed, score, cover, blockers = assessment(report)
    return (report.killed_mutants == killed and report.mutation_score == score and report.minimal_cover_indices == cover
            and report.eligible == (not blockers and not report.blocking_reasons)
            and set(blockers) <= set(report.blocking_reasons)
            and (not report.eligible or bool(report.overlay_sha256 and report.command_runs == 1 + 2 * (1 + len(report.mutants)))))


async def calibrate_suite(task: CodeTask, baseline: Path, suite: BehavioralProbeSuite, policy: OracleCalibrationPolicy,
                          reference_repository: Path | None, *, sandbox_factory=DockerSandbox) -> OracleCalibrationReport:
    report = OracleCalibrationReport(policy=policy, suite_sha256=fingerprint(suite.model_dump(mode="json")), probe_count=len(suite.probes))
    try:
        await asyncio.wait_for(_calibrate_suite(task, baseline, suite, policy, reference_repository, report, sandbox_factory),
                               timeout=policy.timeout_seconds)
    except TimeoutError:
        report.blocking_reasons.append("oracle_calibration_deadline_exceeded")
    except Exception as exc:
        report.blocking_reasons.append(f"oracle_calibration_cleanup_error:{type(exc).__name__}")
    killed, score, cover, blockers = assessment(report)
    report.killed_mutants, report.mutation_score, report.minimal_cover_indices = killed, score, cover
    report.blocking_reasons = list(dict.fromkeys([*report.blocking_reasons, *blockers]))
    report.eligible = not report.blocking_reasons
    report.fingerprint = report_digest(report)
    return report


async def _calibrate_suite(task, baseline, suite, policy, reference_repository, report, sandbox_factory):
    reference = None
    frozen_reference = None
    sandbox = None
    reference_workspace = None
    try:
        if reference_repository is None or not reference_repository.is_absolute():
            raise ValueError("operator must supply an absolute reference repository path")
        reference = reference_repository.resolve()
        report.reference_sha256 = await blocking_safely(lambda: reference_fingerprint(reference, task.policy.max_files,
                                                                                      task.policy.max_repository_bytes))
        if report.reference_sha256 != policy.reference_sha256:
            raise ValueError("reference fingerprint does not match the pinned operator policy")
        reference_workspace = EphemeralWorkspace(reference, task.policy)
        await blocking_safely(reference_workspace.prepare)
        frozen_reference = reference_workspace.root
        if policy.reference_sha256 != await blocking_safely(lambda: reference_fingerprint(
                frozen_reference, task.policy.max_files, task.policy.max_repository_bytes)):
            raise ValueError("frozen reference does not match the pinned source identity")
        overlays, target_names = {}, {}
        for target in sorted({probe.target for probe in suite.probes}):
            module, function = target.rsplit(".", 1)
            path = module.replace(".", "/") + ".py"
            if not (baseline / path).is_file():
                path = module.replace(".", "/") + "/__init__.py"
            overlays[path] = reference_workspace.read_file(path)
            target_names.setdefault(path, set()).add(function)
        report.overlay_sha256 = {path: hashlib.sha256(source.encode()).hexdigest() for path, source in overlays.items()}
        mutants = generate_mutants(overlays, target_names, policy.max_mutants)
        sandbox = sandbox_factory(baseline, task.policy)
        await blocking_safely(sandbox.start)
        root = sandbox.workspace.root.resolve()
        if any(root.is_relative_to(source) or source.is_relative_to(root) for source in (baseline.resolve(), reference, frozen_reference)):
            raise ValueError("reference evaluation workspace is not isolated")
        for path, source in overlays.items():
            await blocking_safely(lambda path=path, source=source: sandbox.workspace.write_file(path, source))
        before = await blocking_safely(sandbox.workspace.unified_diff)
        report.command_runs += 1
        owner = await blocking_safely(lambda: sandbox.run(task.test_command))
        report.reference_owner_tests_passed = (owner.exit_code == 0 and not owner.timed_out and owner.command == task.test_command
            and await blocking_safely(sandbox.workspace.unified_diff) == before)
        if not report.reference_owner_tests_passed:
            raise ValueError("reference overlay did not preserve the owner tests and workspace")

        class CountedSandbox:
            workspace = sandbox.workspace
            def run(self, command):
                report.command_runs += 1
                return sandbox.run(command)

        report.reference_execution = await execute_suite(task, CountedSandbox(), suite)
        if not candidate_matches(report.reference_execution):
            raise ValueError("generated expectations disagree with the operator reference")
        for number, mutant in enumerate(mutants):
            row = MutationProbeEvidence(mutant_id=f"mutant-{number + 1}", path=mutant.path, line=mutant.line,
                operator=mutant.operator, source_sha256=hashlib.sha256(mutant.source.encode()).hexdigest())
            report.mutants.append(row)
            try:
                await blocking_safely(lambda: sandbox.workspace.write_file(mutant.path, mutant.source))
                row.execution = await execute_suite(task, CountedSandbox(), suite)
                row.status, row.killing_probe_indices = execution_status(row.execution, report.suite_sha256, report.probe_count)
            except asyncio.CancelledError:
                raise
            except Exception:
                row.status = "invalid"
            finally:
                await blocking_safely(lambda: sandbox.workspace.write_file(mutant.path, overlays[mutant.path]))
        if await blocking_safely(sandbox.workspace.unified_diff) != before:
            report.blocking_reasons.append("reference_verification_workspace_changed")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        report.blocking_reasons.append(f"oracle_calibration_error:{type(exc).__name__}")
    finally:
        try:
            if reference is not None:
                try:
                    report.reference_unchanged = report.reference_sha256 == await blocking_safely(lambda: reference_fingerprint(
                        reference, task.policy.max_files, task.policy.max_repository_bytes))
                except (OSError, ValueError):
                    report.reference_unchanged = False
        finally:
            try:
                if sandbox is not None:
                    await blocking_safely(sandbox.close)
            finally:
                if reference_workspace is not None:
                    await blocking_safely(reference_workspace.cleanup)
