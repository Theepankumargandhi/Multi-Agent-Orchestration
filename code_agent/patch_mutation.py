"""Patch-targeted test sensitivity. Parse on host, execute only in fresh Docker copies."""

from __future__ import annotations

import ast
import hashlib
import io
import re
import subprocess
import time
import tokenize
from bisect import bisect_left
from dataclasses import dataclass

from code_agent import counterfactual
from code_agent.models import MutationTestTrial, PatchMutationEvidence, PatchMutationReport, QualityGateResult
from code_agent.patch_minimization import file_manifest
from code_agent.regression_challenges import blocking_safely, fingerprint
from code_agent.sandbox import DockerSandbox, DockerUnavailableError
from code_agent.security_policy import classify_untrusted_content
from code_agent.verification import changed_lines_by_file, is_test_path, repository_fingerprint

_COMPARE = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "=="}
_ARITHMETIC = {"+": "-", "-": "+", "*": "//", "//": "*"}
_CONFIG = {"conftest.py", "setup.py", "noxfile.py", "fabfile.py", "settings.py", "config.py"}


@dataclass(frozen=True)
class PatchMutant:
    evidence: PatchMutationEvidence
    source: str  # Private controller-only text; never persisted in evidence or prompts.


def source_mutants(path, source, changed_lines, limit):
    """One-token edits inside function bodies; preserve every other source character."""
    if not 1 <= limit <= 6:
        raise ValueError("mutant limit must be between one and six")
    if len(source) > 100_000 or "\ufffd" in source:
        return []
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    if len(nodes) > 20_000:
        raise ValueError("mutation AST exceeds bounds")
    lines = source.splitlines(keepends=True)
    offsets, total = [], 0
    for line in lines:
        offsets.append(total)
        total += len(line)
    def ast_offset(node, end=False):
        row, column = (node.end_lineno, node.end_col_offset) if end else (node.lineno, node.col_offset)
        # AST columns are UTF-8 byte offsets; tokenize columns are characters.
        return offsets[row - 1] + len(lines[row - 1].encode("utf-8")[:column].decode("utf-8"))
    allowed = {id(item) for function in nodes if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
               for statement in function.body for item in ast.walk(statement)}
    candidates = []
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    if len(tokens) > 20_000:
        raise ValueError("mutation token stream exceeds bounds")
    records = [(offsets[token.start[0] - 1] + token.start[1], offsets[token.end[0] - 1] + token.end[1], token)
               for token in tokens if token.type in {tokenize.OP, tokenize.NAME}]
    starts = [left for left, _, _ in records]
    for node in nodes:
        if id(node) not in allowed:
            continue
        choices = {}
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            start, end = ast_offset(node.left, True), ast_offset(node.comparators[0])
            choices = _COMPARE
        elif isinstance(node, ast.BinOp):
            start, end = ast_offset(node.left, True), ast_offset(node.right)
            choices = _ARITHMETIC
        elif isinstance(node, ast.Constant) and type(node.value) is bool:
            start, end = ast_offset(node), ast_offset(node, True)
            choices = {"True": "False", "False": "True"}
        else:
            continue
        for left, right, token in records[bisect_left(starts, start):bisect_left(starts, end)]:
            if token.string not in choices or token.start[0] not in changed_lines:
                continue
            if not start <= left < right <= end:
                continue
            operator = ("boolean_flip" if isinstance(node, ast.Constant) else "arithmetic_swap" if isinstance(node, ast.BinOp)
                        else "comparison_flip" if token.string in {"==", "!="} else "boundary_shift")
            candidates.append((left, right, token.start[0], token.start[1], operator, choices[token.string]))
    result, seen = [], set()
    for left, right, line, column, operator, replacement in sorted(candidates):
        if (left, right, replacement) in seen:
            continue
        seen.add((left, right, replacement))
        mutated = source[:left] + replacement + source[right:]
        ast.parse(mutated)
        before = hashlib.sha256(source.encode()).hexdigest()
        after = hashlib.sha256(mutated.encode()).hexdigest()
        identity = fingerprint([path, line, column, operator, before, after])
        result.append(PatchMutant(PatchMutationEvidence(mutant_id=identity, path=path, line=line, column=column,
            operator=operator, original_file_sha256=before, mutated_file_sha256=after, expected_source_sha256=""), mutated))
        if len(result) == limit:
            break
    return result


def mutation_plan(workspace, patch, policy):
    result = []
    for path, lines in sorted(changed_lines_by_file(patch).items()):
        if (not path.endswith(".py") or is_test_path(path) or path.split("/")[-1] in _CONFIG
                or not re.fullmatch(r"[A-Za-z0-9_./-]+\.py", path) or path.startswith(("/", "-", ".agentforge-"))
                or ".." in path.split("/")):
            continue
        source = workspace.read_file(path)
        if set(classify_untrusted_content(source)) & {"secret_material", "instruction_injection"}:
            continue
        result.extend(source_mutants(path, source, lines, policy.mutation_testing_max_mutants - len(result)))
        if len(result) == policy.mutation_testing_max_mutants:
            break
    return result


def pair_status(rows, expected=None):
    if (len(rows) != 2 or [row.repeat for row in rows] != [0, 1] or not rows[0].outcomes
            or rows[0].outcomes != rows[1].outcomes or rows[0].source_sha256 != rows[1].source_sha256
            or rows[0].manifest_sha256 != rows[1].manifest_sha256 or rows[0].receipt_sha256 != rows[1].receipt_sha256
            or rows[0].exit_code != rows[1].exit_code
            or (expected is not None and set(rows[0].outcomes) != set(expected))
            or any(value not in {"passed", "assertion"} for row in rows for value in row.outcomes.values())
            or rows[0].exit_code != int("assertion" in rows[0].outcomes.values())):
        return "invalid", []
    killed = sorted(node for node, outcome in rows[0].outcomes.items() if outcome == "assertion")
    return ("killed" if killed else "survived"), killed


async def _trial(task, sandbox, manifest, overlays, eligible, mutant, repeat, factory):
    policy = task.policy.model_copy(update={"command_timeout_seconds": min(30, task.policy.command_timeout_seconds)})
    clone = factory(sandbox.workspace.root, policy)
    try:
        await blocking_safely(clone.start)
        workspace = clone.workspace
        if any(workspace.resolve(path, allow_missing=True).exists() for path in (counterfactual.RUNNER_PATH, counterfactual.RECEIPT_PATH)):
            raise ValueError("reserved mutation harness path exists")
        if await blocking_safely(lambda: file_manifest(workspace, policy)) != manifest:
            raise ValueError("candidate source changed during copy")
        expected = dict(manifest)
        if mutant:
            await blocking_safely(lambda: workspace.write_file(mutant.evidence.path, mutant.source))
            expected[mutant.evidence.path] = mutant.evidence.mutated_file_sha256
        if await blocking_safely(lambda: file_manifest(workspace, policy)) != expected:
            raise ValueError("mutant source binding invalid")
        source_sha = fingerprint(expected)
        for path, content in overlays.items():
            if await blocking_safely(lambda path=path: workspace.read_file(path)) != content:
                raise ValueError("selected tests changed")
        harness, manifest_sha = counterfactual.runner_source(overlays, eligible)
        await blocking_safely(lambda: workspace.write_file(counterfactual.RUNNER_PATH, harness))
        execution = await blocking_safely(lambda: clone.run(list(counterfactual.COMMAND)))
        receipt_path = workspace.resolve(counterfactual.RECEIPT_PATH)
        if execution.timed_out or receipt_path.is_symlink() or receipt_path.stat().st_size > 100_000:
            raise ValueError("mutation receipt or execution invalid")
        raw = await blocking_safely(lambda: workspace.read_file(counterfactual.RECEIPT_PATH))
        outcomes, receipt_sha = counterfactual.decode_receipt(raw, manifest_sha, eligible, execution.exit_code)
        for path in (counterfactual.RUNNER_PATH, counterfactual.RECEIPT_PATH):
            await blocking_safely(lambda path=path: workspace.delete_file(path))
        if await blocking_safely(lambda: file_manifest(workspace, policy)) != expected:
            raise ValueError("mutation tests changed source")
        return MutationTestTrial(phase="mutant" if mutant else "control", mutant_id=mutant.evidence.mutant_id if mutant else "",
            repeat=repeat, source_sha256=source_sha, manifest_sha256=manifest_sha, receipt_sha256=receipt_sha,
            outcomes=outcomes, exit_code=execution.exit_code, duration_ms=execution.duration_ms)
    finally:
        await blocking_safely(clone.close)


async def test_patch_mutations(task, sandbox, patch, changed_files, *, original_sha256, deadline, sandbox_factory=None):
    started = time.perf_counter()
    report = PatchMutationReport(mode=task.policy.mutation_testing_mode, min_mutants=task.policy.mutation_testing_min_mutants,
                                 min_score=task.policy.mutation_testing_min_score, command_sha256=fingerprint(task.test_command))
    manifest = None
    try:
        if (task.test_command not in (["python", "-m", "pytest", "-q"], ["python3", "-m", "pytest", "-q"], ["pytest", "-q"])
                or task.policy.network_enabled or "python" not in task.policy.allowed_test_executables):
            report.status, report.reason = "unsupported", "canonical_offline_pytest_required"
            return report
        report.original_sha256 = await blocking_safely(lambda: repository_fingerprint(sandbox.repository, task.policy.max_files, task.policy.max_repository_bytes))
        if report.original_sha256 != original_sha256:
            report.status, report.reason = "integrity_error", "original_source_changed"
            return report
        manifest = await blocking_safely(lambda: file_manifest(sandbox.workspace, task.policy))
        report.candidate_sha256 = fingerprint(manifest)
        overlays, eligible = await blocking_safely(lambda: counterfactual.select_tests(sandbox.workspace, sandbox.repository, changed_files))
        report.test_file_sha256 = {path: fingerprint(content) for path, content in overlays.items()}
        report.eligible_test_sha256 = sorted(fingerprint(node) for node in eligible)
        plan = await blocking_safely(lambda: mutation_plan(sandbox.workspace, patch, task.policy))
        for mutant in plan:
            expected = {**manifest, mutant.evidence.path: mutant.evidence.mutated_file_sha256}
            mutant.evidence.expected_source_sha256 = fingerprint(expected)
        report.mutants = [mutant.evidence for mutant in plan]
        if not eligible or len(plan) < report.min_mutants:
            report.status, report.reason = "insufficient_evidence", "changed_tests_and_mutation_targets_required"
            return report
        end = min(deadline, started + task.policy.mutation_testing_timeout_seconds)
        async def pair(mutant):
            rows = []
            for repeat in (0, 1):
                if time.perf_counter() + min(30, task.policy.command_timeout_seconds) + 30 >= end:
                    raise TimeoutError("verification reserve")
                row = await _trial(task, sandbox, manifest, overlays, eligible, mutant, repeat, sandbox_factory or DockerSandbox)
                report.trials.append(row)
                rows.append(row)
            return rows
        controls = await pair(None)
        if pair_status(controls)[0] != "survived":
            report.status, report.reason = "inconclusive", "candidate_control_not_clean_and_stable"
            return report
        for mutant in plan:
            rows = await pair(mutant)
            mutant.evidence.status, mutant.evidence.killing_test_sha256 = pair_status(rows, controls[0].outcomes)
        report.status = "inconclusive" if any(row.status == "invalid" for row in report.mutants) else "measured"
        report.reason = "invalid_mutation_outcomes" if report.status == "inconclusive" else "repeated_assertion_sensitivity"
    except DockerUnavailableError:
        report.status, report.reason = "unavailable", "docker_unavailable"
    except TimeoutError:
        report.status, report.reason = "budget_exhausted", "verification_reserve_exhausted"
    except (OSError, ValueError, RuntimeError, TypeError, SyntaxError, RecursionError, tokenize.TokenError, subprocess.SubprocessError):
        report.status, report.reason = "error", "mutation_execution_invalid"
    finally:
        report.killed = sum(row.status == "killed" for row in report.mutants)
        report.survived = sum(row.status == "survived" for row in report.mutants)
        report.invalid = sum(row.status == "invalid" for row in report.mutants)
        report.score = report.killed / len(report.mutants) if report.mutants else 0
        if manifest is not None:
            try:
                report.sources_unchanged = (manifest == await blocking_safely(lambda: file_manifest(sandbox.workspace, task.policy))
                    and report.original_sha256 == await blocking_safely(lambda: repository_fingerprint(sandbox.repository, task.policy.max_files, task.policy.max_repository_bytes)))
            except (OSError, ValueError):
                report.sources_unchanged = False
            if not report.sources_unchanged:
                report.status, report.reason = "integrity_error", "input_source_integrity_changed"
        report.duration_ms = (time.perf_counter() - started) * 1000
        report.fingerprint = fingerprint(report.model_dump(mode="json", exclude={"fingerprint"}))
    return report


def valid_measurement(report):
    """Check complete measurement lineage, not whether it meets an owner's threshold."""
    try:
        if (report is None or report.status != "measured" or not report.sources_unchanged
                or report.fingerprint != fingerprint(report.model_dump(mode="json", exclude={"fingerprint"}))
                or not report.eligible_test_sha256 or not report.test_file_sha256 or not report.mutants
                or len({row.mutant_id for row in report.mutants}) != len(report.mutants)
                or len(report.trials) != 2 + 2 * len(report.mutants)):
            return False
        controls = report.trials[:2]
        if (any(row.phase != "control" or row.mutant_id or row.source_sha256 != report.candidate_sha256 for row in controls)
                or pair_status(controls)[0] != "survived"):
            return False
        for index, mutant in enumerate(report.mutants):
            rows = report.trials[2 + 2 * index:4 + 2 * index]
            status, kills = pair_status(rows, controls[0].outcomes)
            if (status not in {"killed", "survived"} or mutant.status != status or mutant.killing_test_sha256 != kills
                    or mutant.original_file_sha256 == mutant.mutated_file_sha256
                    or mutant.mutant_id != fingerprint([mutant.path, mutant.line, mutant.column, mutant.operator,
                                                       mutant.original_file_sha256, mutant.mutated_file_sha256])
                    or any(row.phase != "mutant" or row.mutant_id != mutant.mutant_id
                           or row.source_sha256 != mutant.expected_source_sha256 for row in rows)):
                return False
        hashes = [report.original_sha256, report.candidate_sha256, report.command_sha256, *report.test_file_sha256.values(), *report.eligible_test_sha256]
        hashes += [value for row in report.trials for value in [row.source_sha256, row.manifest_sha256, row.receipt_sha256, *row.outcomes]]
        hashes += [value for row in report.mutants for value in [row.original_file_sha256, row.mutated_file_sha256, row.expected_source_sha256]]
        return (all(re.fullmatch(r"[a-f0-9]{64}", value) for value in hashes)
                and len({row.manifest_sha256 for row in report.trials}) == 1
                and report.killed == sum(row.status == "killed" for row in report.mutants)
                and report.survived == len(report.mutants) - report.killed and report.invalid == 0
                and report.score == report.killed / len(report.mutants))
    except (ValueError, TypeError, AttributeError, IndexError):
        return False


def mutation_gate(report):
    sealed = report.fingerprint == fingerprint(report.model_dump(mode="json", exclude={"fingerprint"}))
    qualified = (valid_measurement(report) and len(report.mutants) >= report.min_mutants and report.score >= report.min_score)
    failed = (not sealed or report.status == "integrity_error" or (report.status == "measured" and not valid_measurement(report))
              or (report.mode == "enforce" and not qualified))
    return QualityGateResult(name="patch_mutation_sensitivity", status="failed" if failed else "passed" if qualified and report.mode == "enforce" else "skipped",
        summary=f"Patch mutation sensitivity ({report.mode}): {report.status}/{report.reason}; "
                f"killed={report.killed}, survived={report.survived}, invalid={report.invalid}, planned={len(report.mutants)}; "
                f"score={report.score:.1%}, required={report.min_score:.1%}. Conditional test evidence, not correctness proof.")
