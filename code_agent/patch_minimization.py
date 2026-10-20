"""Bounded file-level patch ablation; execution evidence is conditional on the tests."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import time

from code_agent.models import PatchAblationTrial, PatchMinimizationReport, QualityGateResult
from code_agent.regression_challenges import blocking_safely, fingerprint
from code_agent.sandbox import DockerSandbox, DockerUnavailableError
from code_agent.security_policy import classify_untrusted_content
from code_agent.verification import is_test_path, repository_fingerprint

RUNNER_PATH = ".agentforge-minimize.py"
RECEIPT_PATH = ".agentforge-minimize.json"
COMMAND = ["python", "-I", "-B", RUNNER_PATH]


def file_manifest(workspace, policy):
    """Hash the filtered copy view, including binary assets, without persisting bytes."""
    manifest, size = {}, 0
    paths = workspace.list_files(min(50_000, policy.max_files + 1))
    if len(paths) > policy.max_files or len(paths) >= 50_000:
        raise ValueError("minimization manifest file limit reached")
    for path in paths:
        resolved = workspace.resolve(path)
        size += resolved.stat().st_size
        if size > policy.max_repository_bytes:
            raise ValueError("minimization manifest exceeds bounds")
        manifest[path] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return manifest


def restoration_plan(sandbox, task, limit):
    selected = {}
    for path in sorted(sandbox.workspace.changed_files()):
        if (not path.endswith(".py") or is_test_path(path) or path.startswith(".agentforge-")
                or path.split("/")[-1] in {"conftest.py", "setup.py", "noxfile.py", "fabfile.py", "settings.py", "config.py"}):
            continue
        if not re.fullmatch(r"[A-Za-z0-9_./-]+\.py", path) or path.startswith(("/", "-")) or ".." in path.split("/"):
            continue
        original = sandbox.workspace.original_file(path)
        # Missing original text is ambiguous for an existing binary/oversized file.
        if original is None and (sandbox.repository / path).exists():
            continue
        current_path = sandbox.workspace.resolve(path, allow_missing=True)
        current = sandbox.workspace.read_file(path) if current_path.exists() else None
        texts = [text for text in (original, current) if text is not None]
        if any(len(text) > 100_000 or "\ufffd" in text for text in texts):
            continue
        if any(set(classify_untrusted_content(text)) & {"secret_material", "instruction_injection"} for text in texts):
            continue
        try:
            for text in texts:
                ast.parse(text)
        except (SyntaxError, RecursionError):
            continue
        if original != current:
            selected[path] = original
            if len(selected) == limit:
                break
    return selected


def reduced_manifest(manifest, restorations):
    reduced = dict(manifest)
    for path, text in restorations.items():
        if text is None:
            reduced.pop(path, None)
        else:
            reduced[path] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return reduced


def restore_file(workspace, path, text):
    if text is None:
        workspace.delete_file(path)
    else:
        workspace.write_file(path, text)


# This fixed harness records test identities/outcomes, not generated executable code.
# Repository hooks can forge results; fingerprints are not authentication.
_HARNESS = '''import hashlib
import os
import sys
import tempfile
from pathlib import Path
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
os.environ["PYTEST_ADDOPTS"] = ""
sys.pycache_prefix = tempfile.mkdtemp(prefix="af-minimize-cache-")
import pytest
sys.path.insert(0, str(Path.cwd()))
collected, reports = [], {}
collection_errors = 0
overflow = False
def node_hash(node):
    return hashlib.sha256(json.dumps(node, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
class Plugin:
    def pytest_collection_modifyitems(self, session, config, items):
        global overflow
        if len(items) > 200:
            overflow = True
            pytest.exit("minimization test limit exceeded", returncode=3)
        collected.extend(node_hash(item.nodeid) for item in items)
    def pytest_collectreport(self, report):
        global collection_errors
        collection_errors += int(report.failed)
    def pytest_runtest_logreport(self, report):
        phases = reports.setdefault(node_hash(report.nodeid), {})
        status = "skipped" if report.skipped or hasattr(report, "wasxfail") else report.outcome
        if report.when in phases:
            status = "error"
        phases[report.when] = status
exit_code = int(pytest.main(["--rootdir=.", "-q", "-p", "no:cacheprovider", "-o", "addopts="], plugins=[Plugin()]))
tests = {}
for node in collected:
    phases = reports.get(node, {})
    if set(phases) != {"setup", "call", "teardown"} or phases.get("setup") != "passed" or phases.get("teardown") != "passed":
        tests[node] = "error"
    else:
        tests[node] = phases["call"] if phases["call"] in {"passed", "failed", "skipped"} else "error"
Path(RECEIPT_PATH).write_text(json.dumps({"manifest_sha256": MANIFEST_HASH, "tests": tests,
    "collected": sorted(collected), "collection_errors": collection_errors,
    "overflow": overflow, "exit_code": exit_code}), encoding="utf-8")
sys.exit(exit_code)
'''


def runner_source(source_sha256):
    return f"MANIFEST_HASH = {source_sha256!r}\nRECEIPT_PATH = {RECEIPT_PATH!r}\nimport json\n" + _HARNESS


def receipt_data(source_sha256, outcomes, exit_code):
    return {"manifest_sha256": source_sha256, "tests": outcomes, "collected": sorted(outcomes),
            "collection_errors": 0, "overflow": False, "exit_code": exit_code}


def decode_receipt(raw, source_sha256, exit_code):
    if len(raw) > 200_000:
        raise ValueError("minimization receipt exceeds bounds")
    data = json.loads(raw)
    if (not isinstance(data, dict) or set(data) != {"manifest_sha256", "tests", "collected", "collection_errors", "overflow", "exit_code"}
            or data["manifest_sha256"] != source_sha256 or data["overflow"] is not False
            or type(data["collection_errors"]) is not int or data["collection_errors"] != 0
            or type(data["exit_code"]) is not int or data["exit_code"] != exit_code or exit_code not in {0, 1}
            or not isinstance(data["tests"], dict) or not 1 <= len(data["tests"]) <= 200
            or any(not isinstance(node, str) or not re.fullmatch(r"[a-f0-9]{64}", node) for node in data["tests"])
            or any(not isinstance(value, str) or value not in {"passed", "failed", "skipped", "error"} for value in data["tests"].values())
            or data["collected"] != sorted(data["tests"])):
        raise ValueError("minimization receipt invalid")
    if exit_code != int(any(status in {"failed", "error"} for status in data["tests"].values())):
        raise ValueError("minimization exit status inconsistent")
    return data["tests"], fingerprint(data)


async def _trial(task, sandbox, manifest, restorations, phase, repeat, factory):
    policy = task.policy.model_copy(update={"command_timeout_seconds": min(30, task.policy.command_timeout_seconds)})
    clone = factory(sandbox.workspace.root, policy)
    try:
        await blocking_safely(clone.start)
        workspace = clone.workspace
        if any(workspace.resolve(path, allow_missing=True).exists() for path in (RUNNER_PATH, RECEIPT_PATH)):
            raise ValueError("reserved minimization path exists")
        if await blocking_safely(lambda: file_manifest(workspace, policy)) != manifest:
            raise ValueError("candidate changed during copy")
        for path, text in restorations.items():
            await blocking_safely(lambda path=path, text=text: restore_file(workspace, path, text))
        prepared = await blocking_safely(lambda: file_manifest(workspace, policy))
        if prepared != reduced_manifest(manifest, restorations):
            raise ValueError("ablation source binding invalid")
        source_sha = fingerprint(prepared)
        await blocking_safely(lambda: workspace.write_file(RUNNER_PATH, runner_source(source_sha)))
        result = await blocking_safely(lambda: clone.run(list(COMMAND)))
        if result.timed_out or workspace.resolve(RECEIPT_PATH).stat().st_size > min(200_000, policy.max_file_bytes):
            raise ValueError("minimization execution exceeded bounds")
        raw = await blocking_safely(lambda: workspace.read_file(RECEIPT_PATH))
        outcomes, receipt_sha = decode_receipt(raw, source_sha, result.exit_code)
        for path in (RUNNER_PATH, RECEIPT_PATH):
            await blocking_safely(lambda path=path: workspace.delete_file(path))
        if await blocking_safely(lambda: file_manifest(workspace, policy)) != prepared:
            raise ValueError("ablation tests mutated source")
        return PatchAblationTrial(phase=phase, removed_paths=sorted(restorations), repeat=repeat,
            source_sha256=source_sha, receipt_sha256=receipt_sha, outcomes=outcomes, exit_code=result.exit_code,
            duration_ms=result.duration_ms)
    finally:
        await blocking_safely(clone.close)


def matching_passes(trials, expected=None):
    return (len(trials) == 2 and [row.repeat for row in trials] == [0, 1]
            and trials[0].source_sha256 == trials[1].source_sha256
            and trials[0].receipt_sha256 == trials[1].receipt_sha256
            and all(row.exit_code == 0 and row.outcomes and set(row.outcomes.values()) == {"passed"} for row in trials)
            and (expected is None or trials[0].outcomes == expected))


def _commit(sandbox, task, manifest, restorations, report):
    """Atomic in intent: synchronous commit verifies bytes and restores on any failure."""
    workspace = sandbox.workspace
    if file_manifest(workspace, task.policy) != manifest:
        raise ValueError("candidate changed before commit")
    saved = {path: workspace.read_file(path) if workspace.resolve(path, allow_missing=True).exists() else None
             for path in restorations}
    attempted = []
    try:
        for path, text in restorations.items():
            attempted.append(path)
            report.mutation_writes += 1
            restore_file(workspace, path, text)
        after = file_manifest(workspace, task.policy)
        if (after != reduced_manifest(manifest, restorations) or len(workspace.unified_diff()) >= report.patch_chars_before
                or repository_fingerprint(sandbox.repository, task.policy.max_files, task.policy.max_repository_bytes) != report.original_sha256):
            raise ValueError("committed minimization binding invalid")
        report.status, report.reason = "minimized", "repeated_test_preserving_file_ablation"
        report.removed_paths = sorted(restorations)
        report.restored_file_sha256 = {path: fingerprint(text) for path, text in restorations.items()}
    except (OSError, ValueError, RuntimeError):
        report.status, report.reason = "error", "commit_failed_original_candidate_restored"
        for path in reversed(attempted):
            report.mutation_writes += 1
            try:
                if saved[path] is None and not workspace.resolve(path, allow_missing=True).exists():
                    continue
                restore_file(workspace, path, saved[path])
            except (OSError, ValueError, RuntimeError):
                report.status, report.reason = "integrity_error", "commit_rollback_failed"
        if file_manifest(workspace, task.policy) != manifest:
            report.status, report.reason = "integrity_error", "commit_rollback_failed"


async def minimize_patch(task, sandbox, *, remaining_writes, original_sha256, deadline, sandbox_factory=None):
    started = time.perf_counter()
    report = PatchMinimizationReport(status="error", reason="minimization_execution_invalid",
                                     command_sha256=fingerprint(task.test_command))
    manifest = None
    try:
        if task.test_command not in (["python", "-m", "pytest", "-q"], ["python3", "-m", "pytest", "-q"], ["pytest", "-q"]):
            report.status, report.reason = "unsupported", "canonical_pytest_command_required"
            return report
        if task.policy.network_enabled or "python" not in task.policy.allowed_test_executables:
            report.status, report.reason = "unsupported", "sandbox_policy_unsupported"
            return report
        limit = min(task.policy.patch_minimization_max_files, max(0, remaining_writes // 2))
        if not limit:
            report.status, report.reason = "budget_exhausted", "commit_and_rollback_write_reserve"
            return report
        original = await blocking_safely(lambda: repository_fingerprint(sandbox.repository, task.policy.max_files, task.policy.max_repository_bytes))
        report.original_sha256 = original
        if original != original_sha256:
            report.status, report.reason = "integrity_error", "original_source_changed"
            return report
        manifest = await blocking_safely(lambda: file_manifest(sandbox.workspace, task.policy))
        report.candidate_before_sha256 = fingerprint(manifest)
        report.patch_chars_before = len(await blocking_safely(sandbox.workspace.unified_diff))
        plan = await blocking_safely(lambda: restoration_plan(sandbox, task, limit))
        report.eligible_paths = sorted(plan)
        if not plan:
            report.status, report.reason = "unchanged", "no_eligible_python_file_changes"
            return report
        end = min(deadline, started + task.policy.patch_minimization_timeout_seconds)

        async def pair(restorations, phase):
            rows = []
            for repeat in range(2):
                if time.perf_counter() + min(30, task.policy.command_timeout_seconds) + 30 >= end:
                    raise TimeoutError("verification reserve")
                trial = await _trial(task, sandbox, manifest, restorations, phase, repeat, sandbox_factory or DockerSandbox)
                report.trials.append(trial)
                rows.append(trial)
            return rows

        controls = await pair({}, "control")
        if not matching_passes(controls):
            report.status, report.reason = "rejected", "candidate_control_not_clean_and_stable"
            return report
        expected = controls[0].outcomes
        removals = {}
        for path, original_text in plan.items():
            proposal = {**removals, path: original_text}
            trials = await pair(proposal, "ablation")
            if matching_passes(trials, expected):
                removals = proposal
        if not removals:
            report.status, report.reason = "unchanged", "no_test_preserving_file_removal"
            return report
        confirmation = await pair(removals, "confirmation")
        if not matching_passes(confirmation, expected):
            report.status, report.reason = "rejected", "reduced_candidate_confirmation_failed"
            return report
        if time.perf_counter() + 30 >= end:
            raise TimeoutError("verification reserve")
        await blocking_safely(lambda: _commit(sandbox, task, manifest, removals, report))
    except DockerUnavailableError:
        report.status, report.reason = "unavailable", "docker_unavailable"
    except TimeoutError:
        report.status, report.reason = "budget_exhausted", "verification_reserve_exhausted"
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, subprocess.SubprocessError):
        pass
    finally:
        if manifest is not None:
            try:
                report.candidate_after_sha256 = fingerprint(await blocking_safely(lambda: file_manifest(sandbox.workspace, task.policy)))
                report.patch_chars_after = len(await blocking_safely(sandbox.workspace.unified_diff))
                original_after = await blocking_safely(lambda: repository_fingerprint(sandbox.repository, task.policy.max_files, task.policy.max_repository_bytes))
                if (original_after != report.original_sha256 or
                        (report.status != "minimized" and report.candidate_after_sha256 != report.candidate_before_sha256)):
                    report.status, report.reason = "integrity_error", "input_source_integrity_changed"
            except (OSError, ValueError):
                report.status, report.reason = "integrity_error", "input_source_integrity_unreadable"
        report.duration_ms = (time.perf_counter() - started) * 1000
        report.fingerprint = fingerprint(report.model_dump(mode="json", exclude={"fingerprint"}))
    return report


def valid_minimized_report(report):
    try:
        if (report is None or report.status != "minimized" or not report.removed_paths
                or report.fingerprint != fingerprint(report.model_dump(mode="json", exclude={"fingerprint"}))
                or not set(report.removed_paths) <= set(report.eligible_paths)
                or report.eligible_paths != sorted(set(report.eligible_paths))
                or report.removed_paths != sorted(set(report.removed_paths))
                or set(report.removed_paths) != set(report.restored_file_sha256)
                or report.mutation_writes != len(report.removed_paths)
                or report.patch_chars_after >= report.patch_chars_before
                or report.candidate_before_sha256 == report.candidate_after_sha256
                or any(not re.fullmatch(r"[a-f0-9]{64}", value) for value in
                       [report.original_sha256, report.candidate_before_sha256, report.candidate_after_sha256,
                        report.command_sha256, *report.restored_file_sha256.values()])):
            return False
        controls = [row for row in report.trials if row.phase == "control"]
        confirmation = [row for row in report.trials if row.phase == "confirmation"]
        if (len(report.trials) != 4 + 2 * len(report.eligible_paths)
                or any(row.phase != "control" for row in report.trials[:2])
                or any(row.phase != "ablation" for row in report.trials[2:-2])
                or any(row.phase != "confirmation" for row in report.trials[-2:])):
            return False
        accepted = []
        ablations = [row for row in report.trials if row.phase == "ablation"]
        for index, path in enumerate(report.eligible_paths):
            pair = ablations[index * 2:index * 2 + 2]
            if len(pair) != 2 or any(row.removed_paths != sorted([*accepted, path]) for row in pair):
                return False
            if matching_passes(pair, controls[0].outcomes):
                accepted.append(path)
        if sorted(accepted) != report.removed_paths:
            return False
        return (matching_passes(controls) and matching_passes(confirmation, controls[0].outcomes)
                and all(row.removed_paths == [] and row.source_sha256 == report.candidate_before_sha256 for row in controls)
                and all(row.removed_paths == report.removed_paths and row.source_sha256 == report.candidate_after_sha256 for row in confirmation)
                and all(row.receipt_sha256 == fingerprint(receipt_data(row.source_sha256, row.outcomes, row.exit_code)) for row in report.trials))
    except (ValueError, TypeError, AttributeError, IndexError):
        return False


def minimization_gate(report):
    sealed = report.fingerprint == fingerprint(report.model_dump(mode="json", exclude={"fingerprint"}))
    status = "failed" if not sealed or report.status == "integrity_error" or (report.status == "minimized" and not valid_minimized_report(report)) else "passed" if report.status == "minimized" else "skipped"
    return QualityGateResult(name="patch_minimization", status=status,
        summary=f"Advisory file ablation: {report.status}/{report.reason}; {len(report.removed_paths)} production-file change(s) removed.")
