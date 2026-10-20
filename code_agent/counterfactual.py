"""Execution-confirmed regression discrimination in fresh Docker-only copies.

This is conditional test evidence, not a proof of the issue's oracle or causal
localization to a particular changed production line. Repository hooks can forge
receipts; hashes detect drift, not hostile-container authentication.
"""

from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path

from code_agent.models import CounterfactualReport, CounterfactualTrial, QualityGateResult
from code_agent.regression_challenges import blocking_safely, fingerprint
from code_agent.sandbox import DockerSandbox, DockerUnavailableError
from code_agent.verification import is_test_path, repository_fingerprint

RUNNER_PATH = ".agentforge-counterfactual.py"
RECEIPT_PATH = ".agentforge-counterfactual.json"
COMMAND = ["python", "-I", "-B", RUNNER_PATH]
MAX_TESTS = 200


def _functions(source):
    if len(source) > 100_000:
        raise ValueError("test source exceeds counterfactual parse bound")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        raise ValueError("test source is not valid Python") from None
    if sum(1 for _ in ast.walk(tree)) > 20_000:
        raise ValueError("test AST exceeds counterfactual parse bound")
    functions = {}
    for node in tree.body:
        candidates = [("", node)]
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            candidates = [(node.name + "::", child) for child in node.body]
        for prefix, function in candidates:
            if isinstance(function, ast.FunctionDef) and function.name.startswith("test_"):
                key = prefix + function.name
                if key in functions:
                    raise ValueError("duplicate test function identity")
                functions[key] = (ast.dump(function, include_attributes=False),
                                  function.lineno, function.end_lineno,
                                  any(isinstance(item, ast.Assert) for item in ast.walk(function)))
    return functions


def select_tests(workspace, original: Path, changed_files):
    """Parse data only; never import or execute repository source on the host."""
    overlays, eligible = {}, {}
    for path in sorted(changed_files):
        if not is_test_path(path) or not path.endswith(".py") or Path(path).name == "conftest.py":
            continue
        if not re.fullmatch(r"[A-Za-z0-9_./-]+\.py", path) or path.startswith(("/", "-")) or ".." in path.split("/"):
            raise ValueError("counterfactual test path is not supported")
        current_path = workspace.resolve(path, allow_missing=True)
        if not current_path.is_file() or current_path.is_symlink():
            raise ValueError("deleted or symlinked test files cannot supply discrimination")
        source = workspace.read_file(path)
        baseline = original / path
        if baseline.is_symlink() or not baseline.resolve().is_relative_to(original.resolve()):
            raise ValueError("original test path escapes the repository")
        if baseline.exists() and (not baseline.is_file() or baseline.stat().st_size > 100_000):
            raise ValueError("original test source exceeds counterfactual read bound")
        previous = _functions(baseline.read_text(encoding="utf-8") if baseline.is_file() else "")
        for name, (body, start, end, has_assertion) in _functions(source).items():
            if has_assertion and (name not in previous or previous[name][0] != body):
                eligible[path + "::" + name] = [start, end]
        if any(key.startswith(path + "::") for key in eligible):
            overlays[path] = source
        if len(overlays) > 6 or len(eligible) > 40:
            raise ValueError("counterfactual selection exceeds file/function limits")
    return overlays, eligible


# Fixed controller-authored harness. Generated tests run only inside Docker.
# Pytest hook contracts: https://docs.pytest.org/en/stable/reference/reference.html
_HARNESS = '''import os
import sys
import tempfile
from pathlib import Path

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
os.environ["PYTEST_ADDOPTS"] = ""
sys.pycache_prefix = tempfile.mkdtemp(prefix="af-counterfactual-cache-")
import pytest
sys.path.insert(0, str(Path.cwd()))
errors = {}
reports = {}
collected = []
collection_errors = 0
overflow = False

class ReceiptPlugin:
    def pytest_collection_modifyitems(self, session, config, items):
        global overflow
        selected = [item for item in items if item.nodeid.split("[", 1)[0] in ELIGIBLE]
        if len(selected) > MAX_TESTS:
            overflow = True
            pytest.exit("counterfactual selection exceeded", returncode=3)
        deselected = [item for item in items if item not in selected]
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
        collected.extend(item.nodeid for item in selected)

    def pytest_collectreport(self, report):
        global collection_errors
        if report.failed:
            collection_errors += 1

    def pytest_runtest_makereport(self, item, call):
        kind = "error"
        if call.excinfo is not None and call.excinfo.type is AssertionError:
            frame = call.excinfo.traceback[-1]
            try:
                path = Path(str(frame.path)).resolve().relative_to(Path.cwd().resolve()).as_posix()
                base = item.nodeid.split("[", 1)[0]
                bounds = ELIGIBLE.get(base)
                if bounds and path == base.split("::", 1)[0] and bounds[0] <= frame.lineno + 1 <= bounds[1]:
                    kind = "assertion"
            except (ValueError, OSError):
                pass
        errors[(item.nodeid, call.when)] = kind

    def pytest_runtest_logreport(self, report):
        phases = reports.setdefault(report.nodeid, {})
        if report.when in phases:
            phases[report.when] = "error"
        elif report.skipped or hasattr(report, "wasxfail"):
            phases[report.when] = "skipped"
        elif report.failed:
            phases[report.when] = errors.get((report.nodeid, report.when), "error") if report.when == "call" else "error"
        else:
            phases[report.when] = "passed"

exit_code = int(pytest.main(["--rootdir=.", "-q", "-p", "no:cacheprovider", "-o", "addopts=", *FILES], plugins=[ReceiptPlugin()]))
outcomes = {}
for node in collected:
    phases = reports.get(node, {})
    if set(phases) != {"setup", "call", "teardown"} or "error" in phases.values():
        status = "error"
    elif "skipped" in phases.values():
        status = "skipped"
    else:
        status = phases["call"]
    outcomes[node] = status
Path(RECEIPT_PATH).write_text(json.dumps({"manifest_sha256": MANIFEST_HASH, "exit_code": exit_code,
    "collected": collected, "outcomes": outcomes, "collection_errors": collection_errors,
    "overflow": overflow}), encoding="utf-8")
sys.exit(exit_code)
'''


def runner_source(overlays, eligible):
    manifest = {"files": sorted(overlays), "eligible": eligible, "max_tests": MAX_TESTS}
    source = (f"FILES = {manifest['files']!r}\nELIGIBLE = {eligible!r}\nMAX_TESTS = {MAX_TESTS}\n"
              f"MANIFEST_HASH = {fingerprint(manifest)!r}\nRECEIPT_PATH = {RECEIPT_PATH!r}\n"
              "import json\n" + _HARNESS)
    return source, fingerprint(manifest)


def decode_receipt(raw, manifest_hash, eligible, exit_code):
    if len(raw) > 100_000:
        raise ValueError("counterfactual receipt exceeds read bound")
    receipt = json.loads(raw)
    if (not isinstance(receipt, dict) or set(receipt) != {"manifest_sha256", "exit_code", "collected", "outcomes", "collection_errors", "overflow"}
            or receipt["manifest_sha256"] != manifest_hash or receipt["exit_code"] != exit_code
            or type(receipt["exit_code"]) is not int or type(receipt["collection_errors"]) is not int
            or receipt["collection_errors"] != 0 or receipt["overflow"] is not False
            or not isinstance(receipt["collected"], list) or not 1 <= len(receipt["collected"]) <= MAX_TESTS
            or not all(isinstance(node, str) and len(node) <= 1000 for node in receipt["collected"])
            or not isinstance(receipt["outcomes"], dict)):
        raise ValueError("counterfactual receipt binding or collection is invalid")
    nodes = receipt["collected"]
    if (len(nodes) != len(set(nodes)) or set(nodes) != set(receipt["outcomes"])
            or {node.split("[", 1)[0] for node in nodes} != set(eligible)
            or any(status not in {"passed", "assertion", "error", "skipped"} for status in receipt["outcomes"].values())):
        raise ValueError("counterfactual test identities or outcomes drifted")
    statuses = receipt["outcomes"].values()
    expected_exit = 1 if any(value in {"assertion", "error"} for value in statuses) else 0
    if exit_code != expected_exit:
        raise ValueError("counterfactual exit status contradicts receipt")
    return {fingerprint(node): status for node, status in sorted(receipt["outcomes"].items())}, fingerprint(receipt)


async def _trial(task, source, overlays, eligible, arm, repeat, policy, sandbox_factory):
    clone = sandbox_factory(source, policy)
    trial = CounterfactualTrial(arm=arm, repeat=repeat)
    try:
        await blocking_safely(clone.start)
        workspace = clone.workspace
        if any(workspace.resolve(path, allow_missing=True).exists() for path in (RUNNER_PATH, RECEIPT_PATH)):
            raise ValueError("reserved counterfactual paths already exist")
        if arm == "original":
            for path, content in overlays.items():
                await blocking_safely(lambda path=path, content=content: workspace.write_file(path, content))
        for path, content in overlays.items():
            if await blocking_safely(lambda path=path: workspace.read_file(path)) != content:
                raise ValueError("counterfactual overlay source drifted")
        before = await blocking_safely(lambda: repository_fingerprint(workspace.root, policy.max_files, policy.max_repository_bytes))
        trial.source_sha256 = before
        source_code, manifest_hash = runner_source(overlays, eligible)
        trial.manifest_sha256 = manifest_hash
        await blocking_safely(lambda: workspace.write_file(RUNNER_PATH, source_code))
        result = await blocking_safely(lambda: clone.run(list(COMMAND)))
        trial.exit_code = result.exit_code
        if result.timed_out:
            raise ValueError("counterfactual command timed out")
        receipt_path = workspace.resolve(RECEIPT_PATH)
        if receipt_path.is_symlink() or receipt_path.stat().st_size > 100_000:
            raise ValueError("counterfactual receipt exceeds safe read bound")
        raw = await blocking_safely(lambda: workspace.read_file(RECEIPT_PATH))
        trial.outcomes, trial.receipt_sha256 = decode_receipt(raw, manifest_hash, eligible, result.exit_code)
        for path in (RUNNER_PATH, RECEIPT_PATH):
            await blocking_safely(lambda path=path: workspace.delete_file(path))
        after = await blocking_safely(lambda: repository_fingerprint(workspace.root, policy.max_files, policy.max_repository_bytes))
        trial.workspace_unchanged = before == after
        trial.duration_ms = result.duration_ms
        return trial
    finally:
        await blocking_safely(clone.close)


def assess_trials(trials):
    if len(trials) != 4 or {(trial.arm, trial.repeat) for trial in trials} != {
            (arm, repeat) for arm in ("patched", "original") for repeat in (0, 1)}:
        return "incomplete_execution", []
    if any(not trial.workspace_unchanged or not trial.outcomes for trial in trials):
        return "workspace_or_receipt_invalid", []
    patched = [trial for trial in trials if trial.arm == "patched"]
    original = [trial for trial in trials if trial.arm == "original"]
    if (patched[0].outcomes != patched[1].outcomes or original[0].outcomes != original[1].outcomes
            or patched[0].source_sha256 != patched[1].source_sha256 or original[0].source_sha256 != original[1].source_sha256
            or len({trial.manifest_sha256 for trial in trials}) != 1
            or set(patched[0].outcomes) != set(original[0].outcomes)):
        return "unstable_or_mismatched_execution", []
    if any(status != "passed" for trial in patched for status in trial.outcomes.values()):
        return "patched_tests_not_all_passed", []
    if any(status not in {"passed", "assertion"} for trial in original for status in trial.outcomes.values()):
        return "original_errors_or_skips", []
    discriminating = sorted(node for node, status in original[0].outcomes.items() if status == "assertion")
    return ("stable_discrimination" if discriminating else "no_discriminating_test"), discriminating


async def verify_counterfactual(task, sandbox, changed_files, *, deadline=None, original_sha256="", sandbox_factory=None):
    report = CounterfactualReport()
    end = min(deadline if deadline is not None else float("inf"), time.perf_counter() + task.policy.counterfactual_timeout_seconds)
    candidate_before = original_before = ""
    try:
        if task.test_command not in (["python", "-m", "pytest", "-q"], ["python3", "-m", "pytest", "-q"], ["pytest", "-q"]):
            raise ValueError("unsupported_owner_command")
        if "python" not in task.policy.allowed_test_executables or task.policy.network_enabled:
            raise ValueError("unsupported_sandbox_policy")
        if not any(not is_test_path(path) for path in changed_files):
            raise ValueError("no_production_change")
        overlays, eligible = await blocking_safely(lambda: select_tests(sandbox.workspace, sandbox.repository, changed_files))
        if not eligible:
            raise ValueError("no_changed_assertion_tests")
        report.test_file_sha256 = {path: fingerprint(content) for path, content in overlays.items()}
        report.eligible_test_sha256 = sorted(fingerprint(node) for node in eligible)
        original_before = await blocking_safely(lambda: repository_fingerprint(sandbox.repository, task.policy.max_files, task.policy.max_repository_bytes))
        candidate_before = await blocking_safely(lambda: repository_fingerprint(sandbox.workspace.root, task.policy.max_files, task.policy.max_repository_bytes))
        report.original_sha256, report.candidate_sha256 = original_before, candidate_before
        if original_sha256 and original_sha256 != original_before:
            raise ValueError("original_source_drift")
        policy = task.policy.model_copy(update={"command_timeout_seconds": min(30, task.policy.command_timeout_seconds)})
        for arm in ("patched", "original"):
            for repeat in range(2):
                # Reserve one bounded command plus 30 seconds for final review/cleanup.
                if time.perf_counter() + policy.command_timeout_seconds + 30 >= end:
                    report.status, report.reason = "budget_exhausted", "verification_reserve_exhausted"
                    return _seal(report)
                source = sandbox.workspace.root if arm == "patched" else sandbox.repository
                report.trials.append(await _trial(task, source, overlays, eligible, arm, repeat, policy, sandbox_factory or DockerSandbox))
        report.reason, report.discriminating_test_sha256 = assess_trials(report.trials)
        report.status = "passed" if report.reason == "stable_discrimination" else "failed"
    except DockerUnavailableError:
        report.status, report.reason = "unavailable", "docker_unavailable"
    except (ValueError, OSError, RuntimeError, TypeError) as exc:
        # Do not persist repository-controlled exception strings or test output.
        codes = {"unsupported_owner_command", "unsupported_sandbox_policy", "no_production_change", "no_changed_assertion_tests", "original_source_drift"}
        report.reason = str(exc) if str(exc) in codes else "counterfactual_execution_invalid"
    finally:
        if candidate_before and original_before:
            try:
                original_after = await blocking_safely(lambda: repository_fingerprint(sandbox.repository, task.policy.max_files, task.policy.max_repository_bytes))
                candidate_after = await blocking_safely(lambda: repository_fingerprint(sandbox.workspace.root, task.policy.max_files, task.policy.max_repository_bytes))
                report.sources_unchanged = original_before == original_after and candidate_before == candidate_after
            except (ValueError, OSError):
                report.sources_unchanged = False
            if not report.sources_unchanged:
                report.status, report.reason = "failed", "source_integrity_changed"
        # The finally block also seals early budget returns after integrity checks.
        _seal(report)
    return report


def _seal(report):
    report.fingerprint = fingerprint(report.model_dump(mode="json", exclude={"fingerprint"}))
    return report


def counterfactual_gate(report):
    return QualityGateResult(name="counterfactual_regression", status="passed" if valid_report(report) else "failed",
        summary=f"Counterfactual regression evidence: {report.reason}; {len(report.discriminating_test_sha256)} discriminating test(s).")


def valid_report(report):
    """Successful publication requires intact, complete exposure/execution bindings."""
    try:
        return _valid_report(report)
    except (ValueError, TypeError, AttributeError):
        return False


def _valid_report(report):
    if report is None or report.status != "passed" or not report.sources_unchanged:
        return False
    if report.fingerprint != fingerprint(report.model_dump(mode="json", exclude={"fingerprint"})):
        return False
    hashes = [report.original_sha256, report.candidate_sha256, *report.eligible_test_sha256,
              *report.test_file_sha256.values()]
    if not report.eligible_test_sha256 or not report.test_file_sha256 or any(not re.fullmatch(r"[a-f0-9]{64}", value) for value in hashes):
        return False
    for trial in report.trials:
        if (any(not re.fullmatch(r"[a-f0-9]{64}", value) for value in
                [trial.source_sha256, trial.manifest_sha256, trial.receipt_sha256, *trial.outcomes])
                or trial.exit_code != (1 if "assertion" in trial.outcomes.values() else 0)):
            return False
    reason, nodes = assess_trials(report.trials)
    return reason == report.reason == "stable_discrimination" and nodes == report.discriminating_test_sha256
