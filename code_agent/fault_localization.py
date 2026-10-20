"""Repeated Docker spectra ground repair context in executed production lines.

Ochiai scores are associations, not calibrated probabilities or causal proofs.
Repository/container hooks can forge instrumentation: hashes detect drift only.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
from pathlib import Path

from code_agent.models import FaultLocalizationReceipt, SuspiciousLine
from code_agent.regression_challenges import blocking_safely, fingerprint
from code_agent.sandbox import DockerSandbox, DockerUnavailableError
from code_agent.security_policy import classify_untrusted_content
from code_agent.verification import is_test_path, repository_fingerprint

RUNNER_PATH = ".agentforge-spectrum.py"
RECEIPT_PATH = ".agentforge-spectrum.json"
COMMAND = ["python", "-I", "-B", RUNNER_PATH]
MAX_TESTS = 200
MAX_LINES = 20_000


class InstrumentationUnavailable(RuntimeError):
    pass


def production_sources(workspace, policy):
    sources = {}
    total = 0
    for path in workspace.list_files(policy.max_files):
        if not path.endswith(".py") or is_test_path(path) or path.startswith(".agentforge-") or Path(path).name == "conftest.py":
            continue
        if not re.fullmatch(r"[A-Za-z0-9_./-]+\.py", path) or path.startswith(("/", "-")) or ".." in path.split("/"):
            raise ValueError("unsupported spectrum source path")
        source = workspace.read_file(path)
        total += len(source)
        if len(sources) >= 300 or len(source) > 100_000 or total > 2_000_000:
            raise ValueError("spectrum source bounds exceeded")
        sources[path] = source
    return sources


# Trusted harness, not generated test code. Coverage configuration and ambient
# pytest options are disabled; conftest hooks still execute inside the container.
_HARNESS = '''import hashlib
import os
import sys
import tempfile
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
os.environ["PYTEST_ADDOPTS"] = ""
sys.pycache_prefix = tempfile.mkdtemp(prefix="af-spectrum-cache-")
try:
    import coverage
    import pytest
except ImportError:
    Path(RECEIPT_PATH).write_text(json.dumps({"manifest_sha256": MANIFEST_HASH, "unavailable": True}), encoding="utf-8")
    sys.exit(3)
sys.path.insert(0, str(Path.cwd()))
def sources_match():
    return all(digest(Path(path).read_bytes().decode("utf-8", errors="replace")) == expected for path, expected in SOURCES.items())
if not sources_match():
    sys.exit(3)
cov = coverage.Coverage(data_file=None, config_file=False, include=[str(Path.cwd() / path) for path in SOURCES])
cov.start()
collected = []
reports = {}
collection_errors = 0
overflow = False

class SpectrumPlugin:
    def pytest_collection_modifyitems(self, session, config, items):
        global overflow
        if len(items) > MAX_TESTS:
            overflow = True
            pytest.exit("spectrum test bound exceeded", returncode=3)
        collected.extend(digest(item.nodeid) for item in items)

    def pytest_collectreport(self, report):
        global collection_errors
        if report.failed:
            collection_errors += 1

    def pytest_runtest_logstart(self, nodeid, location):
        cov.switch_context(digest(nodeid))

    def pytest_runtest_logfinish(self, nodeid, location):
        cov.switch_context("")

    def pytest_runtest_logreport(self, report):
        phases = reports.setdefault(digest(report.nodeid), {})
        status = "skipped" if report.skipped or hasattr(report, "wasxfail") else report.outcome
        if report.when in phases:
            status = "error"
        phases[report.when] = status

exit_code = int(pytest.main(["--rootdir=.", "-q", "-p", "no:cacheprovider", "-o", "addopts="], plugins=[SpectrumPlugin()]))
cov.stop()
tests = {}
for node in collected:
    phases = reports.get(node, {})
    if set(phases) != {"setup", "call", "teardown"} or phases.get("setup") != "passed" or phases.get("teardown") != "passed":
        tests[node] = "error"
    elif phases.get("call") in {"passed", "failed", "skipped"}:
        tests[node] = phases["call"]
    else:
        tests[node] = "error"
spectra = {}
data = cov.get_data()
for filename in sorted(data.measured_files()):
    try:
        path = Path(filename).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (ValueError, OSError):
        continue
    if path not in SOURCES:
        continue
    counts = {}
    for line, contexts in data.contexts_by_lineno(filename).items():
        failed = sum(tests.get(context) == "failed" for context in set(contexts))
        passed = sum(tests.get(context) == "passed" for context in set(contexts))
        if failed or passed:
            counts[str(line)] = [failed, passed]
    spectra[path] = counts
    if sum(len(lines) for lines in spectra.values()) > MAX_LINES:
        overflow = True
        spectra = {}
        break
if not sources_match():
    sys.exit(3)
Path(RECEIPT_PATH).write_text(json.dumps({"manifest_sha256": MANIFEST_HASH, "tests": tests,
    "collected": collected, "spectra": spectra, "collection_errors": collection_errors,
    "overflow": overflow, "exit_code": exit_code}), encoding="utf-8")
sys.exit(exit_code)
'''


def runner_source(sources):
    manifest = {"sources": {path: fingerprint(text) for path, text in sorted(sources.items())},
                "max_tests": MAX_TESTS, "max_lines": MAX_LINES}
    manifest_hash = fingerprint(manifest)
    source = (f"SOURCES = {manifest['sources']!r}\nMAX_TESTS = {MAX_TESTS}\nMAX_LINES = {MAX_LINES}\n"
              f"MANIFEST_HASH = {manifest_hash!r}\nRECEIPT_PATH = {RECEIPT_PATH!r}\nimport json\n" + _HARNESS)
    return source, manifest_hash


def decode_spectrum(raw, manifest_hash, sources, exit_code):
    if len(raw) > 2_000_000:
        raise ValueError("spectrum receipt exceeds read bound")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("manifest_sha256") != manifest_hash:
        raise ValueError("spectrum manifest binding invalid")
    if set(data) == {"manifest_sha256", "unavailable"} and data["unavailable"] is True and exit_code == 3:
        raise InstrumentationUnavailable("instrumentation unavailable")
    if (set(data) != {"manifest_sha256", "tests", "collected", "spectra", "collection_errors", "overflow", "exit_code"}
            or data["overflow"] is not False or type(data["collection_errors"]) is not int or data["collection_errors"] != 0
            or type(data["exit_code"]) is not int or data["exit_code"] != exit_code or exit_code not in {0, 1}
            or not isinstance(data["tests"], dict) or not 1 <= len(data["tests"]) <= MAX_TESTS
            or not isinstance(data["collected"], list) or len(data["collected"]) != len(data["tests"])
            or any(not isinstance(node, str) or not re.fullmatch(r"[a-f0-9]{64}", node) for node in data["collected"])
            or set(data["collected"]) != set(data["tests"])
            or any(value not in {"passed", "failed", "skipped", "error"} for value in data["tests"].values())
            or not isinstance(data["spectra"], dict) or not set(data["spectra"]) <= set(sources)):
        raise ValueError("spectrum outcomes or collection invalid")
    failed = sum(value == "failed" for value in data["tests"].values())
    passed = sum(value == "passed" for value in data["tests"].values())
    if exit_code != (1 if failed or "error" in data["tests"].values() else 0):
        raise ValueError("spectrum exit status inconsistent")
    if sum(len(lines) for lines in data["spectra"].values() if isinstance(lines, dict)) > MAX_LINES:
        raise ValueError("spectrum line count exceeds bound")
    for path, lines in data["spectra"].items():
        if not isinstance(lines, dict):
            raise ValueError("spectrum line map invalid")
        length = len(sources[path].splitlines())
        for line, counts in lines.items():
            if (not re.fullmatch(r"[1-9][0-9]{0,6}", line) or not 1 <= int(line) <= length
                    or not isinstance(counts, list) or len(counts) != 2
                    or any(type(count) is not int for count in counts)
                    or not 0 <= counts[0] <= failed or not 0 <= counts[1] <= passed or not any(counts)):
                raise ValueError("spectrum counts or source lines invalid")
    return data, failed, passed


def rank_lines(spectra, sources, failing_tests):
    if not failing_tests:
        return []
    rows = []
    for path, lines in spectra.items():
        for line, (failed, passed) in lines.items():
            if failed:
                rows.append(SuspiciousLine(path=path, line=int(line), source_sha256=fingerprint(sources[path]),
                    failed_cover=failed, passed_cover=passed, score=failed / math.sqrt(failing_tests * (failed + passed))))
    return sorted(rows, key=lambda row: (-row.score, -row.failed_cover, row.passed_cover, row.path, row.line))[:12]


def pack_lines(ranked, sources, max_tokens):
    sections, chosen = [], []
    seen = set()
    for row in ranked:
        lines = sources[row.path].splitlines()
        start, end = max(1, row.line - 1), min(len(lines), row.line + 1)
        indices = [number for number in range(start, end + 1) if (row.path, number) not in seen]
        if row.line not in indices:
            continue
        section = f"{row.path}:{row.line} source={row.source_sha256} score={row.score:.4f} failed={row.failed_cover} passed={row.passed_cover}\n"
        section += "\n".join(f"{number}: {lines[number - 1]}" for number in indices)
        if len("\n\n".join([*sections, section])) > max_tokens * 4:
            continue
        sections.append(section)
        chosen.append(row)
        seen.update((row.path, number) for number in indices)
    context = "\n\n".join(sections)
    if set(classify_untrusted_content(context)) & {"secret_material", "instruction_injection"}:
        return "", [], True
    return context, chosen, False


async def collect_spectrum(task, sandbox, sources, sandbox_factory=None):
    clone = (sandbox_factory or DockerSandbox)(sandbox.workspace.root, task.policy.model_copy(
        update={"command_timeout_seconds": min(30, task.policy.command_timeout_seconds)}))
    try:
        await blocking_safely(clone.start)
        workspace = clone.workspace
        for path in (RUNNER_PATH, RECEIPT_PATH):
            if workspace.resolve(path, allow_missing=True).exists():
                raise ValueError("reserved spectrum path already exists")
        for path, source in sources.items():
            if await blocking_safely(lambda path=path: workspace.read_file(path)) != source:
                raise ValueError("spectrum source changed during copy")
        before = await blocking_safely(lambda: repository_fingerprint(workspace.root, task.policy.max_files, task.policy.max_repository_bytes))
        runner, manifest_hash = runner_source(sources)
        await blocking_safely(lambda: workspace.write_file(RUNNER_PATH, runner))
        result = await blocking_safely(lambda: clone.run(list(COMMAND)))
        if result.timed_out:
            raise ValueError("spectrum command timed out")
        receipt_path = workspace.resolve(RECEIPT_PATH)
        if receipt_path.stat().st_size > min(task.policy.max_file_bytes, 2_000_000):
            raise ValueError("spectrum receipt exceeds safe read bound")
        raw = await blocking_safely(lambda: workspace.read_file(RECEIPT_PATH))
        decoded, failed, passed = decode_spectrum(raw, manifest_hash, sources, result.exit_code)
        for path in (RUNNER_PATH, RECEIPT_PATH):
            await blocking_safely(lambda path=path: workspace.delete_file(path))
        after = await blocking_safely(lambda: repository_fingerprint(workspace.root, task.policy.max_files, task.policy.max_repository_bytes))
        if before != after:
            raise ValueError("spectrum trial modified workspace")
        # Repeat binding includes source bytes, exact test identities/outcomes and spectra.
        return decoded, failed, passed, fingerprint({"source": before, "receipt": decoded})
    finally:
        await blocking_safely(clone.close)


class SpectrumLocalizer:
    """One per coding stage; edits clear context, and probes await a fresh owner failure."""

    def __init__(self, stage, *, sandbox_factory=None):
        self.stage = stage
        self.sandbox_factory = sandbox_factory
        self.context = ""
        self.receipts = []
        self.refreshes = 0
        self.last_key = None
        self.source_hash = ""

    async def refresh(self, task, sandbox, observations, deadline):
        tests = [row for row in observations if row.action in {"test", "finish"}]
        if not tests or tests[-1].ok:
            self.context = ""
            return ""
        failure = tests[-1]
        revision = max((row.iteration for row in observations if row.ok and row.action in {"write", "edit", "delete"}), default=-1)
        key = (failure.iteration, fingerprint(failure.output), revision)
        receipt = FaultLocalizationReceipt(stage=self.stage, iteration=max(row.iteration for row in observations),
            status="error", reason="instrumentation_invalid", command_sha256=fingerprint(task.test_command))
        started = time.perf_counter()
        cached = False
        try:
            current_hash = await blocking_safely(lambda: repository_fingerprint(sandbox.workspace.root, task.policy.max_files, task.policy.max_repository_bytes))
            if key == self.last_key and current_hash == self.source_hash:
                cached = True
                return self.context
            self.context = ""
            previous_key = self.last_key
            self.last_key, self.source_hash = key, current_hash
            receipt.source_sha256 = current_hash
            if revision > failure.iteration:
                receipt.status, receipt.reason = "awaiting_owner_test", "edits_invalidated_old_spectrum"
                return ""
            if previous_key == key:
                receipt.status, receipt.reason = "source_changed", "unobserved_source_drift"
                return ""
            if (task.test_command not in (["python", "-m", "pytest", "-q"], ["python3", "-m", "pytest", "-q"], ["pytest", "-q"])
                    or "python" not in task.policy.allowed_test_executables or task.policy.network_enabled):
                receipt.status, receipt.reason = "unsupported", "owner_command_or_policy_unsupported"
                return ""
            if self.refreshes >= task.policy.fault_localization_max_refreshes:
                receipt.status, receipt.reason = "budget_exhausted", "refresh_cap_reached"
                return ""
            self.refreshes += 1
            sources = await blocking_safely(lambda: production_sources(sandbox.workspace, task.policy))
            if not sources:
                receipt.status, receipt.reason = "insufficient_evidence", "no_python_production_sources"
                return ""
            end = min(deadline, started + task.policy.fault_localization_timeout_seconds)
            trials = []
            for _ in range(2):
                if time.perf_counter() + min(30, task.policy.command_timeout_seconds) + 30 >= end:
                    receipt.status, receipt.reason = "budget_exhausted", "planning_and_cleanup_reserve"
                    return ""
                trial = await collect_spectrum(task, sandbox, sources, self.sandbox_factory)
                trials.append(trial)
                receipt.trial_sha256.append(trial[3])
            after = await blocking_safely(lambda: repository_fingerprint(sandbox.workspace.root, task.policy.max_files, task.policy.max_repository_bytes))
            if after != current_hash:
                receipt.status, receipt.reason = "source_changed", "source_drift_during_probe"
                return ""
            if trials[0][3] != trials[1][3]:
                receipt.status, receipt.reason = "unstable", "test_or_spectrum_repeat_mismatch"
                return ""
            data, failed, passed, _ = trials[0]
            receipt.failing_tests, receipt.passing_tests = failed, passed
            receipt.spectrum_sha256 = fingerprint(data["spectra"])
            if not failed or not passed or any(status in {"skipped", "error"} for status in data["tests"].values()):
                receipt.status, receipt.reason = "insufficient_evidence", "need_clean_passing_and_failing_tests"
                return ""
            ranked = rank_lines(data["spectra"], sources, failed)
            context, chosen, unsafe = pack_lines(ranked, sources, task.policy.fault_localization_max_tokens)
            receipt.ranked_lines = chosen
            if unsafe:
                receipt.status, receipt.reason = "unsafe_context", "tainted_source_snippet_withheld"
            elif not context:
                receipt.status, receipt.reason = "insufficient_evidence", "no_rankable_lines_within_budget"
            else:
                self.context = context
                receipt.status, receipt.reason = "selected", "repeated_ochiai_association"
                receipt.context_sha256 = fingerprint(context)
                receipt.approximate_tokens = (len(context) + 3) // 4
            return self.context
        except (DockerUnavailableError, InstrumentationUnavailable):
            receipt.status, receipt.reason = "unavailable", "docker_or_instrumentation_unavailable"
            self.context = ""
            return ""
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError):
            self.context = ""
            return ""
        finally:
            # Cached reads don't create another exposure attempt or spend a refresh.
            receipt.duration_ms = (time.perf_counter() - started) * 1000
            if not cached:
                self.receipts.append(receipt)
