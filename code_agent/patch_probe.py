"""Controller-owned harness, copied into Docker; never imports submitted code on the host."""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import os
import signal
import sys
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class DiscardOutput(io.TextIOBase):
    def write(self, value):
        return len(value)


def timeout(_signum, _frame):
    raise TimeoutError("behavior probe deadline")


def behavior(cases):
    outcomes = {}
    signal.signal(signal.SIGALRM, timeout)
    for case in cases:
        signal.alarm(2)
        phase = "import"
        try:
            with contextlib.redirect_stdout(DiscardOutput()), contextlib.redirect_stderr(DiscardOutput()):
                module, name = case["entrypoint"].split(":", 1)
                function = getattr(importlib.import_module(module), name)
                phase = "call"
                value = function(*case["args"], **case["kwargs"])
                phase = "serialize"
                result = {"kind": "value", "value": value}
                if len(canonical(result).encode()) > 16_000:
                    raise ValueError("behavior result exceeds size limit")
        except Exception as exc:
            result = {"kind": "error" if phase == "call" and not isinstance(exc, TimeoutError) else "harness_error",
                      "type": type(exc).__name__}
        finally:
            signal.alarm(0)
        outcomes[case["case_id"]] = {"sha256": digest(result), "kind": result["kind"], "type": result.get("type")}
    return outcomes


def tests(paths):
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["PYTEST_ADDOPTS"] = ""
    import pytest
    collected, phases = [], {}
    collection_errors = []

    class Plugin:
        def pytest_collection_modifyitems(self, items):
            if len(items) > 500:
                pytest.exit("test count limit", returncode=3)
            collected.extend(item.nodeid for item in items)

        def pytest_collectreport(self, report):
            if report.failed or report.skipped:
                collection_errors.append(True)

        def pytest_runtest_logreport(self, report):
            node = phases.setdefault(report.nodeid, {})
            status = "passed" if report.passed and not hasattr(report, "wasxfail") else "not_passed"
            if report.when in node:
                status = "not_passed"
            node[report.when] = status

    with contextlib.redirect_stdout(DiscardOutput()), contextlib.redirect_stderr(DiscardOutput()):
        exit_code = int(pytest.main(["-q", "-p", "no:cacheprovider", "-o", "addopts=", *paths], plugins=[Plugin()]))
    passed = (exit_code == 0 and not collection_errors and 0 < len(collected) <= 500
              and len(collected) == len(set(collected)) and set(collected) == set(phases)
              and all(node == {"setup": "passed", "call": "passed", "teardown": "passed"} for node in phases.values()))
    return {"passed": passed, "exit_code": exit_code, "collected": len(collected),
            "node_ids_sha256": digest(sorted(collected)), "collection_errors": len(collection_errors)}


def main():
    request = json.loads(Path(".agentforge-verify-input.json").read_bytes())
    sys.path.insert(0, str(Path.cwd()))
    mode = sys.argv[1]
    result = tests(request["test_paths"]) if mode == "tests" else behavior(request["cases"])
    receipt = {"binding": request["binding"], "mode": mode, "result": result}
    Path(".agentforge-verify-receipt.json").write_text(canonical(receipt), encoding="utf-8")


if __name__ == "__main__":
    main()
