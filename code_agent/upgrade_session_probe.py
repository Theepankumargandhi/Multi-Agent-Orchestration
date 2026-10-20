"""Image-owned session harness. Execute ONLY in Docker, never on the host."""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import signal
import sys
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate session JSON keys")
        result[key] = value
    return result


class DiscardOutput(io.TextIOBase):
    def write(self, value):
        return len(value)


class Deadline(BaseException):
    pass


def timeout(_signum, _frame):
    raise Deadline()


def main():
    import pydantic
    if pydantic.__version__ != sys.argv[2]:
        raise RuntimeError("dependency identity mismatch")
    raw = Path(sys.argv[1]).read_bytes()
    if len(raw) > 12000:
        raise ValueError("session input byte limit")
    scenario = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite session JSON")))
    if not 1 <= len(scenario["steps"]) <= 12:
        raise ValueError("session step count")
    sys.path.insert(0, "/workspace")
    signal.signal(signal.SIGALRM, timeout)
    rows = []
    with contextlib.redirect_stdout(DiscardOutput()), contextlib.redirect_stderr(DiscardOutput()):
        signal.alarm(2)
        try:
            module, name = scenario["factory"].split(":", 1)
            session = getattr(importlib.import_module(module), name)(*scenario["args"], **scenario["kwargs"])
        finally:
            signal.alarm(0)
        for index, step in enumerate(scenario["steps"]):
            phase = "method"
            signal.alarm(1)
            try:
                function = getattr(session, step["method"])
                if not callable(function):
                    raise TypeError("session operation is not callable")
                phase = "call"
                value = function(*step["args"], **step["kwargs"])
                phase = "serialize"
                encoded = canonical({"kind": "value", "value": value})
                if len(encoded.encode()) > 8000:
                    raise ValueError("session result byte limit")
                # Snapshot now: later mutations must not rewrite an earlier observation.
                result = json.loads(encoded)
            except Exception as exc:
                if phase != "call":
                    raise RuntimeError("session harness unavailable") from None
                result = {"kind": "error", "type": type(exc).__name__}
                encoded = canonical(result)
            finally:
                signal.alarm(0)
            rows.append({"index": index, "method": step["method"], "result": result,
                "sha256": hashlib.sha256(encoded.encode()).hexdigest()})
            if len(canonical(rows).encode()) > 60000:
                raise ValueError("session output byte limit")
    print(canonical({"protocol": "upgrade-session-v1", "pydantic_version": pydantic.__version__,
        "scenario_id": scenario["scenario_id"], "steps": rows}))


if __name__ == "__main__":
    main()
