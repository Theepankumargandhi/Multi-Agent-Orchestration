"""Image-owned differential probe harness. Application imports happen only inside Docker."""
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
            raise ValueError("duplicate JSON keys")
        result[key] = value
    return result


def timeout(_signum, _frame):
    raise TimeoutError("contract deadline")


class DiscardOutput(io.TextIOBase):
    def write(self, value):
        return len(value)


def main():
    import pydantic
    expected = sys.argv[2]
    if pydantic.__version__ != expected:
        raise RuntimeError("dependency identity mismatch")
    raw = Path(sys.argv[1]).read_bytes()
    if len(raw) > 40000:
        raise ValueError("contract byte limit")
    spec = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not 1 <= len(spec["contracts"]) <= 50:
        raise ValueError("contract count limit")
    sys.path.insert(0, "/workspace")
    signal.signal(signal.SIGALRM, timeout)
    outcomes = []
    for case in spec["contracts"]:
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
                encoded = canonical(result)
                if len(encoded.encode()) > 16000:
                    raise ValueError("result size limit")
        except Exception as exc:
            result = {"kind": "error" if phase == "call" and not isinstance(exc, TimeoutError) else "harness_error", "type": type(exc).__name__}
            encoded = canonical(result)
        finally:
            signal.alarm(0)
        outcomes.append({"case_id": case["case_id"], "result": result,
                         "sha256": hashlib.sha256(encoded.encode()).hexdigest()})
        if len(canonical(outcomes).encode()) > 60000:
            raise ValueError("aggregate contract output limit")
    print(canonical({"protocol": "upgradepilot-probe-v1", "pydantic_version": pydantic.__version__, "outcomes": outcomes}))


if __name__ == "__main__":
    main()
