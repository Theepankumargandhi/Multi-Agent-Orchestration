"""Image-owned Pydantic validation-schema extraction. Never execute this on the host."""
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
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class Discard(io.TextIOBase):
    def write(self, value):
        return len(value)


def deadline(*_args):
    raise TimeoutError("schema deadline")


def main():
    import pydantic
    expected = sys.argv[2]
    if pydantic.__version__ != expected:
        raise RuntimeError("dependency identity mismatch")
    raw = Path(sys.argv[1]).read_bytes()
    if len(raw) > 8000:
        raise ValueError("schema policy byte limit")
    policy = json.loads(raw)
    if not 1 <= len(policy["targets"]) <= 4:
        raise ValueError("schema model limit")
    signal.signal(signal.SIGALRM, deadline)
    sys.path.insert(0, "/workspace")
    rows = []
    for target in policy["targets"]:
        signal.alarm(3)
        try:
            with contextlib.redirect_stdout(Discard()), contextlib.redirect_stderr(Discard()):
                module, name = target["model"].split(":", 1)
                model = getattr(importlib.import_module(module), name)
                if not isinstance(model, type) or not issubclass(model, pydantic.BaseModel) or model.__module__ != module:
                    raise TypeError("explicit locally defined Pydantic model required")
                schema = model.schema(by_alias=True) if expected.startswith("1.") else model.model_json_schema(by_alias=True, mode="validation")
                if not isinstance(schema, dict) or len(canonical(schema).encode()) > 16000:
                    raise ValueError("schema byte limit")
            row = {"model": target["model"], "schema": schema, "sha256": hashlib.sha256(canonical(schema).encode()).hexdigest()}
        finally:
            signal.alarm(0)
        rows.append(row)
    result = canonical({"protocol": "upgrade-model-schema-v1", "pydantic_version": expected, "models": rows})
    if len(result.encode()) > 60000:
        raise ValueError("aggregate schema limit")
    print(result)


if __name__ == "__main__":
    main()
