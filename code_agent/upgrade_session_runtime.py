"""Attested, repeatable stateful replay and bounded ordered-trace counterexamples."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from code_agent.delivery import digest, manifest
from code_agent.upgrade_contracts import _unique_pairs, result_hash
from code_agent.upgrade_differential import DifferentialVerifier, settled_call
from code_agent.upgrade_models import SOURCE_VERSION, TARGET_VERSION, V1_PYTHON, V2_PYTHON
from code_agent.upgrade_sessions import (
    SESSION_CONFIG,
    SESSION_PROBE,
    SESSION_PROTOCOL,
    SessionScenario,
    session_key,
)


def parse_session(stdout, scenario, version):
    if len(stdout.encode()) > 65000:
        raise ValueError("session output byte budget exceeded")
    data = json.loads(stdout, object_pairs_hook=_unique_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite session JSON")))
    if (not isinstance(data, dict) or set(data) != {"protocol", "pydantic_version", "scenario_id", "steps"}
            or data["protocol"] != SESSION_PROTOCOL or data["pydantic_version"] != version
            or data["scenario_id"] != scenario.scenario_id or not isinstance(data["steps"], list)
            or len(data["steps"]) != len(scenario.steps)):
        raise ValueError("session protocol or membership mismatch")
    for index, (row, step) in enumerate(zip(data["steps"], scenario.steps, strict=True)):
        if (not isinstance(row, dict) or set(row) != {"index", "method", "result", "sha256"}
                or type(row["index"]) is not int or row["index"] != index or row["method"] != step.method
                or not isinstance(row["result"], dict) or row["sha256"] != result_hash(row["result"])):
            raise ValueError("session trace integrity mismatch")
        result = row["result"]
        if result.get("kind") == "value" and set(result) == {"kind", "value"}:
            continue
        if result.get("kind") == "error" and set(result) == {"kind", "type"} and isinstance(result["type"], str):
            continue
        raise ValueError("session harness unavailable")
    return data["steps"]


def compare_sessions(scenario, source, target):
    differences = [index for index, (old, new) in enumerate(zip(source, target, strict=True)) if old["sha256"] != new["sha256"]]
    first = differences[0] if differences else None
    return {"scenario_id": scenario.scenario_id, "passed": not differences, "steps": len(source),
        "different_steps": len(differences), "first_divergence": None if first is None else {
            "index": first, "method": scenario.steps[first].method, "source": source[first]["result"],
            "target": target[first]["result"], "source_sha256": source[first]["sha256"], "target_sha256": target[first]["sha256"]},
        "source_trace_sha256": result_hash(source), "target_trace_sha256": result_hash(target),
        "source_value_steps": sum(row["result"]["kind"] == "value" for row in source),
        "source_error_steps": sum(row["result"]["kind"] == "error" for row in source),
        "target_error_steps": sum(row["result"]["kind"] == "error" for row in target)}


class SessionReader:
    """Reusable single-copy observer; imports and execution remain inside Docker."""
    def __init__(self, spec):
        self.spec, self.commands = spec, 0

    async def attest_one(self, sandbox):
        # Bind both the stateless and session harnesses to reviewed controller source.
        paths = (("upgrade_session_probe.py", SESSION_PROBE), ("upgrade_probe.py", "/opt/upgradepilot/probe.py"))
        for local, installed in paths:
            expected = hashlib.sha256(Path(__file__).with_name(local).read_bytes()).hexdigest()
            code = f"import hashlib;print(hashlib.sha256(open({installed!r},'rb').read()).hexdigest())"
            self.commands += 1
            result = await settled_call(sandbox.run, [V2_PYTHON, "-I", "-c", code])
            if result.exit_code or result.timed_out or result.stdout.strip() != expected:
                raise ValueError("session harness identity unavailable; rebuild the session image")

    async def observe(self, sandbox, scenario, version):
        if self.spec.command_timeout_seconds < 2 + len(scenario.steps):
            raise ValueError("command budget cannot contain the entire session")
        sandbox.workspace.write_file(SESSION_CONFIG, scenario.model_dump_json())
        before = digest(manifest(sandbox.workspace))
        observations = []
        interpreter = V1_PYTHON if version == SOURCE_VERSION else V2_PYTHON
        for _ in range(2):
            self.commands += 1
            result = await settled_call(sandbox.run, [interpreter, "-I", SESSION_PROBE, "/workspace/" + SESSION_CONFIG, version])
            if result.exit_code or result.timed_out or digest(manifest(sandbox.workspace)) != before:
                raise ValueError("session unavailable or changed its workspace")
            observations.append(parse_session(result.stdout, scenario, version))
        if observations[0] != observations[1]:
            raise ValueError("session observations unstable across fresh processes")
        return observations[0]


class SessionVerifier(DifferentialVerifier, SessionReader):
    async def attest(self):
        for sandbox in (self.source, self.target):
            await self.attest_one(sandbox)

    async def compare_one(self, scenario):
        old = await self.observe(self.source, scenario, SOURCE_VERSION)
        new = await self.observe(self.target, scenario, TARGET_VERSION)
        return compare_sessions(scenario, old, new)

    async def reduce_trace(self, scenario, original, limit, excluded):
        current, diagnosis, evaluated = scenario, original, 0

        def signature(row):
            first = row["first_divergence"]
            return (first["method"], *((first[key]["kind"], first[key].get("type")) for key in ("source", "target")))

        # Greedy ordered deletion: bounded, not a global minimality claim.
        while len(current.steps) > 1 and evaluated < limit:
            progress = False
            for index in range(len(current.steps)):
                candidate = SessionScenario.model_validate({**current.model_dump(),
                    "steps": [step.model_dump() for number, step in enumerate(current.steps) if number != index]})
                if session_key(candidate) in excluded:
                    continue
                row = await self.compare_one(candidate)
                evaluated += 1
                if not row["passed"] and signature(row) == signature(original):
                    current, diagnosis, progress = candidate, row, True
                    break
                if evaluated >= limit:
                    break
            if not progress:
                break
        # Every compare_one already replays independently twice per version.
        return {"scenario": current.model_dump(), "first_divergence": diagnosis["first_divergence"],
            "original_steps": len(scenario.steps), "reduced_steps": len(current.steps), "shrink_evaluations": evaluated,
            "fresh_process_repeats_per_version": 2, "minimality": "bounded_ordered_deletion_not_global_minimum"}


async def verify_sessions(original, candidate, spec, sandbox_policy, scenarios, *, policy, shrink=False):
    if not scenarios:
        raise ValueError("empty session evidence cannot pass")
    async with SessionVerifier(original, candidate, spec, sandbox_policy) as verifier:
        await verifier.attest()
        rows, examples = [], []
        excluded = {session_key(row) for row in policy.final}
        for scenario in scenarios:
            row = await verifier.compare_one(scenario)
            if not row["source_value_steps"]:
                raise ValueError("all-error source sessions are not positive behavioral evidence")
            rows.append(row)
            if shrink and not row["passed"] and len(examples) < policy.max_counterexamples:
                examples.append(await verifier.reduce_trace(scenario, row, policy.max_shrink_evaluations, excluded))
        return {"passed": all(row["passed"] for row in rows), "total": len(rows),
            "regressions": sum(not row["passed"] for row in rows), "scenarios": rows, "counterexamples": examples,
            "controller_commands": verifier.commands, "fresh_process_repeats_per_version": 2,
            "isolation": "new_process_per_scenario_and_version", "coverage_claim": "owner_authored_sequences_only"}
