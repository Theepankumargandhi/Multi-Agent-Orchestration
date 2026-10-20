"""Isolated old/new execution, repeated observations and bounded counterexample shrinking."""
from __future__ import annotations

import asyncio

from code_agent.delivery import digest, manifest
from code_agent.sandbox import DockerSandbox
from code_agent.upgrade_contracts import parse_probe
from code_agent.upgrade_inputs import CounterexamplePolicy, complexity, input_key, shrink_candidates
from code_agent.upgrade_models import (
    CONTRACT_PATH,
    PROBE_PATH,
    SOURCE_VERSION,
    TARGET_VERSION,
    V1_PYTHON,
    V2_PYTHON,
)


async def settled_call(function, *args):
    """Cancellation cannot race a still-running Docker operation with resource cleanup."""
    pending = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        await asyncio.gather(pending, return_exceptions=True)
        raise


def _batches(cases, spec):
    batch = []
    for case in cases:
        prospective = [*batch, case]
        # Leave room for profile metadata; probe's aggregate input ceiling is 40 KB.
        size = sum(len(row.model_dump_json().encode()) for row in prospective)
        capacity = min(8, max(1, (spec.command_timeout_seconds - 1) // 2))
        if batch and (len(prospective) > capacity or size > 30000):
            yield batch
            batch = []
        batch.append(case)
    if batch:
        yield batch


class DifferentialVerifier:
    """Separate workspaces: generated inputs never enter the repair agent's filesystem."""
    def __init__(self, original, candidate, spec, sandbox_policy):
        self.spec = spec
        self.source = DockerSandbox(original, sandbox_policy)
        self.target = DockerSandbox(candidate, sandbox_policy)
        self.commands = 0

    async def __aenter__(self):
        try:
            await settled_call(self.source.start)
            await settled_call(self.target.start)
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *_args):
        try:
            await settled_call(self.target.close)
        finally:
            await settled_call(self.source.close)

    async def _observe(self, sandbox, cases, version):
        recipe = self.spec.model_copy(update={"contracts": list(cases), "accepted_differences": []})
        # Validate the complete batch rather than relying on model_copy validation.
        recipe = type(self.spec).model_validate(recipe.model_dump())
        sandbox.workspace.write_file(CONTRACT_PATH, recipe.model_dump_json())
        before = digest(manifest(sandbox.workspace))
        results = []
        for _ in range(2):
            interpreter = V1_PYTHON if version == SOURCE_VERSION else V2_PYTHON
            self.commands += 1
            result = await settled_call(sandbox.run, [interpreter, "-I", PROBE_PATH, "/workspace/" + CONTRACT_PATH, version])
            if result.exit_code or result.timed_out or digest(manifest(sandbox.workspace)) != before:
                raise ValueError("generated probe unavailable or changed its workspace")
            results.append(parse_probe(result.stdout, recipe, version)["outcomes"])
        if results[0] != results[1]:
            raise ValueError("generated observations unstable across fresh processes")
        return results[0]

    async def compare(self, cases):
        rows = []
        for batch in _batches(cases, self.spec):
            old = await self._observe(self.source, batch, SOURCE_VERSION)
            new = await self._observe(self.target, batch, TARGET_VERSION)
            for case, source, target in zip(batch, old, new, strict=True):
                rows.append({"case": case, "source": source["result"], "target": target["result"],
                    "source_sha256": source["sha256"], "target_sha256": target["sha256"],
                    "regression": source["sha256"] != target["sha256"]})
        return rows

    async def reduce(self, row, budget, excluded_inputs):
        original, current, evaluated = row, row, 0
        while evaluated < budget:
            candidates = [case for case in shrink_candidates(current["case"], 512)
                if input_key(case) not in excluded_inputs][:min(8, budget - evaluated)]
            if not candidates:
                break
            trials = await self.compare(candidates)
            evaluated += len(trials)
            # Preserve the observed failure category, not just any incidental exception.
            def signature(trial):
                return tuple((trial[key]["kind"], trial[key].get("type")) for key in ("source", "target"))
            witnesses = [trial for trial in trials if trial["regression"] and signature(trial) == signature(original)]
            if not witnesses:
                break
            current = min(witnesses, key=lambda trial: complexity(trial["case"]))
        standalone = (await self.compare([current["case"]]))[0]
        if not standalone["regression"] or any(
            (standalone[key]["kind"], standalone[key].get("type")) != (original[key]["kind"], original[key].get("type"))
            for key in ("source", "target")
        ):
            raise ValueError("counterexample did not reproduce independently of batch order")
        current = standalone
        return {"input": current["case"].model_dump(), "source": current["source"], "target": current["target"],
            "discovery_case_id": original["case"].case_id, "standalone_reproduced": True,
            "source_sha256": current["source_sha256"], "target_sha256": current["target_sha256"],
            "original_input_bytes": complexity(original["case"])[0], "reduced_input_bytes": complexity(current["case"])[0],
            "shrink_evaluations": evaluated, "budget_exhausted": evaluated >= budget,
            "minimality": "bounded_reduction_only_not_global_minimum"}


async def verify_generated(original, candidate, spec, sandbox_policy, cases, *, policy: CounterexamplePolicy, shrink=False,
                           excluded_inputs=frozenset()):
    if not cases:
        raise ValueError("empty generated evidence is not a passing evaluation")
    async with DifferentialVerifier(original, candidate, spec, sandbox_policy) as verifier:
        rows = await verifier.compare(cases)
        failures = sorted((row for row in rows if row["regression"]), key=lambda row: complexity(row["case"]), reverse=True)
        examples = []
        if shrink:
            for row in failures[:policy.max_counterexamples]:
                examples.append(await verifier.reduce(row, policy.max_shrink_evaluations, excluded_inputs))
        return {"passed": not failures, "total": len(rows), "regressions": len(failures),
            "source_value_cases": sum(row["source"]["kind"] == "value" for row in rows),
            "source_error_cases": sum(row["source"]["kind"] == "error" for row in rows),
            "target_value_cases": sum(row["target"]["kind"] == "value" for row in rows),
            "target_error_cases": sum(row["target"]["kind"] == "error" for row in rows),
            "preserved": len(rows) - len(failures), "controller_commands": verifier.commands,
            "cases": [{"case_id": row["case"].case_id, "source_sha256": row["source_sha256"],
                "target_sha256": row["target_sha256"], "regression": row["regression"]} for row in rows],
            "counterexamples": examples, "stability_repeats": 2}
