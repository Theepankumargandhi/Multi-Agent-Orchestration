"""Execute generated laws in disposable copies; final relations never enter repair files."""
from __future__ import annotations

from code_agent.sandbox import DockerSandbox
from code_agent.upgrade_differential import settled_call
from code_agent.upgrade_relations import assess_relation, relation_inventory
from code_agent.upgrade_session_runtime import SessionReader


async def verify_relations(repository, spec, sandbox_policy, cases, version, *, feedback_limit=0):
    if not cases:
        raise ValueError("empty relation evidence cannot pass")
    sandbox = DockerSandbox(repository, sandbox_policy)
    reader = SessionReader(spec)
    rows, feedback, cache = [], [], {}
    try:
        await settled_call(sandbox.start)
        await reader.attest_one(sandbox)
        for case in cases:
            traces = []
            for scenario in case.scenarios:
                # Cache only within this immutable copy, version and invocation; never across patches.
                key = scenario.model_dump_json()
                if key not in cache:
                    cache[key] = await reader.observe(sandbox, scenario, version)
                traces.append(cache[key])
            row = assess_relation(case, traces)
            rows.append(row)
            if not row["passed"] and len(feedback) < feedback_limit:
                feedback.append({"case": case.payload(), "violations": row["violations"],
                    "observations": [{"scenario_id": scenario.scenario_id, "steps": trace}
                        for scenario, trace in zip(case.scenarios, traces, strict=True)]})
    finally:
        await settled_call(sandbox.close)
    return {"passed": all(row["passed"] for row in rows), "total": len(rows),
        "violations": sum(not row["passed"] for row in rows), "cases": rows, "counterexamples": feedback,
        "property_inventory": relation_inventory(cases), "controller_commands": reader.commands,
        "fresh_process_repeats_per_scenario": 2, "dependency_version": version,
        "oracle": "owner_declared_metamorphic_laws_not_old_new_equality"}
