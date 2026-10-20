"""Execution-guided reduction of one structured counterexample, with isolated confirmation."""

from __future__ import annotations

import time

from code_agent.patch_json import REDUCER, complexity, reduction_candidates
from code_agent.upgrade_models import fingerprint


async def reduce_counterexample(domain, parent, policy, evaluate):
    """evaluate(cases) executes fresh original/candidate snapshots twice each.

    Search batches are exploratory; only a fresh single-case confirmation can
    certify a reported input. Budget exhaustion never becomes a passing result.
    """
    started = time.monotonic()
    deadline = started + policy.max_seconds
    record = {
        "reducer": REDUCER,
        "domain_id": domain.domain_id,
        "parent_case_id": parent.case_id,
        "parent_input_sha256": fingerprint(parent.model_dump()),
        "status": "unconfirmed",
        "minimality": "bounded_greedy_not_global",
        "rounds": [],
        "confirmations": [],
        "confirmed_case": None,
        "input_bytes_before": complexity(domain.value(parent))[0],
        "input_bytes_after": None,
        "search_candidates_submitted": 0,
        "deadline_mode": "admission_with_cleanup_settlement",
    }

    async def confirm(case):
        if time.monotonic() >= deadline:
            return False
        result = await evaluate([case], deadline)
        record["confirmations"].append({"case": case.model_dump(), **result})
        return result["status"] == "regression_found" and case.case_id in result["changed"]

    if not await confirm(parent):
        record["duration_ms"] = (time.monotonic() - started) * 1000
        record["fingerprint"] = fingerprint(record)
        return record
    current = parent
    for number in range(policy.max_rounds if policy.enabled else 0):
        # Keep a reserve for the final isolated confirmation. An in-flight
        # operation is settled before cleanup, so this is admission, not an SLA.
        if time.monotonic() + 20 >= deadline:
            record["status"] = "budget_exhausted"
            break
        values = reduction_candidates(domain.input_schema, domain.value(current), policy.max_candidates)
        if not values:
            break
        candidates = [
            domain.case(value, f"generated.reduce.{fingerprint(domain.domain_id)[:12]}.{number}.{index}")
            for index, value in enumerate(values)
        ]
        result = await evaluate(candidates, deadline)
        record["search_candidates_submitted"] += len(candidates)
        record["rounds"].append({"cases": [case.model_dump() for case in candidates], **result})
        if result["status"] != "regression_found":
            break
        hits = [case for case in candidates if case.case_id in result["changed"]]
        if not hits:
            break
        candidate = min(hits, key=lambda case: complexity(domain.value(case)))
        # Batch order/module state cannot establish a standalone reproducer.
        if not await confirm(candidate):
            break
        current = candidate
    record["confirmed_case"] = current.model_dump()
    record["input_bytes_after"] = complexity(domain.value(current))[0]
    if record["status"] != "budget_exhausted":
        record["status"] = (
            "reduced"
            if complexity(domain.value(current)) < complexity(domain.value(parent))
            else "confirmed_unchanged"
        )
    record["duration_ms"] = (time.monotonic() - started) * 1000
    record["fingerprint"] = fingerprint(record)
    return record
